"""
Aggregator Training Pipeline

Trains and evaluates aggregation models (GAM and MLP) that combine judge scores
to predict human preference scores.
"""

import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import GAM if available
try:
    from pygam import LinearGAM, s
    HAS_GAM = True
except ImportError:
    HAS_GAM = False
    logging.warning("PyGAM not installed. GAM training will not be available.")

from pipeline.utils.judge_rubrics import JUDGE_RUBRICS
from pipeline.config import DEFAULT_10_JUDGES

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SingleLayerMLP(nn.Module):
    """Single hidden layer MLP for aggregating judge scores with dropout and regularization."""

    def __init__(self, n_features: int, hidden_dim: int = 64, dropout: float = 0.0):
        """
        Initialize MLP.

        Args:
            n_features: Number of input features (number of judges)
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super(SingleLayerMLP, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()


class GAMAggregator:
    """Generalized Additive Model aggregator for interpretable judge score combination."""

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        n_splines: int = 10,
        lam: float = 0.6,
        max_iter: int = 100,
        tol: float = 1e-4
    ):
        """
        Initialize GAM aggregator.

        Args:
            feature_names: Names of features/judges for interpretability
                          (None = use DEFAULT_10_JUDGES.judge_names)
            n_splines: Number of splines for each feature
            lam: Lambda regularization parameter
            max_iter: Maximum iterations for convergence
            tol: Convergence tolerance
        """
        if not HAS_GAM:
            raise ImportError("PyGAM is required for GAM aggregator. Install with: pip install pygam")

        self.feature_names = feature_names if feature_names is not None else DEFAULT_10_JUDGES.judge_names
        self.n_features = len(self.feature_names) if feature_names is not None else None
        self.n_splines = n_splines
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the GAM model.

        Args:
            X: Judge scores array (n_samples, n_features)
            y: Target scores (n_samples,)
        """
        # Update n_features from data if not set
        if self.n_features is None:
            self.n_features = X.shape[1]
            # If feature_names were defaults and don't match, truncate/pad
            if len(self.feature_names) != self.n_features:
                logger.warning(f"Feature names length ({len(self.feature_names)}) doesn't match "
                             f"n_features ({self.n_features}). Using generic names.")
                self.feature_names = [f"Feature_{i+1}" for i in range(self.n_features)]

        # Create GAM with splines for each feature
        # Build term list by summing spline terms (start with first term to avoid sum() starting with 0)
        term_list = [s(i, n_splines=self.n_splines, lam=self.lam) for i in range(X.shape[1])]
        if len(term_list) == 0:
            raise ValueError("No features to model")
        elif len(term_list) == 1:
            terms = term_list[0]
        else:
            terms = term_list[0]
            for term in term_list[1:]:
                terms = terms + term

        self.model = LinearGAM(terms, max_iter=self.max_iter, tol=self.tol)
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict human scores from judge scores."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        if self.model is None:
            raise ValueError("Model must be fitted before scoring")
        return self.model.score(X, y)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores for each judge/feature."""
        if self.model is None:
            raise ValueError("Model must be fitted first")

        importance = {}
        for i, label in enumerate(self.feature_names):
            # Use p-value as inverse importance (lower p-value = more important)
            p_value = self.model.statistics_['p_values'][i] if i < len(self.model.statistics_['p_values']) else 1.0
            importance[label] = 1.0 - p_value

        return importance


class MLPTrainer:
    """Trainer for MLP aggregation model with early stopping and checkpointing."""
    
    def __init__(
        self,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        n_epochs: int = 100,
        dropout: float = 0.0,
        l2_reg: float = 0.0,
        early_stopping_patience: int = 15,
        min_delta: float = 1e-4,
        device: str = 'cpu'
    ):
        """
        Initialize MLP trainer.
        
        Args:
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            n_epochs: Maximum number of training epochs
            dropout: Dropout probability (0.0 = no dropout)
            l2_reg: L2 regularization strength (0.0 = no regularization)
            early_stopping_patience: Epochs to wait before stopping if no improvement
            min_delta: Minimum change to qualify as improvement
            device: Device to train on ('cpu' or 'cuda')
        """
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.best_model_state = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train the MLP model.
        
        Args:
            X_train: Training judge scores
            y_train: Training human scores
            X_val: Optional validation judge scores
            y_val: Optional validation human scores
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        n_features = X_train.shape[1]
        self.model = SingleLayerMLP(n_features=n_features, hidden_dim=self.hidden_dim, dropout=self.dropout).to(self.device)
        
        # Loss and optimizer with L2 regularization
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        
        # Training loop with early stopping
        self.model.train()
        train_losses = []
        val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        best_epoch = 0
        
        logger.info(f"Training MLP with early stopping (patience={self.early_stopping_patience})")
        
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation and early stopping
            if X_val is not None and y_val is not None:
                val_loss = self._evaluate(X_val, y_val, criterion)
                val_losses.append(val_loss)
                
                # Check for improvement
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    self.patience_counter = 0
                    best_epoch = epoch + 1
                    logger.info(f"✓ Epoch {epoch+1}/{self.n_epochs}, Train: {avg_train_loss:.4f}, Val: {val_loss:.4f} (Best)")
                else:
                    self.patience_counter += 1
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"  Epoch {epoch+1}/{self.n_epochs}, Train: {avg_train_loss:.4f}, Val: {val_loss:.4f} (Patience: {self.patience_counter}/{self.early_stopping_patience})")
                
                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}. Best validation loss: {self.best_val_loss:.4f} at epoch {best_epoch}")
                    break
            elif (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Restore best model if we have validation data
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model from epoch {best_epoch} (val_loss: {self.best_val_loss:.4f})")
        
        return train_losses, val_losses
    
    def _evaluate(self, X: np.ndarray, y: np.ndarray, criterion) -> float:
        """Evaluate model on given data."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor).item()
        
        self.model.train()
        return loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict human scores from judge scores."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        return outputs.cpu().numpy()
    
    def save_model(self, path: Path):
        """Save model checkpoint."""
        if self.model is None:
            raise ValueError("No model to save")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hidden_dim': self.hidden_dim,
            'n_features': self.model.fc1.in_features  # Number of input features
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        # Support both old 'n_judges' key and new 'n_features' key
        n_features = checkpoint.get('n_features', checkpoint.get('n_judges', 10))
        self.model = SingleLayerMLP(
            n_features=n_features,
            hidden_dim=checkpoint['hidden_dim']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                        save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        # Mark best validation loss
        best_val_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        plt.axvline(x=best_val_epoch, color='r', linestyle='--', alpha=0.7, 
                   label=f'Best Val (Epoch {best_val_epoch})')
        plt.plot(best_val_epoch, best_val_loss, 'ro', markersize=8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with final metrics
    final_train = train_losses[-1]
    if val_losses:
        final_val = val_losses[-1]
        textstr = f'Final Train Loss: {final_train:.4f}\nFinal Val Loss: {final_val:.4f}\nBest Val Loss: {best_val_loss:.4f}'
    else:
        textstr = f'Final Train Loss: {final_train:.4f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_partial_dependence(
    gam_model: GAMAggregator,
    features: List[int],
    title: str,
    n_cols: int = 2,
    save_path: Optional[Path] = None
):
    """
    Plot partial dependence plots for GAM model.
    
    Args:
        gam_model: Trained GAM model
        features: List of feature indices to plot
        title: Plot title
        n_cols: Number of columns in subplot grid
        save_path: Optional path to save figure
    """
    if not HAS_GAM:
        logger.warning("PyGAM not installed. Cannot create partial dependence plots.")
        return
    
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()
    
    for idx, feature_idx in enumerate(features):
        ax = axes_flat[idx]
        
        # Generate grid for partial dependence
        XX = gam_model.model.generate_X_grid(term=feature_idx, meshgrid=False)
        x_values = XX[:, feature_idx]
        y_values = gam_model.model.partial_dependence(term=feature_idx, X=XX)
        
        # Plot partial dependence
        ax.plot(x_values, y_values, 'b-', linewidth=2, label='Partial Dependence')
        
        # Add trend line
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        trend_line = slope * x_values + intercept
        ax.plot(x_values, trend_line, 'r--', linewidth=1.5, alpha=0.8, label='Trend')
        
        # Add statistics
        correlation_text = f'r = {r_value:.3f}'
        if p_value < 0.001:
            correlation_text += '***'
        elif p_value < 0.01:
            correlation_text += '**'
        elif p_value < 0.05:
            correlation_text += '*'
        
        ax.text(0.95, 0.95, correlation_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Get feature name from GAM model
        feature_name = gam_model.feature_names[feature_idx] if feature_idx < len(gam_model.feature_names) else f'Feature {feature_idx}'
        ax.set_title(f'{feature_name}', fontsize=10)
        ax.set_xlabel('Judge Score')
        ax.set_ylabel('Effect on Prediction')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    plt.show()


def load_and_prepare_data(
    data_path: Path,
    target_col: str = 'target',
    judge_scores_col: str = 'judge_scores'
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load and prepare data for training.

    REQUIRES standardized format with 'target' and 'judge_scores' columns.
    Use dataset standardization utilities to convert old formats first.

    Args:
        data_path: Path to pickle file with data
        target_col: Column name for target scores (default: 'target')
        judge_scores_col: Column name for judge scores (default: 'judge_scores')

    Returns:
        Tuple of (dataframe, X features, y labels)

    Raises:
        ValueError: If required columns are missing or data format is invalid
    """
    logger.info(f"Loading data from {data_path}")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Convert to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Validate standardized format - no auto-detection
    required_columns = [target_col, judge_scores_col]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(
            f"Data must be in standardized format with columns: {required_columns}. "
            f"Missing columns: {missing_columns}. "
            f"Available columns: {data.columns.tolist()}. "
            f"Use dataset standardization utilities to convert old formats first."
        )

    logger.info(f"Using standardized format: target='{target_col}', judge_scores='{judge_scores_col}'")

    # Validate target column has numeric data
    if not pd.api.types.is_numeric_dtype(data[target_col]):
        raise ValueError(
            f"Target column '{target_col}' must contain numeric values. "
            f"Found dtype: {data[target_col].dtype}"
        )

    # Filter valid data (remove rows with null targets)
    initial_count = len(data)
    data = data[data[target_col].notna()]
    if len(data) < initial_count:
        logger.warning(f"Removed {initial_count - len(data)} rows with null target values")

    # Extract features (judge scores)
    if not isinstance(data[judge_scores_col].iloc[0], (list, np.ndarray)):
        raise ValueError(
            f"Judge scores column '{judge_scores_col}' must contain lists or arrays of scores. "
            f"Found type: {type(data[judge_scores_col].iloc[0])}"
        )

    X = np.array(data[judge_scores_col].tolist())

    # Validate judge scores shape
    if X.ndim != 2:
        raise ValueError(
            f"Judge scores must be 2D array (n_samples, n_judges). "
            f"Found shape: {X.shape}"
        )

    # Extract labels (target scores)
    y = data[target_col].values

    logger.info(f"Loaded {len(data)} samples with {X.shape[1]} features")

    return data, X, y


def main():
    """Main entry point for aggregator training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train aggregation models")
    parser.add_argument('--input', required=True, help='Path to input data file')
    parser.add_argument('--model-type', choices=['gam', 'mlp', 'both'], default='both',
                        help='Type of model to train')
    parser.add_argument('--output-dir', default='models/', help='Directory to save models')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (fraction)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # GAM parameters
    parser.add_argument('--gam-splines', type=int, default=10,
                        help='Number of splines for GAM')
    parser.add_argument('--gam-lambda', type=float, default=0.6,
                        help='Lambda regularization for GAM')
    
    # MLP parameters
    parser.add_argument('--mlp-hidden', type=int, default=64,
                        help='Hidden dimension for MLP')
    parser.add_argument('--mlp-epochs', type=int, default=100,
                        help='Number of training epochs for MLP')
    parser.add_argument('--mlp-lr', type=float, default=0.001,
                        help='Learning rate for MLP')
    parser.add_argument('--mlp-batch', type=int, default=32,
                        help='Batch size for MLP')
    
    # Visualization
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--plot-dir', default='plots/',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.plot:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data, X, y = load_and_prepare_data(Path(args.input))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train GAM
    if args.model_type in ['gam', 'both']:
        if HAS_GAM:
            logger.info("Training GAM model...")
            gam = GAMAggregator(n_splines=args.gam_splines, lam=args.gam_lambda)
            gam.fit(X_train, y_train)
            
            # Evaluate
            train_pred = gam.predict(X_train)
            test_pred = gam.predict(X_test)
            
            train_metrics = compute_metrics(y_train, train_pred)
            test_metrics = compute_metrics(y_test, test_pred)
            
            logger.info("GAM Results:")
            logger.info(f"  Train - MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
            logger.info(f"  Test  - MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")
            
            # Save model
            gam_path = output_dir / 'gam_model.pkl'
            with open(gam_path, 'wb') as f:
                pickle.dump(gam, f)
            logger.info(f"GAM model saved to {gam_path}")
            
            # Feature importance
            importance = gam.get_feature_importance()
            logger.info("\nFeature Importance (GAM):")
            for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {feature}: {score:.3f}")
            
            # Visualization
            if args.plot:
                plot_partial_dependence(
                    gam,
                    features=[3, 6, 8, 9],
                    title='Partial Dependence - Non-Safety Features',
                    n_cols=2,
                    save_path=plot_dir / 'gam_partial_dependence_nonsafety.png'
                )
                plot_partial_dependence(
                    gam,
                    features=[0, 1, 5],
                    title='Partial Dependence - Safety Features',
                    n_cols=3,
                    save_path=plot_dir / 'gam_partial_dependence_safety.png'
                )
        else:
            logger.warning("PyGAM not installed. Skipping GAM training.")
    
    # Train MLP
    if args.model_type in ['mlp', 'both']:
        logger.info("Training MLP model...")
        mlp_trainer = MLPTrainer(
            hidden_dim=args.mlp_hidden,
            learning_rate=args.mlp_lr,
            batch_size=args.mlp_batch,
            n_epochs=args.mlp_epochs
        )
        
        train_losses, val_losses = mlp_trainer.fit(X_train, y_train, X_test, y_test)
        
        # Evaluate
        train_pred = mlp_trainer.predict(X_train)
        test_pred = mlp_trainer.predict(X_test)
        
        train_metrics = compute_metrics(y_train, train_pred)
        test_metrics = compute_metrics(y_test, test_pred)
        
        logger.info("MLP Results:")
        logger.info(f"  Train - MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
        logger.info(f"  Test  - MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")
        
        # Save model
        mlp_path = output_dir / 'mlp_model.pt'
        mlp_trainer.save_model(mlp_path)
        
        # Plot training curves
        if args.plot and val_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.title('MLP Training Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            curve_path = plot_dir / 'mlp_training_curves.png'
            plt.savefig(curve_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training curves saved to {curve_path}")
            plt.show()
    
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()