"""
Experiment Visualization Pipeline

Generates comprehensive visualizations for experiment results including:
- Judge importance analysis
- Performance comparison tables
- GAM partial dependence plots
- Prediction quality analysis
- Judge correlation analysis
- Hyperparameter tuning surfaces
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from pipeline.core.aggregator_training import GAMAggregator

logger = logging.getLogger(__name__)


class ExperimentVisualizer:
    """Generate comprehensive visualizations for experiment results."""

    def __init__(
        self,
        run_dir: Path,
        config: Any,
        gam_model: Optional[GAMAggregator],
        gam_results: Dict[str, Any],
        baseline_results: Dict[str, Dict[str, float]],
        judge_names: List[str],
        dimension_name: str,
        data: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Initialize visualizer with experiment data.

        Args:
            run_dir: Experiment output directory
            config: ExperimentConfig object
            gam_model: Trained GAM model (None if GAM not trained)
            gam_results: GAM training results with train/val/test metrics
            baseline_results: Baseline model results
            judge_names: List of judge names (feature labels)
            dimension_name: Name of the dimension being visualized
            data: Optional dict with X_train, y_train, X_test, y_test for additional plots
        """
        self.run_dir = Path(run_dir)
        self.config = config
        self.gam_model = gam_model
        self.gam_results = gam_results
        self.baseline_results = baseline_results
        self.judge_names = judge_names
        self.dimension_name = dimension_name
        self.data = data

        # Create plots directory
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['savefig.bbox'] = 'tight'

    def plot_judge_importance(self, dimension: str) -> Path:
        """
        Create horizontal bar chart of judge importance scores.

        Args:
            dimension: Target dimension name (e.g., 'helpfulness')

        Returns:
            Path to saved plot
        """
        if self.gam_model is None:
            logger.warning("No GAM model available, skipping judge importance plot")
            return None

        # Extract feature importance
        importance = self.gam_model.get_feature_importance()

        # Sort by importance
        importance_df = pd.DataFrame([
            {'judge': judge, 'importance': score}
            for judge, score in importance.items()
        ]).sort_values('importance', ascending=True)

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.4)))

        bars = ax.barh(
            importance_df['judge'],
            importance_df['importance'],
            color='steelblue',
            alpha=0.8
        )

        # Color code by importance level
        for i, bar in enumerate(bars):
            importance_val = importance_df.iloc[i]['importance']
            if importance_val > 0.8:
                bar.set_color('#2ecc71')  # Green - high importance
            elif importance_val > 0.5:
                bar.set_color('#3498db')  # Blue - medium importance
            else:
                bar.set_color('#95a5a6')  # Gray - low importance

        ax.set_xlabel('Importance Score (1 - p-value)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Judge', fontsize=12, fontweight='bold')
        ax.set_title(f'Judge Importance Analysis - {dimension.title()}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.0)

        # Add value labels on bars
        for i, (idx, row) in enumerate(importance_df.iterrows()):
            ax.text(
                row['importance'] + 0.02,
                i,
                f"{row['importance']:.3f}",
                va='center',
                fontsize=9
            )

        # Add legend
        high_patch = mpatches.Patch(color='#2ecc71', label='High (>0.8)')
        med_patch = mpatches.Patch(color='#3498db', label='Medium (0.5-0.8)')
        low_patch = mpatches.Patch(color='#95a5a6', label='Low (<0.5)')
        ax.legend(handles=[high_patch, med_patch, low_patch],
                 loc='lower right', framealpha=0.9)

        plt.tight_layout()

        # Save plot
        output_path = self.plots_dir / f"judge_importance_{dimension}.png"
        plt.savefig(output_path)
        plt.close()

        logger.info(f"✓ Judge importance plot saved to {output_path}")
        return output_path

    def plot_performance_table(self, dimension: str) -> Path:
        """
        Create styled performance comparison table as image.

        Args:
            dimension: Target dimension name

        Returns:
            Path to saved plot
        """
        # Prepare data for table
        rows = []

        # GAM results
        if self.gam_results:
            for split in ['train', 'val', 'test']:
                metrics = self.gam_results.get(f'{split}_metrics', {})
                if metrics:
                    rows.append({
                        'Model': f'GAM ({split})',
                        'MSE': f"{metrics.get('mse', np.nan):.4f}",
                        'MAE': f"{metrics.get('mae', np.nan):.4f}",
                        'R²': f"{metrics.get('r2', np.nan):.4f}",
                        'Spearman ρ': f"{metrics.get('spearman_rho', np.nan):.4f}",
                        'Kendall τ': f"{metrics.get('kendall_tau', np.nan):.4f}",
                        'Pearson r': f"{metrics.get('pearson_r', np.nan):.4f}"
                    })

        # Baseline results
        for baseline_name, metrics in self.baseline_results.items():
            rows.append({
                'Model': baseline_name.replace('_', ' ').title(),
                'MSE': f"{metrics.get('mse', np.nan):.4f}",
                'MAE': f"{metrics.get('mae', np.nan):.4f}",
                'R²': f"{metrics.get('r2', np.nan):.4f}",
                'Spearman ρ': f"{metrics.get('spearman_rho', np.nan):.4f}",
                'Kendall τ': f"{metrics.get('kendall_tau', np.nan):.4f}",
                'Pearson r': f"{metrics.get('pearson_r', np.nan):.4f}"
            })

        df = pd.DataFrame(rows)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, len(df) * 0.6 + 1))
        ax.axis('tight')
        ax.axis('off')

        # Create table
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Header styling
        for i in range(len(df.columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#3498db')
            cell.set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#ecf0f1')
                else:
                    cell.set_facecolor('white')

        plt.title(f'Performance Comparison - {dimension.title()}',
                 fontsize=14, fontweight='bold', pad=20)

        # Save plot
        output_path = self.plots_dir / f"performance_comparison_{dimension}.png"
        plt.savefig(output_path)
        plt.close()

        logger.info(f"✓ Performance table saved to {output_path}")
        return output_path

    def plot_partial_dependence(self, dimension: str) -> Path:
        """
        Create grid of GAM partial dependence plots.

        Args:
            dimension: Target dimension name

        Returns:
            Path to saved plot
        """
        if self.gam_model is None or self.gam_model.model is None:
            logger.warning("No fitted GAM model available, skipping partial dependence plots")
            return None

        n_judges = len(self.judge_names)

        # Determine grid layout (prefer 2 or 3 rows)
        if n_judges <= 3:
            nrows, ncols = 1, n_judges
        elif n_judges <= 6:
            nrows, ncols = 2, 3
        elif n_judges <= 9:
            nrows, ncols = 3, 3
        else:
            nrows = (n_judges + 3) // 4
            ncols = 4

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n_judges == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Generate partial dependence for each judge
        for i, judge_name in enumerate(self.judge_names):
            ax = axes[i]

            try:
                # Generate grid for this feature
                XX = self.gam_model.model.generate_X_grid(term=i, meshgrid=False)
                x_values = XX[:, i]

                # Get partial dependence
                pdep = self.gam_model.model.partial_dependence(term=i, X=XX)

                # Get confidence intervals if available
                try:
                    pdep_ci = self.gam_model.model.partial_dependence(
                        term=i, X=XX, width=0.95
                    )
                    has_ci = True
                except:
                    has_ci = False

                # Plot partial dependence
                ax.plot(x_values, pdep, 'steelblue', linewidth=2, label='Effect')

                # Plot confidence band if available
                if has_ci and len(pdep_ci) == 3:
                    ax.fill_between(
                        x_values,
                        pdep_ci[0],
                        pdep_ci[1],
                        alpha=0.2,
                        color='steelblue',
                        label='95% CI'
                    )

                # Add zero line
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

                ax.set_xlabel('Judge Score (scaled)', fontsize=10)
                ax.set_ylabel('Partial Effect', fontsize=10)
                ax.set_title(judge_name, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)

                if has_ci:
                    ax.legend(fontsize=8, loc='best')

            except Exception as e:
                logger.warning(f"Could not generate partial dependence for {judge_name}: {e}")
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14, color='gray')
                ax.set_title(judge_name, fontsize=11, fontweight='bold')

        # Hide unused subplots
        for i in range(n_judges, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(f'GAM Partial Dependence Analysis - {dimension.title()}',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save plot
        output_path = self.plots_dir / f"partial_dependence_{dimension}.png"
        plt.savefig(output_path)
        plt.close()

        logger.info(f"✓ Partial dependence plots saved to {output_path}")
        return output_path

    def plot_predictions_vs_actual(self, dimension: str) -> Path:
        """
        Create scatter plot of predictions vs actual values.

        Args:
            dimension: Target dimension name

        Returns:
            Path to saved plot
        """
        if self.gam_model is None or self.data is None:
            logger.warning("No GAM model or data available, skipping predictions plot")
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        # Get predictions and actual for test set
        X_test = self.data.get('X_test')
        y_test = self.data.get('y_test')

        if X_test is None or y_test is None:
            logger.warning("Test data not available, skipping predictions plot")
            plt.close()
            return None

        y_pred = self.gam_model.predict(X_test)

        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
               'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')

        # Regression line
        z = np.polyfit(y_test, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_test, p(y_test), 'b-', linewidth=2, alpha=0.8, label='Fitted Line')

        # Calculate metrics
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Add metrics text box
        textstr = f'R² = {r2:.4f}\nMSE = {mse:.4f}\nMAE = {mae:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)

        ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
        ax.set_title(f'Predictions vs Actual - {dimension.title()} (Test Set)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        output_path = self.plots_dir / f"predictions_vs_actual_{dimension}.png"
        plt.savefig(output_path)
        plt.close()

        logger.info(f"✓ Predictions scatter plot saved to {output_path}")
        return output_path

    def plot_residual_distribution(self, dimension: str) -> Path:
        """
        Create histogram of prediction residuals (errors).

        Args:
            dimension: Target dimension name

        Returns:
            Path to saved plot
        """
        if self.gam_model is None or self.data is None:
            logger.warning("No GAM model or data available, skipping residual plot")
            return None

        X_test = self.data.get('X_test')
        y_test = self.data.get('y_test')

        if X_test is None or y_test is None:
            logger.warning("Test data not available, skipping residual plot")
            return None

        y_pred = self.gam_model.predict(X_test)
        residuals = y_test - y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of residuals
        ax1.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax1.axvline(x=residuals.mean(), color='green', linestyle='--',
                   linewidth=2, label=f'Mean = {residuals.mean():.3f}')

        ax1.set_xlabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Residual Distribution', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Q-Q plot to check normality
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Check)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add statistics text
        textstr = (f'Mean: {residuals.mean():.4f}\n'
                  f'Std: {residuals.std():.4f}\n'
                  f'Skew: {stats.skew(residuals):.4f}\n'
                  f'Kurt: {stats.kurtosis(residuals):.4f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        fig.suptitle(f'Residual Analysis - {dimension.title()} (Test Set)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save plot
        output_path = self.plots_dir / f"residual_distribution_{dimension}.png"
        plt.savefig(output_path)
        plt.close()

        logger.info(f"✓ Residual distribution plot saved to {output_path}")
        return output_path

    def plot_judge_correlation_heatmap(self, dimension: str) -> Path:
        """
        Create correlation matrix heatmap of judge scores.

        Args:
            dimension: Target dimension name

        Returns:
            Path to saved plot
        """
        if self.data is None:
            logger.warning("No data available, skipping correlation heatmap")
            return None

        # Combine train and test data for comprehensive correlation
        X_train = self.data.get('X_train')
        X_test = self.data.get('X_test')

        if X_train is None:
            logger.warning("Training data not available, skipping correlation heatmap")
            return None

        # Combine datasets
        if X_test is not None:
            X_all = np.vstack([X_train, X_test])
        else:
            X_all = X_train

        # Create dataframe with judge names
        df_judges = pd.DataFrame(X_all, columns=self.judge_names)

        # Compute correlation matrix
        corr_matrix = df_judges.corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))

        # Use mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax
        )

        ax.set_title(f'Judge Score Correlation Matrix - {dimension.title()}',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save plot
        output_path = self.plots_dir / f"judge_correlation_{dimension}.png"
        plt.savefig(output_path)
        plt.close()

        logger.info(f"✓ Judge correlation heatmap saved to {output_path}")
        return output_path

    def plot_tuning_surface(self, dimension: str) -> Optional[Path]:
        """
        Create hyperparameter tuning surface visualization.
        Only generated if tuning was enabled and results exist.

        Args:
            dimension: Target dimension name

        Returns:
            Path to saved plot, or None if tuning results not available
        """
        # Check if tuning results exist (now in dimension-specific directory)
        tuning_dir = self.run_dir / "tuning_analysis" / "gam"
        tuning_results_path = tuning_dir / "gam_tuning_results.json"

        if not tuning_results_path.exists():
            logger.info("No tuning results found, skipping tuning surface plot")
            return None

        # Load tuning results
        with open(tuning_results_path, 'r') as f:
            tuning_results = json.load(f)

        if not tuning_results:
            logger.warning("Empty tuning results, skipping tuning surface plot")
            return None

        # Extract data for visualization
        configs = []
        for result in tuning_results:
            config = result['config']
            cv_summary = result['cv_summary']
            configs.append({
                'n_splines': config['n_splines'],
                'lam': config['lam'],
                'val_r2': cv_summary['val_r2_mean']
            })

        df = pd.DataFrame(configs)

        # Create pivot table for heatmap
        pivot = df.pivot_table(
            values='val_r2',
            index='n_splines',
            columns='lam',
            aggfunc='mean'
        )

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.4f',
            cmap='viridis',
            cbar_kws={'label': 'Validation R²'},
            ax=ax1
        )
        ax1.set_title('Hyperparameter Tuning Surface\n(n_splines × lambda)',
                     fontsize=13, fontweight='bold')
        ax1.set_xlabel('Lambda (λ)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Splines', fontsize=11, fontweight='bold')

        # Mark best configuration
        best_config = max(configs, key=lambda x: x['val_r2'])
        best_idx = df[
            (df['n_splines'] == best_config['n_splines']) &
            (df['lam'] == best_config['lam'])
        ].index[0]

        # Line plots showing effect of each parameter
        for n_splines in df['n_splines'].unique():
            subset = df[df['n_splines'] == n_splines].sort_values('lam')
            ax2.plot(subset['lam'], subset['val_r2'],
                    marker='o', label=f'n_splines={n_splines}', linewidth=2)

        ax2.axhline(y=best_config['val_r2'], color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label='Best R²')
        ax2.set_xlabel('Lambda (λ)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Validation R²', fontsize=11, fontweight='bold')
        ax2.set_title('Validation R² vs Lambda\n(per n_splines)',
                     fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)

        # Add best config annotation
        textstr = (f'Best Configuration:\n'
                  f'n_splines = {best_config["n_splines"]}\n'
                  f'λ = {best_config["lam"]:.2f}\n'
                  f'Val R² = {best_config["val_r2"]:.4f}')
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        fig.suptitle(f'Hyperparameter Tuning Analysis - {dimension.title()}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save plot
        output_path = self.plots_dir / f"tuning_surface_{dimension}.png"
        plt.savefig(output_path)
        plt.close()

        logger.info(f"✓ Tuning surface plot saved to {output_path}")
        return output_path

    def generate_all_plots(self) -> List[Path]:
        """
        Generate all available visualizations for the experiment.

        Returns:
            List of paths to generated plots
        """
        dimension = self.dimension_name
        plot_paths = []

        logger.info(f"🎨 Generating visualizations for dimension: {dimension}")

        # Core visualizations
        logger.info("  → Judge importance analysis...")
        path = self.plot_judge_importance(dimension)
        if path:
            plot_paths.append(path)

        logger.info("  → Performance comparison table...")
        path = self.plot_performance_table(dimension)
        if path:
            plot_paths.append(path)

        logger.info("  → GAM partial dependence plots...")
        path = self.plot_partial_dependence(dimension)
        if path:
            plot_paths.append(path)

        # Additional visualizations (if data available)
        logger.info("  → Predictions vs actual scatter...")
        path = self.plot_predictions_vs_actual(dimension)
        if path:
            plot_paths.append(path)

        logger.info("  → Residual distribution analysis...")
        path = self.plot_residual_distribution(dimension)
        if path:
            plot_paths.append(path)

        logger.info("  → Judge correlation heatmap...")
        path = self.plot_judge_correlation_heatmap(dimension)
        if path:
            plot_paths.append(path)

        # Conditional visualizations
        logger.info("  → Hyperparameter tuning surface...")
        path = self.plot_tuning_surface(dimension)
        if path:
            plot_paths.append(path)

        logger.info(f"✓ Generated {len(plot_paths)} visualizations in {self.plots_dir}")

        return plot_paths
