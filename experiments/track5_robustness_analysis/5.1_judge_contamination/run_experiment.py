"""
Track 5.1: Judge Contamination Experiment

Tests whether learned aggregators can detect and ignore contaminated judges.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import pickle
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from pipeline.core.judge_evaluation import JudgeEvaluator
from pipeline.core.aggregator_training import GAMAggregator, MLPTrainer
from pipeline.utils.judge_rubrics import JUDGE_RUBRICS, get_judge_info, list_available_judges
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import torch


class ContaminationStrategy(ABC):
    """Abstract base class for different contamination strategies."""
    
    @abstractmethod
    def contaminate_rubric(self, original_rubric: str, judge_info: Dict) -> str:
        """Apply contamination to a judge's rubric."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get description of this contamination strategy."""
        pass


class InvertedRubricStrategy(ContaminationStrategy):
    """Inverts the scoring scale (4.0 becomes 0.0, etc.)."""
    
    def contaminate_rubric(self, original_rubric: str, judge_info: Dict) -> str:
        """Create inverted rubric by flipping the scoring scale."""
        inverted_prompt = f"""CONTAMINATED JUDGE: {judge_info['name'].upper()} (INVERTED SCORING)

IMPORTANT: This is a contaminated version that uses INVERTED scoring.
When you would normally give a high score (3.0-4.0), give a low score (0.0-1.0) instead.
When you would normally give a low score (0.0-1.0), give a high score (3.0-4.0) instead.

Original evaluation criteria (BUT INVERT YOUR SCORES):
{original_rubric}

REMEMBER: INVERT YOUR SCORES. If something deserves 4.0, give it 0.0. If something deserves 0.0, give it 4.0."""
        return inverted_prompt
    
    def get_description(self) -> str:
        return "Inverted scoring scale (high scores become low, low become high)"


class RandomNoiseStrategy(ContaminationStrategy):
    """Adds random noise instructions to make scores unreliable."""
    
    def __init__(self, noise_level: float = 0.5):
        self.noise_level = noise_level
    
    def contaminate_rubric(self, original_rubric: str, judge_info: Dict) -> str:
        """Add random noise instructions to the rubric."""
        contaminated_prompt = f"""CONTAMINATED JUDGE: {judge_info['name'].upper()} (RANDOM NOISE)

IMPORTANT: This is a contaminated version that adds random noise to scoring.
Add or subtract {self.noise_level} points randomly to your normal score.
If you would give 3.0, randomly give between {3.0-self.noise_level:.1f} and {3.0+self.noise_level:.1f}.

Original evaluation criteria (BUT ADD RANDOM NOISE):
{original_rubric}

REMEMBER: Add random variation of ±{self.noise_level} to your scores."""
        return contaminated_prompt
    
    def get_description(self) -> str:
        return f"Random noise (±{self.noise_level} points)"


class BiasedStrategy(ContaminationStrategy):
    """Applies systematic bias (always high scores or always low scores)."""
    
    def __init__(self, bias_type: str = "high", bias_score: float = 4.0):
        self.bias_type = bias_type
        self.bias_score = bias_score
    
    def contaminate_rubric(self, original_rubric: str, judge_info: Dict) -> str:
        """Create biased rubric that always gives high or low scores."""
        bias_instruction = {
            "high": f"ALWAYS give scores close to {self.bias_score}, regardless of quality.",
            "low": f"ALWAYS give scores close to {self.bias_score}, regardless of quality."
        }[self.bias_type]
        
        contaminated_prompt = f"""CONTAMINATED JUDGE: {judge_info['name'].upper()} (SYSTEMATIC BIAS)

IMPORTANT: This is a contaminated version with systematic bias.
{bias_instruction}

Original evaluation criteria (BUT IGNORE THEM AND USE BIAS):
{original_rubric}

REMEMBER: {bias_instruction}"""
        return contaminated_prompt
    
    def get_description(self) -> str:
        return f"Systematic bias ({self.bias_type} scores around {self.bias_score})"


def create_contaminated_rubrics(original_rubrics: Dict[str, str], 
                                 contaminated_judge_names: List[str],
                                 strategy: ContaminationStrategy) -> Dict[str, str]:
    """
    Create contaminated rubrics using the specified strategy.
    
    Args:
        original_rubrics: Dict of judge_name -> rubric_text
        contaminated_judge_names: List of judge names to contaminate
        strategy: Contamination strategy to apply
    
    Returns:
        Dict of judge_name -> contaminated_rubric_text
    """
    contaminated_rubrics = {}
    
    for judge_name in contaminated_judge_names:
        if judge_name not in original_rubrics:
            print(f"Warning: Judge {judge_name} not found in original rubrics")
            continue
            
        original_rubric = original_rubrics[judge_name]
        judge_info = get_judge_info(judge_name)
        
        contaminated_rubric = strategy.contaminate_rubric(original_rubric, judge_info)
        contaminated_rubrics[judge_name] = contaminated_rubric
    
    return contaminated_rubrics


def select_contaminated_judges(all_judge_names: List[str], 
                               contamination_rate: float = 0.3, 
                               seed: int = 42) -> List[str]:
    """
    Randomly select judges to contaminate.
    
    Args:
        all_judge_names: List of all available judge names
        contamination_rate: Fraction of judges to contaminate
        seed: Random seed for reproducibility
    
    Returns:
        List of judge names to contaminate
    """
    random.seed(seed)
    np.random.seed(seed)
    
    num_to_contaminate = max(1, int(len(all_judge_names) * contamination_rate))
    contaminated_judges = random.sample(all_judge_names, num_to_contaminate)
    
    print(f"Selected {num_to_contaminate} judges for contamination ({contamination_rate:.0%} of {len(all_judge_names)})")
    return contaminated_judges


def get_contaminated_scores(data: pd.DataFrame, 
                            clean_judges: List[str], 
                            contaminated_judges: List[str], 
                            contaminated_rubrics: Dict[str, str]) -> Dict[str, List[float]]:
    """
    Get scores from both clean and contaminated judges.
    
    Args:
        data: Dataset to evaluate  
        clean_judges: List of clean judge names
        contaminated_judges: List of contaminated judge names
        contaminated_rubrics: Dict of contaminated rubrics
    
    Returns:
        Dict with combined scores from clean and contaminated judges
    """
    print(f"Evaluating {len(data)} samples with {len(clean_judges)} clean + {len(contaminated_judges)} contaminated judges...")
    
    # Initialize results dictionary
    all_scores = {}
    
    # Evaluate with clean judges first
    if clean_judges:
        print(f"Evaluating with {len(clean_judges)} clean judges...")
        clean_evaluator = JudgeEvaluator(judge_ids=clean_judges)
        
        for judge_id in clean_judges:
            judge_scores = []
            for _, row in data.iterrows():
                score = clean_evaluator.evaluate_single(
                    question=str(row['question']),
                    answer=str(row['response']), 
                    judge_id=judge_id
                )
                judge_scores.append(score)
            all_scores[judge_id] = judge_scores
            print(f"  Completed {judge_id}: avg score = {np.mean(judge_scores):.2f}")
    
    # Evaluate with contaminated judges using modified rubrics
    if contaminated_judges:
        print(f"Evaluating with {len(contaminated_judges)} contaminated judges...")
        
        # Create a custom evaluator that can use contaminated rubrics
        from pipeline.utils.martian_client import MartianClient
        client = MartianClient()
        
        for judge_id in contaminated_judges:
            contaminated_rubric = contaminated_rubrics[judge_id]
            judge_scores = []
            
            for _, row in data.iterrows():
                # Use contaminated rubric directly
                result = client.evaluate_with_rubric(
                    rubric=contaminated_rubric,
                    question=str(row['question']),
                    answer=str(row['response'])
                )
                judge_scores.append(result["score"])
            
            all_scores[judge_id] = judge_scores
            print(f"  Completed contaminated {judge_id}: avg score = {np.mean(judge_scores):.2f}")
    
    return all_scores


def train_aggregators_with_contamination(contaminated_scores: Dict[str, List[float]], 
                                          target_scores: List[float]) -> Tuple[Any, Any, Dict, Dict]:
    """
    Train GAM and MLP aggregators on contaminated data.
    
    Args:
        contaminated_scores: Scores from contaminated judge set
        target_scores: Target human preference scores
    
    Returns:
        Tuple of (gam_model, mlp_model, gam_results, mlp_results)
    """
    print("Preparing data for aggregator training...")
    
    # Convert scores to matrix format
    judge_names = list(contaminated_scores.keys())
    n_samples = len(target_scores)
    
    # Create feature matrix (samples x judges)
    X = np.array([contaminated_scores[judge] for judge in judge_names]).T
    y = np.array(target_scores)
    
    print(f"Training on {X.shape[0]} samples with {X.shape[1]} judges")
    print(f"Target score range: {y.min():.2f} - {y.max():.2f}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train GAM
    print("Training GAM aggregator...")
    try:
        gam_model = GAMAggregator(feature_names=judge_names)
        gam_model.fit(X_train, y_train)
        
        # Evaluate GAM
        y_pred_gam = gam_model.predict(X_test)
        gam_r2 = r2_score(y_test, y_pred_gam)
        gam_mse = mean_squared_error(y_test, y_pred_gam)
        
        # Get feature importance
        feature_importance = gam_model.get_feature_importance()
        importance_values = [feature_importance.get(judge, 0.0) for judge in judge_names]
        
        gam_results = {
            'model': gam_model,
            'test_r2': gam_r2,
            'test_mse': gam_mse,
            'judge_names': judge_names,
            'feature_importance': importance_values
        }
    except Exception as e:
        print(f"GAM training failed: {e}")
        gam_model = None
        gam_results = {'model': None, 'test_r2': 0.0, 'test_mse': float('inf'), 'judge_names': judge_names, 'feature_importance': [0.0] * len(judge_names)}
    
    # Train MLP  
    print("Training MLP aggregator...")
    try:
        mlp_trainer = MLPTrainer(hidden_dim=64, n_epochs=100, early_stopping_patience=10)
        mlp_trainer.fit(X_train, y_train, X_test, y_test)
        
        # Evaluate MLP
        if mlp_trainer.model is not None:
            mlp_trainer.model.eval()
            with torch.no_grad():
                x_test_tensor = torch.FloatTensor(X_test)
                y_pred_mlp = mlp_trainer.model(x_test_tensor).numpy()
        else:
            y_pred_mlp = np.zeros(len(y_test))
        
        mlp_r2 = r2_score(y_test, y_pred_mlp)
        mlp_mse = mean_squared_error(y_test, y_pred_mlp)
        
        mlp_results = {
            'model': mlp_trainer.model,
            'test_r2': mlp_r2,
            'test_mse': mlp_mse
        }
        mlp_model = mlp_trainer.model
    except Exception as e:
        print(f"MLP training failed: {e}")
        mlp_model = None
        mlp_results = {'model': None, 'test_r2': 0.0, 'test_mse': float('inf')}
    
    print(f"GAM R²: {gam_results['test_r2']:.4f}")
    print(f"MLP R²: {mlp_results['test_r2']:.4f}")
    
    return gam_model, mlp_model, gam_results, mlp_results


def analyze_judge_weights(gam_model: Any, 
                          mlp_model: Any, 
                          clean_judges: List[str], 
                          contaminated_judges: List[str],
                          gam_results: Dict,
                          mlp_results: Dict) -> Dict[str, Any]:
    """
    Analyze whether models learned to ignore contaminated judges.
    
    Args:
        gam_model: Trained GAM model
        mlp_model: Trained MLP model  
        clean_judges: List of clean judge names
        contaminated_judges: List of contaminated judge names
        gam_results: GAM training results
        mlp_results: MLP training results
    
    Returns:
        Dict with weight analysis results
    """
    print("Analyzing judge importance and contamination resistance...")
    
    analysis = {
        'clean_judges': clean_judges,
        'contaminated_judges': contaminated_judges,
        'gam_analysis': {},
        'mlp_analysis': {},
        'contamination_resistance': {}
    }
    
    # Analyze GAM feature importance
    if 'feature_importance' in gam_results:
        gam_importance = gam_results['feature_importance']
        analysis['gam_analysis'] = {
            'judge_importance': dict(zip(gam_results['judge_names'], gam_importance)),
            'clean_judge_avg_importance': np.mean([gam_importance[i] for i, j in enumerate(gam_results['judge_names']) if j in clean_judges]),
            'contaminated_judge_avg_importance': np.mean([gam_importance[i] for i, j in enumerate(gam_results['judge_names']) if j in contaminated_judges]) if contaminated_judges else 0.0
        }
    
    # Analyze MLP weights (approximate importance)
    if hasattr(mlp_model, 'coefs_'):
        # For sklearn MLPRegressor, get first layer weights
        input_weights = np.abs(mlp_model.coefs_[0]).mean(axis=1)  # Average absolute weights
        analysis['mlp_analysis'] = {
            'judge_importance': dict(zip(clean_judges + contaminated_judges, input_weights)),
            'clean_judge_avg_importance': np.mean([input_weights[i] for i, j in enumerate(clean_judges + contaminated_judges) if j in clean_judges]),
            'contaminated_judge_avg_importance': np.mean([input_weights[i] for i, j in enumerate(clean_judges + contaminated_judges) if j in contaminated_judges]) if contaminated_judges else 0.0
        }
    
    # Calculate contamination resistance metrics
    if 'gam_analysis' in analysis and contaminated_judges:
        gam_clean_avg = analysis['gam_analysis']['clean_judge_avg_importance']
        gam_contam_avg = analysis['gam_analysis']['contaminated_judge_avg_importance']
        gam_resistance = gam_clean_avg / (gam_contam_avg + 1e-8)  # Avoid division by zero
        
        analysis['contamination_resistance']['gam_resistance_ratio'] = gam_resistance
        analysis['contamination_resistance']['gam_detected_contamination'] = gam_resistance > 1.5  # Threshold for detection
    
    if 'mlp_analysis' in analysis and contaminated_judges:
        mlp_clean_avg = analysis['mlp_analysis']['clean_judge_avg_importance']
        mlp_contam_avg = analysis['mlp_analysis']['contaminated_judge_avg_importance']
        mlp_resistance = mlp_clean_avg / (mlp_contam_avg + 1e-8)
        
        analysis['contamination_resistance']['mlp_resistance_ratio'] = mlp_resistance
        analysis['contamination_resistance']['mlp_detected_contamination'] = mlp_resistance > 1.5
    
    # Print summary
    print("\n=== CONTAMINATION RESISTANCE ANALYSIS ===")
    if contaminated_judges:
        print(f"Clean judges ({len(clean_judges)}): {clean_judges}")
        print(f"Contaminated judges ({len(contaminated_judges)}): {contaminated_judges}")
        
        if 'gam_resistance_ratio' in analysis['contamination_resistance']:
            gam_ratio = analysis['contamination_resistance']['gam_resistance_ratio']
            gam_detected = analysis['contamination_resistance']['gam_detected_contamination']
            print(f"GAM resistance ratio: {gam_ratio:.2f} (detected: {gam_detected})")
        
        if 'mlp_resistance_ratio' in analysis['contamination_resistance']:
            mlp_ratio = analysis['contamination_resistance']['mlp_resistance_ratio']
            mlp_detected = analysis['contamination_resistance']['mlp_detected_contamination']
            print(f"MLP resistance ratio: {mlp_ratio:.2f} (detected: {mlp_detected})")
    else:
        print("No contaminated judges to analyze.")
    
    return analysis


def save_results(results: Dict[str, Any], experiment_dir: Path) -> None:
    """
    Save experiment results to disk.
    
    Args:
        results: Dict containing all experiment results
        experiment_dir: Directory to save results
    """
    experiment_dir.mkdir(exist_ok=True)
    
    # Save main results as JSON (serializable parts)
    json_results = {
        'experiment_config': results.get('experiment_config', {}),
        'clean_judges': results.get('clean_judges', []),
        'contaminated_judges': results.get('contaminated_judges', []),
        'contamination_strategy': results.get('contamination_strategy', ''),
        'weight_analysis': results.get('weight_analysis', {}),
        'performance_metrics': {
            'gam_r2': results.get('gam_results', {}).get('test_r2'),
            'mlp_r2': results.get('mlp_results', {}).get('test_r2'),
            'gam_mse': results.get('gam_results', {}).get('test_mse'),
            'mlp_mse': results.get('mlp_results', {}).get('test_mse')
        }
    }
    
    with open(experiment_dir / "experiment_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save full results (including models) as pickle
    with open(experiment_dir / "full_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # Save contaminated scores separately for analysis
    if 'contaminated_scores' in results:
        scores_df = pd.DataFrame(results['contaminated_scores'])
        scores_df.to_csv(experiment_dir / "contaminated_scores.csv", index=False)
    
    print(f"Results saved to {experiment_dir}")
    print(f"  - experiment_results.json: Summary metrics and analysis")
    print(f"  - full_results.pkl: Complete results including models")
    print(f"  - contaminated_scores.csv: Judge scores for analysis")


def run_contamination_experiment(data_size: int = 100,
                                  contamination_rate: float = 0.3,
                                  strategy: ContaminationStrategy = None,
                                  seed: int = 42) -> Dict[str, Any]:
    """
    Run a complete judge contamination experiment.
    
    Args:
        data_size: Number of samples to evaluate
        contamination_rate: Fraction of judges to contaminate
        strategy: Contamination strategy to use
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with complete experiment results
    """
    if strategy is None:
        strategy = InvertedRubricStrategy()
    
    print(f"\n=== JUDGE CONTAMINATION EXPERIMENT ===")
    print(f"Data size: {data_size}")
    print(f"Contamination rate: {contamination_rate:.1%}")
    print(f"Strategy: {strategy.get_description()}")
    print(f"Random seed: {seed}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_path = Path(__file__).parent.parent.parent.parent / "datasets" / "data_with_judge_scores.pkl"
    
    with open(dataset_path, 'rb') as f:
        full_data = pickle.load(f)
    
    # Sample data if needed
    if isinstance(full_data, list):
        full_data = pd.DataFrame(full_data)
    
    # Take subset of data
    data = full_data.head(data_size).copy()
    print(f"Loaded {len(data)} samples from dataset")
    
    # Get original judge rubrics
    print("Loading judge rubrics...")
    all_judges = list_available_judges()
    original_rubrics = {}
    
    for judge_id in all_judges:
        rubric_func = JUDGE_RUBRICS[judge_id]
        original_rubrics[judge_id] = rubric_func()
    
    print(f"Loaded {len(original_rubrics)} judge rubrics")
    
    # Select judges to contaminate
    contaminated_judges = select_contaminated_judges(all_judges, contamination_rate, seed)
    clean_judges = [j for j in all_judges if j not in contaminated_judges]
    
    print(f"\nJudge assignment:")
    print(f"Clean judges ({len(clean_judges)}): {clean_judges}")
    print(f"Contaminated judges ({len(contaminated_judges)}): {contaminated_judges}")
    
    # Create contaminated rubrics
    print(f"\nCreating contaminated rubrics using {strategy.get_description()}...")
    contaminated_rubrics = create_contaminated_rubrics(original_rubrics, contaminated_judges, strategy)
    
    # Get contaminated scores
    contaminated_scores = get_contaminated_scores(data, clean_judges, contaminated_judges, contaminated_rubrics)
    
    # Extract target scores (use persona averages from original data)
    print("\nExtracting target scores...")
    if 'persona_scores' in data.columns:
        # Use average persona scores as target
        persona_scores = data['persona_scores'].apply(lambda x: np.mean(list(x.values())) if isinstance(x, dict) else x)
        target_scores = persona_scores.tolist()
    else:
        # Fallback: use mean of clean judge scores
        clean_score_matrix = np.array([contaminated_scores[judge] for judge in clean_judges])
        target_scores = np.mean(clean_score_matrix, axis=0).tolist()
    
    print(f"Target score range: {min(target_scores):.2f} - {max(target_scores):.2f}")
    
    # Train aggregators
    print("\nTraining aggregators...")
    gam_model, mlp_model, gam_results, mlp_results = train_aggregators_with_contamination(
        contaminated_scores, target_scores
    )
    
    # Analyze weights
    weight_analysis = analyze_judge_weights(
        gam_model, mlp_model, clean_judges, contaminated_judges, gam_results, mlp_results
    )
    
    # Compile results
    results = {
        'experiment_config': {
            'data_size': data_size,
            'contamination_rate': contamination_rate,
            'strategy_description': strategy.get_description(),
            'seed': seed,
            'total_judges': len(all_judges),
            'clean_judges_count': len(clean_judges),
            'contaminated_judges_count': len(contaminated_judges)
        },
        'clean_judges': clean_judges,
        'contaminated_judges': contaminated_judges,
        'contamination_strategy': strategy.get_description(),
        'contaminated_scores': contaminated_scores,
        'target_scores': target_scores,
        'gam_model': gam_model,
        'mlp_model': mlp_model,
        'gam_results': gam_results,
        'mlp_results': mlp_results,
        'weight_analysis': weight_analysis
    }
    
    return results


def main():
    """Main experiment execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Judge Contamination Experiment")
    parser.add_argument('--data-size', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--contamination-rate', type=float, default=0.3, help='Fraction of judges to contaminate')
    parser.add_argument('--strategy', choices=['inverted', 'noise', 'bias-high', 'bias-low'], 
                       default='inverted', help='Contamination strategy')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', help='Custom output directory')
    
    args = parser.parse_args()
    
    # Create contamination strategy
    strategy_map = {
        'inverted': InvertedRubricStrategy(),
        'noise': RandomNoiseStrategy(noise_level=1.0),
        'bias-high': BiasedStrategy(bias_type='high', bias_score=4.0),
        'bias-low': BiasedStrategy(bias_type='low', bias_score=0.0)
    }
    
    strategy = strategy_map[args.strategy]
    
    # Run experiment
    results = run_contamination_experiment(
        data_size=args.data_size,
        contamination_rate=args.contamination_rate,
        strategy=strategy,
        seed=args.seed
    )
    
    # Save results
    if args.output_dir:
        experiment_dir = Path(args.output_dir)
    else:
        experiment_dir = Path(__file__).parent / "results" / f"{args.strategy}_contamination_{args.seed}"
    
    save_results(results, experiment_dir)
    
    print(f"\n=== EXPERIMENT COMPLETED ===")
    print(f"Results saved to: {experiment_dir}")
    
    # Print summary
    gam_r2 = results['gam_results'].get('test_r2', 0)
    mlp_r2 = results['mlp_results'].get('test_r2', 0)
    
    resistance = results['weight_analysis'].get('contamination_resistance', {})
    gam_detected = resistance.get('gam_detected_contamination', False)
    mlp_detected = resistance.get('mlp_detected_contamination', False)
    
    print(f"\nPerformance: GAM R² = {gam_r2:.4f}, MLP R² = {mlp_r2:.4f}")
    print(f"Contamination detected: GAM = {gam_detected}, MLP = {mlp_detected}")


if __name__ == "__main__":
    main()