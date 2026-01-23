"""
Experiment 4.1: Synthetic Preference Decomposition

Tests if GAM aggregators can recover known ground truth weights.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

from pipeline.core.aggregator_training import GAMAggregator, compute_metrics


def generate_synthetic_data(n_samples=1000, true_weights=None, seed=42, noise_std=0.5):
    """Generate synthetic judge scores and ground truth human scores.

    Args:
        n_samples: Number of samples to generate
        true_weights: Dict mapping judge names to weights (must sum to 1.0)
        seed: Random seed for reproducibility
        noise_std: Standard deviation of Gaussian noise to add to human scores

    Returns:
        X: Judge scores array (n_samples, n_judges)
        y: Human scores array (n_samples,)
        true_weights: Ground truth weights dict
        judge_names: List of judge names
    """
    np.random.seed(seed)

    # Default ground truth weights (Recipe A)
    if true_weights is None:
        true_weights = {
            'Truthfulness': 0.30,
            'Helpfulness': 0.50,
            'Clarity': 0.20
        }

    judge_names = list(true_weights.keys())
    n_judges = len(judge_names)

    # Generate random judge scores (uniform between 1-4 to match real judge scale)
    X = np.random.uniform(1.0, 4.0, size=(n_samples, n_judges))

    # Compute ground truth human scores as weighted sum
    weights_array = np.array([true_weights[name] for name in judge_names])
    y_clean = X @ weights_array

    # Add Gaussian noise to simulate human variance
    noise = np.random.normal(0, noise_std, size=n_samples)
    y = y_clean + noise

    # Scale to human rating range [0, 10]
    y = (y - y.min()) / (y.max() - y.min()) * 10

    print(f"✓ Generated {n_samples} synthetic samples")
    print(f"  Judge score range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  Human score range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Ground truth weights: {true_weights}")

    return X, y, true_weights, judge_names


def train_gam(X_train, y_train, judge_names, n_splines=5, lam=20.0):
    """Train GAM aggregator with Track 2 hyperparameters.

    Args:
        X_train: Training judge scores
        y_train: Training human scores
        judge_names: List of judge names
        n_splines: Number of splines (default from Track 2)
        lam: Lambda regularization (default from Track 2)

    Returns:
        Trained GAMAggregator model
    """
    print(f"\n📊 Training GAM (n_splines={n_splines}, lam={lam})...")

    gam = GAMAggregator(
        feature_names=judge_names,
        n_splines=n_splines,
        lam=lam,
        max_iter=200,
        tol=1e-4
    )

    gam.fit(X_train, y_train)

    print("✓ GAM training complete")
    return gam


def extract_learned_weights(model, X_data, judge_names):
    """Extract feature importance as learned weights from GAM.

    Args:
        model: Trained GAMAggregator
        X_data: Data to compute importance over (usually training set)
        judge_names: List of judge names

    Returns:
        Dict mapping judge names to normalized importance weights
    """
    # Get raw feature importance from GAM
    importance_dict = model.get_feature_importance(X_data)

    # Normalize to sum to 1.0 (to match true_weights format)
    total_importance = sum(importance_dict.values())

    if total_importance == 0:
        print("⚠️  Warning: Total importance is zero, using uniform weights")
        learned_weights = {name: 1.0 / len(judge_names) for name in judge_names}
    else:
        learned_weights = {
            name: importance / total_importance
            for name, importance in importance_dict.items()
        }

    return learned_weights


def compute_weight_recovery_error(learned_weights, true_weights):
    """Compute absolute error between learned and true weights.

    Args:
        learned_weights: Dict of learned weights (normalized)
        true_weights: Dict of true weights

    Returns:
        mean_absolute_error: Mean |learned - true| across all judges
        max_absolute_error: Max |learned - true|
        error_dict: Per-judge errors
    """
    errors = {}
    for judge_name in true_weights.keys():
        true_w = true_weights[judge_name]
        learned_w = learned_weights.get(judge_name, 0.0)
        errors[judge_name] = abs(learned_w - true_w)

    mae = np.mean(list(errors.values()))
    max_err = np.max(list(errors.values()))

    return mae, max_err, errors


def compute_rank_correlation(learned_weights, true_weights):
    """Compute Spearman rank correlation between learned and true weights."""
    judge_names = list(true_weights.keys())

    true_ranks = [true_weights[name] for name in judge_names]
    learned_ranks = [learned_weights[name] for name in judge_names]

    rho, pval = spearmanr(true_ranks, learned_ranks)
    return rho, pval


def visualize_results(true_weights, learned_weights, errors, save_path):
    """Create visualization comparing true vs learned weights."""
    judge_names = list(true_weights.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: True vs Learned weights
    x = np.arange(len(judge_names))
    width = 0.35

    ax1.bar(x - width/2, [true_weights[n] for n in judge_names],
            width, label='True Weights', alpha=0.8)
    ax1.bar(x + width/2, [learned_weights[n] for n in judge_names],
            width, label='Learned Weights', alpha=0.8)

    ax1.set_xlabel('Judge')
    ax1.set_ylabel('Weight')
    ax1.set_title('Weight Recovery: True vs Learned')
    ax1.set_xticks(x)
    ax1.set_xticklabels(judge_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Absolute errors
    ax2.bar(judge_names, [errors[n] for n in judge_names], alpha=0.8, color='red')
    ax2.set_xlabel('Judge')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Weight Recovery Error by Judge')
    ax2.set_xticklabels(judge_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {save_path}")
    plt.close()


def run_single_experiment(n_samples, true_weights, seed, output_dir):
    """Run a single weight recovery experiment.

    Returns:
        results: Dict with all experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: n={n_samples}, seed={seed}")
    print(f"{'='*60}")

    # 1. Generate synthetic data
    X, y, true_weights, judge_names = generate_synthetic_data(
        n_samples=n_samples,
        true_weights=true_weights,
        seed=seed
    )

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    print(f"\n📦 Data split: {len(X_train)} train, {len(X_test)} test")

    # 3. Train GAM
    gam = train_gam(X_train, y_train, judge_names)

    # 4. Evaluate on test set
    y_pred = gam.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)

    print(f"\n📈 Test Performance:")
    print(f"  R² = {test_r2:.4f}")
    print(f"  MSE = {test_mse:.4f}")

    # 5. Extract learned weights
    learned_weights = extract_learned_weights(gam, X_train, judge_names)

    print(f"\n🎯 Weight Recovery:")
    for name in judge_names:
        true_w = true_weights[name]
        learned_w = learned_weights[name]
        print(f"  {name:15s}: true={true_w:.3f}, learned={learned_w:.3f}, "
              f"error={abs(true_w - learned_w):.3f}")

    # 6. Compute error metrics
    mae, max_err, error_dict = compute_weight_recovery_error(learned_weights, true_weights)
    rho, pval = compute_rank_correlation(learned_weights, true_weights)

    print(f"\n📊 Summary Metrics:")
    print(f"  Mean Absolute Error: {mae:.4f}")
    print(f"  Max Absolute Error: {max_err:.4f}")
    print(f"  Rank Correlation (ρ): {rho:.4f} (p={pval:.4f})")

    # 7. Save results
    results = {
        'n_samples': n_samples,
        'seed': seed,
        'true_weights': true_weights,
        'learned_weights': learned_weights,
        'weight_errors': error_dict,
        'mean_absolute_error': mae,
        'max_absolute_error': max_err,
        'rank_correlation': rho,
        'rank_correlation_pval': pval,
        'test_r2': test_r2,
        'test_mse': test_mse
    }

    # Save to JSON
    results_path = output_dir / f'results_n{n_samples}_seed{seed}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {results_path}")

    # Visualize
    viz_path = output_dir / f'weights_n{n_samples}_seed{seed}.png'
    visualize_results(true_weights, learned_weights, error_dict, viz_path)

    return results


if __name__ == "__main__":
    # Setup output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Experiment 4.1: Synthetic Weight Recovery")
    print("=" * 60)

    # Test with small dataset first
    print("\n🧪 PHASE 1: Small-scale test (n=50)")
    small_results = run_single_experiment(
        n_samples=50,
        true_weights={'Truthfulness': 0.30, 'Helpfulness': 0.50, 'Clarity': 0.20},
        seed=42,
        output_dir=output_dir
    )

    print("\n" + "=" * 60)
    print("✅ Experiment 4.1 Complete!")
    print("=" * 60)
