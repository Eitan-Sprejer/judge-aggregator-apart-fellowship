#!/usr/bin/env python3
"""
Analyze judge model comparison results: GPT-5-Mini vs GPT-5-Nano

This script:
1. Loads judge scores from both models
2. Computes per-judge correlation (mini vs nano)
3. Computes correlation with human labels
4. Generates comparison report and visualizations

Usage:
    python experiments/track1_baseline_performance/judge_model_comparison/analyze_comparison.py <results_dir>

Example:
    python experiments/track1_baseline_performance/judge_model_comparison/analyze_comparison.py results/comparison_20241129_120000
"""

import sys
from pathlib import Path
import argparse
import pickle
import json

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def load_results(results_dir: Path) -> dict:
    """Load combined results from comparison run."""
    combined_path = results_dir / "combined_results.pkl"
    if not combined_path.exists():
        raise FileNotFoundError(f"Results not found: {combined_path}")

    with open(combined_path, 'rb') as f:
        return pickle.load(f)


def compute_mini_nano_correlation(mini_scores: dict, nano_scores: dict) -> pd.DataFrame:
    """Compute per-judge correlation between mini and nano scores."""
    results = []

    for judge_id in mini_scores.keys():
        if judge_id not in nano_scores:
            continue

        mini = mini_scores[judge_id]
        nano = nano_scores[judge_id]

        # Filter out NaN values
        mask = ~(np.isnan(mini) | np.isnan(nano))
        if mask.sum() < 3:
            continue

        mini_clean = mini[mask]
        nano_clean = nano[mask]

        # Compute correlations
        pearson_r, pearson_p = stats.pearsonr(mini_clean, nano_clean)
        spearman_rho, spearman_p = stats.spearmanr(mini_clean, nano_clean)

        # Compute agreement within tolerance
        diff = np.abs(mini_clean - nano_clean)
        agreement_05 = (diff <= 0.5).mean() * 100
        agreement_10 = (diff <= 1.0).mean() * 100

        # Compute mean absolute difference
        mae = diff.mean()

        results.append({
            "judge": judge_id,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "agreement_0.5": agreement_05,
            "agreement_1.0": agreement_10,
            "mae": mae,
            "n_samples": len(mini_clean)
        })

    return pd.DataFrame(results)


def compute_human_correlation(
    scores: dict,
    human_labels: dict,
    model_name: str
) -> pd.DataFrame:
    """Compute correlation between judge scores and human labels."""
    results = []

    for judge_id, judge_scores in scores.items():
        # Determine which dimension this judge belongs to
        dimension = None
        for dim in human_labels.keys():
            if dim in judge_id.lower():
                dimension = dim
                break

        if dimension is None:
            continue

        human = human_labels[dimension]

        # Filter out NaN values
        mask = ~(np.isnan(judge_scores) | np.isnan(human))
        if mask.sum() < 3:
            continue

        judge_clean = judge_scores[mask]
        human_clean = human[mask]

        # Compute correlations
        pearson_r, pearson_p = stats.pearsonr(judge_clean, human_clean)
        spearman_rho, spearman_p = stats.spearmanr(judge_clean, human_clean)

        results.append({
            "model": model_name,
            "judge": judge_id,
            "dimension": dimension,
            "pearson_r": pearson_r,
            "spearman_rho": spearman_rho,
            "n_samples": len(judge_clean)
        })

    return pd.DataFrame(results)


def compute_dimension_summary(
    mini_scores: dict,
    nano_scores: dict,
    human_labels: dict
) -> pd.DataFrame:
    """Compute aggregated correlation with human labels per dimension."""
    results = []

    for dimension, human in human_labels.items():
        # Get all judges for this dimension
        mini_dim_judges = [j for j in mini_scores.keys() if dimension in j.lower()]
        nano_dim_judges = [j for j in nano_scores.keys() if dimension in j.lower()]

        if not mini_dim_judges or not nano_dim_judges:
            continue

        # Average judge scores for this dimension
        mini_avg = np.nanmean([mini_scores[j] for j in mini_dim_judges], axis=0)
        nano_avg = np.nanmean([nano_scores[j] for j in nano_dim_judges], axis=0)

        # Filter valid samples
        mask = ~(np.isnan(mini_avg) | np.isnan(nano_avg) | np.isnan(human))
        if mask.sum() < 3:
            continue

        mini_clean = mini_avg[mask]
        nano_clean = nano_avg[mask]
        human_clean = human[mask]

        # Compute correlations
        mini_pearson, _ = stats.pearsonr(mini_clean, human_clean)
        mini_spearman, _ = stats.spearmanr(mini_clean, human_clean)

        nano_pearson, _ = stats.pearsonr(nano_clean, human_clean)
        nano_spearman, _ = stats.spearmanr(nano_clean, human_clean)

        results.append({
            "dimension": dimension,
            "mini_pearson": mini_pearson,
            "mini_spearman": mini_spearman,
            "nano_pearson": nano_pearson,
            "nano_spearman": nano_spearman,
            "pearson_diff": nano_pearson - mini_pearson,
            "spearman_diff": nano_spearman - mini_spearman,
            "n_judges": len(mini_dim_judges)
        })

    return pd.DataFrame(results)


def plot_mini_nano_scatter(
    mini_scores: dict,
    nano_scores: dict,
    output_path: Path
):
    """Create scatter plot comparing mini vs nano scores."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    judge_ids = sorted(mini_scores.keys())[:6]  # Plot first 6 judges

    for idx, judge_id in enumerate(judge_ids):
        ax = axes[idx]

        mini = mini_scores[judge_id]
        nano = nano_scores[judge_id]

        # Filter valid
        mask = ~(np.isnan(mini) | np.isnan(nano))
        mini_clean = mini[mask]
        nano_clean = nano[mask]

        # Scatter plot
        ax.scatter(mini_clean, nano_clean, alpha=0.6, s=30)

        # Add diagonal line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='y=x')

        # Compute correlation for title
        r, _ = stats.pearsonr(mini_clean, nano_clean)

        ax.set_xlabel('Mini Score')
        ax.set_ylabel('Nano Score')
        ax.set_title(f'{judge_id}\n(r={r:.3f})', fontsize=10)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle('Mini vs Nano Judge Scores', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved scatter plot: {output_path}")


def plot_correlation_comparison(
    mini_human_corr: pd.DataFrame,
    nano_human_corr: pd.DataFrame,
    output_path: Path
):
    """Create bar plot comparing human correlations."""
    # Merge results
    mini_summary = mini_human_corr.groupby('dimension')['spearman_rho'].mean().reset_index()
    mini_summary.columns = ['dimension', 'mini_spearman']

    nano_summary = nano_human_corr.groupby('dimension')['spearman_rho'].mean().reset_index()
    nano_summary.columns = ['dimension', 'nano_spearman']

    merged = pd.merge(mini_summary, nano_summary, on='dimension')

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(merged))
    width = 0.35

    bars1 = ax.bar(x - width/2, merged['mini_spearman'], width, label='Mini', color='steelblue')
    bars2 = ax.bar(x + width/2, merged['nano_spearman'], width, label='Nano', color='coral')

    ax.set_xlabel('Dimension')
    ax.set_ylabel('Spearman Correlation with Human Labels')
    ax.set_title('Judge-Human Correlation by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(merged['dimension'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved correlation comparison: {output_path}")


def generate_report(
    mini_nano_corr: pd.DataFrame,
    dim_summary: pd.DataFrame,
    config: dict,
    output_path: Path
):
    """Generate markdown report."""
    report = []
    report.append("# Judge Model Comparison Report")
    report.append(f"\n**Date**: {config.get('timestamp', 'N/A')}")
    report.append(f"\n**Samples**: {config.get('n_samples', 'N/A')}")
    report.append(f"\n**Dimensions**: {', '.join(config.get('dimensions', []))}")
    report.append(f"\n**Models**: Mini ({config['models']['mini']}) vs Nano ({config['models']['nano']})")

    report.append("\n\n## Summary\n")

    # Overall statistics
    avg_pearson = mini_nano_corr['pearson_r'].mean()
    avg_spearman = mini_nano_corr['spearman_rho'].mean()
    avg_agreement = mini_nano_corr['agreement_0.5'].mean()

    report.append(f"**Mini-Nano Agreement**:")
    report.append(f"- Average Pearson r: {avg_pearson:.3f}")
    report.append(f"- Average Spearman ρ: {avg_spearman:.3f}")
    report.append(f"- Average Agreement (±0.5): {avg_agreement:.1f}%")

    report.append("\n\n## Mini vs Nano Correlation (Per Judge)\n")
    report.append("| Judge | Pearson r | Spearman ρ | Agreement (±0.5) | MAE |")
    report.append("|-------|-----------|------------|------------------|-----|")

    for _, row in mini_nano_corr.iterrows():
        report.append(
            f"| {row['judge'][:40]} | {row['pearson_r']:.3f} | {row['spearman_rho']:.3f} | "
            f"{row['agreement_0.5']:.1f}% | {row['mae']:.3f} |"
        )

    report.append("\n\n## Human Label Correlation (By Dimension)\n")
    report.append("| Dimension | Mini → Human | Nano → Human | Difference |")
    report.append("|-----------|--------------|--------------|------------|")

    for _, row in dim_summary.iterrows():
        diff_str = f"+{row['spearman_diff']:.3f}" if row['spearman_diff'] >= 0 else f"{row['spearman_diff']:.3f}"
        report.append(
            f"| {row['dimension']} | {row['mini_spearman']:.3f} | {row['nano_spearman']:.3f} | {diff_str} |"
        )

    report.append("\n\n## Recommendation\n")

    # Decision logic
    avg_diff = dim_summary['spearman_diff'].mean()
    if abs(avg_diff) < 0.05 and avg_agreement > 80:
        report.append("✅ **Nano appears viable**: High agreement with Mini scores and similar human correlation.")
        report.append(f"   - Average human correlation difference: {avg_diff:.3f}")
        report.append(f"   - Mini-Nano agreement: {avg_agreement:.1f}%")
        report.append("   - Cost savings: ~80%")
    elif abs(avg_diff) < 0.1 and avg_agreement > 70:
        report.append("⚠️ **Nano may be acceptable**: Moderate agreement, consider for less critical applications.")
        report.append(f"   - Average human correlation difference: {avg_diff:.3f}")
        report.append(f"   - Mini-Nano agreement: {avg_agreement:.1f}%")
    else:
        report.append("❌ **Mini recommended**: Significant difference in scores or human correlation.")
        report.append(f"   - Average human correlation difference: {avg_diff:.3f}")
        report.append(f"   - Mini-Nano agreement: {avg_agreement:.1f}%")

    report_text = "\n".join(report)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"  Saved report: {output_path}")
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Analyze judge model comparison results")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        # Try relative to script
        script_dir = Path(__file__).parent
        results_dir = script_dir / args.results_dir

    if not results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Judge Model Comparison Analysis")
    print("=" * 60)
    print(f"Results directory: {results_dir}")

    # Load results
    print("\nLoading results...")
    data = load_results(results_dir)

    mini_scores = data["mini"]
    nano_scores = data["nano"]
    human_labels = data["human_labels"]
    config = data["config"]

    print(f"  Mini judges: {len(mini_scores)}")
    print(f"  Nano judges: {len(nano_scores)}")
    print(f"  Dimensions: {list(human_labels.keys())}")

    # Compute correlations
    print("\nComputing correlations...")

    print("  Mini vs Nano per-judge correlation...")
    mini_nano_corr = compute_mini_nano_correlation(mini_scores, nano_scores)

    print("  Mini vs Human correlation...")
    mini_human_corr = compute_human_correlation(mini_scores, human_labels, "mini")

    print("  Nano vs Human correlation...")
    nano_human_corr = compute_human_correlation(nano_scores, human_labels, "nano")

    print("  Dimension summary...")
    dim_summary = compute_dimension_summary(mini_scores, nano_scores, human_labels)

    # Save analysis results
    print("\nSaving analysis results...")

    mini_nano_corr.to_csv(results_dir / "mini_nano_correlation.csv", index=False)
    print(f"  Saved: mini_nano_correlation.csv")

    mini_human_corr.to_csv(results_dir / "mini_human_correlation.csv", index=False)
    print(f"  Saved: mini_human_correlation.csv")

    nano_human_corr.to_csv(results_dir / "nano_human_correlation.csv", index=False)
    print(f"  Saved: nano_human_correlation.csv")

    dim_summary.to_csv(results_dir / "dimension_summary.csv", index=False)
    print(f"  Saved: dimension_summary.csv")

    # Generate plots
    print("\nGenerating plots...")
    plot_mini_nano_scatter(mini_scores, nano_scores, results_dir / "mini_nano_scatter.png")
    plot_correlation_comparison(mini_human_corr, nano_human_corr, results_dir / "human_correlation_comparison.png")

    # Generate report
    print("\nGenerating report...")
    report = generate_report(mini_nano_corr, dim_summary, config, results_dir / "comparison_report.md")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print("\n" + report)


if __name__ == "__main__":
    main()
