#!/usr/bin/env python3
"""
Analyze Persona Score Reproducibility

Loads results from multiple runs and computes agreement/stability metrics.

Usage:
    python rebutal_augusto/analyze_reproducibility.py
    python rebutal_augusto/analyze_reproducibility.py --results-dir rebutal_augusto/results
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_runs(results_dir):
    """Load all run results from directory.

    Args:
        results_dir: Path to results directory containing run_*.pkl files

    Returns:
        List of run data dictionaries, sorted by run_id
    """
    results_dir = Path(results_dir)
    run_files = sorted(results_dir.glob("run_*.pkl"))

    if not run_files:
        print(f"ERROR: No run_*.pkl files found in {results_dir}")
        sys.exit(1)

    runs = []
    for f in run_files:
        with open(f, "rb") as fh:
            runs.append(pickle.load(fh))

    runs.sort(key=lambda r: r["run_id"])
    print(f"Loaded {len(runs)} runs: {[r['run_id'] for r in runs]}")
    return runs


def compute_per_sample_stats(runs):
    """Compute per-sample score statistics across runs.

    For each sample, computes the mean, std, min, max of its average
    persona score across all runs.

    Args:
        runs: List of run data dictionaries

    Returns:
        DataFrame with per-sample statistics
    """
    n_samples = runs[0]["n_samples"]
    n_runs = len(runs)

    # Build matrix: (n_samples, n_runs)
    score_matrix = np.full((n_samples, n_runs), np.nan)
    for run_idx, run in enumerate(runs):
        for sample_idx, score in enumerate(run["scores"]["average_scores"]):
            if score is not None:
                score_matrix[sample_idx, run_idx] = score

    df = pd.DataFrame({
        "mean": np.nanmean(score_matrix, axis=1),
        "std": np.nanstd(score_matrix, axis=1),
        "min": np.nanmin(score_matrix, axis=1),
        "max": np.nanmax(score_matrix, axis=1),
        "range": np.nanmax(score_matrix, axis=1) - np.nanmin(score_matrix, axis=1),
    })

    return df, score_matrix


def compute_per_persona_stats(runs):
    """Compute per-persona score statistics across runs.

    Args:
        runs: List of run data dictionaries

    Returns:
        DataFrame with per-persona statistics
    """
    persona_names = list(runs[0]["scores"]["per_persona"].keys())
    n_samples = runs[0]["n_samples"]
    n_runs = len(runs)

    results = []
    for persona in persona_names:
        # Build matrix: (n_samples, n_runs)
        matrix = np.full((n_samples, n_runs), np.nan)
        for run_idx, run in enumerate(runs):
            scores = run["scores"]["per_persona"][persona]
            for sample_idx, score in enumerate(scores):
                if score is not None:
                    matrix[sample_idx, run_idx] = score

        # Per-sample std across runs, then average
        per_sample_std = np.nanstd(matrix, axis=1)
        per_sample_range = np.nanmax(matrix, axis=1) - np.nanmin(matrix, axis=1)

        # Exact agreement: same score in all runs
        exact_agreement = np.sum(per_sample_range == 0) / n_samples * 100

        results.append({
            "persona": persona,
            "mean_score": np.nanmean(matrix),
            "mean_std_across_runs": np.nanmean(per_sample_std),
            "mean_range_across_runs": np.nanmean(per_sample_range),
            "exact_agreement_pct": exact_agreement,
            "max_range": np.nanmax(per_sample_range),
        })

    return pd.DataFrame(results)


def compute_pairwise_correlations(runs):
    """Compute Pearson and Spearman correlations between all run pairs.

    Args:
        runs: List of run data dictionaries

    Returns:
        Dictionary with correlation matrices
    """
    n_runs = len(runs)

    pearson_matrix = np.ones((n_runs, n_runs))
    spearman_matrix = np.ones((n_runs, n_runs))

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            scores_i = np.array(runs[i]["scores"]["average_scores"], dtype=float)
            scores_j = np.array(runs[j]["scores"]["average_scores"], dtype=float)

            # Only compare where both are valid
            valid = ~np.isnan(scores_i) & ~np.isnan(scores_j)
            si = scores_i[valid]
            sj = scores_j[valid]

            if len(si) > 2:
                r_pearson, _ = stats.pearsonr(si, sj)
                r_spearman, _ = stats.spearmanr(si, sj)
            else:
                r_pearson = np.nan
                r_spearman = np.nan

            pearson_matrix[i, j] = r_pearson
            pearson_matrix[j, i] = r_pearson
            spearman_matrix[i, j] = r_spearman
            spearman_matrix[j, i] = r_spearman

    return {
        "pearson": pearson_matrix,
        "spearman": spearman_matrix,
    }


def compute_score_change_stats(score_matrix):
    """Compute statistics about score changes across runs.

    Args:
        score_matrix: (n_samples, n_runs) array of average scores

    Returns:
        Dictionary with change statistics
    """
    n_samples, n_runs = score_matrix.shape

    # Pairwise absolute differences
    all_diffs = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            diffs = np.abs(score_matrix[:, i] - score_matrix[:, j])
            valid = ~np.isnan(diffs)
            all_diffs.extend(diffs[valid].tolist())

    all_diffs = np.array(all_diffs)

    # Per-sample range
    ranges = np.nanmax(score_matrix, axis=1) - np.nanmin(score_matrix, axis=1)

    return {
        "mean_abs_diff": np.mean(all_diffs),
        "median_abs_diff": np.median(all_diffs),
        "pct_diff_ge_1": np.sum(all_diffs >= 1.0) / len(all_diffs) * 100,
        "pct_diff_ge_2": np.sum(all_diffs >= 2.0) / len(all_diffs) * 100,
        "pct_exact_same": np.sum(all_diffs == 0) / len(all_diffs) * 100,
        "mean_range": np.nanmean(ranges),
        "median_range": np.nanmedian(ranges),
        "pct_zero_range": np.sum(ranges == 0) / n_samples * 100,
    }


def plot_score_distributions(runs, output_dir):
    """Plot score distributions for each run as overlaid histograms.

    Args:
        runs: List of run data
        output_dir: Where to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for run in runs:
        scores = [s for s in run["scores"]["average_scores"] if s is not None]
        ax.hist(scores, bins=30, alpha=0.3, label=f"Run {run['run_id']}", density=True)

    ax.set_xlabel("Average Persona Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distributions Across Runs", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "score_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved score_distributions.png")


def plot_per_sample_variance(sample_stats, output_dir):
    """Plot histogram of per-sample standard deviation across runs.

    Args:
        sample_stats: DataFrame with per-sample statistics
        output_dir: Where to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Std distribution
    axes[0].hist(sample_stats["std"], bins=30, color="#4C72B0", edgecolor="white")
    axes[0].axvline(sample_stats["std"].mean(), color="red", linestyle="--",
                     label=f"Mean={sample_stats['std'].mean():.3f}")
    axes[0].set_xlabel("Std Dev Across Runs", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Per-Sample Score Variability", fontsize=14)
    axes[0].legend()

    # Range distribution
    axes[1].hist(sample_stats["range"], bins=30, color="#DD8452", edgecolor="white")
    axes[1].axvline(sample_stats["range"].mean(), color="red", linestyle="--",
                     label=f"Mean={sample_stats['range'].mean():.3f}")
    axes[1].set_xlabel("Score Range Across Runs (max-min)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Per-Sample Score Range", fontsize=14)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "per_sample_variance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved per_sample_variance.png")


def plot_pairwise_scatter(runs, output_dir):
    """Plot scatter plots comparing run pairs.

    Args:
        runs: List of run data
        output_dir: Where to save plots
    """
    n_runs = len(runs)
    if n_runs < 2:
        return

    # Plot first 6 pairs max
    pairs = [(i, j) for i in range(n_runs) for j in range(i+1, n_runs)][:6]
    n_plots = len(pairs)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for plot_idx, (i, j) in enumerate(pairs):
        ax = axes[plot_idx]
        si = np.array(runs[i]["scores"]["average_scores"], dtype=float)
        sj = np.array(runs[j]["scores"]["average_scores"], dtype=float)
        valid = ~np.isnan(si) & ~np.isnan(sj)

        ax.scatter(si[valid], sj[valid], alpha=0.3, s=10, color="#4C72B0")
        # Perfect agreement line
        lims = [min(np.min(si[valid]), np.min(sj[valid])),
                max(np.max(si[valid]), np.max(sj[valid]))]
        ax.plot(lims, lims, "r--", alpha=0.5)

        r, _ = stats.pearsonr(si[valid], sj[valid])
        ax.set_xlabel(f"Run {runs[i]['run_id']}", fontsize=11)
        ax.set_ylabel(f"Run {runs[j]['run_id']}", fontsize=11)
        ax.set_title(f"r = {r:.4f}", fontsize=12)

    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Pairwise Run Comparisons", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "pairwise_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved pairwise_scatter.png")


def plot_persona_stability(persona_stats, output_dir):
    """Bar chart of per-persona stability metrics.

    Args:
        persona_stats: DataFrame with per-persona statistics
        output_dir: Where to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Sort by mean std
    sorted_stats = persona_stats.sort_values("mean_std_across_runs", ascending=True)

    # Mean std across runs
    axes[0].barh(sorted_stats["persona"], sorted_stats["mean_std_across_runs"],
                  color="#4C72B0")
    axes[0].set_xlabel("Mean Std Dev Across Runs", fontsize=12)
    axes[0].set_title("Persona Scoring Variability (lower = more stable)", fontsize=13)

    # Exact agreement percentage
    sorted_stats2 = persona_stats.sort_values("exact_agreement_pct", ascending=True)
    axes[1].barh(sorted_stats2["persona"], sorted_stats2["exact_agreement_pct"],
                  color="#55A868")
    axes[1].set_xlabel("% Samples With Exact Same Score", fontsize=12)
    axes[1].set_title("Exact Agreement Across Runs", fontsize=13)

    fig.tight_layout()
    fig.savefig(output_dir / "persona_stability.png", dpi=150)
    plt.close(fig)
    print(f"  Saved persona_stability.png")


def plot_correlation_heatmap(corr_matrices, output_dir):
    """Plot correlation heatmaps for Pearson and Spearman.

    Args:
        corr_matrices: Dictionary with 'pearson' and 'spearman' matrices
        output_dir: Where to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (name, matrix) in zip(axes, corr_matrices.items()):
        n_runs = matrix.shape[0]
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0)
        ax.set_title(f"{name.title()} Correlation", fontsize=13)
        ax.set_xticks(range(n_runs))
        ax.set_yticks(range(n_runs))
        ax.set_xticklabels([f"Run {i+1}" for i in range(n_runs)])
        ax.set_yticklabels([f"Run {i+1}" for i in range(n_runs)])

        # Annotate cells
        for i in range(n_runs):
            for j in range(n_runs):
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                        fontsize=9, color="black")

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmaps.png", dpi=150)
    plt.close(fig)
    print(f"  Saved correlation_heatmaps.png")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze persona score reproducibility across runs"
    )
    parser.add_argument(
        "--results-dir", type=str, default="rebutal_augusto/results",
        help="Directory containing run_*.pkl files"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load runs
    runs = load_runs(results_dir)

    print(f"\n📊 Analyzing {len(runs)} runs of {runs[0]['n_samples']} samples each\n")

    # 1. Per-sample stats
    print("1. Per-sample score statistics:")
    sample_stats, score_matrix = compute_per_sample_stats(runs)
    print(f"   Mean std across runs:   {sample_stats['std'].mean():.4f}")
    print(f"   Median std across runs: {sample_stats['std'].median():.4f}")
    print(f"   Mean range across runs: {sample_stats['range'].mean():.4f}")
    print(f"   Samples with zero variance: "
          f"{(sample_stats['std'] == 0).sum()} / {len(sample_stats)} "
          f"({(sample_stats['std'] == 0).mean() * 100:.1f}%)")

    # 2. Score change stats
    print("\n2. Score change magnitude:")
    change_stats = compute_score_change_stats(score_matrix)
    print(f"   Mean |Δscore| between run pairs:       {change_stats['mean_abs_diff']:.4f}")
    print(f"   Median |Δscore| between run pairs:     {change_stats['median_abs_diff']:.4f}")
    print(f"   % pairs with exact same score:         {change_stats['pct_exact_same']:.1f}%")
    print(f"   % pairs differing by ≥1 point:         {change_stats['pct_diff_ge_1']:.1f}%")
    print(f"   % pairs differing by ≥2 points:        {change_stats['pct_diff_ge_2']:.1f}%")
    print(f"   % samples with same score in ALL runs: {change_stats['pct_zero_range']:.1f}%")

    # 3. Per-persona stats
    print("\n3. Per-persona stability:")
    persona_stats = compute_per_persona_stats(runs)
    print(persona_stats.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 4. Pairwise correlations
    print("\n4. Pairwise run correlations:")
    corr_matrices = compute_pairwise_correlations(runs)

    pearson_off_diag = corr_matrices["pearson"][
        np.triu_indices_from(corr_matrices["pearson"], k=1)
    ]
    spearman_off_diag = corr_matrices["spearman"][
        np.triu_indices_from(corr_matrices["spearman"], k=1)
    ]
    print(f"   Pearson  — mean: {np.nanmean(pearson_off_diag):.4f}, "
          f"min: {np.nanmin(pearson_off_diag):.4f}, max: {np.nanmax(pearson_off_diag):.4f}")
    print(f"   Spearman — mean: {np.nanmean(spearman_off_diag):.4f}, "
          f"min: {np.nanmin(spearman_off_diag):.4f}, max: {np.nanmax(spearman_off_diag):.4f}")

    # 5. Generate plots
    print("\n5. Generating plots...")
    plot_score_distributions(runs, plots_dir)
    plot_per_sample_variance(sample_stats, plots_dir)
    plot_pairwise_scatter(runs, plots_dir)
    plot_persona_stability(persona_stats, plots_dir)
    plot_correlation_heatmap(corr_matrices, plots_dir)

    # 6. Save summary
    summary = {
        "n_runs": len(runs),
        "n_samples": runs[0]["n_samples"],
        "model": runs[0]["model"],
        "temperature": runs[0]["temperature"],
        "per_sample": {
            "mean_std": float(sample_stats["std"].mean()),
            "median_std": float(sample_stats["std"].median()),
            "mean_range": float(sample_stats["range"].mean()),
            "pct_zero_variance": float((sample_stats["std"] == 0).mean() * 100),
        },
        "score_changes": change_stats,
        "correlations": {
            "pearson_mean": float(np.nanmean(pearson_off_diag)),
            "spearman_mean": float(np.nanmean(spearman_off_diag)),
        },
        "persona_stats": persona_stats.to_dict(orient="records"),
    }

    with open(results_dir / "reproducibility_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Analysis complete!")
    print(f"   Summary: {results_dir}/reproducibility_summary.json")
    print(f"   Plots:   {plots_dir}/")


if __name__ == "__main__":
    main()
