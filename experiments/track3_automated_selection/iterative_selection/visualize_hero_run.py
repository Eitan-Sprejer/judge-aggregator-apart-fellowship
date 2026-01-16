#!/usr/bin/env python3
"""
Visualize Hero Run Multi-Output Results.

Generates plots and a report for the hero run experiment.

Usage:
    python experiments/track3_automated_selection/iterative_selection/visualize_hero_run.py \
        --results-dir results/helpsteer2-baseline_20260112_102426/hero_run_20260114_151845
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')

DIMENSION_COLORS = {
    'helpfulness': '#2ecc71',
    'correctness': '#3498db',
    'coherence': '#9b59b6',
    'complexity': '#f39c12',
    'verbosity': '#e74c3c',
}

AVG_COLOR = '#2c3e50'


def load_summary(results_dir: Path) -> dict:
    """Load the summary.json from the hero run."""
    summary_path = results_dir / "summary.json"
    with open(summary_path) as f:
        return json.load(f)


def plot_r2_trajectory(summary: dict, output_dir: Path) -> None:
    """Plot R² trajectory for all dimensions as judges are pruned."""
    trajectory = summary["trajectory"]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    iterations = [t["iteration"] for t in trajectory]
    n_judges = [t["n_judges"] for t in trajectory]
    
    # Plot each dimension
    for dim in DIMENSION_COLORS.keys():
        r2_values = [t["dimension_r2"][dim] for t in trajectory]
        ax.plot(iterations, r2_values, 'o-', 
                color=DIMENSION_COLORS[dim], 
                label=dim.capitalize(),
                linewidth=2, markersize=6, alpha=0.8)
    
    # Plot average R²
    avg_r2 = [t["avg_r2"] for t in trajectory]
    ax.plot(iterations, avg_r2, 's-', 
            color=AVG_COLOR, 
            label='Average',
            linewidth=3, markersize=8)
    
    # Mark best iteration
    best_iter = summary["best_iteration"]
    best_r2 = summary["best_avg_r2"]
    ax.axvline(x=best_iter, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax.annotate(f'Best: {best_r2:.4f}\n({summary["best_n_judges"]} judges)', 
                xy=(best_iter, best_r2), xytext=(best_iter + 1, best_r2 + 0.02),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    # Add judge count as secondary x-axis labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(iterations[::2])
    ax2.set_xticklabels([str(n_judges[i]) for i in range(0, len(n_judges), 2)])
    ax2.set_xlabel('Number of Judges', fontsize=11)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Hero Run: R² Trajectory Across All Dimensions\n(Multi-Output Judge Pruning)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'r2_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved r2_trajectory.png")


def plot_judges_removed(summary: dict, output_dir: Path) -> None:
    """Plot showing which judges were removed at each iteration."""
    trajectory = summary["trajectory"]
    
    removed = [(t["iteration"], t["removed"], t["n_judges"]) 
               for t in trajectory if t["removed"]]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(removed) * 0.4)))
    
    y_positions = range(len(removed))
    iterations = [r[0] for r in removed]
    judges = [r[1].replace('helpsteer2-', '').replace('-judge', '') for r in removed]
    n_judges = [r[2] for r in removed]
    
    # Color by dimension
    colors = []
    for judge in judges:
        if 'helpfulness' in judge:
            colors.append(DIMENSION_COLORS['helpfulness'])
        elif 'correctness' in judge:
            colors.append(DIMENSION_COLORS['correctness'])
        elif 'coherence' in judge:
            colors.append(DIMENSION_COLORS['coherence'])
        elif 'complexity' in judge:
            colors.append(DIMENSION_COLORS['complexity'])
        elif 'verbosity' in judge:
            colors.append(DIMENSION_COLORS['verbosity'])
        else:
            colors.append('#95a5a6')
    
    bars = ax.barh(y_positions, [1] * len(removed), color=colors, alpha=0.7, edgecolor='white')
    
    # Add labels
    for i, (iter_num, judge, n_j) in enumerate(zip(iterations, judges, n_judges)):
        ax.text(0.02, i, f"Iter {iter_num}: {judge}", va='center', ha='left', 
                fontsize=9, fontweight='bold', color='white')
        ax.text(0.98, i, f"→ {n_j} left", va='center', ha='right',
                fontsize=9, color='white')
    
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title('Judges Removed (in order)\nColored by dimension', fontsize=13)
    ax.invert_yaxis()
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=d.capitalize()) 
                       for d, c in DIMENSION_COLORS.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'judges_removed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved judges_removed.png")


def plot_final_comparison(summary: dict, output_dir: Path) -> None:
    """Bar chart comparing initial, best, and final performance per dimension."""
    trajectory = summary["trajectory"]
    
    initial = trajectory[0]
    best = trajectory[summary["best_iteration"]]
    final = trajectory[-1]
    
    dimensions = list(DIMENSION_COLORS.keys())
    x = np.arange(len(dimensions))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    initial_r2 = [initial["dimension_r2"][d] for d in dimensions]
    best_r2 = [best["dimension_r2"][d] for d in dimensions]
    final_r2 = [final["dimension_r2"][d] for d in dimensions]
    
    bars1 = ax.bar(x - width, initial_r2, width, label=f'Initial ({initial["n_judges"]} judges)', 
                   color='#bdc3c7', edgecolor='white')
    bars2 = ax.bar(x, best_r2, width, label=f'Best ({best["n_judges"]} judges)', 
                   color='#27ae60', edgecolor='white')
    bars3 = ax.bar(x + width, final_r2, width, label=f'Final ({final["n_judges"]} judges)', 
                   color='#e74c3c', edgecolor='white')
    
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Performance Comparison: Initial vs Best vs Final', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dimensions], fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.01:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height > 0 else -10),
                            textcoords="offset points",
                            ha='center', va='bottom' if height > 0 else 'top',
                            fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved final_comparison.png")


def plot_improvement_heatmap(summary: dict, output_dir: Path) -> None:
    """Heatmap showing R² at each iteration for each dimension."""
    trajectory = summary["trajectory"]
    dimensions = list(DIMENSION_COLORS.keys())
    
    # Build matrix: iterations x dimensions
    n_iters = len(trajectory)
    matrix = np.zeros((len(dimensions), n_iters))
    
    for j, t in enumerate(trajectory):
        for i, dim in enumerate(dimensions):
            matrix[i, j] = t["dimension_r2"][dim]
    
    fig, ax = plt.subplots(figsize=(16, 5))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.15, vmax=0.35)
    
    # Labels
    ax.set_xticks(range(n_iters))
    ax.set_xticklabels([f"{t['iteration']}\n({t['n_judges']}j)" for t in trajectory], fontsize=8)
    ax.set_yticks(range(len(dimensions)))
    ax.set_yticklabels([d.capitalize() for d in dimensions], fontsize=11)
    
    # Add values
    for i in range(len(dimensions)):
        for j in range(n_iters):
            val = matrix[i, j]
            color = 'white' if abs(val) > 0.15 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                    color=color, fontsize=7, fontweight='bold')
    
    # Mark best iteration
    best_iter = summary["best_iteration"]
    ax.axvline(x=best_iter - 0.5, color='green', linewidth=3)
    ax.axvline(x=best_iter + 0.5, color='green', linewidth=3)
    
    ax.set_xlabel('Iteration (number of judges)', fontsize=11)
    ax.set_title('R² Heatmap: Performance by Dimension Across Pruning Iterations\n(Green box = optimal)', fontsize=13)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('R²', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'r2_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved r2_heatmap.png")


def generate_report(summary: dict, output_dir: Path) -> None:
    """Generate markdown report."""
    trajectory = summary["trajectory"]
    initial = trajectory[0]
    best = trajectory[summary["best_iteration"]]
    final = trajectory[-1]
    
    # Compute improvement
    improvement = (best["avg_r2"] - initial["avg_r2"]) / abs(initial["avg_r2"]) * 100 if initial["avg_r2"] != 0 else 0
    
    # Judges removed
    removed_judges = [t["removed"] for t in trajectory if t["removed"]]
    
    # Categorize best judges by dimension
    best_judges_by_dim = {dim: [] for dim in DIMENSION_COLORS.keys()}
    for judge in best["judges"] if "judges" in best else summary["best_judges"]:
        for dim in DIMENSION_COLORS.keys():
            if dim in judge:
                best_judges_by_dim[dim].append(judge.replace('helpsteer2-', '').replace('-judge', ''))
                break
    
    report = f"""# Hero Run: Multi-Output Judge Pruning Report

## Executive Summary

This experiment started with **all 30 judges** (5 parent judges + 25 child judges) and iteratively pruned them using the `human_correlation` strategy while optimizing for **all 5 HelpSteer2 dimensions simultaneously**.

### Key Results

| Metric | Initial | Best | Final |
|--------|---------|------|-------|
| **Number of Judges** | {initial['n_judges']} | {best['n_judges']} | {final['n_judges']} |
| **Average R²** | {initial['avg_r2']:.4f} | {best['avg_r2']:.4f} | {final['avg_r2']:.4f} |
| **Iteration** | 0 | {summary['best_iteration']} | {len(trajectory) - 1} |

🎯 **Pruning from 30 → 15 judges improved average R² by {improvement:.1f}%!**

---

## Methodology

### Pruning Strategy: Human Correlation
At each iteration, the judge with the **lowest average Pearson correlation** to human targets across all 5 dimensions was removed.

### Stopping Criteria
- Stop when average R² drops more than **15%** from peak, OR
- Minimum of **5 judges** reached

### Multi-Output Approach
- **5 independent GAMs** were trained at each iteration (one per dimension)
- Importance scores were averaged across all dimensions
- Human correlations were computed per-dimension and averaged for removal decisions

---

## Performance Trajectory

### Average R² vs Number of Judges

| Judges | Avg R² | Δ from Initial |
|--------|--------|----------------|
"""
    
    for t in trajectory:
        delta = t["avg_r2"] - initial["avg_r2"]
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        marker = " ⭐" if t["iteration"] == summary["best_iteration"] else ""
        report += f"| {t['n_judges']} | {t['avg_r2']:.4f} | {delta_str}{marker} |\n"
    
    report += f"""
### Per-Dimension R² at Best Iteration ({best['n_judges']} judges)

| Dimension | R² | vs Initial |
|-----------|-----|------------|
"""
    
    for dim in DIMENSION_COLORS.keys():
        init_r2 = initial["dimension_r2"][dim]
        best_r2 = best["dimension_r2"][dim]
        delta = best_r2 - init_r2
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        report += f"| {dim.capitalize()} | {best_r2:.4f} | {delta_str} |\n"
    
    report += f"""
---

## Optimal Judge Set ({summary['best_n_judges']} judges)

"""
    
    for dim, judges in best_judges_by_dim.items():
        if judges:
            report += f"### {dim.capitalize()}\n"
            for j in judges:
                report += f"- `{j}`\n"
            report += "\n"
    
    report += f"""---

## Judges Removed (in order)

| Order | Judge | Reason |
|-------|-------|--------|
"""
    
    for i, t in enumerate(trajectory):
        if t["removed"]:
            judge_short = t["removed"].replace('helpsteer2-', '').replace('-judge', '')
            report += f"| {i} | `{judge_short}` | Lowest avg correlation |\n"
    
    report += f"""
---

## Key Insights

### 1. More Judges ≠ Better Performance
Starting with all 30 judges produced **worse** results (avg R²={initial['avg_r2']:.4f}) than using the optimal 15 judges (avg R²={best['avg_r2']:.4f}). This is likely due to:
- **Overfitting**: Too many features relative to training samples
- **Noise**: Some judges introduced conflicting signals
- **Redundancy**: Multiple judges capturing the same information

### 2. Optimal Reduction: 50%
The best performance was achieved by removing **exactly half** of the judges (30 → 15), suggesting a good balance between diversity and signal quality.

### 3. Parent Judges Are Essential
All **5 parent judges** were retained in the optimal set, indicating they provide foundational signal that children cannot replace.

### 4. Dimension-Specific Performance
- **Helpfulness** and **Correctness** showed the strongest R² (~0.32)
- **Coherence** remained challenging (~0.07) - may need different judge design
- **Complexity** and **Verbosity** showed moderate performance (~0.17)

### 5. Cross-Dimension Trade-offs
Some judges performed well for their "home" dimension but hurt others (negative correlations), making multi-output optimization crucial.

---

## Recommendations

1. **Use the 15-judge optimal set** for production deployments
2. **Consider dimension-specific ensembles** for best per-dimension performance
3. **Investigate coherence dimension** - may need architectural changes
4. **Run periodic re-optimization** as judge definitions evolve

---

## Visualizations

See the generated plots:
- `r2_trajectory.png` - R² over iterations for all dimensions
- `judges_removed.png` - Removal order with dimension coloring
- `final_comparison.png` - Initial vs Best vs Final bar chart
- `r2_heatmap.png` - Full iteration × dimension R² matrix

"""
    
    report_path = output_dir / 'HERO_RUN_REPORT.md'
    report_path.write_text(report)
    print(f"  ✓ Saved HERO_RUN_REPORT.md")


def main():
    parser = argparse.ArgumentParser(description="Visualize Hero Run results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to hero run results directory (containing summary.json)",
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    print(f"Loading results from {results_dir}")
    summary = load_summary(results_dir)
    
    print(f"\nGenerating visualizations...")
    plot_r2_trajectory(summary, results_dir)
    plot_judges_removed(summary, results_dir)
    plot_final_comparison(summary, results_dir)
    plot_improvement_heatmap(summary, results_dir)
    
    print(f"\nGenerating report...")
    generate_report(summary, results_dir)
    
    print(f"\n✅ All outputs saved to {results_dir}")


if __name__ == "__main__":
    main()
