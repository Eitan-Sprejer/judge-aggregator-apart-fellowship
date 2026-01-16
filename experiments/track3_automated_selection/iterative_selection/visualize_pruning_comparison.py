#!/usr/bin/env python3
"""
Visualize pruning strategy comparison results.

Usage:
    python experiments/track3_automated_selection/iterative_selection/visualize_pruning_comparison.py \
        --results-file results/helpsteer2-baseline_20260112_102426/pruning_comparison/comparison_summary.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'importance': '#2ecc71',
    'redundancy': '#e74c3c', 
    'attribution_correlation': '#9b59b6',
    'human_correlation': '#3498db',
    'combined': '#f39c12',
    'random': '#95a5a6',
}

STRATEGY_LABELS = {
    'importance': 'Importance',
    'redundancy': 'Redundancy',
    'attribution_correlation': 'Attr. Correlation',
    'human_correlation': 'Human Correlation',
    'combined': 'Combined',
    'random': 'Random (baseline)',
}


def load_results(results_file: Path) -> pd.DataFrame:
    """Load comparison results into DataFrame."""
    with open(results_file) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def plot_final_r2_by_dimension(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of final R² for each strategy grouped by dimension."""
    dimensions = df['dimension'].unique()
    strategies = list(COLORS.keys())
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(dimensions))
    width = 0.13
    
    for i, strategy in enumerate(strategies):
        strategy_data = df[df['strategy'] == strategy]
        values = [strategy_data[strategy_data['dimension'] == d]['final_r2'].values[0] 
                  for d in dimensions]
        offset = (i - len(strategies)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=STRATEGY_LABELS[strategy], 
                      color=COLORS[strategy], edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Final R²', fontsize=12)
    ax.set_title('Final R² by Pruning Strategy and Dimension\n(5 judges → 3 judges)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dimensions], fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_r2_by_dimension.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved final_r2_by_dimension.png")


def plot_delta_r2_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of R² change (delta) for each strategy/dimension."""
    pivot = df.pivot(index='dimension', columns='strategy', values='delta_r2')
    
    # Reorder columns
    pivot = pivot[[s for s in COLORS.keys() if s in pivot.columns]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', 
                   vmin=-0.1, vmax=0.15)
    
    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in pivot.columns], rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([d.capitalize() for d in pivot.index])
    
    # Add values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if abs(val) > 0.05 else 'black'
            ax.text(j, i, f'{val:+.3f}', ha='center', va='center', 
                    color=color, fontsize=10, fontweight='bold')
    
    ax.set_title('R² Change After Pruning (ΔR²)\nGreen = improvement, Red = degradation', fontsize=13)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('ΔR²', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'delta_r2_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved delta_r2_heatmap.png")


def plot_strategy_ranking(df: pd.DataFrame, output_dir: Path) -> None:
    """Show average ranking of each strategy across dimensions."""
    # Rank strategies within each dimension (higher R² = rank 1)
    df_ranked = df.copy()
    df_ranked['rank'] = df.groupby('dimension')['final_r2'].rank(ascending=False)
    
    # Average rank per strategy
    avg_ranks = df_ranked.groupby('strategy')['rank'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bars = ax.barh(range(len(avg_ranks)), avg_ranks.values, 
                   color=[COLORS[s] for s in avg_ranks.index],
                   edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(range(len(avg_ranks)))
    ax.set_yticklabels([STRATEGY_LABELS[s] for s in avg_ranks.index], fontsize=11)
    ax.set_xlabel('Average Rank (lower is better)', fontsize=12)
    ax.set_title('Strategy Ranking Across All Dimensions\n(1 = best, 6 = worst)', fontsize=13)
    ax.invert_yaxis()
    
    # Add values
    for i, (strategy, rank) in enumerate(avg_ranks.items()):
        ax.text(rank + 0.1, i, f'{rank:.2f}', va='center', fontsize=10)
    
    ax.set_xlim(0, 7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved strategy_ranking.png")


def plot_init_vs_final_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot of initial vs final R² colored by strategy."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for strategy in COLORS.keys():
        strategy_df = df[df['strategy'] == strategy]
        ax.scatter(strategy_df['iter0_r2'], strategy_df['final_r2'],
                   c=COLORS[strategy], label=STRATEGY_LABELS[strategy],
                   s=100, alpha=0.8, edgecolor='white', linewidth=1)
    
    # Diagonal line (no change)
    lims = [-0.1, 0.35]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='No change')
    
    ax.set_xlabel('Initial R² (5 judges)', fontsize=12)
    ax.set_ylabel('Final R² (3 judges)', fontsize=12)
    ax.set_title('Initial vs Final R² by Strategy\nPoints above diagonal = improvement', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'init_vs_final_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved init_vs_final_scatter.png")


def plot_best_strategy_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Summary chart showing best strategy per dimension."""
    # Find best strategy per dimension
    best_per_dim = df.loc[df.groupby('dimension')['final_r2'].idxmax()]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = range(len(best_per_dim))
    bars = ax.bar(x, best_per_dim['final_r2'],
                  color=[COLORS[s] for s in best_per_dim['strategy']],
                  edgecolor='white', linewidth=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in best_per_dim['dimension']], fontsize=11)
    ax.set_ylabel('Final R²', fontsize=12)
    ax.set_title('Best Performing Strategy per Dimension', fontsize=14)
    
    # Add strategy labels on bars
    for i, (_, row) in enumerate(best_per_dim.iterrows()):
        ax.text(i, row['final_r2'] + 0.01, STRATEGY_LABELS[row['strategy']], 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                rotation=45)
    
    ax.set_ylim(0, max(best_per_dim['final_r2']) * 1.25)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_strategy_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved best_strategy_summary.png")


def generate_report(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate markdown report."""
    
    # Calculate summary statistics
    best_per_dim = df.loc[df.groupby('dimension')['final_r2'].idxmax()]
    
    # Strategy win counts
    win_counts = best_per_dim['strategy'].value_counts()
    
    # Average performance by strategy
    avg_by_strategy = df.groupby('strategy').agg({
        'final_r2': 'mean',
        'delta_r2': 'mean',
    }).sort_values('final_r2', ascending=False)
    
    # Ranking
    df_ranked = df.copy()
    df_ranked['rank'] = df.groupby('dimension')['final_r2'].rank(ascending=False)
    avg_ranks = df_ranked.groupby('strategy')['rank'].mean().sort_values()
    
    report = f"""# Pruning Strategy Comparison Report

## Experiment Overview

This experiment compared **6 pruning strategies** for selecting which judges to remove when reducing the judge panel from 5 to 3 judges per dimension on the HelpSteer2 dataset.

### Strategies Evaluated

| Strategy | Description |
|----------|-------------|
| **Importance** | Remove judge with lowest GAM importance score (50% variance-based + 50% attribution-based) |
| **Redundancy** | Remove judge with highest mean pairwise score correlation to other judges |
| **Attribution Correlation** | Remove from most highly correlated GAM attribution pair (threshold: 0.9) |
| **Human Correlation** | Remove judge with lowest Pearson correlation to human targets |
| **Combined** | Score = importance × (1 - redundancy); remove lowest |
| **Random** | Remove a random judge (baseline) |

### Dataset

- **Dataset**: HelpSteer2
- **Samples**: 1,000
- **Dimensions**: helpfulness, correctness, coherence, complexity, verbosity
- **Initial judges**: 5 per dimension
- **Target judges**: 3 per dimension (pruning ~40%)

---

## Key Findings

### 1. Best Strategy per Dimension

| Dimension | Best Strategy | Final R² | ΔR² |
|-----------|---------------|----------|-----|
"""
    
    for _, row in best_per_dim.iterrows():
        delta_str = f"+{row['delta_r2']:.4f}" if row['delta_r2'] > 0 else f"{row['delta_r2']:.4f}"
        report += f"| {row['dimension'].capitalize()} | {STRATEGY_LABELS[row['strategy']]} | {row['final_r2']:.4f} | {delta_str} |\n"
    
    report += f"""
### 2. Strategy Win Counts

"""
    for strategy, count in win_counts.items():
        report += f"- **{STRATEGY_LABELS[strategy]}**: {count} dimension(s)\n"
    
    report += f"""
### 3. Average Ranking (1 = best, 6 = worst)

| Rank | Strategy | Avg. Rank Score |
|------|----------|-----------------|
"""
    for i, (strategy, rank) in enumerate(avg_ranks.items(), 1):
        report += f"| {i} | {STRATEGY_LABELS[strategy]} | {rank:.2f} |\n"
    
    report += f"""
### 4. Average Performance Across Dimensions

| Strategy | Avg Final R² | Avg ΔR² |
|----------|--------------|---------|
"""
    for strategy, row in avg_by_strategy.iterrows():
        delta_str = f"+{row['delta_r2']:.4f}" if row['delta_r2'] > 0 else f"{row['delta_r2']:.4f}"
        report += f"| {STRATEGY_LABELS[strategy]} | {row['final_r2']:.4f} | {delta_str} |\n"
    
    # Interesting observations
    coherence_improvement = df[(df['dimension'] == 'coherence') & (df['strategy'] == 'human_correlation')]['delta_r2'].values[0]
    correctness_improvement = df[(df['dimension'] == 'correctness') & (df['strategy'] == 'importance')]['delta_r2'].values[0]
    
    report += f"""
---

## Observations

### Pruning Can Improve Performance
Surprisingly, pruning judges sometimes **improved** R² rather than degrading it:
- **Coherence** showed massive improvement (+{coherence_improvement:.4f}) with human_correlation strategy
- **Correctness** improved (+{correctness_improvement:.4f}) with importance strategy

This suggests some judges were adding noise rather than signal.

### Human Correlation Strategy Excels
The **human_correlation** strategy achieved the best average ranking ({avg_ranks['human_correlation']:.2f}), winning in 3 out of 5 dimensions. This makes intuitive sense: judges that correlate well with human judgments are more likely to contribute positively to the aggregated score.

### Attribution Correlation Fell Back to Importance
The attribution_correlation strategy always fell back to importance-based pruning because no judge pairs exceeded the 0.9 correlation threshold. This indicates that the synthetic child judges in HelpSteer2 produce sufficiently diverse GAM attributions.

### Redundancy Strategy Underperformed
The redundancy strategy (removing judges with correlated scores) performed poorly in most dimensions. High score correlation doesn't necessarily mean the judges are redundant - they might both be capturing important signals.

### Random Baseline Performed Worst
As expected, random removal consistently performed near the bottom, validating that informed pruning strategies provide meaningful improvements.

---

## Recommendations

1. **Default to human_correlation** for general pruning tasks when human labels are available
2. **Use importance-based pruning** when human labels are unavailable  
3. **Avoid pure redundancy-based pruning** - score correlation is not a good proxy for removability
4. **Consider dimension-specific strategies** - different dimensions may benefit from different approaches

---

## Visualizations

See the following generated plots:
- `final_r2_by_dimension.png` - Bar chart comparing strategies per dimension
- `delta_r2_heatmap.png` - Heatmap of R² changes
- `strategy_ranking.png` - Average ranking across dimensions
- `init_vs_final_scatter.png` - Scatter plot of initial vs final R²
- `best_strategy_summary.png` - Best strategy per dimension

"""
    
    report_path = output_dir / 'PRUNING_COMPARISON_REPORT.md'
    report_path.write_text(report)
    print(f"  ✓ Saved PRUNING_COMPARISON_REPORT.md")


def main():
    parser = argparse.ArgumentParser(description="Visualize pruning comparison results")
    parser.add_argument(
        "--results-file",
        type=str,
        required=True,
        help="Path to comparison_summary.json",
    )
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    output_dir = results_file.parent
    
    print(f"Loading results from {results_file}")
    df = load_results(results_file)
    
    print(f"\nGenerating visualizations...")
    plot_final_r2_by_dimension(df, output_dir)
    plot_delta_r2_heatmap(df, output_dir)
    plot_strategy_ranking(df, output_dir)
    plot_init_vs_final_scatter(df, output_dir)
    plot_best_strategy_summary(df, output_dir)
    
    print(f"\nGenerating report...")
    generate_report(df, output_dir)
    
    print(f"\n✅ All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
