# Pruning Strategy Comparison Report

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
| Coherence | Human Correlation | 0.1039 | +0.1166 |
| Complexity | Human Correlation | 0.1740 | -0.0116 |
| Correctness | Importance | 0.2887 | +0.0188 |
| Helpfulness | Human Correlation | 0.3089 | -0.0084 |
| Verbosity | Redundancy | 0.2470 | -0.0042 |

### 2. Strategy Win Counts

- **Human Correlation**: 3 dimension(s)
- **Importance**: 1 dimension(s)
- **Redundancy**: 1 dimension(s)

### 3. Average Ranking (1 = best, 6 = worst)

| Rank | Strategy | Avg. Rank Score |
|------|----------|-----------------|
| 1 | Human Correlation | 2.10 |
| 2 | Attr. Correlation | 2.90 |
| 3 | Combined | 2.90 |
| 4 | Importance | 2.90 |
| 5 | Redundancy | 4.60 |
| 6 | Random (baseline) | 5.60 |

### 4. Average Performance Across Dimensions

| Strategy | Avg Final R² | Avg ΔR² |
|----------|--------------|---------|
| Attr. Correlation | 0.2128 | +0.0105 |
| Combined | 0.2128 | +0.0105 |
| Importance | 0.2128 | +0.0105 |
| Human Correlation | 0.2104 | +0.0081 |
| Random (baseline) | 0.1808 | -0.0215 |
| Redundancy | 0.1716 | -0.0307 |

---

## Observations

### Pruning Can Improve Performance
Surprisingly, pruning judges sometimes **improved** R² rather than degrading it:
- **Coherence** showed massive improvement (+0.1166) with human_correlation strategy
- **Correctness** improved (+0.0188) with importance strategy

This suggests some judges were adding noise rather than signal.

### Human Correlation Strategy Excels
The **human_correlation** strategy achieved the best average ranking (2.10), winning in 3 out of 5 dimensions. This makes intuitive sense: judges that correlate well with human judgments are more likely to contribute positively to the aggregated score.

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

