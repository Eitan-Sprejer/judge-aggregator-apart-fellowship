# Judge Model Comparison: GPT-5-Mini vs GPT-5-Nano

## ✅ Decision: Use GPT-5-Nano

**Conclusion**: GPT-5-Nano is **acceptable for judge evaluation** despite ~0.25 lower correlation with humans. The 5x cost savings justify the tradeoff for most research applications.

👉 **See [FINDINGS.md](./FINDINGS.md) for complete analysis and detailed results**

## Overview

This experiment compared judge scoring between `gpt-5-mini` (~$0.25/1M input) and `gpt-5-nano` (~5x cheaper) to determine if nano is a viable cost-saving alternative for judge evaluation.

## Research Question

**Is `gpt-5-nano` sufficiently accurate compared to `gpt-5-mini` for judge scoring?**

We examined:
1. **Mini-Nano Score Correlation**: Do both models give similar scores per judge?
2. **Human Label Correlation**: Which model correlates better with human annotations?
3. **Judgment Quality**: Are nano's judgments sensible and intuitive?

## Methodology

### Dataset
- **Source**: HelpSteer2
- **Samples**: 100 (same subset for both models)
- **Dimensions**: helpfulness, correctness
- **Random seed**: 42 (reproducibility)

### Judges
- Auto-created judges at depth=1 (parent + children)
- Same judge definitions for both models
- Only the scoring model differs

### Metrics
- **Pearson r**: Linear correlation between mini and nano scores
- **Spearman ρ**: Rank correlation (more robust to outliers)
- **Agreement**: % of samples where scores differ by ≤0.5
- **MAE**: Mean absolute difference between scores

## Usage

### Run the Experiment

```bash
# From repository root
python experiments/track1_baseline_performance/judge_model_comparison/run_comparison.py
```

This will:
1. Load 100 samples from HelpSteer2
2. Evaluate all judges with gpt-5-mini
3. Evaluate all judges with gpt-5-nano
4. Save results to `results/comparison_<timestamp>/`

### Analyze Results

```bash
python experiments/track1_baseline_performance/judge_model_comparison/analyze_comparison.py results/comparison_<timestamp>
```

This generates:
- `mini_nano_correlation.csv` - Per-judge correlation between models
- `dimension_summary.csv` - Human correlation by dimension
- `mini_nano_scatter.png` - Visual score comparison
- `comparison_report.md` - Summary with recommendation

## Expected Cost

- Mini: ~$0.04 for 100 samples × 12 judges
- Nano: ~$0.008 for 100 samples × 12 judges
- Total: ~$0.05

## Interpretation Guide

### Mini-Nano Agreement
| Metric | Good | Acceptable | Concerning |
|--------|------|------------|------------|
| Pearson r | >0.9 | 0.8-0.9 | <0.8 |
| Agreement (±0.5) | >85% | 70-85% | <70% |

### Human Correlation Difference
| Difference | Interpretation |
|------------|----------------|
| |Δ| < 0.05 | Negligible - Nano equivalent |
| |Δ| < 0.10 | Minor - Nano acceptable for most uses |
| |Δ| > 0.10 | Significant - Consider Mini for critical evaluation |

## File Structure

```
judge_model_comparison/
├── README.md                   # This file
├── FINDINGS.md                 # Complete analysis and results
├── run_comparison.py           # Main runner script
├── analyze_comparison.py       # Analysis script
└── results/                    # Output directory
    └── comparison_20251129_133938/
        ├── config.json                    # Experiment configuration
        ├── combined_results.pkl           # All scores + human labels
        ├── mini_scores.pkl                # Mini judge scores
        ├── nano_scores.pkl                # Nano judge scores
        ├── mini_nano_correlation.csv      # Per-judge Mini-Nano correlation
        ├── mini_nano_scatter.png          # Visual comparison
        └── parent_human_correlation.csv   # Parent judges vs human annotations
```

## Key Results Summary

### Mini vs Nano Correlation (12 judges, 100 samples)
- Average Pearson r: **0.62**
- Agreement (±0.5): **68%**
- Agreement (±1.0): **88%**

### Parent Judges vs Human Annotations
| Dimension | Mini r | Nano r | Difference |
|-----------|--------|--------|------------|
| **Helpfulness** | 0.55 | 0.30 | -0.25 |
| **Correctness** | -0.05 | -0.00 | -0.05 (both broken) |

### Decision
✅ **Use Nano** - sensible judgments, 5x cheaper, acceptable correlation tradeoff

---

**See [FINDINGS.md](./FINDINGS.md) for detailed analysis including qualitative judgment examples**

## Related Work

This experiment is part of **Track 1: Baseline Performance Comparison** in the Apart Fellowship research.

See `docs/methodology_proposals.md` for the full research plan.
