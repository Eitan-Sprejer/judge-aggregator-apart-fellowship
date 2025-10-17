# Rubric Sensitivity Analysis

**Research Question**: How sensitive are learned aggregators (GAM/MLP) to variations in judge rubrics compared to simple averaging?

## Setup

We created 5 semantically equivalent variants of each judge's rubric (e.g., "harmlessness" rephrased 5 different ways) and evaluated the same responses. This simulates real-world rubric design variations or evaluator interpretation differences.

Each response gets scored by all 50 judge variants (10 judge types × 5 rubric variants), allowing us to measure how much each aggregation method's predictions vary based on rubric phrasing alone.

## Results

Simple judge averaging is surprisingly robust compared to learned models:

| Model | R² Score | Std Dev | Robustness |
|-------|----------|---------|------------|
| Judge Mean | 0.636 | 0.009 | **Baseline** |
| GAM | 0.640 | 0.013 | 1.4× less robust |
| MLP | 0.650 | 0.016 | 1.8× less robust |

The learned models achieve only 2-4% better performance but show significantly more variance across rubric formulations. This suggests that for many applications, the engineering effort of training and maintaining learned aggregators may not be worth the modest performance gains.

**Key insight**: When rubrics vary (as they do in practice), the simple mean baseline becomes relatively more attractive due to its stability.

## Running the Experiment

```bash
python rubric_robustness_analysis.py
```

Results are saved to `results_full_20250818_215910/`.

## Files

- `rubric_robustness_analysis.py` - Main analysis script
- `results_full_20250818_215910/` - Complete results directory
  - `restructured_scores_fixed.pkl` - Judge scores (1000 samples × 50 judges)
  - `plots_corrected/rubric_robustness_analysis.png` - Visualization
- `src/` - Utility modules for data processing
