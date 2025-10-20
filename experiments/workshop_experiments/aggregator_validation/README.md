# Aggregator Validation on Less Varied Data

**Research Question**: Does our aggregator's modest R² (~0.58) reflect genuine performance limitations, or is it an artifact of our simulated persona data being too varied?

## Motivation

When human annotators have diverse preferences, even a perfect aggregator will show lower R² scores. To isolate the aggregator's actual performance from natural human variance, we test on two less-varied targets:

1. **UltraFeedback's overall_score**: Single GPT-4 reviewer's judgment
2. **Single-persona preferences**: Train and test on one consistent persona at a time

If R² scores improve significantly on these targets, it confirms that multi-persona variance (not aggregator quality) was the bottleneck.

## Approach

The experiment uses the same 10 judges but evaluates against ground truth with lower variance. This lets us measure the aggregator's ceiling performance when human feedback is more consistent.

## Running the Experiment

```bash
python run_experiment.py
```

## Files

- `run_experiment.py` - Main experiment runner
- `data_preparation.py` - Data loading and preprocessing
- `training_functions.py` - Model training utilities
- `visualizations.py` - Result plotting
- `experiment_results_full/` - Saved results and models
- `short_description.md` - Brief motivation notes
