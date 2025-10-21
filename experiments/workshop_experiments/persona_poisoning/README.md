# Aggregator Robustness to Contaminated Training Data

**Research Question**: How well does a learned MLP aggregator handle contaminated human feedback during training?

## Approach

Train MLP aggregators on human feedback with varying levels of contamination, then test all models on a clean test set. We compare against single judge baselines to see if multi-judge aggregation offers robustness benefits.

**Contamination types tested**:
- **Random noise**: ±3 random error per rating (simulates inconsistent annotators)
- **Systematic bias**: Consistent +2/-2 offset per annotator (simulates scale misalignment)
- **Scale compression**: Compress [0,10] → [3,7] (simulates annotators avoiding extremes)

**Contamination rates**: 0%, 5%, 10%, 15%, 20%, 25%, 30%, 40%, 50%

## Results

At 25% contamination, the learned aggregator retains 79% of its clean performance while the best single judge drops to 58%.

| Method | Clean R² | 25% Contamination | Drop |
|--------|----------|-------------------|------|
| Learned Aggregator | 0.514 | 0.404 | 21% |
| Mean of Judges | 0.518 | 0.334 | 36% |
| Best Single Judge | 0.466 | 0.271 | 42% |

**Breaking points**: Aggregator degrades significantly after 30% contamination, while single judges fail around 25%.

**Vulnerability by type**: Most robust to random noise (2.4% drop at 50% contamination), most vulnerable to scale compression (24% drop).

## Running the Experiment

```bash
# Train models at different contamination levels
python run_aggregator_robustness.py --data ../../dataset/data_with_judge_scores.pkl

# Generate comparison figures
python analyze_with_baselines.py
```

## Files

- `run_aggregator_robustness.py` - Main training script
- `results/contamination_analysis.png` - 6-panel comparison figure
- `results/complete_analysis.json` - Raw results for all methods
- `CORRECTED_METHODOLOGY_REPORT.md` - Detailed methodology notes
