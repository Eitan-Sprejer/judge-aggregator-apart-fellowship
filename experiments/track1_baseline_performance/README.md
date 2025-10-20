# Track 1: Baseline Performance Comparison

**Priority**: PRIMARY (25% effort)

**Goal**: Validate aggregator performance against baselines on synthetic and real human data

## Research Questions

1. Do learned aggregators outperform naive baselines (mean, median, best-judge)?
2. How do aggregators perform on real human annotations vs. synthetic persona data?
3. Can aggregators trained on synthetic data generalize to real human preferences?

## Planned Experiments

### 1.1 Persona Synthetic Data
- **Status**: Planned
- **Dataset**: UltraFeedback with 8 simulated personas
- **Baselines**: Mean, median, max, best-single-judge
- **Metrics**: R², MAE, correlation with persona consensus
- **Directory**: `1.1_persona_synthetic/`

### 1.2 MAJ-Eval Comparison
- **Status**: Planned
- **Dataset**: MAJ-Eval benchmark data
- **Baseline**: MAJ-Eval's multi-agent debate aggregator
- **Metrics**: Head-to-head performance comparison
- **Directory**: `1.2_maj_eval_comparison/`
- **Note**: Have MAJ-Eval code for comparison

### 1.3 JUDGE-BENCH Validation
- **Status**: Planned
- **Dataset**: JUDGE-BENCH (20 NLP tasks with human annotations)
- **Task Subset**: TBD (need to define which tasks to use)
- **Metrics**: Correlation with human judgments, cross-task generalization
- **Directory**: `1.3_judge_bench/`

## Key Contribution

Demonstrates practical applicability of learned aggregators beyond synthetic scenarios and establishes performance ceiling for interpretability-focused approaches.

## Dependencies

- Track 1.1 results inform Track 2.3 (persona-based judge importance)
- Track 1.3 results feed into Track 2.2 (cross-task judge importance)
- All baselines establish performance benchmarks for other tracks

## Expected Outcomes

- Quantitative evidence that learned aggregation > naive baselines
- Performance gap analysis: synthetic vs. real human data
- Generalization limits: when do aggregators fail?
