# Track 4: Aggregator Validation

**Priority**: SECONDARY (15% effort)

**Goal**: Validate that aggregators correctly decompose known preference structures

## Research Questions

1. Can aggregators recover ground truth aggregation recipes from synthetic data?
2. Can we reverse-engineer persona internal preference systems from their ratings?
3. What aggregation functions are learnable vs. fundamentally ambiguous?

## Planned Experiments

### 4.1 Synthetic Preference Decomposition
- **Status**: Planned
- **Methodology**:
  1. Create synthetic ground truth: `human_score = 0.3*truthfulness + 0.5*helpfulness + 0.2*clarity`
  2. Generate dataset with known aggregation recipe
  3. Train GAM/MLP aggregators
  4. Compare learned weights to ground truth
- **Directory**: `4.1_synthetic_decomposition/`
- **Validation**: Can aggregators recover the 0.3/0.5/0.2 weights?

### 4.2 Persona Preference Recovery
- **Status**: Planned
- **Dataset**: Persona simulation data (from Track 1.1)
- **Analysis**: Reverse-engineer each persona's internal weighting
- **Directory**: `4.2_persona_preference_recovery/`
- **Key Question**: Does "Professor" actually weight logical consistency highly? Can we prove it?

## Key Contribution

Validates that judge importance measures are meaningful, not artifacts:
- If aggregators can't recover known preferences → importance scores unreliable
- If aggregators succeed → confidence in Track 2 interpretability results

## Methodology

**Synthetic Ground Truth**:
```python
# Generate data with known recipe
def generate_ground_truth(judge_scores):
    weights = {
        'truthfulness': 0.3,
        'helpfulness': 0.5,
        'clarity': 0.2
    }
    return sum(weights[j] * judge_scores[j] for j in weights)

# Train aggregator, compare learned weights
```

**Persona Analysis**:
- Train separate aggregator per persona
- Extract judge importance for each
- Compare to persona definitions (from `pipeline/core/persona_simulation.py`)
- Expected: "Professor" → high logical consistency, "Child" → high clarity

**Failure Mode Analysis**:
- Test ambiguous cases: Can aggregators distinguish 0.5X + 0.5Y from 0.3X + 0.7Y?
- Identify when multiple aggregation recipes fit data equally well
- Characterize identifiability limits

## Expected Outcomes

- Confidence bounds on judge importance estimates
- Validation: "Aggregators recover ground truth within ±0.05 weight error"
- Persona preference profiles:
  - Professor: [Logical Consistency: 0.35, Truthfulness: 0.30, ...]
  - Child: [Clarity: 0.40, Helpfulness: 0.25, ...]
- Identifiability analysis: Which preference structures are recoverable?

## Dependencies

- Uses synthetic data generation capabilities
- Requires persona simulation from Track 1.1
- Validates Track 2 interpretability claims

## Technical Notes

**Validation Metrics**:
- Weight recovery error: `|learned_weight - true_weight|`
- Rank correlation: Do top judges match ground truth top judges?
- Prediction accuracy: R² on held-out synthetic data

**Failure Criteria**:
- If weight recovery error > 0.15 → aggregator unreliable
- If rank correlation < 0.7 → importance rankings questionable
- Informs confidence in Track 2 conclusions
