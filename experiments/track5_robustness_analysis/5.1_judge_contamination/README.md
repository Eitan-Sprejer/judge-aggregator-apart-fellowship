# Track 5.1: Judge Contamination Experiment

This experiment tests whether learned aggregators (GAM and MLP) can detect and resist contaminated judges in the evaluation pipeline.

## Overview

**Research Question**: Can interpretable aggregation models identify and downweight contaminated judges while maintaining performance on clean evaluation data?

**Key Hypothesis**: GAM and MLP models should learn to assign lower importance to contaminated judges compared to clean judges, demonstrating robustness against adversarial evaluation scenarios.

## Contamination Framework

### Extensible Architecture

The experiment implements an extensible contamination framework using the Strategy pattern:

```python
class ContaminationStrategy(ABC):
    @abstractmethod
    def contaminate_rubric(self, original_rubric: str, judge_info: Dict) -> str:
        """Apply contamination to a judge's rubric."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get description of this contamination strategy."""
        pass
```

### Available Contamination Strategies

1. **InvertedRubricStrategy**: Flips scoring scale (4.0 → 0.0, 0.0 → 4.0)
   - Most direct test of contamination resistance
   - High scores become low, low scores become high
   - Completely reverses judge preferences

2. **RandomNoiseStrategy**: Adds ±noise to all scores
   - Configurable noise level (default: ±1.0 points)
   - Tests robustness to unreliable judges
   - Simulates evaluation inconsistency

3. **BiasedStrategy**: Always gives high or low scores
   - Can be configured for systematic high bias (always ~4.0) or low bias (always ~0.0)
   - Tests resistance to judges that ignore content quality
   - Simulates lenient or harsh evaluators

### Adding New Contamination Types

To add a new contamination strategy:

```python
class MyContaminationStrategy(ContaminationStrategy):
    def contaminate_rubric(self, original_rubric: str, judge_info: Dict) -> str:
        # Implement your contamination logic
        return modified_rubric
    
    def get_description(self) -> str:
        return "Description of your strategy"
```

## Pipeline Architecture

### Step-by-Step Process

1. **Dataset Loading**: Loads pre-evaluated dataset with human preference scores
2. **Judge Selection**: Randomly selects subset of judges for contamination
3. **Rubric Contamination**: Applies contamination strategy to selected judges' rubrics
4. **Parallel Evaluation**: Evaluates samples with both clean and contaminated judges
5. **Score Aggregation**: Combines all judge scores into training matrix
6. **Model Training**: Trains GAM and MLP aggregators on contaminated data
7. **Resistance Analysis**: Analyzes whether models learned to ignore contaminated judges

### Why This Works

**GAM Interpretability**: GAM provides direct feature importance scores showing which judges matter most. If working correctly, clean judges should have higher importance than contaminated ones.

**MLP Weight Analysis**: While less interpretable, MLP input layer weights provide proxy for judge importance. Contamination resistance manifests as lower average weights for contaminated judges.

**Resistance Metrics**: The resistance ratio (clean_importance / contaminated_importance) quantifies how well models distinguished between clean and contaminated judges. Ratios > 1.5 indicate successful contamination detection.

### Expected Behavior

- **Inverted Strategy**: Should be easily detected as it completely reverses preferences
- **Noise Strategy**: Moderate detection difficulty - depends on noise level vs signal strength  
- **Bias Strategy**: Detection depends on how different bias is from true score distribution

## Usage

### Basic Usage
```bash
# Run with default settings (inverted contamination, 30% rate, 100 samples)
python run_experiment.py

# Custom configuration
python run_experiment.py --data-size 500 --contamination-rate 0.2 --strategy noise --seed 123
```

### Available Arguments
- `--data-size`: Number of samples to evaluate (default: 100)
- `--contamination-rate`: Fraction of judges to contaminate (default: 0.3)
- `--strategy`: Contamination strategy (choices: inverted, noise, bias-high, bias-low)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-dir`: Custom output directory

### Example Commands
```bash
# Test inverted scoring with high contamination
python run_experiment.py --strategy inverted --contamination-rate 0.5 --data-size 200

# Test noise resistance with large dataset
python run_experiment.py --strategy noise --data-size 1000 --contamination-rate 0.2

# Test bias resistance
python run_experiment.py --strategy bias-high --contamination-rate 0.3
```

## Key Metrics

### Performance Metrics
- **R² Score**: Coefficient of determination for both GAM and MLP
- **MSE**: Mean Squared Error on test set
- **Judge Importance**: Feature importance scores for each judge

### Contamination Resistance Metrics
- **Resistance Ratio**: `clean_judge_avg_importance / contaminated_judge_avg_importance`
- **Detection Success**: Boolean flag for resistance ratio > 1.5
- **Importance Distribution**: Per-judge importance scores

### Expected Results
- **Good Resistance**: Clean judges get higher importance than contaminated judges
- **Detection Success**: Resistance ratio > 1.5 indicates contamination detection
- **Performance Maintenance**: R² scores remain reasonable despite contamination

## Output Structure

Results are saved to `results/{strategy}_contamination_{seed}/`:

```
results/
├── experiment_results.json     # Summary metrics and analysis
├── full_results.pkl           # Complete results including models
└── contaminated_scores.csv    # Judge scores for detailed analysis
```

### experiment_results.json Structure
```json
{
  "experiment_config": {
    "data_size": 100,
    "contamination_rate": 0.3,
    "strategy_description": "Inverted scoring scale",
    "total_judges": 10,
    "clean_judges_count": 7,
    "contaminated_judges_count": 3
  },
  "clean_judges": ["truthfulness-judge", "helpfulness-judge", ...],
  "contaminated_judges": ["harmlessness-judge", "clarity-judge", ...],
  "weight_analysis": {
    "contamination_resistance": {
      "gam_resistance_ratio": 2.67,
      "gam_detected_contamination": true,
      "mlp_resistance_ratio": 1.8,
      "mlp_detected_contamination": true
    }
  },
  "performance_metrics": {
    "gam_r2": 0.65,
    "mlp_r2": 0.68
  }
}
```

## Research Applications

### Extending the Framework
This contamination framework can be easily extended for new research directions:

1. **New Contamination Types**: Implement additional `ContaminationStrategy` classes
2. **Multi-Strategy Testing**: Combine multiple contamination types
3. **Adaptive Contamination**: Dynamic contamination based on judge performance
4. **Cross-Task Analysis**: Test contamination resistance across different tasks

### Future Enhancements

1. **Multi-Strategy Contamination**: Apply different strategies to different judges
2. **Adaptive Contamination Rate**: Adjust contamination based on dataset characteristics  
3. **Contamination Detection**: Automated detection without ground truth
4. **Hierarchical Contamination**: Contaminate judge categories rather than individual judges
5. **Temporal Contamination**: Contamination that varies across evaluation samples
3. **Score Collection**: Get scores from both clean and contaminated judges
4. **Aggregator Training**: Train GAM/MLP models on contaminated data
5. **Weight Analysis**: Analyze if models learned to ignore contaminated judges

## Expected Outcomes

- Clean judges should receive higher weights
- Contaminated judges should receive near-zero weights
- Model performance should remain robust despite contamination

## Implementation Notes

This experiment will use existing pipeline components and extend them with contamination logic.