# Codebase Refactoring Guide

**Last Updated**: 2025-01

This guide explains the refactored codebase structure for the fellowship stage. The codebase was refactored from workshop-paper code into a flexible framework supporting diverse fellowship experiments.

## Quick Start: What Changed?

### For Workshop Code Users
Workshop code still works! Old experiments moved to `experiments/workshop_experiments/` and use the legacy system.

### For New Fellowship Experiments
New experiments use:
- **Standardized data format** (all datasets → same columns)
- **Flexible judge configuration** (5, 7, 10, or any number of judges)
- **YAML configs** instead of hardcoded values
- **Modular experiment structure** (track-based organization)

---

## Key Changes

### 1. Standardized Data Format

**Before** (workshop): Different column names per dataset
```python
# UltraFeedback had: instruction, answer, source
# Other formats varied
```

**After** (fellowship): All datasets use same columns
```python
{
    'question': str,      # The input/prompt
    'response': str,      # The model's answer
    'dataset': str,       # Source (e.g., 'ultrafeedback')
    'target': float,      # Optimization target (human/persona score)
    'target_type': str,   # 'persona', 'human', 'aggregate', etc.
    'target_score_range': tuple  # (min, max) e.g., (1.0, 10.0)
}
```

### 2. Flexible Judge Configuration

**Before** (workshop): Always 10 hardcoded judges
```python
# Hardcoded in pipeline/utils/judge_rubrics.py
JUDGE_IDS = [
    "truthfulness-judge",
    "harmlessness-judge",
    # ... always exactly 10
]
```

**After** (fellowship): Configure any judge set
```python
from pipeline.config import JudgeConfig

# Use 5 judges
judges = JudgeConfig(
    judge_ids=["truthfulness-judge", "harmlessness-judge", "helpfulness-judge",
               "clarity-judge", "conciseness-judge"],
    judge_names=["Truthfulness", "Harmlessness", "Helpfulness", "Clarity", "Conciseness"],
    score_range=(0.0, 4.0)
)

# Or use all 10 (default)
from pipeline.config import DEFAULT_10_JUDGES
judges = DEFAULT_10_JUDGES
```

### 3. Configuration System

**Before** (workshop): Hardcoded params + `training_config.json`
```python
# Hardcoded in code
n_judges = 10
hidden_dim = 64
# ...
```

**After** (fellowship): YAML configs or programmatic
```yaml
# config/my_experiment.yaml
name: "track1-experiment"
dataset: "ultrafeedback"
dataset_kwargs:
  n_samples: 2000
  with_personas: true

judges:
  judge_ids: [...]
  judge_names: [...]

models:
  gam:
    n_splines: 10
    lam: 0.6
  mlp:
    hidden_dim: 64
    learning_rate: 0.005
```

```python
# Load from YAML
from pipeline.config import ExperimentConfig
config = ExperimentConfig.from_yaml("config/my_experiment.yaml")

# Or create programmatically
config = ExperimentConfig(
    name="my-experiment",
    dataset="ultrafeedback",
    judges=my_judge_config,
    dataset_kwargs={'n_samples': 2000}
)
```

### 4. Directory Structure

**Before** (workshop):
```
experiments/
├── persona_poisoning/
├── rubric_sensitivity/
└── aggregator_validation/
```

**After** (fellowship):
```
experiments/
├── workshop_experiments/          # Legacy workshop code
│   ├── persona_poisoning/
│   ├── rubric_sensitivity/
│   └── aggregator_validation/
├── track1_baseline_performance/   # NEW: Fellowship experiments
│   ├── 1.1_persona_synthetic/
│   ├── 1.2_maj_eval_comparison/
│   └── 1.3_judge_bench/
├── track2_judge_interpretability/
├── track3_automated_selection/
└── track4_aggregator_validation/
```

---

## New Components

### `pipeline/config/`
Configuration classes for experiments:
- `ExperimentConfig`: Top-level experiment configuration
- `JudgeConfig`: Judge selection and scoring
- `ModelConfig`: GAM and MLP hyperparameters
- `GAMConfig`, `MLPConfig`: Individual model configs

**Example**:
```python
from pipeline.config import ExperimentConfig, create_default_config

# Create with defaults
config = create_default_config(
    name="my-experiment",
    dataset="ultrafeedback",
    n_samples=2000
)

# Or load from YAML
config = ExperimentConfig.from_yaml("config/my_experiment.yaml")

# Access components
print(config.judges.n_judges)  # Number of judges
print(config.models.mlp.hidden_dim)  # MLP hidden dimension
```

### Updated Components

#### `pipeline/core/dataset_loader.py`
- **New method**: `load(dataset_name, **kwargs)` - returns standardized format
- **Preprocessors**: `_preprocess_ultrafeedback()`, `_preprocess_judge_bench()`, `_preprocess_maj_eval()`
- **Validation**: `_validate_standardized_format()` checks required columns

**Example**:
```python
from pipeline.core.dataset_loader import DatasetLoader

loader = DatasetLoader()
data = loader.load(
    dataset_name='ultrafeedback',
    n_samples=2000,
    with_personas=True
)

# Data has standardized columns: question, response, target, dataset, ...
print(data.columns)
```

#### `pipeline/core/judge_evaluation.py`
- **New parameter**: `judge_ids` - list of judges to use
- **Requires standardized format**: Data must have `question` and `response` columns

**Example**:
```python
from pipeline.core.judge_evaluation import JudgeEvaluator

# Use specific judges
evaluator = JudgeEvaluator(
    judge_ids=["truthfulness-judge", "clarity-judge", "helpfulness-judge"]
)

# Requires standardized format with 'question' and 'response' columns
data = evaluator.evaluate_dataset(data)
```

#### `pipeline/core/aggregator_training.py`
- **GAM**: Accepts `feature_names` parameter (defaults to `DEFAULT_10_JUDGES.judge_names`)
- **MLP**: Uses `n_features` instead of hardcoded `n_judges=10`
- **Data loading**: Requires standardized format with `target` and `judge_scores` columns

**Example**:
```python
from pipeline.core.aggregator_training import GAMAggregator, MLPTrainer

# GAM with custom feature names
gam = GAMAggregator(
    feature_names=["Truthfulness", "Clarity", "Helpfulness"],
    n_splines=10,
    lam=0.6
)
gam.fit(X, y)

# MLP automatically detects n_features from X.shape[1]
mlp_trainer = MLPTrainer(hidden_dim=64, learning_rate=0.005)
mlp_trainer.fit(X_train, y_train, X_val, y_val)
```

---

## Migration Guide

### Migrating Workshop Experiments

Workshop experiments have been updated to use the new structure. They're located in `experiments/workshop_experiments/` and now use:
- Standardized column names (question/response)
- Config-based judge selection
- Hardcoded MLP defaults (no config file)
- Direct imports from `pipeline.config`

### Creating New Fellowship Experiments

**Step 1: Create experiment directory**
```bash
mkdir -p experiments/track1_baseline_performance/1.1_persona_synthetic
cd experiments/track1_baseline_performance/1.1_persona_synthetic
```

**Step 2: Create README.md** (see track READMEs for templates)

**Step 3: Create config** (optional, or use programmatic config)
```yaml
# config.yaml
name: "1.1-persona-synthetic"
dataset: "ultrafeedback"
dataset_kwargs:
  n_samples: 2000
  with_personas: true

judges:
  judge_ids: ["truthfulness-judge", "harmlessness-judge", ...]
  judge_names: ["Truthfulness", "Harmlessness", ...]
  score_range: [0.0, 4.0]

models:
  train_gam: true
  train_mlp: true
  gam:
    n_splines: 10
    lam: 0.6
  mlp:
    hidden_dim: 64
    learning_rate: 0.005
```

**Step 4: Create run_experiment.py**
```python
from pathlib import Path
from pipeline.config import ExperimentConfig
from pipeline.core.dataset_loader import DatasetLoader
from pipeline.core.judge_evaluation import JudgeEvaluator
from pipeline.core.aggregator_training import GAMAggregator, MLPTrainer, load_and_prepare_data

# Load config
config = ExperimentConfig.from_yaml("config.yaml")

# Load dataset (automatically standardized)
loader = DatasetLoader()
data = loader.load(config.dataset, **config.dataset_kwargs)

# Evaluate with judges
evaluator = JudgeEvaluator(judge_ids=config.judges.judge_ids)
data = evaluator.evaluate_dataset(data)

# Save intermediate results
data.to_pickle("results/data_with_scores.pkl")

# Train models
_, X, y = load_and_prepare_data("results/data_with_scores.pkl")

# GAM
gam = GAMAggregator(
    feature_names=config.judges.judge_names,
    n_splines=config.models.gam.n_splines,
    lam=config.models.gam.lam
)
gam.fit(X_train, y_train)

# MLP
mlp = MLPTrainer(
    hidden_dim=config.models.mlp.hidden_dim,
    learning_rate=config.models.mlp.learning_rate
)
mlp.fit(X_train, y_train, X_val, y_val)
```

---

## Common Patterns

### Pattern 1: Using Different Judge Sets

**Track 2.2** (cross-task analysis) might use different judge subsets per task:

```python
# Task 1: Safety-focused judges
safety_judges = JudgeConfig(
    judge_ids=["truthfulness-judge", "harmlessness-judge", "honesty-judge"],
    judge_names=["Truthfulness", "Harmlessness", "Honesty"],
    score_range=(0.0, 4.0)
)

# Task 2: Quality-focused judges
quality_judges = JudgeConfig(
    judge_ids=["clarity-judge", "conciseness-judge", "logical-consistency-judge"],
    judge_names=["Clarity", "Conciseness", "Logical Consistency"],
    score_range=(0.0, 4.0)
)
```

### Pattern 2: Different Target Types

**Track 1.1** (persona scores) vs **Track 1.3** (human annotations):

```python
# Persona target (1-10 scale)
data = loader.load('ultrafeedback', with_personas=True)
# data['target_type'] = 'persona'
# data['target_score_range'] = (1.0, 10.0)

# Human annotation target (varies by dataset)
data = loader.load('judge_bench', task_name='summarization')
# data['target_type'] = 'human'
# data['target_score_range'] = (0.0, 5.0)  # Example
```

### Pattern 3: Hyperparameter Tuning

Hyperparameter tuning now works with flexible judge counts:

```python
from analysis.gam_hyperparameter_tuning import GAMHyperparameterTuner

tuner = GAMHyperparameterTuner(
    feature_names=config.judges.judge_names,  # Dynamic!
    param_grid={
        'n_splines': [8, 10, 12],
        'lam': [0.4, 0.6, 0.8]
    }
)

best_gam = tuner.tune(X_train, y_train, X_val, y_val)
```

---

## Breaking Changes

### What Was Removed?

1. **`load_training_config()`**: Deleted - use `ExperimentConfig` or hardcoded defaults
2. **`training_config.json`**: Deleted - use YAML configs or hardcoded values
3. **Auto-detection logic**: Removed - data must use standardized column names
4. **`load_existing_personas()`**: Removed - use `DatasetLoader.load()` instead
5. **Individual judge getters**: Removed - use `JUDGE_RUBRICS` dictionary
6. **`DEFAULT_FEATURE_LABELS`**: Removed - use `DEFAULT_10_JUDGES.judge_names`

### Migration Required For

**Old data with non-standard columns**:
- Must preprocess to standardized format (`question`, `response`, `target`, `judge_scores`)
- No auto-detection - data must have exact column names

**Code using removed functions**:
- Replace `load_training_config()` with hardcoded defaults or config objects
- Replace `get_*_judge_rubric()` with `JUDGE_RUBRICS['judge-id']()`
- Replace `load_existing_personas()` with `DatasetLoader.load()`

**Code assuming 10 judges**:
- Update to use `config.judges.n_judges` dynamically
- Pass `feature_names` explicitly to GAM

---

## File Organization

### Configuration Files
```
config/
└── example_config.yaml          # Template for fellowship experiments
```

### Pipeline Components
```
pipeline/
├── config/                      # NEW: Configuration classes
│   ├── __init__.py
│   └── experiment_config.py
├── core/                        # UPDATED: Flexible components
│   ├── dataset_loader.py        # Standardized format
│   ├── judge_evaluation.py      # Dynamic judge_ids
│   ├── aggregator_training.py   # Flexible n_features
│   ├── persona_simulation.py
│   └── baseline_models.py
└── utils/
    ├── judge_rubrics.py
    └── data_merger.py
```

### Experiments
```
experiments/
├── workshop_experiments/        # Legacy code (backward compat)
│   ├── persona_poisoning/
│   ├── rubric_sensitivity/
│   └── aggregator_validation/
└── track{1-4}_{name}/          # NEW: Fellowship experiments
    ├── README.md               # Track overview
    └── {N}.{M}_{experiment}/
        ├── README.md           # Experiment methodology
        ├── config.yaml         # Optional config
        ├── run_experiment.py   # Execution script
        └── results/            # Outputs
```

---

## Troubleshooting

### "Data missing required columns: ['question', 'response']"
**Cause**: Data not in standardized format
**Fix**: Use `DatasetLoader.load()` to preprocess data, or manually rename columns to standardized format

### "Data must have 'target' and 'judge_scores' columns"
**Cause**: Missing required columns for training
**Fix**: Run persona simulation (sets 'target') and judge evaluation (sets 'judge_scores') first

### "Judge names length doesn't match n_features"
**Cause**: Provided feature_names list doesn't match number of judges
**Fix**: Ensure config.judges.judge_names has same length as config.judges.judge_ids

### Import errors with `pipeline.config`
**Cause**: Module not in path
**Fix**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline.config import ExperimentConfig
```

---

## Quick Reference

### Loading Data (Standardized)
```python
from pipeline.core.dataset_loader import DatasetLoader
loader = DatasetLoader()
data = loader.load('ultrafeedback', n_samples=2000, with_personas=True)
```

### Configuring Judges
```python
from pipeline.config import JudgeConfig, DEFAULT_10_JUDGES
judges = DEFAULT_10_JUDGES  # All 10 judges
# Or custom: JudgeConfig(judge_ids=[...], judge_names=[...])
```

### Training Models
```python
from pipeline.core.aggregator_training import GAMAggregator, MLPTrainer

gam = GAMAggregator(feature_names=config.judges.judge_names)
gam.fit(X, y)

mlp = MLPTrainer(hidden_dim=64)
mlp.fit(X_train, y_train, X_val, y_val)
```

### Full Config-Based Workflow
```python
config = ExperimentConfig.from_yaml("config.yaml")
loader = DatasetLoader()
data = loader.load(config.dataset, **config.dataset_kwargs)
evaluator = JudgeEvaluator(judge_ids=config.judges.judge_ids)
data = evaluator.evaluate_dataset(data)
# ... train models with config.models.*
```

---

## Additional Resources

- **Track READMEs**: `experiments/track*/README.md` - Research goals and experiments
- **CLAUDE.md**: Updated with fellowship context and new structure
- **methodology_proposals.md**: Detailed track descriptions and methodologies
- **example_config.yaml**: Full config file template with all options
