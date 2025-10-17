# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

Multi-Judge Interpretability is a research framework for evaluating AI outputs using multiple specialized judges and learned aggregation models. The system trains interpretable models (GAM and MLP) that combine judge scores to match human preferences.

**Status**: Completed hackathon project (2nd place at Apart x Martian), with cleaned repository for reproducibility and potential paper submission.

### Core Problem

Current AI evaluation systems have limitations:
- Single judges have limited perspectives
- Naive averaging treats all dimensions equally
- Fixed rules can't capture context-dependent preferences

### Solution

Learn interpretable aggregation functions from data:
- 10 specialized judges evaluate different dimensions
- Human feedback from 8 simulated personas
- GAM/MLP models learn to combine judge scores
- Interpretability analysis shows which judges matter

## Repository Structure

```
judge-aggregator/
├── analysis/                        # Analysis library code (imported by scripts)
│   ├── gam_hyperparameter_tuning.py
│   ├── mlp_hyperparameter_tuning.py
│   ├── correlation_analysis.py
│   └── run_correlation_analysis.py
├── experiments/                     # Self-contained research experiments
│   ├── persona_poisoning/
│   ├── rubric_sensitivity/
│   └── aggregator_validation/
├── pipeline/                        # Core pipeline components
│   ├── core/                        # Judge evaluation, model training, baselines
│   └── utils/                       # Judge rubrics, data utilities
├── dataset/                         # Main dataset
│   └── data_with_judge_scores.pkl   # 2000 samples with judge scores + human feedback
├── results/full_experiments/
│   └── main_experiment_results/     # Primary experiment results
├── config/                          # Training configurations
│   └── training_config.json
├── utils/                           # Logging utilities
│   └── logging_setup.py
├── run_full_experiment.py           # Main CLI: Full experiment pipeline
├── analyze_existing_experiment.py   # Main CLI: Post-hoc GAM analysis
└── gam_stability_analysis.py        # Main CLI: Feature stability analysis
```

## Quick Start

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env  # Add your API keys

# Run full experiment
python run_full_experiment.py --data-source ultrafeedback --data-size 2000

# Add GAM analysis to existing results
python analyze_existing_experiment.py --experiment-dir results/full_experiments/main_experiment_results

# Stability analysis
python gam_stability_analysis.py --experiment-dir results/full_experiments/main_experiment_results --n-runs 20
```

## Key Files

### Main Scripts (User-facing CLI)

- **`run_full_experiment.py`**: Main experiment pipeline - loads UltraFeedback, simulates personas, gets judge scores, trains models
- **`analyze_existing_experiment.py`**: Adds GAM hyperparameter tuning and baseline comparisons to existing results (non-destructive)
- **`gam_stability_analysis.py`**: Analyzes stability of GAM feature importance across model variants

### Analysis Library (Imported modules)

- **`analysis/gam_hyperparameter_tuning.py`**: GAM tuning logic
- **`analysis/mlp_hyperparameter_tuning.py`**: MLP tuning logic
- **`analysis/correlation_analysis.py`**: Judge-persona correlation analysis
- **`analysis/run_correlation_analysis.py`**: Standalone correlation runner

### Pipeline Components

- **`pipeline/core/dataset_loader.py`**: Loads UltraFeedback dataset
- **`pipeline/core/persona_simulation.py`**: Simulates 8 human personas
- **`pipeline/core/judge_evaluation.py`**: Evaluates with 10 judges
- **`pipeline/core/aggregator_training.py`**: Trains GAM/MLP models
- **`pipeline/core/baseline_models.py`**: Baseline comparison methods
- **`pipeline/utils/judge_rubrics.py`**: Full rubric text for all judges

### Configuration

- **`config/training_config.json`**: Default training parameters for different dataset sizes (small/medium/large/enterprise)

## Data Flow

1. **UltraFeedback** → Load questions and answers
2. **Persona Simulation** → 8 personas rate each response (1-10 scale)
3. **Judge Evaluation** → 10 judges score each response (1-4 scale)
4. **Model Training** → GAM/MLP learn to predict persona scores from judge scores
5. **Analysis** → Interpret which judges matter, test robustness

## Models

### GAM (Generalized Additive Model)
- Highly interpretable
- Individual judge contribution via feature importance
- Partial dependence plots show judge-score relationships
- Can enforce monotonicity

### MLP (Multi-Layer Perceptron)
- Single hidden layer neural network
- Better performance on complex interactions
- Less interpretable than GAM
- Hyperparameter tuning supported

## Experiments

Three self-contained experiments in `experiments/`:

1. **persona_poisoning**: Robustness to contaminated training data (random noise, systematic bias, scale compression)
2. **rubric_sensitivity**: Stability across semantic rubric variations
3. **aggregator_validation**: Performance on low-variance ground truth (UltraFeedback overall_score, single personas)

Each has its own README, run scripts, and results.

## Common Tasks

### Adding New Analysis

Analysis scripts import from `analysis/` library:

```python
from analysis.gam_hyperparameter_tuning import GAMHyperparameterTuner
from analysis.mlp_hyperparameter_tuning import HyperparameterTuner
```

### Reading Experiment Results

Results are in `results/full_experiments/main_experiment_results/`:
- `experiment_summary.json` - All metrics and configurations
- `data/data_with_judge_scores.pkl` - Full dataset
- `model_comparison.png` - Performance visualization
- `gam_analysis/` - GAM tuning results
- `gam_stability_analysis_*/` - Stability analysis

### Understanding Training

Training configs are in `config/training_config.json`:
- Automatically selects scale based on dataset size
- `large_scale` used for 2000-sample experiments
- Hyperparameter tuning searches around these defaults

## Important Notes

- **Dataset**: Only one dataset directory (`dataset/`), contains 2000 samples
- **Models**: Trained models saved in experiment directories, not root `models/`
- **Analysis scripts**: Stay at root (user-facing CLI), library code in `analysis/`
- **Experiments**: Self-contained with own READMEs, run scripts, and results

## Performance Expectations

From main experiment results:
- **Best Heuristic Baseline** (10-judge mean): R² ≈ 0.58
- **GAM**: R² ≈ 0.62 (interpretable, shows judge importance)
- **MLP**: R² ≈ 0.64 (best performance, less interpretable)

The R² ceiling is limited by natural human preference variance across personas.

## Common Issues

### Import Errors
- Check `sys.path` setup in scripts
- Analysis modules import from `analysis/` not `tools.analysis/`

### Data Not Found
- Main dataset is `dataset/data_with_judge_scores.pkl`
- No `data/` directory (was removed)

### Outdated References
- No `project/` directory structure
- Run via CLI scripts not notebooks
- Experiments are in `experiments/` not scattered

## Citation

```bibtex
@misc{multi-judge-interpretability-2025,
  title={Multi-Judge Interpretability: Learning to Aggregate AI Evaluations},
  author={Your Team},
  year={2025},
  howpublished={Apart x Martian Hackathon, 2nd Place}
}
```
