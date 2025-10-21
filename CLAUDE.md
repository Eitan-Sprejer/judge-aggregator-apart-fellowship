# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

Multi-Judge Interpretability is a research framework for evaluating AI outputs using multiple specialized judges and learned aggregation models. The system trains interpretable models (GAM and MLP) that combine judge scores to match human preferences.

**Status**: Apart Fellowship stage (Jan 2025 - ongoing)
- **Phase 1 (Completed)**: Hackathon - 2nd place at Apart x Martian
- **Phase 2 (Completed)**: Workshop papers accepted to 2 venues
- **Phase 3 (Current)**: Fellowship - Full paper development for conference submission (ICML target, Jan 2025 deadline TBD)

**Research Focus**: This repository is transitioning from workshop-paper codebase to conference-paper research platform. We're expanding from preliminary synthetic-persona experiments to comprehensive validation across real human data, systematic judge selection, and cross-task generalization.

### Core Problem

Current AI evaluation systems have limitations:
- Single judges have limited perspectives
- Naive averaging treats all dimensions equally
- Fixed rules can't capture context-dependent preferences

### Solution

Learn interpretable aggregation functions from data:
- 10 specialized judges evaluate different dimensions
- Human feedback from 8 simulated personas (workshop) + real human annotations (fellowship)
- GAM/MLP models learn to combine judge scores
- Interpretability analysis shows which judges matter

## Fellowship Research Tracks

The fellowship focuses on 4 primary research tracks (see `docs/methodology_proposals.md` for full details):

### Track 1: Baseline Performance Comparison (PRIMARY - 25% effort)
**Goal**: Validate aggregator performance against baselines on synthetic and real human data

Experiments:
- **1.1**: Performance on persona-annotated synthetic data (8 personas, self-confidence reports)
- **1.2**: Comparison against MAJ-Eval's multi-agent aggregator (have their code)
- **1.3**: Performance on JUDGE-BENCH human-annotated data (20 NLP tasks, existing dataset)

**Key Contribution**: Demonstrates practical applicability beyond synthetic scenarios

### Track 2: Judge Interpretability (PRIMARY - 40% effort)
**Goal**: Understand which judges matter and why (main differentiator from related work)

Experiments:
- **2.1**: Judge importance analysis via GAM feature importance + MLP ablation
- **2.2**: Cross-task judge importance heatmap (JUDGE-BENCH tasks)
- **2.3**: Persona-based judge importance variation

**Key Contribution**: Fills gap left by MAJ-Eval (debates but no dimension analysis), JUDGE-BENCH (evaluates but doesn't aggregate)

### Track 3: Automated Judge Selection (SECONDARY - 15% effort)
**Goal**: Develop systematic pipeline for selecting optimal judge sets

Experiments:
- **3.1**: Iterative judge selection pipeline (propose → identify least important → propose complements)
- **3.2**: Extract generalizable heuristics across JUDGE-BENCH tasks

**Key Contribution**: Actionable methodology for building evaluation systems

### Track 4: Aggregator Validation (SECONDARY - 15% effort)
**Goal**: Validate that aggregators correctly decompose known preference structures

Experiments:
- **4.1**: Synthetic preference decomposition (ground truth aggregation recipes)
- **4.2**: Persona internal preference system recovery

**Key Contribution**: Validates that judge importance measures are meaningful

### Out of Scope (Fellowship)
- **Track 5**: Robustness/Bias Analysis - Mostly completed for workshop (persona contamination, rubric sensitivity)
- **Track 6**: Aggregator Architecture - Optional future work (prompt-aware, confidence-weighted)

## Repository Structure

```
judge-aggregator/
├── docs/                            # Fellowship documentation
│   ├── overview.md                  # Research proposal overview
│   ├── methodology_proposals.md     # Detailed track descriptions (PRIMARY)
│   └── literature_review.md         # Related work catalog
├── analysis/                        # Analysis library code (imported by scripts)
│   ├── gam_hyperparameter_tuning.py
│   ├── mlp_hyperparameter_tuning.py
│   ├── correlation_analysis.py
│   └── run_correlation_analysis.py
├── experiments/                     # Fellowship research experiments (track-based)
│   ├── track1_baseline_performance/
│   │   ├── 1.1_persona_synthetic/
│   │   ├── 1.2_maj_eval_comparison/
│   │   └── 1.3_judge_bench/
│   ├── track2_judge_interpretability/
│   │   ├── 2.1_importance_analysis/
│   │   ├── 2.2_cross_task_heatmap/
│   │   └── 2.3_persona_variation/
│   ├── track3_automated_selection/
│   │   ├── 3.1_selection_pipeline/
│   │   └── 3.2_generalizable_heuristics/
│   ├── track4_aggregator_validation/
│   │   ├── 4.1_synthetic_decomposition/
│   │   └── 4.2_persona_preference_recovery/
│   └── workshop_experiments/        # Previous hackathon/workshop work
│       ├── persona_poisoning/
│       ├── rubric_sensitivity/
│       └── aggregator_validation/
├── pipeline/                        # Core pipeline components
│   ├── core/                        # Judge evaluation, model training, baselines
│   └── utils/                       # Judge rubrics, data utilities
├── dataset/                         # Workshop dataset (may not be used for fellowship)
│   └── data_with_judge_scores.pkl   # 2000 samples from UltraFeedback (workshop)
├── results/                         # Experiment results (organized by track)
│   ├── workshop/                    # Workshop paper results
│   └── fellowship/                  # Fellowship experiment results
├── config/                          # Training configurations
│   └── training_config.json
├── utils/                           # Logging utilities
│   └── logging_setup.py
├── run_full_experiment.py           # Main CLI: Full experiment pipeline (workshop)
├── analyze_existing_experiment.py   # Main CLI: Post-hoc GAM analysis
└── gam_stability_analysis.py        # Main CLI: Feature stability analysis
```

**Note**: Repository is transitioning from workshop structure to fellowship structure. Each track experiment will have its own subdirectory with:
- `README.md` - Detailed methodology and experiment design
- `run_experiment.py` - Execution script
- `results/` - Experiment outputs

## Quick Start

### Initial Setup
```bash
# Setup environment
pip install -r requirements.txt
cp .env.example .env  # Add your API keys (OpenAI, etc.)
```

### Workshop Experiments (Legacy)
```bash
# Run workshop pipeline (UltraFeedback synthetic personas)
python run_full_experiment.py --data-source ultrafeedback --data-size 2000

# Add GAM analysis to existing results
python analyze_existing_experiment.py --experiment-dir results/full_experiments/main_experiment_results

# Stability analysis
python gam_stability_analysis.py --experiment-dir results/full_experiments/main_experiment_results --n-runs 20
```

### Fellowship Experiments (Current)
```bash
# Navigate to specific track experiment
cd experiments/track1_baseline_performance/1.1_persona_synthetic/

# Run experiment (each has own README and run script)
python run_experiment.py

# Results saved to experiments/track1_baseline_performance/1.1_persona_synthetic/results/
```

**Note**: Fellowship experiments are self-contained. Check each experiment's README.md for specific instructions and methodology details.

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

### Workshop Experiments (Completed)

Three robustness experiments in `experiments/workshop_experiments/` (Track 5 - completed for workshop paper):

1. **persona_poisoning**: Robustness to contaminated training data (random noise, systematic bias, scale compression)
2. **rubric_sensitivity**: Stability across semantic rubric variations
3. **aggregator_validation**: Performance on low-variance ground truth (UltraFeedback overall_score, single personas)

### Fellowship Experiments (In Progress)

Track-based experiments organized by research question (see "Fellowship Research Tracks" section above for details):

**Track 1 - Baseline Performance**:
- 1.1: Synthetic persona validation
- 1.2: MAJ-Eval comparison (have their code)
- 1.3: JUDGE-BENCH human annotations (existing dataset, need to define subsets)

**Track 2 - Judge Interpretability** (PRIMARY):
- 2.1: GAM/MLP importance analysis
- 2.2: Cross-task heatmap (predefined judge list across tasks)
- 2.3: Persona-based variation

**Track 3 - Automated Selection**:
- 3.1: Iterative judge selection pipeline
- 3.2: Generalizable heuristics extraction

**Track 4 - Aggregator Validation**:
- 4.1: Synthetic ground truth decomposition
- 4.2: Persona preference recovery

Each fellowship experiment has its own subdirectory with README.md (detailed methodology), run scripts, and results.

## Common Tasks

### Fellowship Workflow

**Creating a New Experiment**:
1. Create branch: `git checkout -b experiment/1.1-persona-synthetic`
2. Create experiment directory: `experiments/track1_baseline_performance/1.1_persona_synthetic/`
3. Add `README.md` with detailed methodology
4. Implement `run_experiment.py` (can import from `pipeline/` and `analysis/`)
5. Run experiment and save results to `results/` subdirectory
6. Create GitHub issue for experiment tracking

**Branch Naming Convention**:
- Format: `experiment/[track].[number]-[short-name]`
- Examples: `experiment/1.1-persona-synthetic`, `experiment/2.2-cross-task-heatmap`

**Git Workflow**:
- Each experiment gets its own issue and branch
- Branch from `main`, merge back via PR when experiment complete
- Results committed to repository for reproducibility

### Adding New Analysis

Analysis scripts import from `analysis/` library:

```python
from analysis.gam_hyperparameter_tuning import GAMHyperparameterTuner
from analysis.mlp_hyperparameter_tuning import HyperparameterTuner
```

### Reading Experiment Results

**Workshop Results**: `results/full_experiments/main_experiment_results/`
- `experiment_summary.json` - All metrics and configurations
- `data/data_with_judge_scores.pkl` - Full dataset
- `model_comparison.png` - Performance visualization
- `gam_analysis/` - GAM tuning results
- `gam_stability_analysis_*/` - Stability analysis

**Fellowship Results**: `experiments/[track]/[experiment]/results/`
- Each experiment self-contained with own results directory

### Understanding Training

Training configs are in `config/training_config.json`:
- Automatically selects scale based on dataset size
- `large_scale` used for 2000-sample experiments
- Hyperparameter tuning searches around these defaults

## Important Notes

### Workshop → Fellowship Transition
- **Repository Purpose**: Transitioning from workshop-paper codebase to conference-paper research platform
- **Current Phase**: Setting up track-based experiment structure, not yet executing experiments
- **Workshop Code**: Remains functional for reproducibility, but not primary focus

### Data Management
- **Workshop Dataset**: `dataset/data_with_judge_scores.pkl` (2000 UltraFeedback samples, may not be used)
- **JUDGE-BENCH**: External dataset (20 NLP tasks with human annotations), need to define subsets for Track 1.3 and Track 2.2
- **MAJ-Eval**: Have their code for comparison (Track 1.2)
- **New Data**: Fellowship experiments will generate own datasets in respective experiment directories

### Code Organization
- **Models**: Trained models saved in experiment directories, not root `models/`
- **Analysis Library**: Reusable code in `analysis/`, imported by experiment scripts
- **Experiments**: Self-contained with own README.md, run scripts, and results
- **Pipeline Components**: Core functionality in `pipeline/` for reuse across experiments

### Key Datasets for Fellowship

**JUDGE-BENCH** (Track 1.3, 2.2):
- 20 diverse NLP evaluation tasks with human annotations
- Existing dataset, need to define task subsets
- Used for: Real human validation, cross-task judge importance analysis

**MAJ-Eval Data** (Track 1.2):
- Multi-agent debate evaluation data
- Have their code for running comparison
- Used for: Baseline comparison against state-of-the-art

**Synthetic Personas** (Track 1.1, 2.3, 4.2):
- 8 simulated human personas (from workshop)
- May expand or modify for fellowship experiments
- Used for: Controlled validation, persona-specific analysis

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
- Fellowship experiments should add parent directories to path for pipeline imports

### Data Not Found
- Workshop dataset: `dataset/data_with_judge_scores.pkl`
- Fellowship datasets: Each experiment manages own data in `experiments/[track]/[experiment]/data/`
- No root `data/` directory (was removed during workshop cleanup)

### Outdated References
- No `project/` directory structure
- Workshop experiments run via CLI scripts (root level), fellowship experiments via track-specific scripts
- Current experiments in `experiments/` are workshop experiments (will move to `workshop_experiments/`)

### Track Organization
- Don't create experiments outside track structure
- Each experiment must have README.md with methodology before implementation
- Use git branches/issues for tracking: `experiment/[track].[number]-[name]`

### Dataset Confusion
- **Workshop dataset** (dataset/): UltraFeedback with synthetic personas, may not be used
- **JUDGE-BENCH**: External dataset, details in experiment-specific READMEs
- **MAJ-Eval**: Using their code, not their exact data format

## Publication Timeline & Citation

### Project Phases
1. **Hackathon** (Dec 2024): 2nd place at Apart x Martian
2. **Workshop Papers** (Early 2025): Accepted to 2 workshop venues
3. **Conference Paper** (Target: ICML Jan 2025): Full paper with extended experiments (Tracks 1-4)

### Citation

**Workshop Paper** (TBD - update with actual venue):
```bibtex
@inproceedings{multi-judge-interpretability-workshop-2025,
  title={Multi-Judge Interpretability: Learning to Aggregate AI Evaluations},
  author={Your Team},
  booktitle={Workshop on [Venue Name]},
  year={2025},
  note={Apart Fellowship Research}
}
```

**Conference Paper** (Target):
```bibtex
@inproceedings{multi-judge-interpretability-icml-2025,
  title={Multi-Judge Interpretability: Systematic Evaluation and Selection for AI Assessment},
  author={Your Team},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025},
  note={Under Review - Fellowship Stage}
}
```

### Related Work
See `docs/literature_review.md` for comprehensive related work catalog.
