# Experiments Directory

This directory contains all research experiments for the Multi-Judge Interpretability project.

## Structure

### Fellowship Experiments (Current Phase)

Fellowship experiments are organized by research track:

- **`track1_baseline_performance/`** (PRIMARY - 25% effort)
  - Validate aggregator performance against baselines
  - Real human data vs. synthetic personas
  - See [Track 1 README](track1_baseline_performance/README.md)

- **`track2_judge_interpretability/`** (PRIMARY - 40% effort)
  - Main research contribution
  - Which judges matter and why
  - Cross-task and persona-specific analysis
  - See [Track 2 README](track2_judge_interpretability/README.md)

- **`track3_automated_selection/`** (SECONDARY - 15% effort)
  - Systematic judge selection pipeline
  - Generalizable heuristics
  - See [Track 3 README](track3_automated_selection/README.md)

- **`track4_aggregator_validation/`** (SECONDARY - 15% effort)
  - Validate interpretability claims
  - Synthetic ground truth recovery
  - See [Track 4 README](track4_aggregator_validation/README.md)

### Workshop Experiments (Completed)

- **`workshop_experiments/`**
  - Experiments from hackathon and workshop papers
  - Track 5 (Robustness) experiments completed here
  - See [Workshop README](workshop_experiments/README.md)

## Research Timeline

**Phase 1** (Dec 2024): Hackathon - 2nd place at Apart x Martian
**Phase 2** (Early 2025): Workshop papers accepted to 2 venues
**Phase 3** (Current): Fellowship - Full paper development

Target: ICML submission (Jan 2025 deadline TBD)

## Track Dependencies

```
Track 1.1 (Persona Data)
    ↓
Track 2.3 (Persona Variation)

Track 1.3 (JUDGE-BENCH)
    ↓
Track 2.2 (Cross-Task Heatmap)

Track 2 (All) → Judge Importance Rankings
    ↓
Track 3 (Automated Selection)

Track 1 + Track 2
    ↓
Track 4 (Validation)
```

## Experiment Structure

Each fellowship experiment follows this structure:

```
track{N}_{name}/{X}.{Y}_{experiment}/
├── README.md              # Detailed methodology
├── run_experiment.py      # Execution script
├── config.yaml            # Optional configuration (can use programmatic config)
├── data/                  # Experiment-specific data
└── results/               # Outputs, figures, tables
```

## Quick Start

### Running a Fellowship Experiment

```bash
# Navigate to specific experiment
cd experiments/track1_baseline_performance/1.1_persona_synthetic/

# Read methodology
cat README.md

# Run experiment
python run_experiment.py

# Results saved to results/ directory
```

### Creating a New Experiment

1. Create directory: `experiments/track{N}_{name}/{X}.{Y}_{experiment}/`
2. Write `README.md` with methodology (see track READMEs for templates)
3. Implement `run_experiment.py` (can import from `pipeline/` and `analysis/`)
4. Create git branch: `experiment/{track}.{number}-{short-name}`
5. Create GitHub issue for tracking

## Configuration System

Fellowship experiments use the new flexible configuration system:

```python
from pipeline.config import ExperimentConfig, JudgeConfig
from pipeline.core.dataset_loader import DatasetLoader
from pipeline.core.judge_evaluation import JudgeEvaluator

# Load or create config
config = ExperimentConfig.from_yaml("config.yaml")
# Or: config = create_default_config(name="my-experiment", dataset="ultrafeedback")

# Load dataset (automatically standardized)
loader = DatasetLoader()
data = loader.load(config.dataset, **config.dataset_kwargs)

# Evaluate with judges
evaluator = JudgeEvaluator(judge_ids=config.judges.judge_ids)
data = evaluator.evaluate_dataset(data)

# Train models
# ... (see REFACTORING_GUIDE.md for details)
```

See `docs/REFACTORING_GUIDE.md` for full configuration system documentation.

## Key Datasets

**UltraFeedback** (Track 1.1, 2.3, 4.2):
- 2000 samples from workshop
- Synthetic persona annotations (8 personas)
- Question-answer pairs with preference scores

**JUDGE-BENCH** (Track 1.3, 2.2):
- 20 NLP tasks with real human annotations
- External dataset, need to define task subsets
- Used for cross-task generalization

**MAJ-Eval** (Track 1.2):
- Multi-agent debate benchmark
- Have their code for comparison
- Head-to-head baseline validation

## Related Documentation

- `docs/methodology_proposals.md` - Detailed track descriptions and research questions
- `docs/REFACTORING_GUIDE.md` - Configuration system and migration guide
- `CLAUDE.md` - Project overview and quick start
- `docs/literature_review.md` - Related work catalog
