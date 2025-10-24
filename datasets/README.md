# Datasets

This directory contains all datasets used in the Multi-Judge Interpretability project.

## Directory Structure

```
datasets/
├── data_with_judge_scores.pkl          # Workshop dataset (2000 samples)
├── judge-bench/                        # JUDGE-BENCH (19 NLP tasks)
├── mslr-annotated/                     # MSLR medical summarization
├── processed/                          # Processed datasets ready for experiments
│   ├── .gitignore
│   └── README.md
└── README.md                           # This file
```

## Dataset Categories

### 1. Workshop Data (Legacy)
**File**: `data_with_judge_scores.pkl`
- **Source**: UltraFeedback with 8 synthetic personas
- **Size**: 2000 samples, ~10MB
- **Contents**: Questions, responses, judge scores, persona annotations
- **Usage**: Workshop experiments (persona poisoning, rubric sensitivity)
- **Status**: May be regenerated for fellowship experiments

### 2. JUDGE-BENCH (Human Annotations)
**Directory**: `judge-bench/`
- **Source**: [dmg-illc/JUDGE-BENCH](https://github.com/dmg-illc/JUDGE-BENCH)
- **Size**: 19 diverse NLP evaluation tasks, ~227MB
- **Contents**: Human annotations for various NLP tasks
- **Usage**: Track 1.3 (baseline performance), Track 2.2 (cross-task analysis)
- **Setup**: Manually cloned from GitHub

**Available Tasks**:
- CoLA, DICES, LLMBar, QAGS, SummEval
- Medical safety, toxic chat, newsroom
- Recipe crowdsourcing, persona chat, topical chat
- Roscoe, WMT-23, WMT-human
- And more (see `judge-bench/README.md`)

### 3. MSLR Annotated (Medical Summarization)
**Directory**: `mslr-annotated/`
- **Source**: [allenai/mslr-annotated-dataset](https://github.com/allenai/mslr-annotated-dataset)
- **Size**: 470 medical review summaries, ~5.6MB
- **Contents**: Human facet annotations (fluency, population, intervention, outcome)
- **Usage**: Track 1.2 (MAJ-Eval comparison)
- **Setup**: Manually downloaded (see `mslr-annotated/VERSION` for version info)

**Key Files**:
- `data/data_with_overlap_scores.json` - Main dataset
- `LICENSE` - Apache 2.0
- `README.md` - Original dataset documentation
- `VERSION` - Version tracking (commit: 3317358, 2023-05-18)

### 4. Processed Datasets
**Directory**: `processed/`
- **Purpose**: Preprocessed datasets in standardized format
- **Format**: Dual-column schema (human + synthetic targets)
- **Usage**: Ready for training and experiments
- **Details**: See `processed/README.md`

## Standardized Format

All datasets are preprocessed into a common schema:

```python
{
    'question': str,                              # Input/prompt
    'response': str,                              # Model output
    'dataset': str,                               # Source dataset name
    'target_human': Optional[Dict[str, float]],   # Human annotations by dimension
                                                  # e.g., {"fluency": 2.0, "population": 1.5}
                                                  # or {"overall": 7.5} for single dimension
    'target_synthetic': Optional[Dict[str, float]], # Synthetic/persona scores by dimension
                                                    # e.g., {"overall": 7.5}
    'score_range_human': Optional[Tuple],         # (min, max) for human scores
    'score_range_synthetic': Optional[Tuple],     # (min, max) for synthetic scores
    # ... dataset-specific fields
}
```

## Loading Datasets

```python
from pipeline.core.dataset_loader import DatasetLoader

loader = DatasetLoader()

# UltraFeedback (auto-downloads from HuggingFace)
uf = loader.load('ultrafeedback', n_samples=2000)

# MSLR (from local download)
mslr = loader.load('mslr', n_samples=100)

# JUDGE-BENCH (not yet implemented)
# judge_bench = loader.load('judge_bench', task_name='cola')

# StorySparkQA (auto-downloads from HuggingFace)
# story = loader.load('story_spark_qa', n_samples=100)
```

## Score Ranges by Dataset

| Dataset | Human Range | Synthetic Range | Notes |
|---------|-------------|-----------------|-------|
| UltraFeedback | None | (0-10) | Persona scores |
| MSLR | (0-2) | None | Facet averages |
| JUDGE-BENCH | Task-specific | None | Varies by task |
| StorySparkQA | TBD | TBD | To be determined |

## Git Tracking

- **Tracked**: `judge-bench/`, `mslr-annotated/` (raw data, documentation)
- **Not tracked**: `processed/*.pkl` (large processed files)
- **Excluded**: Workshop data may not be used in fellowship experiments

## External Dependencies

### Auto-Downloaded (HuggingFace)
- UltraFeedback: `openbmb/UltraFeedback`
- StorySparkQA: `NEU-HAI/StorySparkQA`

### Manually Downloaded (GitHub)
- JUDGE-BENCH: Already included in repo
- MSLR: Already included in repo (version tracked)

## Dataset Sizes

```
Total: ~243MB

datasets/
├── data_with_judge_scores.pkl    # 10MB (workshop)
├── judge-bench/                  # 227MB (19 tasks)
├── mslr-annotated/               # 5.6MB (470 reviews)
└── processed/                    # Variable (gitignored)
```

## Fellowship Experiments

Different tracks use different datasets:

- **Track 1.1**: UltraFeedback with synthetic personas
- **Track 1.2**: MSLR, StorySparkQA (MAJ-Eval comparison)
- **Track 1.3**: JUDGE-BENCH (human annotations)
- **Track 2.x**: JUDGE-BENCH (judge importance analysis)
- **Track 3.x**: JUDGE-BENCH (judge selection)
- **Track 4.x**: UltraFeedback (aggregator validation)

See `docs/methodology_proposals.md` for details on each track.
