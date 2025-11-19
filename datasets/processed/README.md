# Processed Datasets

This directory contains preprocessed datasets in standardized format, ready for fellowship experiments.

## Current Datasets

| File | Samples | Dimensions | Human Agg | Human Ind | Size |
|------|---------|------------|-----------|-----------|------|
| `ultrafeedback_full.pkl` | 63,905 | 1 (overall) | ❌ | ❌ | 116 MB |
| `helpsteer2_full.pkl` | 21,362 | 5 | ✅ | ❌ | 47 MB |
| `summeval_full.pkl` | 1,600 | 4 | ✅ | ✅ (8) | 4.9 MB |
| `newsroom_full.pkl` | 420 | 4 | ✅ | ✅ (3) | 1.9 MB |
| `recipe_full.pkl` | 52 | 6 | ✅ | ✅ (20) | 72 KB |
| `wmt_en_de_full.pkl` | 9,871 | 1 (quality) | ✅ | ✅ (3) | 7.7 MB |
| `wmt_zh_en_full.pkl` | 15,981 | 1 (quality) | ✅ | ✅ (3) | 11 MB |

**Total**: 7 datasets, 113,191 samples, ~189 MB

## Standardized Format

All processed datasets follow the 15-column standardized schema:

```python
{
    # Core fields
    'question': str,                          # Input/prompt/source text
    'response': str,                          # Output/completion/generated text
    'dataset': str,                           # Dataset identifier

    # Human annotations
    'target_human_aggregated': Optional[Dict[str, float]],  # Mean scores by dimension
                                                             # e.g., {"fluency": 2.5, "coherence": 3.1}
    'target_human_individual': Optional[List[Dict]],        # Individual annotator scores
                                                             # e.g., [{"fluency": 2, "coherence": 3}, ...]
    'score_range_human': Optional[Dict[str, Tuple]],        # Score ranges by dimension
                                                             # e.g., {"fluency": (1, 5), "coherence": (1, 5)}

    # Synthetic annotations
    'target_synthetic': Optional[Dict[str, float]],         # Synthetic/persona scores
    'score_range_synthetic': Optional[Dict[str, Tuple]],    # Synthetic score ranges

    # Metadata
    'dimensions': List[str],                  # Scoring dimensions
    'task_type': str,                         # Task category
    'reference_output': Optional[str],        # Gold reference (if available)
    'context': Optional[Any],                 # Additional context
    'response_metadata': Optional[Dict],      # Response metadata
    'annotator_metadata': Optional[Dict],     # Annotator information

    # Traceability
    'original_index': Any                     # Original dataset index
}
```

## Usage

### Load a processed dataset

```python
import pickle
import pandas as pd

# Load dataset
with open('datasets/processed/summeval_full.pkl', 'rb') as f:
    df = pickle.load(f)

print(f"Loaded {len(df)} samples")
print(f"Columns: {df.columns.tolist()}")
print(f"Dimensions: {df.iloc[0]['dimensions']}")
```

### Filter for specific samples

```python
# Filter for samples with human annotations
df_annotated = df[df['target_human_aggregated'].notna()]

# Filter by dimension
df_multi_dim = df[df['dimensions'].apply(lambda x: len(x) > 1)]

# Get samples with individual annotators
df_with_individuals = df[df['target_human_individual'].notna()]
```

### Regenerate from source

If you need to regenerate datasets:

```python
from pipeline.core.dataset_loader import DatasetLoader
import pickle

loader = DatasetLoader()

# Load full dataset
df = loader.load('helpsteer2', split='train')

# Save to pickle
with open('datasets/processed/helpsteer2_train.pkl', 'wb') as f:
    pickle.dump(df, f)
```

## Dataset Details

### UltraFeedback (`ultrafeedback_full.pkl`)
- **Task**: Instruction-following
- **Source**: openbmb/UltraFeedback (HuggingFace)
- **Annotations**: None (to be filled by persona simulation)
- **Use case**: Synthetic persona experiments (Track 1.1)

### HelpSteer2 (`helpsteer2_full.pkl`)
- **Task**: Helpfulness evaluation
- **Source**: nvidia/HelpSteer2 (HuggingFace) - train + validation splits merged
- **Dimensions**: helpfulness, correctness, coherence, complexity, verbosity (0-4 scale)
- **Annotations**: Aggregated human scores only (no individual annotators)
- **Use case**: Large-scale experiments, judge importance analysis (Tracks 2.1, 2.2)

### SummEval (`summeval_full.pkl`)
- **Task**: News summarization
- **Source**: JUDGE-BENCH (Yale-LILY/SummEval)
- **Dimensions**: coherence (1-5), consistency (1-5), fluency (1-3), relevance (1-5)
- **Annotations**: 8 individual annotators (3 experts + 5 crowdworkers)
- **Reference**: Gold reference summaries available
- **Use case**: Primary dataset for Track 1.3 (JUDGE-BENCH validation), inter-annotator analysis

### NewsRoom (`newsroom_full.pkl`)
- **Task**: News summarization
- **Source**: JUDGE-BENCH
- **Dimensions**: Informativeness (1-5), Relevance (1-5), Fluency (1-5), Coherence (1-5)
- **Annotations**: 3 individual annotators per sample
- **Use case**: Cross-task comparison (Track 2.2)

### Recipe (`recipe_full.pkl`)
- **Task**: Recipe generation
- **Source**: JUDGE-BENCH
- **Dimensions**: grammar, fluency, verbosity, structure, success, overall (1-6 scale)
- **Annotations**: 20 crowdworker annotators per sample
- **Use case**: Most diverse individual annotations

### WMT en→de (`wmt_en_de_full.pkl`)
- **Task**: English to German translation
- **Source**: JUDGE-BENCH (WMT-20)
- **Dimensions**: quality (0-6 scale)
- **Annotations**: 3 expert annotators per sample
- **Reference**: Human reference translations available
- **Use case**: Expert-quality annotations, cross-lingual experiments

### WMT zh→en (`wmt_zh_en_full.pkl`)
- **Task**: Chinese to English translation
- **Source**: JUDGE-BENCH (WMT-20)
- **Dimensions**: quality (0-6 scale)
- **Annotations**: 3 expert annotators per sample
- **Reference**: Human reference translations available
- **Use case**: Expert-quality annotations, cross-lingual experiments

## Preprocessing Pipeline

All datasets were preprocessed using `pipeline/core/dataset_loader.py`:

```python
from pipeline.core.dataset_loader import DatasetLoader
import pandas as pd

loader = DatasetLoader()

# UltraFeedback
df = loader.load('ultrafeedback', n_samples=None)

# HelpSteer2 (merge train + validation)
df_train = loader.load('helpsteer2', split='train', n_samples=None)
df_val = loader.load('helpsteer2', split='validation', n_samples=None)
df = pd.concat([df_train, df_val], ignore_index=True)

# JUDGE-BENCH tasks
df = loader.load('judge_bench', task_name='summeval', n_samples=None)
df = loader.load('judge_bench', task_name='newsroom', n_samples=None)
df = loader.load('judge_bench', task_name='recipe', n_samples=None)
df = loader.load('judge_bench', task_name='wmt_en_de', n_samples=None)
df = loader.load('judge_bench', task_name='wmt_zh_en', n_samples=None)
```

## Git Tracking

Processed datasets are **gitignored** (binary files, large sizes).

For reproducibility:
1. Raw data sources are documented in `datasets/DATASET_COMPARISON.md`
2. Processing code is tracked in `pipeline/core/dataset_loader.py`
3. Regeneration instructions are provided above

For collaboration:
- Share via cloud storage (Dropbox/Drive)
- Or regenerate using the loader code (takes ~1-2 minutes for all datasets)

## Notes

- **Last updated**: 2025-11-10
- **Total datasets**: 7 preprocessed datasets
- **Total samples**: 113,191
- **Total disk usage**: ~189 MB
- **HelpSteer2**: Merged train + validation splits (21,362 total samples)
- **Datasets excluded**: MSLR (sparse annotations), StorySparkQA (no human annotations)
- **Documentation**: See `datasets/DATASET_COMPARISON.md` for detailed format comparison
- **Validation summary**: See `datasets/PREPROCESSING_VALIDATION_SUMMARY.md`
