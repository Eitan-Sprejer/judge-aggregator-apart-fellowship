# Processed Datasets

This directory contains preprocessed datasets in standardized format, ready for experiments.

## Format

All processed datasets use the standardized schema:
```python
{
    'question': str,
    'response': str,
    'dataset': str,
    'target_human': Optional[Dict[str, float]],    # Human annotations by dimension
                                                   # e.g., {"fluency": 2.0, "population": 1.5}
                                                   # or {"overall": 7.5} for single dimension
    'target_synthetic': Optional[Dict[str, float]], # Persona/synthetic scores by dimension
                                                    # e.g., {"overall": 7.5}
    'score_range_human': Optional[Tuple],          # (min, max) for human scores
    'score_range_synthetic': Optional[Tuple],      # (min, max) for synthetic scores
    # ... dataset-specific fields
}
```

## Workflow

### 1. Initial Processing
```python
from pipeline.core.dataset_loader import DatasetLoader

loader = DatasetLoader()
df = loader.load('ultrafeedback', n_samples=2000)

# Save base dataset (no scores yet)
loader.create_experiment_subset(
    df,
    n_samples=2000,
    output_path='datasets/processed/ultrafeedback_2000.pkl'
)
```

### 2. Add Synthetic Scores (Personas)
```python
import pickle

# Load processed dataset
with open('datasets/processed/ultrafeedback_2000.pkl', 'rb') as f:
    df = pickle.load(f)

# Add persona scores as dict (single "overall" dimension for UltraFeedback)
df['target_synthetic'] = [{"overall": score} for score in persona_scores]

# Overwrite with updated scores
with open('datasets/processed/ultrafeedback_2000.pkl', 'wb') as f:
    pickle.dump(df, f)
```

### 3. Add Human Scores
```python
# Load dataset with synthetic scores
with open('datasets/processed/ultrafeedback_2000.pkl', 'rb') as f:
    df = pickle.load(f)

# Add human annotations as dict
# For single-dimension: [{"overall": 7.5}, {"overall": 8.2}, ...]
# For multi-dimension: [{"fluency": 2.0, "accuracy": 1.5}, ...]
df['target_human'] = human_scores

# Overwrite again
with open('datasets/processed/ultrafeedback_2000.pkl', 'wb') as f:
    pickle.dump(df, f)
```

## Naming Convention

Use descriptive names indicating source and size:
- `ultrafeedback_2000.pkl` - UltraFeedback, 2000 samples
- `mslr_processed.pkl` - MSLR, all samples with human scores
- `judge_bench_cola.pkl` - JUDGE-BENCH CoLA task

Files get updated in-place as annotations are added.

## Current Datasets

*Document processed datasets here as you create them:*

- `ultrafeedback_2000.pkl` - Workshop dataset (2000 samples, 8 personas)
  - Created: 2024-XX-XX
  - Status: Has synthetic scores from personas
  - Score ranges: synthetic (0-10), human (None)

## Git Tracking

Processed datasets are **not tracked in git** (see `.gitignore`). They can be large and are binary files.

For reproducibility:
1. Raw data sources are tracked (JUDGE-BENCH, MSLR manual downloads)
2. Processing code is tracked (dataset_loader.py)
3. Small processed datasets (<10MB) can be committed if needed

For collaboration:
- Share via Dropbox/Drive/cloud storage
- Or regenerate using loader code (usually fast)
