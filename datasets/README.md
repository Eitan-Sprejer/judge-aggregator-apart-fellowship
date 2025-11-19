# Datasets

This directory contains all datasets used in the Multi-Judge Interpretability project.

## Quick Reference Table

| Dataset | Samples | Dimensions | Score Range | Human Agg | Human Ind | Task Type |
|---------|---------|------------|-------------|-----------|-----------|-----------|
| **UltraFeedback** | 64K | 1 (overall) | N/A | ❌ | ❌ | instruction_following |
| **HelpSteer2** | 21K | 5 | 0-4 | ✅ | ❌ | helpfulness_evaluation |
| **SummEval** | 1.6K | 4 | 1-5 | ✅ | ✅ (8) | summarization |
| **NewsRoom** | 420 | 4 | 1-5 | ✅ | ✅ (3) | summarization |
| **Recipe** | 52 | 6 | 1-6 | ✅ | ✅ (20) | recipe_generation |
| **WMT en→de** | 9.9K | 1 (quality) | 0-6 | ✅ | ✅ (3 experts) | translation |
| **WMT zh→en** | 16K | 1 (quality) | 0-6 | ✅ | ✅ (3 experts) | translation |
| **MSLR** | 4.7K (7.8% annotated) | 4 | 0-2, 0-1 | ⚠️ | ⚠️ | summarization |

**Legend**: ✅ Available | ❌ Not available | ⚠️ Partial/Issues | (N) = Number of annotators

## Directory Structure

```
datasets/
├── data_with_judge_scores.pkl          # Workshop dataset (2000 samples)
├── judge-bench/                        # JUDGE-BENCH (19 NLP tasks)
├── mslr-annotated/                     # MSLR medical summarization
├── processed/                          # Processed datasets ready for experiments
│   ├── .gitignore                      # Ignore cached .pkl files
│   └── README.md                       # Processed dataset documentation
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
- **Note**: Only 7.8% of samples (364/4654) have annotations by design

**Key Files**:
- `data/data_with_overlap_scores.json` - Main dataset
- `LICENSE` - Apache 2.0
- `README.md` - Original dataset documentation
- `VERSION` - Version tracking (commit: 3317358, 2023-05-18)

### 4. Processed Datasets
**Directory**: `processed/`
- **Purpose**: Preprocessed datasets in standardized format
- **Format**: 15-column standardized schema (see below)
- **Usage**: Ready for training and experiments
- **Caching**: Automatically cached on first load via `DatasetLoader`
- **Details**: See `processed/README.md`

## Standardized Format

All datasets are preprocessed into a common 15-column schema:

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

**Null Handling**: Consistent use of `None` for missing data across all datasets ✅

## Detailed Dataset Previews

### 1. UltraFeedback

**Format**: Instruction-following with NO human annotations (synthetic only)

**Sample Structure**:
```python
{
  'question': str,                          # Instruction prompt
  'response': str,                          # Model completion
  'dataset': 'ultrafeedback',
  'target_human_aggregated': None,          # No human annotations
  'target_human_individual': None,
  'score_range_human': None,
  'target_synthetic': None,                 # To be filled by persona simulation
  'score_range_synthetic': {'overall': (0.0, 10.0)},
  'dimensions': ['overall'],
  'task_type': 'instruction_following',
  'reference_output': None,
  'context': None,
  'response_metadata': None,
  'annotator_metadata': None,
  'original_index': int
}
```

**Use Case**: Track 1.1 (persona synthetic validation), Track 4.x (aggregator validation)

---

### 2. HelpSteer2

**Format**: Helpfulness evaluation with 5-dimensional aggregated human annotations

**Dimensions**: helpfulness, correctness, coherence, complexity, verbosity (0-4 scale)

**Sample Structure**:
```python
{
  'question': str,                          # User prompt
  'response': str,                          # AI-generated answer
  'dataset': 'helpsteer2',
  'target_human_aggregated': {              # ✅ Aggregated scores
    'helpfulness': float,                   # 0-4 scale
    'correctness': float,
    'coherence': float,
    'complexity': float,
    'verbosity': float
  },
  'target_human_individual': None,          # ❌ Individual scores not provided
  'score_range_human': {
    'helpfulness': (0, 4),
    'correctness': (0, 4),
    'coherence': (0, 4),
    'complexity': (0, 4),
    'verbosity': (0, 4)
  },
  'dimensions': ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity'],
  'task_type': 'helpfulness_evaluation'
}
```

**Sample Data**:
| Field | Value |
|-------|-------|
| dimensions | 5 dimensions (helpfulness, correctness, coherence, complexity, verbosity) |
| target_human_aggregated | {'helpfulness': 1.0, 'correctness': 3.0, 'coherence': 3.0, 'complexity': 1.0, 'verbosity': 2.0} |
| score_range_human | All dimensions: (0, 4) |
| Individual annotators | None (only aggregated scores provided) |

**Use Case**: Large-scale experiments, judge importance analysis (Tracks 2.1, 2.2)

---

### 3. SummEval (JUDGE-BENCH)

**Format**: News summarization with 8 individual annotators (3 experts + 5 crowdworkers)

**Dimensions**: coherence, consistency, fluency, relevance (1-5 scale)

**Sample Structure**:
```python
{
  'question': str,                          # Source news article
  'response': str,                          # Generated summary
  'dataset': 'summeval',
  'target_human_aggregated': {              # ✅ Mean across 8 annotators
    'coherence': float,                     # 1-5 scale
    'consistency': float,                   # 1-5 scale
    'fluency': float,                       # 1-5 scale
    'relevance': float                      # 1-5 scale
  },
  'target_human_individual': [              # ✅ 8 individual annotators
    {'coherence': int, 'consistency': int, 'fluency': int, 'relevance': int},
    # ... 7 more annotators
  ],
  'score_range_human': {
    'coherence': (1, 5),
    'consistency': (1, 5),
    'fluency': (1, 5),
    'relevance': (1, 5)
  },
  'dimensions': ['coherence', 'consistency', 'fluency', 'relevance'],
  'task_type': 'summarization',
  'reference_output': str,                  # ✅ Gold reference summary
  'annotator_metadata': {
    'num_annotators': 8,
    'num_experts': 3,
    'num_crowdworkers': 5
  }
}
```

**Sample Data**:
| Field | Value |
|-------|-------|
| dimensions | 4 dimensions (coherence, consistency, fluency, relevance) |
| target_human_aggregated | {'coherence': 2.5, 'consistency': 4.375, 'fluency': 3.75, 'relevance': 3.125} |
| target_human_individual | 8 annotators (3 experts + 5 crowdworkers) |
| reference_output | Gold reference summary provided ✅ |

**Use Case**: Primary dataset for Track 1.3 (JUDGE-BENCH validation), inter-annotator analysis

---

### 4. NewsRoom (JUDGE-BENCH)

**Format**: News summarization with crowdsourced annotations

**Dimensions**: Informativeness, Relevance, Fluency, Coherence (1-5 scale, capitalized)

**Sample Structure**:
```python
{
  'question': str,                          # Source article
  'response': str,                          # Generated summary
  'dataset': 'newsroom',
  'target_human_aggregated': {              # ✅ Mean scores
    'Informativeness': float,               # 1-5 scale
    'Relevance': float,                     # 1-5 scale
    'Fluency': float,                       # 1-5 scale
    'Coherence': float                      # 1-5 scale
  },
  'target_human_individual': [              # ✅ Variable annotators (typically 3)
    {'Informativeness': int, 'Relevance': int, 'Fluency': int, 'Coherence': int},
    # ...
  ],
  'score_range_human': {
    'Informativeness': (1, 5),
    'Relevance': (1, 5),
    'Fluency': (1, 5),
    'Coherence': (1, 5)
  },
  'dimensions': ['Informativeness', 'Relevance', 'Fluency', 'Coherence'],
  'task_type': 'summarization',
  'reference_output': None                  # ❌ No gold reference
}
```

**Use Case**: Cross-task comparison (Track 2.2)

---

### 5. Recipe (JUDGE-BENCH)

**Format**: Recipe quality evaluation with 6 dimensions and many annotators

**Dimensions**: grammar, fluency, verbosity, structure, success, overall (1-6 scale)

**Sample Structure**:
```python
{
  'question': 'Evaluate this recipe:',      # Standard prompt
  'response': str,                          # Recipe text
  'dataset': 'recipe',
  'target_human_aggregated': {              # ✅ Mean across 20 annotators
    'grammar': float,                       # 1-6 scale
    'fluency': float,
    'verbosity': float,
    'structure': float,
    'success': float,
    'overall': float
  },
  'target_human_individual': [              # ✅ 20 crowdworker annotators
    {'grammar': int, 'fluency': int, ...},
    # ... 19 more
  ],
  'score_range_human': {                    # All dimensions: 1-6
    'grammar': (1, 6),
    # ... same for all dimensions
  },
  'dimensions': ['grammar', 'fluency', 'verbosity', 'structure', 'success', 'overall'],
  'task_type': 'recipe_generation'
}
```

**Use Case**: Most diverse individual annotations

---

### 6. WMT-20 Translation (JUDGE-BENCH)

**Format**: Machine translation quality with expert annotations

**Language Pairs**: English→German (`wmt_en_de`), Chinese→English (`wmt_zh_en`)

**Sample Structure**:
```python
{
  'question': str,                          # Source text (English or Chinese)
  'response': str,                          # Machine translation
  'dataset': 'wmt_en_de' or 'wmt_zh_en',
  'target_human_aggregated': {              # ✅ Mean across 3 experts
    'quality': float                        # 0-6 scale
  },
  'target_human_individual': [              # ✅ 3 expert annotators
    {'quality': int},
    {'quality': int},
    {'quality': int}
  ],
  'score_range_human': {'quality': (0, 6)},
  'dimensions': ['quality'],                # Single dimension
  'task_type': 'translation',
  'reference_output': str,                  # ✅ Human reference translation
  'context': {
    'source_language': 'English'/'Chinese',
    'target_language': 'German'/'English',
    'language_pair': 'en_de'/'zh_en'
  },
  'annotator_metadata': {
    'num_annotators': 3,
    'expert_annotators': True               # ✅ Expert quality
  }
}
```

**Use Case**: Expert-quality annotations, cross-lingual experiments

---

### 7. MSLR

**Format**: Medical literature summarization with MAJ-Eval dimensions

**Status**: ⚠️ **Sparse annotations** - Only 7.8% of samples (364/4654) annotated by design

**Dimensions**: fluency (0-2), pio_consistency (0-2, avg of P/I/O), effect_direction (0-1), evidence_strength (0-1)

**Sample Structure**:
```python
{
  'question': str,                          # Cochrane Review + source docs + reference
  'response': str,                          # Generated summary
  'dataset': 'mslr',
  'target_human_aggregated': {              # ⚠️ Only present in 7.8% of samples
    'fluency': float,                       # 0-2 scale
    'pio_consistency': float,               # 0-2 scale (avg of P/I/O)
    'effect_direction': float,              # 0-1 scale
    'evidence_strength': float              # 0-1 scale
  },
  'target_human_individual': [...],         # ⚠️ Only present in annotated samples
  'score_range_human': {
    'fluency': (0, 2),
    'pio_consistency': (0, 2),
    'effect_direction': (0, 1),
    'evidence_strength': (0, 1)
  },
  'dimensions': ['fluency', 'pio_consistency', 'effect_direction', 'evidence_strength'],
  'task_type': 'summarization',
  'reference_output': str                   # ✅ Reference summary
}
```

**Use Case**: MAJ-Eval comparison (must filter for annotated samples only)

## Consistency Analysis

### ✅ **Standardized Format Compliance**

All datasets conform to the 15-column standardized schema with consistent null handling (using `None` for missing data).

### ⚠️ **Key Differences to Consider**

#### 1. Score Ranges (Normalization Required)
| Dataset | Range | Notes |
|---------|-------|-------|
| HelpSteer2 | 0-4 | Consistent across 5 dimensions |
| SummEval | 1-5 | Consistent across all 4 dimensions |
| NewsRoom | 1-5 | Consistent across 4 dimensions |
| Recipe | 1-6 | Consistent across 6 dimensions |
| WMT | 0-6 | Single quality dimension |
| MSLR | 0-2, 0-1 | Mixed ranges by dimension |
| UltraFeedback | 0-10 | Synthetic only |

**Implication**: Requires normalization before cross-dataset comparison

#### 2. Annotation Availability
| Dataset | Aggregated | Individual | Count | Quality |
|---------|------------|------------|-------|---------|
| UltraFeedback | ❌ | ❌ | 0 | N/A (synthetic only) |
| HelpSteer2 | ✅ | ❌ | 0 | Aggregated only |
| SummEval | ✅ | ✅ | 8 | 3 experts + 5 crowd |
| NewsRoom | ✅ | ✅ | 3 | Crowdsourced |
| Recipe | ✅ | ✅ | 20 | Crowdsourced |
| WMT | ✅ | ✅ | 3 | ✅ **Expert quality** |
| MSLR | ⚠️ | ⚠️ | ? | Sparse (7.8%) |

**Implication**: Only 5 datasets have individual annotators for inter-annotator analysis

#### 3. Reference Outputs
| Dataset | Reference Available |
|---------|---------------------|
| SummEval | ✅ Gold summary |
| WMT | ✅ Human translation |
| MSLR | ✅ Reference summary |
| Others | ❌ |

**Implication**: Reference-based judge evaluation only possible on 3 datasets

#### 4. Dimension Naming (Case Sensitivity)
- **SummEval**: lowercase (`coherence`, `consistency`, `fluency`, `relevance`)
- **NewsRoom**: Capitalized (`Informativeness`, `Relevance`, `Fluency`, `Coherence`)
- **Recipe**: lowercase (`grammar`, `fluency`, ...)
- **HelpSteer2**: lowercase (`helpfulness`, `correctness`, ...)

**Implication**: Dimension name matching should be case-insensitive

## Cross-Dataset Compatibility Matrix

| Use Case | Compatible Datasets | Notes |
|----------|---------------------|-------|
| **Human annotation training** | HelpSteer2, SummEval, NewsRoom, Recipe, WMT, MSLR | 6 datasets with human scores |
| **Individual annotator analysis** | SummEval, NewsRoom, Recipe, WMT, MSLR | Inter-annotator agreement metrics |
| **Expert annotations** | SummEval (3), WMT (3) | Highest annotation quality |
| **Multi-dimensional** | HelpSteer2 (5), SummEval (4), NewsRoom (4), Recipe (6), MSLR (4) | >1 dimension |
| **Reference-based evaluation** | SummEval, WMT, MSLR | Ground truth comparisons |
| **Summarization tasks** | SummEval, NewsRoom, MSLR | Same task type |
| **Cross-lingual** | WMT (en→de, zh→en) | Translation quality |

## Loading Datasets

```python
from pipeline.core.dataset_loader import DatasetLoader

loader = DatasetLoader()

# UltraFeedback (auto-downloads from HuggingFace)
df = loader.load('ultrafeedback', n_samples=2000)

# HelpSteer2 (auto-downloads from HuggingFace)
df_train = loader.load('helpsteer2', split='train', n_samples=None)
df_val = loader.load('helpsteer2', split='validation', n_samples=None)

# MSLR (from local download - automatically filters for annotated samples only)
df = loader.load('mslr', n_samples=100)

# JUDGE-BENCH tasks
df = loader.load('judge_bench', task_name='summeval', n_samples=None)
df = loader.load('judge_bench', task_name='newsroom', n_samples=None)
df = loader.load('judge_bench', task_name='recipe', n_samples=None)
df = loader.load('judge_bench', task_name='wmt_en_de', n_samples=None)
df = loader.load('judge_bench', task_name='wmt_zh_en', n_samples=None)

# Caching behavior
df = loader.load('summeval', use_cache=True)   # Loads from cache if available
df = loader.load('summeval', use_cache=False)  # Forces reload from source
```

**Caching**: Datasets are automatically cached to `datasets/processed/*.pkl` on first load. Subsequent loads retrieve from cache unless `use_cache=False`.

## Recommendations for Fellowship Experiments

### Track 1.3 (JUDGE-BENCH validation)
- **Use SummEval**: Best option (8 annotators, 4 dimensions, gold reference)
- **Use WMT**: Expert quality, single dimension, reference translation
- **Consider NewsRoom**: 4 dimensions, 3 annotators

### Track 2.2 (Cross-task judge importance)
- SummEval vs NewsRoom (both summarization, different dimensions)
- SummEval vs Recipe (different tasks, similar score ranges)
- WMT (translation) vs others (test generalization)

### Normalization Strategy
- Min-max normalization: `(score - min) / (max - min)` → [0, 1]
- Z-score within dataset: `(score - mean) / std`
- Consider dimension-specific normalization

### Data Quality Considerations
1. **MSLR**: Sparse annotations (only 7.8% of samples annotated by design) - must filter for annotated samples
2. **HelpSteer2**: No individual annotators (only aggregated scores)
3. **UltraFeedback**: No human annotations (synthetic persona simulation required)

## Git Tracking

- **Tracked**: `judge-bench/`, `mslr-annotated/` (raw data, documentation)
- **Not tracked**: `processed/*.pkl` (large processed files, automatically regenerated)
- **Excluded**: Workshop data may not be used in fellowship experiments

## External Dependencies

### Auto-Downloaded (HuggingFace)
- UltraFeedback: `openbmb/UltraFeedback`
- HelpSteer2: `nvidia/HelpSteer2`
- MSLR: `allenai/mslr2022` (medical literature summarization)

### Manually Downloaded (GitHub)
- JUDGE-BENCH: Already included in repo
- MSLR annotated: Already included in repo (version tracked)

## Dataset Sizes

```
Total: ~243MB (raw) + variable (processed, gitignored)

datasets/
├── data_with_judge_scores.pkl    # 10MB (workshop)
├── judge-bench/                  # 227MB (19 tasks)
├── mslr-annotated/               # 5.6MB (470 reviews)
└── processed/                    # Variable (automatically cached, gitignored)
```

## Summary

**Status**: ✅ **7 datasets fully preprocessed and validated**

**Consistency**: High - all datasets conform to 15-column standardized format

**Diversity**:
- Tasks: Summarization (4), Translation (2), Recipe (1), Instruction (1), Helpfulness (1)
- Score ranges: 0-4, 1-5, 1-6, 0-6, 0-10
- Annotators: 0, 3, 8, 20
- Annotation quality: Crowdsourced, Expert, Synthetic, Mixed

**Ready for Experiments**:
- ✅ UltraFeedback, HelpSteer2, SummEval, NewsRoom, Recipe, WMT en→de, WMT zh→en
- ⚠️ MSLR (sparse annotations - only 7.8% annotated by design)
