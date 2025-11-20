# Judge Definitions

This directory contains judge definitions organized by dataset, with depth-based file structure for granularity experiments.

## Directory Structure

```
judges/
├── README.md
├── helpsteer2/
│   ├── depth_0_parents.yaml       # Parent judges (2 judges)
│   └── depth_1_children.yaml      # First-generation children (9-10 judges)
└── ultrafeedback/
    └── depth_0_parents.yaml        # Baseline 10 judges
```

## File Organization

### Depth-Based Files

Judges are organized by hierarchy depth to support granularity experiments:

- **depth_0_parents.yaml**: Top-level parent judges created from dataset dimension descriptions
- **depth_1_children.yaml**: First-generation children from decomposing parent judges
- **depth_2_grandchildren.yaml**: Second-generation (if deeper decomposition is used)
- etc.

This organization allows experiments to:
- Test parent judges only (`judge_file: "judges/helpsteer2/depth_0_parents.yaml"`)
- Test children only (`judge_file: "judges/helpsteer2/depth_1_children.yaml"`)
- Compare different granularities (`judge_files: ["judges/helpsteer2/depth_0_parents.yaml", "judges/helpsteer2/depth_1_children.yaml"]`)

## Judge Format

Each YAML file contains judges in this format:

```yaml
judges:
  - id: helpsteer2-helpfulness-judge
    name: Helpfulness Judge
    version: '1.0'
    description: "Evaluates how helpful the response is..."
    scoring_description: "Score from 0-4 based on helpfulness"
    definition: "Detailed explanation of helpfulness..."
    criteria:
      - range: [0.0, 0.9]
        label: "Very Poor"
        indicators:
          - "Response is largely irrelevant..."
          - "Provides little useful information..."
      - range: [1.0, 1.9]
        label: "Poor"
        indicators: ["..."]
      # ... more levels up to [3.5, 4.0]
    guidelines: ["Consider relevance...", "Assess completeness..."]
    score_range: [0.0, 4.0]
    dataset: helpsteer2
    parent_id: null  # null for parent, parent-id for children
    auto_generated: true
```

## Generation

Judges are automatically created by `JudgeCreationOrchestrator`:

1. **Parent Creation**: `ParentJudgeCreatorAgent` creates parent judges from dimension descriptions
2. **Decomposition** (optional): Track 3.0 decomposition pipeline creates child judges
3. **Depth Computation**: Recursive depth calculation based on `parent_id`
4. **File Saving**: Judges grouped by depth and saved to separate files

## Usage in Experiments

### Automatic Judge Creation

```yaml
# config.yaml
dataset: helpsteer2
target_dimensions:
  - helpfulness
  - correctness

judge_cache_strategy: "auto"  # or "force_create", "load_only"
judge_decomposition_depth: 1  # 0 = parents only, 1 = one level of children

judges:
  score_range: [0.0, 4.0]
  # Files will be auto-detected after creation
```

### Manual Judge Selection

```yaml
judges:
  # Option 1: Single file
  judge_file: "judges/helpsteer2/depth_0_parents.yaml"

  # Option 2: Multiple files for mixed depths
  judge_files:
    - "judges/helpsteer2/depth_0_parents.yaml"
    - "judges/helpsteer2/depth_1_children.yaml"
```

## Cache Strategy

- **auto** (default): Load from cache if exists, create if missing
- **force_create**: Always regenerate judges, overwrite existing files
- **load_only**: Only load from cache, error if missing

## Versioning

- Generated judges are tracked in git for reproducibility
- Judge IDs include dataset name (e.g., `helpsteer2-helpfulness-judge`)
- `auto_generated: true` metadata marks LLM-generated judges
- `dataset: <name>` metadata tags judges by source dataset
- `parent_id` tracks hierarchical relationships

## Adding New Datasets

To add judges for a new dataset:

1. Create dimension descriptions in dataset loader
2. Set `target_dimensions` in config
3. Run experiment - judges will be auto-created
4. Commit generated YAML files to git

Or manually create judges following the format above.
