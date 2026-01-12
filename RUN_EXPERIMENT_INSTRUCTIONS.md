# Instructions for Running Judge Selection Experiment

> **Audience**: AI assistant helping run experiments on a different machine
> **Task**: Execute backward elimination to select optimal judges from a pool
> **Expected time**: 20-60 minutes depending on pool size and hardware

---

## Prerequisites Check

Before starting, verify:

```bash
# 1. Check Python version (need 3.8+)
python --version

# 2. Check if CUDA/GPU available (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 3. Verify you're in the correct directory
ls -la | grep -E "pipeline|experiments|config"
```

**Expected**: You should see directories: `pipeline/`, `experiments/`, `config/`, `judges/`

---

## Step 1: Environment Setup

### Option A: Use Existing Virtual Environment

```bash
# Activate if .venv exists
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate  # Windows
```

### Option B: Create New Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check critical packages
python -c "import torch; import pygam; import pandas; print('✓ All packages installed')"
```

---

## Step 2: Prepare Your Data

### Data Format Required

Your dataset must be a **pickled pandas DataFrame** with:
- Column `judge_scores`: List/array of judge scores per sample
- Column `target`: Human preference scores (or similar ground truth)

### Example Data Structure

```python
import pandas as pd
import pickle

# Your dataframe should look like:
df = pd.DataFrame({
    'judge_scores': [[7, 8, 6, 9, ...], [5, 7, 8, 6, ...], ...],  # Each row has N judge scores
    'target': [8.5, 6.2, 9.1, ...],  # Human scores
    # Optional: other metadata columns
})

# Save it
with open('datasets/processed/my_dataset.pkl', 'wb') as f:
    pickle.dump(df, f)
```

### Verify Data Format

```bash
python -c "
import pickle
import pandas as pd

with open('datasets/processed/YOUR_FILE.pkl', 'rb') as f:
    df = pickle.load(f)
    
print(f'Rows: {len(df)}')
print(f'Columns: {df.columns.tolist()}')
print(f'Judge scores shape: {df[\"judge_scores\"].iloc[0].shape if hasattr(df[\"judge_scores\"].iloc[0], \"shape\") else len(df[\"judge_scores\"].iloc[0])}')
print(f'Target column exists: {\"target\" in df.columns}')
"
```

---

## Step 3: Prepare Judge Pool

### Create Judge Definitions File

Create `judges/my_pool/all_candidates.yaml` with your full pool:

```yaml
# judges/my_pool/all_candidates.yaml

truthfulness:
  id: truthfulness
  name: "Truthfulness"
  description: "Factual accuracy and correctness"
  scale: "1-5"

helpfulness:
  id: helpfulness
  name: "Helpfulness"
  description: "How useful the response is"
  scale: "1-5"

clarity:
  id: clarity
  name: "Clarity"
  description: "How clear and understandable"
  scale: "1-5"

# ... add all 50+ judges
```

### Verify Judge File

```bash
python -c "
import yaml
from pathlib import Path

with open('judges/my_pool/all_candidates.yaml') as f:
    judges = yaml.safe_load(f)
    
print(f'Total judges in pool: {len(judges)}')
print(f'Judge IDs: {list(judges.keys())[:5]}...')
"
```

---

## Step 4: Configure Experiment

### Edit Configuration File

```bash
# Copy template
cp config/backward_selection_example.yaml config/my_experiment.yaml

# Edit it
nano config/my_experiment.yaml  # or vim, code, etc.
```

### Required Changes

```yaml
# config/my_experiment.yaml

name: "my-judge-selection"
description: "Select best 10 judges for [YOUR TASK]"

# Point to YOUR judge pool
initial_judge_file: "judges/my_pool/all_candidates.yaml"

# Point to YOUR data
data_file: "datasets/processed/my_dataset.pkl"
target_column: "target"  # Or whatever your ground truth column is called

# Selection parameters
target_judges: 10  # How many judges you want to select
max_iterations: 50  # Should be >= (pool_size - target_judges)

# Output location
output_dir: "results/my_selection_experiment"
```

### Validate Configuration

```bash
python -c "
import yaml
with open('config/my_experiment.yaml') as f:
    config = yaml.safe_load(f)
print('Configuration loaded successfully')
print(f'Target judges: {config[\"target_judges\"]}')
print(f'Data file: {config[\"data_file\"]}')
print(f'Judge pool: {config[\"initial_judge_file\"]}')
"
```

---

## Step 5: Run the Experiment

### Determine Hardware

**If you have GPU:**
```bash
# Check GPU info
nvidia-smi  # Should show GPU name and memory

# Use GPU mode (8-10x faster)
USE_GPU="--gpu"
```

**If CPU only:**
```bash
USE_GPU="--cpu"
```

### Execute Selection

```bash
# Full command
python run_mlp_selection.py \
    --config config/my_experiment.yaml \
    $USE_GPU \
    --hidden-dim 64 \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 0.001 \
    --dropout 0.2 \
    --output results/my_selection_$(date +%Y%m%d_%H%M%S)
```

### Expected Output

You should see:
```
================================================================================
GPU-ACCELERATED JUDGE SELECTION - BACKWARD ELIMINATION
================================================================================
Strategy: Start with ALL candidate judges, iteratively remove
          the least important until reaching target number.
================================================================================

Configuration:
  Config file: config/my_experiment.yaml
  Data file: datasets/processed/my_dataset.pkl
  Target: target
  Starting judges: all in file
  Target judges: 10
  Max iterations: 50
  Output: results/my_selection_20260102_143022

MLP Settings:
  Device: cuda
  Hidden dim: 64
  Batch size: 32
  Max epochs: 100
  Learning rate: 0.001
  Dropout: 0.2

================================================================================
INITIALIZATION
================================================================================
Initialized MLPJudgeSelector on device: cuda
GPU available: True
GPU name: NVIDIA GeForce RTX 3090
Loaded 50 initial judges

================================================================================
RUNNING ITERATIVE SELECTION
================================================================================

============================================================
Iteration 1 / 50
Current judges: 50
============================================================
Training MLP with 50 judges on cuda
✓ Epoch 15/100, Train: 0.1234, Val: 0.1456 (Best)
Early stopping at epoch 27. Best validation loss: 0.1398 at epoch 15
Restored best model from epoch 15 (val_loss: 0.1398)
Computing gradient-based importance...
Computing variance-based importance...
Test R²: 0.6523
Improvement: 0.6523
Composite score: 0.7234
Removing judge: punctuation (importance: 0.0234)

============================================================
Iteration 2 / 50
Current judges: 49
============================================================
...

============================================================
Iteration 40 / 50
Current judges: 10
============================================================
Training MLP with 10 judges on cuda
Test R²: 0.7234
Improvement: 0.0012
Stopping: target_judges_reached_10

================================================================================
RESULTS SUMMARY
================================================================================
Total iterations: 40
Final judge count: 10
Final test R²: 0.7234
Final test Spearman ρ: 0.7456
Stop reason: target_judges_reached_10

Iteration progression:
Iter Judges       R² Spearman            Removed
------------------------------------------------------------
   0      50   0.6523     0.6789         punctuation
   1      49   0.6587     0.6823          word_count
   2      48   0.6612     0.6845        capitalization
...
  39      10   0.7234     0.7456                none

✅ Selection complete! Results saved to results/my_selection_20260102_143022

Final selected judges:
Rank                         Judge   Importance
--------------------------------------------------
   1                 truthfulness       0.8923
   2                  helpfulness       0.8567
   3         logical_consistency       0.8234
   4                     clarity       0.7891
   5                   relevance       0.7645
   6                  creativity       0.7123
   7                   coherence       0.6987
   8                completeness       0.6754
   9                       safety       0.6432
  10                  factuality       0.6210
```

---

## Step 6: Monitor Progress

The experiment will take time. You can monitor it:

### Watch Iteration Files

```bash
# In another terminal
watch -n 5 'ls -lh results/my_selection_*/iteration_*/ | tail -20'
```

### Check Current Metrics

```bash
# See latest iteration results
cat results/my_selection_*/iteration_*/result.json | jq '.test_metrics'
```

### Estimate Time Remaining

```bash
python -c "
import json
from pathlib import Path
import time

results_dir = sorted(Path('results').glob('my_selection_*'))[-1]
iterations = sorted(results_dir.glob('iteration_*/result.json'))

if len(iterations) > 1:
    # Calculate average time per iteration
    times = []
    for i in range(len(iterations)-1):
        t1 = iterations[i].stat().st_mtime
        t2 = iterations[i+1].stat().st_mtime
        times.append(t2 - t1)
    
    avg_time = sum(times) / len(times)
    remaining_iters = 50 - len(iterations)
    
    print(f'Completed: {len(iterations)} iterations')
    print(f'Avg time/iter: {avg_time:.1f} seconds')
    print(f'Est. remaining: {(remaining_iters * avg_time / 60):.1f} minutes')
"
```

---

## Step 7: Retrieve Results

### Final Outputs

After completion, you'll find:

```
results/my_selection_TIMESTAMP/
├── config.yaml              # Saved configuration
├── final_results.json       # Complete metrics
├── final_judges.txt         # List of selected 10 judges
├── iteration_00/
│   ├── result.json          # Metrics at iteration 0
│   ├── judges.txt           # Judge list (50 judges)
│   └── mlp_model.pt         # MLP checkpoint
├── iteration_01/
│   └── ...
└── iteration_39/            # Final iteration
    ├── result.json
    ├── judges.txt           # Final 10 judges
    └── mlp_model.pt
```

### Extract Key Information

```bash
# Get final judge list
cat results/my_selection_*/final_judges.txt

# Get final performance
cat results/my_selection_*/final_results.json | jq '.[-1].test_metrics'

# Get importance scores
cat results/my_selection_*/final_results.json | jq '.[-1].importance_scores'
```

### Create Summary Report

```bash
python -c "
import json
from pathlib import Path

results_dir = sorted(Path('results').glob('my_selection_*'))[-1]
with open(results_dir / 'final_results.json') as f:
    results = json.load(f)

final = results[-1]
print('=' * 60)
print('JUDGE SELECTION SUMMARY')
print('=' * 60)
print(f'Starting judges: {results[0][\"n_judges\"]}')
print(f'Final judges: {final[\"n_judges\"]}')
print(f'Final R²: {final[\"test_metrics\"][\"r2\"]:.4f}')
print(f'Final Spearman ρ: {final[\"test_metrics\"][\"spearman_rho\"]:.4f}')
print(f'Stop reason: {final[\"stop_reason\"]}')
print()
print('Selected Judges (ranked by importance):')
sorted_judges = sorted(final['importance_scores'].items(), key=lambda x: x[1], reverse=True)
for i, (judge, score) in enumerate(sorted_judges, 1):
    print(f'{i:2d}. {judge:30s} {score:.4f}')
" > results/SUMMARY.txt

cat results/SUMMARY.txt
```

---

## Step 8: Transfer Results Back

### Package Results

```bash
# Create archive
tar -czf judge_selection_results.tar.gz results/my_selection_*/

# Check size
ls -lh judge_selection_results.tar.gz
```

### What to Send Back

Minimal (just results):
```bash
# Copy these files
results/my_selection_*/final_judges.txt
results/my_selection_*/final_results.json
results/SUMMARY.txt
```

Full (for debugging/analysis):
```bash
# Send entire archive
judge_selection_results.tar.gz
```

---

## Troubleshooting

### Error: "CUDA out of memory"

**Solutions:**
```bash
# 1. Reduce batch size
--batch-size 16  # Instead of 32

# 2. Reduce hidden dimension
--hidden-dim 32  # Instead of 64

# 3. Fall back to CPU
--cpu
```

### Error: "No module named 'torch'"

**Solution:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Error: "No module named 'pygam'"

**Solution:**
```bash
pip install pygam
```

### Error: "Cannot load data file"

**Check:**
```bash
# Verify file exists
ls -lh datasets/processed/my_dataset.pkl

# Verify format
python -c "
import pickle
with open('datasets/processed/my_dataset.pkl', 'rb') as f:
    df = pickle.load(f)
print(df.head())
"
```

### Selection is Very Slow (CPU)

**Expected times:**
- GPU (RTX 3090): ~30 seconds/iteration
- CPU (16 cores): ~4 minutes/iteration

**To speed up on CPU:**
```bash
# Reduce epochs
--epochs 50

# Smaller model
--hidden-dim 32

# Or just use GPU if available
```

### Results Look Wrong (R² too low)

**Possible causes:**
1. Data mismatch (judge scores don't match judge pool)
2. Wrong target column
3. Insufficient data (<500 samples)

**Debug:**
```bash
python -c "
import pickle
import yaml

# Load data
with open('datasets/processed/my_dataset.pkl', 'rb') as f:
    df = pickle.load(f)

# Load judges
with open('judges/my_pool/all_candidates.yaml') as f:
    judges = yaml.safe_load(f)

print(f'Data samples: {len(df)}')
print(f'Judge pool size: {len(judges)}')
print(f'Judge scores per sample: {len(df[\"judge_scores\"].iloc[0])}')
print(f'Target stats: mean={df[\"target\"].mean():.2f}, std={df[\"target\"].std():.2f}')

# Should match
assert len(df['judge_scores'].iloc[0]) == len(judges), 'Mismatch!'
print('✓ Data and judges match')
"
```

---

## Quick Reference Commands

```bash
# Activate environment
source .venv/bin/activate

# Run experiment (GPU)
python run_mlp_selection.py --config config/my_experiment.yaml --gpu

# Run experiment (CPU)
python run_mlp_selection.py --config config/my_experiment.yaml --cpu

# Check progress
ls -lh results/my_selection_*/iteration_*/

# View final results
cat results/my_selection_*/final_judges.txt

# Package for transfer
tar -czf results.tar.gz results/my_selection_*/
```

---

## Expected Timeline

| Pool Size | Target | GPU Time | CPU Time |
|-----------|--------|----------|----------|
| 20 → 10   | 10 iter| ~5 min   | ~40 min  |
| 50 → 10   | 40 iter| ~20 min  | ~2.5 hrs |
| 100 → 10  | 90 iter| ~45 min  | ~6 hrs   |

---

## Success Criteria

**You know it worked when:**
1. ✅ Final iteration shows: `Stop reason: target_judges_reached_10`
2. ✅ `final_judges.txt` contains exactly 10 judges
3. ✅ Final R² is > 0.6 (task-dependent, but this is reasonable)
4. ✅ No errors in terminal output
5. ✅ All iteration directories created (0 through N)

**If any of these fail, check troubleshooting section above.**

---

## Questions for the Human

Before starting, confirm:
1. Do you have the dataset in pickle format?
2. Do you have the judge pool YAML file?
3. Is GPU available, or should I use CPU?
4. What's the target number of judges to select?
5. Where should results be saved?

Answer these and I'll execute the experiment.
