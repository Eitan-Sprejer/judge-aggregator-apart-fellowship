# Backward Compatibility Review

This document lists all backward compatibility code added during refactoring. Review each item critically to decide what to keep vs. remove.

## Critical Analysis Framework

For each item, ask:
1. **Who needs this?** Workshop experiments only, or new fellowship code too?
2. **Cost of removal?** How much code breaks if we remove it?
3. **Maintenance burden?** Does it complicate the codebase?
4. **Alternative?** Can we achieve the same with a one-time migration script?

---

## 1. Auto-Detection of Column Names

**Files Affected**: `judge_evaluation.py`, `persona_simulation.py`, `aggregator_training.py`

**What it does**: Automatically detects whether data uses old format (instruction/answer) or new format (question/response)

### `pipeline/core/judge_evaluation.py` (lines 254-266)
```python
# Auto-detect column names for backward compatibility
if question_col is None:
    if 'question' in data.columns:
        question_col = 'question'
    elif 'instruction' in data.columns:
        question_col = 'instruction'
    else:
        raise ValueError("Could not find question column...")

if answer_col is None:
    if 'response' in data.columns:
        answer_col = 'response'
    elif 'answer' in data.columns:
        answer_col = 'answer'
```

**Similar code in**:
- `pipeline/core/persona_simulation.py` (lines 272-288)
- `pipeline/core/aggregator_training.py` (lines 591-600)

**Analysis**:
- ✅ **Keep**: Makes new code work with both formats seamlessly
- ✅ **Low cost**: ~20 lines per file, clean logic
- ✅ **Fellowship benefit**: Fellowship experiments might load workshop data for comparison
- ⚠️ **Alternative**: Force all data through standardization step first

**Recommendation**: **KEEP** - This is useful flexibility, not just legacy support

---

## 2. Individual Judge Rubric Getter Functions

**File**: `pipeline/utils/judge_rubrics.py` (lines 182-252)

**What it does**: Maintains individual functions like `get_truthfulness_rubric()` instead of just using `JUDGE_RUBRICS['truthfulness-judge']()`

```python
# Backward compatibility: Expose individual rubric getter functions
# These are kept for any code that might be calling them directly
def get_truthfulness_rubric() -> str:
    """Returns the truthfulness judge rubric."""
    return JUDGE_RUBRICS['truthfulness-judge']()

def get_harmlessness_rubric() -> str:
    """Returns the harmlessness judge rubric."""
    return JUDGE_RUBRICS['harmlessness-judge']()

# ... 8 more functions (70 lines total)
```

**Analysis**:
- ❌ **Remove**: Only workshop code might use these
- ✅ **Low removal cost**: Find/replace in 2-3 files max
- ❌ **Maintenance burden**: 70 lines of boilerplate
- ✅ **Better pattern**: Direct dict access `JUDGE_RUBRICS[judge_id]()`

**Recommendation**: **REMOVE** - Use dict access pattern instead

**Migration**: Search for `get_.*_rubric\(\)` calls, replace with dict access

---

## 3. DEFAULT_FEATURE_LABELS Constant

**File**: `pipeline/core/aggregator_training.py` (lines 38-51)

**What it does**: Provides hardcoded list of 10 judge names for old code that doesn't pass `feature_names`

```python
# Default feature labels for backward compatibility (10 judges)
DEFAULT_FEATURE_LABELS = [
    "Truthfulness",
    "Harmlessness",
    "Helpfulness",
    # ... 7 more
]

# Default judge order for reference (backward compatibility)
DEFAULT_JUDGE_IDS = [
    'truthfulness-judge',
    'harmlessness-judge',
    # ... 8 more
]
```

**Analysis**:
- ⚠️ **Mixed**: Useful as documentation of standard judge set
- ✅ **Low cost**: ~15 lines
- ⚠️ **Fellowship benefit**: Could define DEFAULT_10_JUDGES in config instead
- ❌ **Confusing**: Makes it unclear if feature_names is required or optional

**Recommendation**: **REFACTOR** - Move to `pipeline/config/` as named config, not "default"

**Better approach**:
```python
# In pipeline/config/experiment_config.py
STANDARD_10_JUDGES = JudgeConfig(
    judge_ids=['truthfulness-judge', ...],
    judge_names=['Truthfulness', ...]
)
```

---

## 4. load_training_config() Function

**File**: `pipeline/core/aggregator_training.py` (lines 72-100)

**What it does**: Loads old `training_config.json` format with deprecation warning

```python
def load_training_config(size_category: str = 'large_scale') -> Dict:
    """
    DEPRECATED: This function is for backward compatibility with workshop experiments.
    New experiments should use pipeline.config.ExperimentConfig instead.
    """
    logger.warning("load_training_config() is deprecated. New experiments should use "
                   "pipeline.config.ExperimentConfig instead.")

    config_path = Path(__file__).parent.parent.parent / 'config' / 'training_config.json'
    # ... loads JSON config
```

**Analysis**:
- ✅ **Keep short-term**: Workshop experiments still use this
- ❌ **Remove long-term**: Once workshop experiments migrated
- ✅ **Clear deprecation**: Users warned to migrate
- ⚠️ **Maintenance**: Adds 30 lines to file

**Recommendation**: **KEEP** with clear deprecation, plan removal timeline

**Action**: Add TODO comment with removal date (e.g., "Remove after workshop migration - Q2 2025")

---

## 5. JUDGE_IDS Constant in run_full_experiment.py

**File**: `run_full_experiment.py` (line 53-54)

**What it does**: Imports DEFAULT_JUDGE_IDS for backward compatibility

```python
# For backward compatibility
JUDGE_IDS = DEFAULT_JUDGE_IDS
```

**Analysis**:
- ❌ **Remove**: run_full_experiment.py can use DEFAULT_10_JUDGES from config
- ✅ **Zero cost**: One import change
- ❌ **Confusing**: Hides where judge list comes from

**Recommendation**: **REMOVE** immediately

**Migration**: Replace `JUDGE_IDS` with `DEFAULT_10_JUDGES.judge_ids` throughout file

---

## 6. Legacy Baseline Comparison Methods

**Files**: `run_full_experiment.py` (lines 1156-1239), `analyze_existing_experiment.py` (lines 144-300)

**What it does**: Fallback baseline comparison using old data format

**Analysis**:
- ❌ **Remove**: 100+ lines of duplicated code
- ✅ **High removal benefit**: Eliminates code complexity
- ⚠️ **Risk**: If new baseline code has bugs, no fallback
- ✅ **Better approach**: Fix new baseline code instead of maintaining fallback

**Recommendation**: **REMOVE** - Delete legacy methods, fix new code if needed

---

## 7. load_existing_personas() Method

**File**: `pipeline/core/dataset_loader.py` (lines 253-275)

**What it does**: Loads pre-generated persona data in old format

```python
def load_existing_personas(self, file_path: str, ...) -> pd.DataFrame:
    """
    Load existing dataset with persona annotations (backward compatibility).

    This method is for backward compatibility with workshop experiments.
    Use load() method for new experiments.
    """
    logger.warning("load_existing_personas() is deprecated. Use load() method...")
```

**Analysis**:
- ✅ **Keep short-term**: Workshop experiments use this
- ❌ **Remove long-term**: Once workshop data migrated to new format
- ✅ **Clear deprecation**: Users warned
- ⚠️ **Maintenance**: 25 lines

**Recommendation**: **KEEP** with deprecation, plan migration

**Action**: Create one-time script to migrate workshop persona data to new format

---

## Summary Table

| Item | Lines | Recommendation | Priority | Action |
|------|-------|----------------|----------|---------|
| Auto-detection (columns) | ~60 | **KEEP** | Low | None - useful feature |
| Individual getter functions | ~70 | **REMOVE** | High | Find/replace with dict access |
| DEFAULT_FEATURE_LABELS | ~15 | **REFACTOR** | Medium | Move to config module |
| load_training_config() | ~30 | **KEEP** (temp) | Low | Add removal timeline |
| JUDGE_IDS constant | ~2 | **REMOVE** | High | Replace with config |
| Legacy baseline methods | ~200 | **REMOVE** | High | Delete entirely |
| load_existing_personas() | ~25 | **KEEP** (temp) | Low | Plan migration script |

---

## Recommended Actions

### Immediate (High Priority)
1. **Remove individual getter functions** in `judge_rubrics.py`
2. **Remove JUDGE_IDS constant** in `run_full_experiment.py`
3. **Remove legacy baseline methods** in both files

### Medium Term (Next Sprint)
4. **Refactor DEFAULT_FEATURE_LABELS** to config module as named constants
5. **Create workshop data migration script** for persona data

### Low Priority (Before Conference Paper)
6. **Remove load_training_config()** after workshop migration complete
7. **Remove load_existing_personas()** after data migration

---

## Migration Script Needed

Create `scripts/migrate_workshop_data.py` to:
1. Load old workshop persona data
2. Standardize column names (instruction → question, answer → response)
3. Add target_type and target_score_range metadata
4. Save in new standardized format
5. Update workshop experiment scripts to use new format

Once migration complete, can remove all "KEEP (temp)" items.
