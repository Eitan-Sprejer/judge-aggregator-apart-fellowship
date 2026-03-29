# Cross-Model Persona Comparison Experiment

This experiment tests **cross-model agreement**: given the exact same 14-persona evaluation pipeline and the same QA dataset, do different LLMs produce similar scores and rankings?

This is a companion to the reproducibility experiment in `rebutal_augusto/`, which tested whether the *same* model produces consistent results across multiple identical runs. Here we keep the prompts/personas fixed and vary the *model* to measure inter-model reliability.

## 🧠 Models

| Run | Model | Short Name | Status |
|-----|-------|------------|--------|
| 1 | `meta-llama/llama-3.3-70b-instruct` | Llama 3.3 70B | ✅ Reused from `rebutal_augusto` |
| 2 | `deepseek/deepseek-v3.2` | DeepSeek V3.2 | ✅ Clean run |
| 3 | `meta-llama/llama-4-maverick` | Llama 4 Maverick | ✅ 33 residual errors after repair |
| 4 | `mistralai/mixtral-8x7b-instruct` | Mixtral 8x7B | ✅ 3 residual errors after repair |

All models accessed via the **Martian API** (`https://api.withmartian.com/v1`).

### ❌ Dropped Models

The following models were tested but dropped because they could not reliably produce structured JSON output:

| Model | Problem |
|-------|---------|
| `martian/lobster` | 5,548 / 7,000 persona scores returned empty responses (HTTP 200 but no content) |
| `nvidia/nemotron-3-super-120b-a12b` | 6,719 / 7,000 scores failed — model returned prose instead of JSON |
| `openai/gpt-5.4-mini` | 7,000 / 7,000 scores failed — complete inability to produce JSON output |

These models consistently returned HTTP 200 OK but with non-JSON content (empty bodies, plain text analysis, or markdown-wrapped responses), making them incompatible with the persona simulation pipeline which requires raw JSON `{"score": N, "analysis": "..."}` responses.

## 🛠️ How to Run the Experiment

### Prerequisites
- Python virtual environment at `.venv/` with project dependencies
- `MARTIAN_API_KEY` set in `.env`

### 1. Pre-copy Llama baseline
Run 1 uses the existing result from `rebutal_augusto`:
```fish
cp rebutal_augusto/results/run_1.pkl rebutal_augusto_pt3/results/run_1.pkl
```

### 2. Launch the other models in parallel
```fish
cd /home/augustomb/Desktop/Rebutal
fish rebutal_augusto_pt3/launch_parallel.fish
```

**What it does:**
1. Loads the same 500-sample subset (seed=42) of `datasets/data_with_judge_scores.pkl`.
2. Spawns parallel evaluation runs, each with a different model via `run_reproducibility.py --model <model_name>`.
3. Each model evaluates all 500 samples through 14 persona prompts (7,000 API calls per model).
4. Auto-recovers from HTTP exceptions with exponential backoff and 3 retries.
5. Outputs results to `rebutal_augusto_pt3/results/run_{N}.pkl`.

### 3. Check for errors
```fish
.venv/bin/python rebutal_augusto_pt3/count_errors.py
```

### 4. Repair API errors
```fish
.venv/bin/python rebutal_augusto_pt3/repair_errors.py
```
The repair script reads the model name from each `.pkl` file's metadata and re-queries the correct model. Run it multiple times if needed — it's idempotent.

### 5. Generate the analysis
```fish
.venv/bin/python rebutal_augusto_pt3/analyze_reproducibility.py
```

---

## 📊 Results Summary

| Metric | Value |
|--------|-------|
| Mean Pearson correlation (pairwise) | **0.782** |
| Mean Spearman correlation (pairwise) | **0.729** |
| Mean \|Δscore\| between model pairs | **1.44 points** |
| Samples with same score from all models | **0.4%** |
| Most lenient model | Mixtral 8x7B (mean 6.74) |
| Strictest model | DeepSeek V3.2 (mean 4.94) |
| Most stable persona (cross-model) | Parent (std=1.19) |
| Most variable persona (cross-model) | Privacy Advocate (std=1.72) |

---

## 📈 How to Read the Plots

All plots are saved to `results/plots/`.

### 1. `score_distributions.png`
**What it is:** Density overlaid histograms showing the distribution of average persona scores (0–10) for each model.  
**How to read it:** Models that produce similar score distributions will overlap. Shifted distributions indicate systematic leniency or harshness differences between models.

### 2. `per_sample_variance.png`
**What it is:** Histograms showing per-sample Standard Deviation (left) and Min-Max range (right) across all 4 models.  
**How to read it:** Higher values indicate samples where models disagree most. A spike at 0 means all models gave the same score.

### 3. `pairwise_scatter.png`
**What it is:** Scatter plots comparing every pair of models.  
**How to read it:** Points near the red Y=X line mean both models agreed. The Pearson r value above each plot quantifies linear agreement.

### 4. `persona_stability.png`
**What it is:** Bar charts showing which personas produce the most/least cross-model agreement.  
**How to read it:**
* **Left (Variability):** Mean std dev across models per persona. Small bars = consistent scoring across models.
* **Right (Exact agreement):** % of samples where all 4 models gave the exact same integer score.

### 5. `correlation_heatmaps.png`
**What it is:** Pearson and Spearman correlation matrices between all model pairs.  
**How to read it:** High values (→ 1.0) mean models preserve the same relative ranking of answers, even if absolute scores differ.
