# Persona Reproducibility Experiment

This folder contains the scripts to calculate the empirical stability, variance, and reproducibility of the 14-persona LLM grading pipeline. By treating the simulated persona panel as a stochastic judge, this pipeline tests if exactly the same prompts yield exactly the same quantitative score across multiple identical runs.

## 🛠️ How to Run the Experiment

To launch the full reproducibility test (e.g., 5 parallel independent runs over 500 samples), use the provided Fish script. Under the hood, this executes `run_reproducibility.py` inside background tasks.

```fish
cd rebutal_augusto/
fish launch_parallel.fish
```

**What it does:**
1. Loads a subset (default 500) of the original `data_with_judge_scores.pkl` QA dataset.
2. Spawns 5 independent evaluation runs in parallel using `meta-llama/llama-3.3-70b-instruct` natively patched for `temperature=0.7`.
3. Auto-recovers from HTTP exceptions, 502s, and infinite hangs with exponential backoff and localized retries.
4. Outputs the comprehensive raw dataset, complete with embedded JSON feedbacks and averages, to `rebutal_augusto/results/run_{1..5}.pkl`.

## 📊 Generating the Analysis

Once you have multiple `.pkl` files in the `results/` folder, run the analyzer script to compute metrics and plots:

```bash
.venv/bin/python rebutal_augusto/analyze_reproducibility.py \
    --results-dir rebutal_augusto/results
```

This calculates strict pairwise correlations (Pearson & Spearman), absolute variance, and per-persona stability percentage. It generates 5 plots in `rebutal_augusto/results/plots/`.

---

## 📈 How to Read the Plots

### 1. `score_distributions.png`
**What it is:** Density overlaid histograms showing the macro-level view of scores distributed between 0.0 and 10.0 for each run.
**How to read it:** These are the **average ground-truth scores** (the mean of the 14 personas combined) plotted continuously. Perfect reproducibility means the colored shapes (Run 1 vs Run 2 etc.) should perfectly stack on top of each other, forming one solid shape without overlapping tails.

### 2. `per_sample_variance.png`
**What it is:** Twin histograms showing the absolute Standard Deviation (left) and the Min-Max range (right) of each sample across the 5 runs. 
**How to read it:** Lower is better. A large spike at `0` signifies perfect agreement. If the average range is `0.5`, it means a single sample's score drifted by around half a point when evaluating exactly the same answer in different runs.

### 3. `pairwise_scatter.png`
**What it is:** A matrix of scatter plots comparing two random runs (e.g. Y=Run 2, X=Run 1). 
**How to read it:** The red dotted line is the `Y = X` line of perfect agreement. Dots strictly clinging to that line signify high stability. Above the chart lies your $r$ Pearson Correlation coefficient. You want numbers tight and dense approaching `1.0`.

### 4. `persona_stability.png`
**What it is:** Bar charts grading the individual personas directly (e.g., "Professor", "CEO", "Child").
**How to read it:** 
* **Left (Variability):** Standard deviation ranking. Small bars are good. Personas with low standard deviation are highly consistent.
* **Right (Exact agreement):** Percentage of times the persona gave the *exact same 0-10 integer score* for the same sample in all 5 runs. Large bars are good. This reveals whether a "Therapist" prompt is intrinsically more stable than a "Skeptic" prompt.

### 5. `correlation_heatmaps.png`
**What it is:** Global Pearson and Spearman correlation matrices across all combinations of runs.
**How to read it:** Dark and deeply colored cells (approaching 1.0) imply that despite micro-variances by LLM personas, the overarching ranking layout is successfully robust and preserved equivalently across different identical trials. 
