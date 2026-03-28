# Persona Phrasing Sensitivity Experiment

This experiment is designed to quantify how much an LLM's feedback and scores fluctuate when the *linguistic phrasing* of a given persona definition changes, even while the *semantic meaning* (the core traits of the persona) remains identical.

## 🚀 How to Run the Experiment

To reproduce or re-run the phrasing experiment pipeline, follow these exact steps from the `rebutal_augusto_pt2` directory:

### 1. Define Persona Variants
The personas are defined as python dictionaries mapping persona names to their character bios. 
*   **Baseline**: Define the standard personas in `personas.py`.
*   **Variants**: Define your rewrites (e.g., poetic, technical, staccato) in `persona1.py`, `persona2.py`, `persona3.py`, and `persona4.py`.

### 2. Launch the Simulation (Parallel)
If you are using `fish` shell, launch the simulation pipeline across all 5 variations concurrently:
```bash
fish launch_phrasing.fish
```
This distributes the workload in the background, making 2,500 total evaluation permutations. Logs are stored in `results/variant_{v}.log`.

### 3. Repair API Timeouts
The Martian API is heavily rate-limited and will inevitably throw `502 Bad Gateway` timeouts or invalid JSON during a 2,500-request parallel run. To actively crawl through the completed `.pkl` files and surgically repair missing lines:
```bash
python repair_phrasing_errors.py
```

### 4. Execute Analysis & Plotting
Once the data is complete and repaired, extract the metrics and generate the graphs:
```bash
python analyze_phrasing.py
python plot_extra_phrasing.py
```
This outputs summary JSONs, statistics tables, and a suite of PNG plots to `results/phrasing_plots/`.

---

## 📈 How to Read the Generated Graphs

The analysis scripts automatically dump four critical visualizations into the `results/phrasing_plots/` directory.

### 1. `phrasing_mean_shifts.png`
*   **What it is**: A horizontal bar chart plotting the mean shift in absolute scores. 
*   **How to read it**: The baseline score is `0`. Bars stretching to the **left (negative)** mean that the specific phrasing variant made the persona systematically **stricter / harsher**. Bars stretching to the **right (positive)** mean the LLM became notably more **lenient**.

### 2. `phrasing_correlations.png`
*   **What it is**: A bar chart of the Spearman Rank Correlation between the variant's scores and the baseline's scores.
*   **How to read it**: This tests "relative ranking" rather than strictness. If an LLM gives a baseline score of `5` to Answer A and `9` to Answer B, did the phrasing variant also preserve that Answer B > Answer A? 
    *   **Near 1.0**: The phrasing change preserved the model's logic perfectly. 
    *   **Low values (<0.6)**: The phrasing wildly destabilized the model's internal logic and understanding of what a "good" answer is.

### 3. `phrasing_exact_matches.png`
*   **What it is**: A heatmap matrix comparing Exact Score Determinism.
*   **How to read it**: It tracks the literal percentage of times the LLM gave the exact identical 1-10 integer score (e.g. `8 == 8`) under the new phrasing compared to the baseline phrasing. Lower percentages indicate high hallucination or extreme sensitivity to prompt wording.

### 4. `phrasing_overall_density.png`
*   **What it is**: Layered density histograms tracking the overarching curve of all 1-10 scores across all 14 personas rolled together, split by variant.
*   **How to read it**: Use this to see if a certain phrasing variant drastically shifts the overarching bell curve of the model. For example, if the baseline curve is heavily clustered at `8`, a technical phrasing might shift the entire model's scoring frequency to be flatly distributed between `4` and `9`.
