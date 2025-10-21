# Cohesive Methodology Proposals

The main document's methodology section has a broad range of different ideas for experiments, and doesn't necessarily tell a cohesive story of what we want to do.

This document serves the purpose of proposing more narrowed-down and cohesive methodology proposals. In other words, this is a filtered version of the methodology section as it stands today.

**Fellowship Focus**: We are focusing on **Tracks 1-4** for the fellowship stage. Track 5 (Robustness) was mostly completed for the workshop paper, and Track 6 (Architecture) is future work.

# Eitan:

As discussed, I think we all agree that the main novelties from our approach are:

* Interpretability: by interpreting the aggregator, we can get measures of how relevant each judge is to the final score for a specific human-annotated dataset.  
* Judge / rubric selection: we'll automatically define and select the best judges for a given dataset.  
* Individual human/persona preference decomposition: by training the aggregator on single-human preferences, we can get an overall decomposition of their preferences.  
* Context-dependency: we could measure how the human preference decomposition varies over different tasks / contexts. For example, we could see whether humans give more value to “correctness” judge for coding / math questions, and “creativeness” for story-telling questions or whatever.

Then, the main research question could be thought of as the following: “Which judges actually matter for human preferences, how do we systematically select them, and how robust are these patterns to different settings?”

## Six Parallel Research Tracks

### Track 1: Baseline Performance Comparison (PRIMARY \- 25% effort)

**Goal**: Compare aggregator performance against baselines on both synthetic and real data.

Experiments:

1. Performance on Persona-Annotated Synthetic Data  
   * Define a set of personas, and train aggregators on your synthetic persona data (now with self-confidence reports)  
   * Compare against baselines: single judge, naive mean, linear regression  
   * Metrics: Judge agreement (Spearman/kendall tau)  
2. Performance against [MAJ-Eval's](https://arxiv.org/pdf/2507.21028) aggregator  
   * Train directly on their own multi-judge data, and compare agreement with the human experts. Compare with their LLM that aggregates the scores.  
   * If we have sufficient time, use our own judges and aggregate on their scores. Compare that with their performance.  
3. Performance on [JUDGE-BENCH](https://aclanthology.org/2025.acl-short.20.pdf) Human-Annotated Data  
   * Train aggregators on JUDGE-BENCH human annotations  
   * Compare against same baselines  
   * Metrics: Agreement with human annotations (Spearman/Kendall tau)

Why This Track Matters:

* Demonstrates performance advantage over simple baselines  
* Validates that method works on both controlled synthetic and real human data  
* Shows practical applicability to real evaluation scenarios

---

### Track 2: Judge Interpretability (PRIMARY \- 40% effort)

**Core Contribution**: This is your main differentiator from related work.

Experiments:

1. **Judge Importance Analysis** (kinda already done)  
   * GAM feature importance across 20+ training runs ✓  
   * MLP interpretability: give the MLP another try, and see if we can find the judge importance by running ablation experiments  
   * Dhruv Yadav and Fernando Avalos prob have more ideas on this  
   * Extend: Compute importance across different task types from JUDGE-BENCH (see experiment 3\)  
2. Cross-Task Judge Importance Analysis  
   * Given a set of predefined judges, train separate aggregators on each JUDGE-BENCH task  
   * Build judge importance heatmap across tasks  
   * Discover: Which judges are universally important? Which are task-specific?  
3. Persona-Based Judge Importance Analysis (Related to track 4\)  
   * Measure how the judge importance varies across the different personas on the synthetic data.  
   * Train separate aggregators for each persona  
   * Discover: Do different personas weigh judges differently?

Why This Track Matters:

* MAJ-EVAL focuses on multi-agent debates but doesn't analyze which dimensions matter  
* Dialogue Evaluator focuses on efficiency but not interpretability  
* JUDGE-BENCH evaluates judges but doesn't aggregate or interpret them  
* You fill the gap: Understanding what makes judges useful

---

### Track 3: Automated Judge Selection (SECONDARY \- 15% effort)

**Goal**: Develop and validate a systematic pipeline for selecting optimal judge sets.

Experiments:

1. Judge Selection Pipeline  
   * Propose initial set of judges for a given dataset  
   * Train aggregator and identify the least important judges  
   * Propose new judges that complement the most relevant ones  
   * Re-run the pipeline iteratively  
   * Output: Optimized judge set for the dataset  
2. Selection Heuristic Validation  
   * Apply the pipeline across multiple JUDGE-BENCH tasks  
   * Extract generalizable patterns: "For task type X, use judges {Y, Z, …}"  
   * Validate heuristics on held-out tasks  
   * Key question: Can we predict which judges matter for new task types?

Why This Track Matters:

* Provides actionable methodology for building evaluation systems  
* Demonstrates that judge importance patterns are learnable and generalizable  
* Reduces manual effort in designing judge panels

---

### Track 4: Aggregator Validation (SECONDARY \- 15% effort)

**Goal**: Validate that the aggregator can correctly decompose known preference structures.

Experiments:

1. Synthetic Preference Decomposition  
   * Generate ground truth data for the aggregators, i.e. preference data made by aggregating a set of preferences that coincide with the judges  
   * Train aggregator and verify it recovers the known preference decomposition  
   * Key question: Can the aggregator correctly identify the "recipe" used to create preferences?  
2. Persona Internal Preference System  
   * Assign personas explicit sets of preferences (e.g., "Professor values Truthfulness=0.4, Logical Consistency=0.3, Clarity=0.3")  
   * Train aggregator on that single persona's preferences  
   * Extract the persona's own internal preference system  
   * Compare recovered weights to assigned weights

Why This Track Matters:

* Validates that aggregator learning is working correctly on controllable data  
* Provides confidence that judge importance measures are meaningful  
* Demonstrates that method can decompose individual preference profiles

---

### Track 5: Robustness and Bias Analysis (SUPPORTING \- 10% effort)

**Goal**: Comprehensive safety/reliability characterization.

Experiments Already Done:

* Persona contamination (systematic bias, random noise, scale compression) ✓  
* Rubric sensitivity analysis (bias transformations) ✓

Experiments Planned But Not Done:

1. Judge Contamination

2. … [Fernando Avalos](mailto:favalosdev@gmail.com)

Why This Track Matters:

* Supports safety story: "Our interpretable aggregators are also more robust"  
* Complements Track 2: Understanding judges helps detect bias  
* Relatively quick experiments that strengthen overall contribution

---

### Track 6: Aggregator Architecture and Design (OPTIONAL \- 10% effort)

**Goal**: Explore architectural improvements to the base aggregator.

Experiments:

1. Prompt-Aware Aggregation  
   * Extend aggregator input: \[judge\_scores, prompt\_embedding\]  
   * Use sentence transformer to embed prompts  
   * Key question: Does prompt awareness improve performance or interpretability?  
2. Confidence-Weighted Aggregation  
   * Add self-reported confidence scores to judges  
   * Train aggregator that can weight confident judges more  
   * Measure: Does this improve robustness to uncertain judges?

Why This Track Matters:

* Addresses architectural limitation that aggregator only sees scores  
* Similar to dialogue paper's "prompt granularity"  
* Creates richer interpretability: judge importance could be context-dependent  
* **Status**: Optional \- only pursue if time permits and initial results promising

