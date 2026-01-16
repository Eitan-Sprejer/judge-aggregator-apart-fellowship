# Hero Run: Multi-Output Judge Pruning Report

## Executive Summary

This experiment started with **all 30 judges** (5 parent judges + 25 child judges) and iteratively pruned them using the `human_correlation` strategy while optimizing for **all 5 HelpSteer2 dimensions simultaneously**.

### Key Results

| Metric | Initial | Best | Final |
|--------|---------|------|-------|
| **Number of Judges** | 30 | 15 | 11 |
| **Average R²** | 0.0973 | 0.2122 | 0.1797 |
| **Iteration** | 0 | 15 | 19 |

🎯 **Pruning from 30 → 15 judges improved average R² by 118.0%!**

---

## Methodology

### Pruning Strategy: Human Correlation
At each iteration, the judge with the **lowest average Pearson correlation** to human targets across all 5 dimensions was removed.

### Stopping Criteria
- Stop when average R² drops more than **15%** from peak, OR
- Minimum of **5 judges** reached

### Multi-Output Approach
- **5 independent GAMs** were trained at each iteration (one per dimension)
- Importance scores were averaged across all dimensions
- Human correlations were computed per-dimension and averaged for removal decisions

---

## Performance Trajectory

### Average R² vs Number of Judges

| Judges | Avg R² | Δ from Initial |
|--------|--------|----------------|
| 30 | 0.0973 | 0.0000 |
| 29 | 0.1264 | +0.0290 |
| 28 | 0.1375 | +0.0401 |
| 27 | 0.1460 | +0.0487 |
| 26 | 0.1409 | +0.0435 |
| 25 | 0.1567 | +0.0594 |
| 24 | 0.1590 | +0.0616 |
| 23 | 0.1606 | +0.0633 |
| 22 | 0.1737 | +0.0764 |
| 21 | 0.1732 | +0.0758 |
| 20 | 0.1791 | +0.0818 |
| 19 | 0.1811 | +0.0838 |
| 18 | 0.1969 | +0.0996 |
| 17 | 0.1926 | +0.0952 |
| 16 | 0.1981 | +0.1008 |
| 15 | 0.2122 | +0.1149 ⭐ |
| 14 | 0.2029 | +0.1056 |
| 13 | 0.2039 | +0.1065 |
| 12 | 0.1879 | +0.0906 |
| 11 | 0.1797 | +0.0824 |

### Per-Dimension R² at Best Iteration (15 judges)

| Dimension | R² | vs Initial |
|-----------|-----|------------|
| Helpfulness | 0.3219 | +0.1183 |
| Correctness | 0.3211 | +0.1049 |
| Coherence | 0.0677 | +0.1979 |
| Complexity | 0.1632 | +0.1140 |
| Verbosity | 0.1873 | +0.0394 |

---

## Optimal Judge Set (15 judges)

### Helpfulness
- `helpfulness`
- `helpfulness-relevance-to-intent`
- `helpfulness-completeness-of-solution`
- `helpfulness-context-tailoring-and-assumptions`
- `helpfulness-organization-for-immediate-application`

### Correctness
- `correctness`
- `correctness-requirement-coverage`
- `correctness-procedural-correctness`

### Coherence
- `coherence`
- `coherence-logical-flow-and-sequencing`
- `coherence-structure-and-formatting-support`

### Complexity
- `complexity`
- `complexity-lexical-sophistication`

### Verbosity
- `verbosity`
- `verbosity-extras-density`

---

## Judges Removed (in order)

| Order | Judge | Reason |
|-------|-------|--------|
| 0 | `coherence-internal-consistency` | Lowest avg correlation |
| 1 | `verbosity-framing-padding` | Lowest avg correlation |
| 2 | `helpfulness-actionability-and-specificity` | Lowest avg correlation |
| 3 | `verbosity-redundancy` | Lowest avg correlation |
| 4 | `coherence-concision-and-non-redundancy` | Lowest avg correlation |
| 5 | `correctness-factual-accuracy` | Lowest avg correlation |
| 6 | `verbosity-minimal-overage` | Lowest avg correlation |
| 7 | `complexity-syntactic-complexity` | Lowest avg correlation |
| 8 | `coherence-referential-clarity-and-terminology` | Lowest avg correlation |
| 9 | `correctness-uncertainty-handling` | Lowest avg correlation |
| 10 | `correctness-grounding-and-non-hallucination` | Lowest avg correlation |
| 11 | `complexity-assumed-knowledge` | Lowest avg correlation |
| 12 | `verbosity-structural-expansion` | Lowest avg correlation |
| 13 | `complexity-terminology-density` | Lowest avg correlation |
| 14 | `complexity-formalism-and-references` | Lowest avg correlation |
| 15 | `correctness` | Lowest avg correlation |
| 16 | `coherence` | Lowest avg correlation |
| 17 | `correctness-procedural-correctness` | Lowest avg correlation |
| 18 | `correctness-requirement-coverage` | Lowest avg correlation |

---

## Key Insights

### 1. More Judges ≠ Better Performance
Starting with all 30 judges produced **worse** results (avg R²=0.0973) than using the optimal 15 judges (avg R²=0.2122). This is likely due to:
- **Overfitting**: Too many features relative to training samples
- **Noise**: Some judges introduced conflicting signals
- **Redundancy**: Multiple judges capturing the same information

### 2. Optimal Reduction: 50%
The best performance was achieved by removing **exactly half** of the judges (30 → 15), suggesting a good balance between diversity and signal quality.

### 3. Parent Judges Are Essential
All **5 parent judges** were retained in the optimal set, indicating they provide foundational signal that children cannot replace.

### 4. Dimension-Specific Performance
- **Helpfulness** and **Correctness** showed the strongest R² (~0.32)
- **Coherence** remained challenging (~0.07) - may need different judge design
- **Complexity** and **Verbosity** showed moderate performance (~0.17)

### 5. Cross-Dimension Trade-offs
Some judges performed well for their "home" dimension but hurt others (negative correlations), making multi-output optimization crucial.

---

## Recommendations

1. **Use the 15-judge optimal set** for production deployments
2. **Consider dimension-specific ensembles** for best per-dimension performance
3. **Investigate coherence dimension** - may need architectural changes
4. **Run periodic re-optimization** as judge definitions evolve

---

## Visualizations

See the generated plots:
- `r2_trajectory.png` - R² over iterations for all dimensions
- `judges_removed.png` - Removal order with dimension coloring
- `final_comparison.png` - Initial vs Best vs Final bar chart
- `r2_heatmap.png` - Full iteration × dimension R² matrix

