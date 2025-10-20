# Track 2: Judge Interpretability

**Priority**: PRIMARY (40% effort - main differentiator)

**Goal**: Understand which judges matter and why

## Research Questions

1. Which judges contribute most to predicting human preferences?
2. Does judge importance vary across task types (summarization vs. QA vs. creative writing)?
3. Do different human personas weight judges differently?

## Planned Experiments

### 2.1 Judge Importance Analysis
- **Status**: Planned
- **Methods**:
  - GAM feature importance analysis
  - MLP ablation studies (remove judges, measure performance drop)
  - Shapley value estimation for judge contributions
- **Outputs**: Judge ranking by importance, contribution scores
- **Directory**: `2.1_importance_analysis/`

### 2.2 Cross-Task Judge Importance Heatmap
- **Status**: Planned
- **Dataset**: JUDGE-BENCH (subset of 20 tasks)
- **Analysis**: Judge importance matrix across task types
- **Visualization**: Heatmap showing which judges matter for which tasks
- **Directory**: `2.2_cross_task_heatmap/`
- **Key Question**: Do safety judges matter more for medical QA? Creativity for story generation?

### 2.3 Persona-Based Judge Importance
- **Status**: Planned
- **Dataset**: UltraFeedback with 8 personas (from Track 1.1)
- **Analysis**: Per-persona judge importance
- **Directory**: `2.3_persona_variation/`
- **Key Question**: Does the "Professor" persona weight logical consistency more than "Child"?

## Key Contribution

**Main differentiator from related work**:
- MAJ-Eval does multi-agent debates but doesn't analyze which dimensions matter
- JUDGE-BENCH evaluates judges but doesn't aggregate or analyze importance
- This fills the interpretability gap

## Methodology

**GAM Analysis**:
- Directly extract feature importance from trained GAMs
- Partial dependence plots show judge-score relationships
- Natural interpretability

**MLP Ablation**:
- Train with all judges → baseline performance
- Remove each judge → measure ΔR²
- Larger drop = more important judge

**Cross-Task Analysis**:
- Train separate models per task
- Compare judge rankings across tasks
- Identify task-specific vs. universal judges

## Expected Outcomes

- Judge importance rankings (universal and task-specific)
- Insights: "Truthfulness + Helpfulness explain 80% of variance"
- Actionable guidance: Which judges to prioritize for new domains

## Dependencies

- Requires Track 1.1 (persona data) and Track 1.3 (JUDGE-BENCH data)
- Informs Track 3 (automated selection uses importance rankings)
