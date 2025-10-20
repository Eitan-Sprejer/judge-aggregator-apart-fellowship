# Track 3: Automated Judge Selection

**Priority**: SECONDARY (15% effort)

**Goal**: Develop systematic pipeline for selecting optimal judge sets

## Research Questions

1. Can we programmatically identify redundant judges?
2. Can we iteratively propose complementary judges to fill gaps?
3. Do selection heuristics generalize across task types?

## Planned Experiments

### 3.1 Iterative Judge Selection Pipeline
- **Status**: Planned
- **Pipeline**:
  1. Start with current 10 judges
  2. Train aggregator, analyze importance (from Track 2.1)
  3. Identify least important judge
  4. Propose complementary judge to fill gaps
  5. Evaluate new judge set
  6. Repeat
- **Directory**: `3.1_selection_pipeline/`
- **Output**: Optimized judge sets for different objectives

### 3.2 Generalizable Selection Heuristics
- **Status**: Planned
- **Dataset**: JUDGE-BENCH tasks (from Track 1.3)
- **Analysis**: Extract selection rules that work across tasks
- **Directory**: `3.2_generalizable_heuristics/`
- **Key Question**: What makes a good judge set? Diversity? Coverage? Orthogonality?

## Key Contribution

Actionable methodology for building evaluation systems:
- "Start with these 5 core judges"
- "Add domain-specific judges based on task type"
- "Remove redundant judges that correlate >0.9"

## Methodology

**Judge Redundancy Analysis**:
- Correlation matrix of judge scores
- Identify highly correlated judges (r > 0.8)
- Test if removing one degrades performance

**Gap Identification**:
- Analyze disagreement patterns between aggregator predictions and ground truth
- Identify systematic errors (e.g., "overvalues verbosity")
- Propose judge dimension to address gap

**Judge Proposal**:
- Use YAML-based judge system (from refactoring)
- Generate candidate judge rubrics programmatically
- Test multiple variations (A/B testing)

## Expected Outcomes

- Judge selection algorithm with stopping criteria
- Heuristics: "5-7 judges optimal", "diminishing returns after 8"
- Template judge sets for common scenarios:
  - Safety-critical applications: [Truthfulness, Harmlessness, Honesty]
  - Creative writing: [Creativity, Clarity, Engagement]
  - Technical QA: [Truthfulness, Explanatory Depth, Logical Consistency]

## Dependencies

- Requires Track 2 results (judge importance rankings)
- Benefits from flexible judge system (YAML refactoring completed)
- Uses datasets from Track 1

## Technical Notes

**YAML-Based Judge System** (completed in refactoring):
- `pipeline/utils/judges.yaml` - All judge definitions
- `pipeline/utils/judge_prompt_template.txt` - Prompt template
- Easy to add/modify judges programmatically
- Version control of judge definitions
