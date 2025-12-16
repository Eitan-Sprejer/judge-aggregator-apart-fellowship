# Track 3: Automated Judge Selection

**Priority**: SECONDARY (15% effort)

**Goal**: Develop systematic pipeline for selecting optimal judge sets

## Research Questions

1. Can we programmatically identify redundant judges?
2. Can we iteratively propose complementary judges to fill gaps?
3. Do selection heuristics generalize across task types?

## Completed Work

### 3.0 LLM-Driven Judge Decomposition Pipeline ✅
- **Status**: COMPLETED
- **Implementation**: Three-agent LLM orchestration pipeline for recursive judge decomposition
- **Key Components**:
  - **DecompositionAgent**: Analyzes judges and identifies 3-5 orthogonal sub-dimensions
  - **BrainstormAgent**: Authors detailed 5-level rubrics for each dimension
  - **ValidationAgent**: Validates decomposition coverage and minimal overlap
- **Files**:
  - `llm_judge_decomposer.py`: Core recursive decomposition engine
  - `decompose_all_judges.py`: Batch processor for all judges
- **Output**: 55 hierarchical judges (10 parents + 45 children) with parent-child relationships
  - Generated file: `generated_judges/all-judges-decomposed-*.yaml`
  - Format: Matches canonical `judges.yaml` format exactly
  - Parent tracking: Each child judge includes `parent_id` for lineage analysis

#### Quick Start

Generate decomposed judges:
```bash
# Decompose all judges (10 parents → 55 total with children)
python experiments/track3_automated_selection/decompose_all_judges.py \
    --max-depth 1 \
    --output experiments/track3_automated_selection/generated_judges

# Decompose single judge
python experiments/track3_automated_selection/llm_judge_decomposer.py \
    truthfulness-judge \
    --max-depth 1
```

#### Architecture

**DecompositionAgent** → **BrainstormAgent** → **ValidationAgent**
1. Decompose parent judge into orthogonal dimensions
2. Author 5-level rubric for each dimension
3. Validate coverage and overlap

**Generated Judge Format**:
- Score ranges: [0.0, 0.9], [1.0, 1.9], [2.0, 2.9], [3.0, 3.9], [4.0, 4.0] (no gaps)
- All fields match `judges.yaml` canonical format
- Parent judge included in output with `parent_id` on all children

#### Configuration
```bash
# Environment variables
export MARTIAN_API_URL=https://api.withmartian.com/v1
export MARTIAN_API_KEY=<your-api-key>

# CLI options
--max-depth INT           # Maximum recursion depth (default: 1)
--model STR              # Martian model (default: openai/gpt-5-nano)
--temperature FLOAT      # Sampling temperature (default: 0.4)
--max-tokens INT         # Max tokens per completion (default: 2048)
--output PATH            # Output directory (default: generated_judges/)
--judges JUDGE_IDS       # Specific judges (default: all)
--name STR               # Custom output filename prefix
```

### 3.0.1 Judge Hierarchy Visualizer ✅
- **Status**: COMPLETED
- **Implementation**: Interactive HTML graph visualization using Pyvis
- **File**: `visualize_judges.py`

#### Quick Start

```bash
# Basic visualization (generates HTML file)
python experiments/track3_automated_selection/visualize_judges.py \
    experiments/track3_automated_selection/generated_judges/my-judges.yaml

# With text tree output and auto-open in browser
python experiments/track3_automated_selection/visualize_judges.py \
    path/to/judges.yaml --tree --open

# Custom output path and title
python experiments/track3_automated_selection/visualize_judges.py \
    path/to/judges.yaml \
    --output my_visualization.html \
    --title "My Judge Hierarchy"
```

#### Features
- 🎨 **Color-coded nodes** by depth level (red=root, orange=depth 1, teal=depth 2, etc.)
- 🔍 **Hover tooltips** showing judge name, ID, description, and child count
- 🖱️ **Interactive controls** - drag nodes, zoom, pan, keyboard navigation
- 📊 **Legend panel** with depth colors and node shapes (● parent, ◆ leaf)
- 📈 **Statistics panel** showing total/root/leaf judges and max depth
- 🌳 **Text tree output** (`--tree`) for console visualization

#### CLI Options
```bash
--output, -o PATH    # Output HTML file (default: same as input with .html)
--open               # Open visualization in browser after creation
--tree               # Print text tree representation to console
--title STR          # Custom title for the visualization
```

#### Output
The visualizer generates a self-contained HTML file with:
- Hierarchical graph layout (top-down)
- Physics-based node positioning
- Dark theme with high contrast colors
- Embedded legend and statistics

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
