"""Judge Creation Orchestrator for dataset-specific judge generation.

This module coordinates the creation of dataset-specific judges:
1. Create parent judges from dimension descriptions (using ParentJudgeCreatorAgent)
2. Decompose parent judges into children (using Track 3.0 decomposition pipeline)
3. Save to dataset-specific YAML files
4. Handle caching and loading of previously generated judges

Usage:
    from pipeline.core.judge_creation_orchestrator import JudgeCreationOrchestrator

    orchestrator = JudgeCreationOrchestrator(
        dataset_name="helpsteer2",
        model="openai/gpt-4.1-nano",
        temperature=0.4
    )

    # Create judges for dimensions with auto-caching
    judges = orchestrator.create_judges_for_dataset(
        dimensions=[
            {"name": "helpfulness", "description": "..."},
            {"name": "correctness", "description": "..."}
        ],
        cache_strategy="auto",
        decomposition_depth=1
    )
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

# Add repository root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Add Track 3 to path for agent imports
TRACK3_DIR = REPO_ROOT / "experiments" / "track3_automated_selection"
if str(TRACK3_DIR) not in sys.path:
    sys.path.append(str(TRACK3_DIR))

from experiments.track3_automated_selection.llm_judge_decomposer import (
    ChatCompletionClient,
    DecompositionAgent,
    BrainstormAgent,
    ValidationAgent,
    ParentJudgeCreatorAgent,
    LLMConfig,
    _merge_dimension_to_rubric,
    InlineListDumper,
)

load_dotenv()


class JudgeCreationOrchestrator:
    """Orchestrates creation of dataset-specific judges with caching support."""

    def __init__(
        self,
        dataset_name: str,
        model: str = "openai/gpt-5.1",
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ):
        """Initialize the orchestrator.

        Args:
            dataset_name: Name of the dataset (e.g., 'helpsteer2')
            model: Martian model name for LLM calls
            temperature: Sampling temperature
            max_tokens: Maximum tokens per completion
        """
        self.dataset_name = dataset_name
        self.dataset_judges_dir = REPO_ROOT / "judges" / dataset_name
        # Create dataset subdirectory
        self.dataset_judges_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LLM client
        self.llm_config = LLMConfig(
            model=model, temperature=temperature, max_tokens=max_tokens
        )
        self.client = ChatCompletionClient(self.llm_config)

        # Initialize agents
        self.parent_creator = ParentJudgeCreatorAgent(self.client)
        self.decomposition_agent = DecompositionAgent(self.client)
        self.brainstorm_agent = BrainstormAgent(self.client)
        self.validation_agent = ValidationAgent(self.client)

    def create_judges_for_dataset(
        self,
        dimensions: List[Dict[str, str]],
        cache_strategy: str = "auto",
        decomposition_depth: int = 1,
        sample_qa_pairs: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Create judges for dataset dimensions with caching support.

        Args:
            dimensions: List of dimension dicts with 'name' and 'description' keys
            cache_strategy: Cache behavior - 'auto', 'force_create', or 'load_only'
            decomposition_depth: Depth of recursive decomposition (0 = no decomposition)
            sample_qa_pairs: Optional sample Q&A pairs to inform judge creation

        Returns:
            List of judge definitions (parents + children if decomposed)

        Raises:
            RuntimeError: If cache_strategy='load_only' and cache doesn't exist
            ValueError: If invalid cache_strategy provided
        """
        if cache_strategy not in ["auto", "force_create", "load_only"]:
            raise ValueError(
                f"Invalid cache_strategy: {cache_strategy}. "
                f"Must be 'auto', 'force_create', or 'load_only'"
            )

        # Check cache
        cache_exists = self.dataset_judges_dir.exists()

        if cache_strategy == "load_only":
            if not cache_exists:
                raise RuntimeError(
                    f"Cache strategy is 'load_only' but no cache found at {self.dataset_judges_dir}"
                )
            return self._load_from_cache()

        # For "auto" strategy, check which dimensions are cached
        requested_dimensions = {dim["name"] for dim in dimensions}
        cached_dimensions = self._get_cached_dimensions() if cache_exists else set()
        missing_dimensions = requested_dimensions - cached_dimensions

        if cache_strategy == "auto" and not missing_dimensions:
            # All requested dimensions are cached
            print(f"All {len(requested_dimensions)} dimensions found in cache: {self.dataset_judges_dir}")
            return self._load_from_cache()

        # Need to create some or all judges
        if missing_dimensions:
            print(f"Creating judges for {len(missing_dimensions)} missing dimensions: {missing_dimensions}")
            if cached_dimensions:
                print(f"  (Will preserve {len(cached_dimensions)} cached dimensions: {cached_dimensions})")
        else:
            print(f"Creating judges for {self.dataset_name} with {len(dimensions)} dimensions...")

        # Load existing judges if any
        all_judges = []
        if cache_exists and cached_dimensions:
            try:
                all_judges = self._load_from_cache()
                print(f"  Loaded {len(all_judges)} existing judges from cache")
            except Exception as e:
                print(f"Warning: Failed to load existing cache, will recreate all: {e}")
                all_judges = []

        # Create judges for missing dimensions only
        dimensions_to_create = [d for d in dimensions if d["name"] in missing_dimensions]

        for dimension in dimensions_to_create:
            dim_name = dimension["name"]
            dim_description = dimension["description"]
            print(f"  Creating parent judge for '{dim_name}'...")

            # Step 1: Create parent judge
            parent_judge = self._create_parent_judge(
                dimension_name=dim_name,
                dimension_description=dim_description,
                sample_qa_pairs=sample_qa_pairs,
            )
            # Add explicit dimension field
            parent_judge['dimension'] = dim_name
            all_judges.append(parent_judge)

            # Step 2: Decompose parent if requested
            if decomposition_depth > 0:
                print(f"    Decomposing '{dim_name}' to depth {decomposition_depth}...")
                child_judges = self._decompose_parent(
                    parent_judge=parent_judge, max_depth=decomposition_depth
                )
                # Add explicit dimension field to all children
                for child in child_judges:
                    child['dimension'] = dim_name
                all_judges.extend(child_judges)
                print(f"    Generated {len(child_judges)} child judges for '{dim_name}'")

        # Step 3: Save to YAML (overwrites with complete set)
        print(f"Saving {len(all_judges)} judges to {self.dataset_judges_dir}")
        self._save_to_yaml(all_judges)

        return all_judges

    def _create_parent_judge(
        self,
        dimension_name: str,
        dimension_description: str,
        sample_qa_pairs: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Create a parent judge for a dimension using ParentJudgeCreatorAgent.

        Args:
            dimension_name: Name of the dimension
            dimension_description: Description of what the dimension measures
            sample_qa_pairs: Optional sample Q&A pairs

        Returns:
            Parent judge definition
        """
        try:
            parent_judge = self.parent_creator.create_parent_judge(
                dimension_name=dimension_name,
                dimension_description=dimension_description,
                dataset_name=self.dataset_name,
                sample_qa_pairs=sample_qa_pairs,
            )
            return parent_judge
        except Exception as e:
            raise RuntimeError(
                f"Failed to create parent judge for dimension '{dimension_name}': {e}"
            ) from e

    def _decompose_parent(
        self, parent_judge: Dict[str, Any], max_depth: int
    ) -> List[Dict[str, Any]]:
        """Decompose a parent judge into children using Track 3.0 pipeline.

        Args:
            parent_judge: Parent judge definition
            max_depth: Maximum decomposition depth

        Returns:
            List of child judge definitions
        """
        parent_id = parent_judge["id"]
        child_judges = []

        if max_depth < 1:
            return child_judges

        try:
            # Step 1: Identify dimensions
            dimensions = self.decomposition_agent.decompose(parent_judge)

            # Step 2: Author rubrics for each dimension
            child_rubrics = []
            for dimension in dimensions:
                try:
                    rubric = self.brainstorm_agent.author(parent_judge, dimension)
                    child_rubrics.append((dimension, rubric))
                except Exception as e:
                    print(
                        f"Warning: Failed to author rubric for {dimension.get('dimension_id')}: {e}"
                    )
                    continue

            if not child_rubrics:
                print(f"Warning: No child rubrics generated for {parent_id}.")
                return child_judges

            # Step 3: Validate decomposition
            rubrics_only = [r for _, r in child_rubrics]
            validation = self.validation_agent.validate(parent_judge, rubrics_only)

            if not validation.get("is_valid", False):
                print(
                    f"Warning: Decomposition validation failed for {parent_id}. "
                    f"Gaps: {validation.get('gaps', [])}"
                )

            # Step 4: Merge dimensions into judge definitions
            for dimension, child_rubric in child_rubrics:
                child_judge = _merge_dimension_to_rubric(
                    parent_id, dimension, child_rubric
                )
                # Add dataset metadata
                child_judge["dataset"] = self.dataset_name
                child_judge["auto_generated"] = True
                child_judges.append(child_judge)

            # Note: For simplicity, we only decompose to depth 1
            # Recursive decomposition can be added later if needed

        except Exception as e:
            print(f"Warning: Decomposition failed for {parent_id}: {e}")

        return child_judges

    def _compute_judge_depth(self, judge: Dict[str, Any], all_judges: List[Dict[str, Any]]) -> int:
        """Compute depth of a judge in the hierarchy.

        Args:
            judge: Judge definition
            all_judges: All judges (for looking up parents)

        Returns:
            Depth (0 = parent, 1 = child, 2 = grandchild, etc.)
        """
        if judge.get('parent_id') is None:
            return 0

        # Find parent and recurse
        parent_id = judge['parent_id']
        parent = next((j for j in all_judges if j['id'] == parent_id), None)

        if parent is None:
            # Parent not found - treat as depth 0 (shouldn't happen)
            return 0

        return 1 + self._compute_judge_depth(parent, all_judges)

    def _save_to_yaml(self, judges: List[Dict[str, Any]]) -> None:
        """Save judges to depth-specific YAML files.

        Args:
            judges: List of judge definitions to save
        """
        # Group judges by depth
        judges_by_depth: Dict[int, List[Dict[str, Any]]] = {}

        for judge in judges:
            depth = self._compute_judge_depth(judge, judges)
            if depth not in judges_by_depth:
                judges_by_depth[depth] = []
            judges_by_depth[depth].append(judge)

        # Save each depth to separate file
        depth_names = {
            0: "parents",
            1: "children",
            2: "grandchildren",
            3: "great_grandchildren"
        }

        saved_files = []
        for depth, depth_judges in sorted(judges_by_depth.items()):
            depth_name = depth_names.get(depth, f"depth_{depth}")
            filename = f"depth_{depth}_{depth_name}.yaml"
            filepath = self.dataset_judges_dir / filename

            # Wrap in judges key to match judges.yaml format
            payload = {"judges": depth_judges}

            # Save with inline list formatting for ranges
            with filepath.open("w", encoding="utf-8") as f:
                yaml.dump(payload, f, Dumper=InlineListDumper, sort_keys=False, allow_unicode=False)

            print(f"  Saved {len(depth_judges)} depth-{depth} judges to {filepath.name}")
            saved_files.append(filepath)

        print(f"\n✅ Saved {len(judges)} total judges across {len(saved_files)} files in {self.dataset_judges_dir}")

    def _get_cached_dimensions(self) -> set[str]:
        """Get set of dimensions that have cached judges.

        Returns:
            Set of dimension names found in cached judges

        Raises:
            RuntimeError: If cache files don't exist or are invalid
        """
        # Check if cache exists (look for depth_0_parents.yaml as indicator)
        depth_0_file = self.dataset_judges_dir / "depth_0_parents.yaml"

        if not depth_0_file.exists():
            return set()

        # Load all depth files to extract dimensions
        cached_dimensions = set()
        depth_files = sorted(self.dataset_judges_dir.glob("depth_*.yaml"))

        for depth_file in depth_files:
            try:
                with depth_file.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if not isinstance(data, dict) or "judges" not in data:
                    continue

                judges = data["judges"]
                if not isinstance(judges, list):
                    continue

                # Extract dimensions from judge IDs
                # Format: {dataset}-{dimension}-judge[-subdimension]
                for judge in judges:
                    judge_id = judge.get("id", "")
                    parts = judge_id.split("-")
                    if len(parts) >= 3:
                        # Extract dimension (second part)
                        dimension = parts[1]
                        cached_dimensions.add(dimension)

            except Exception as e:
                print(f"Warning: Failed to parse dimensions from {depth_file}: {e}")
                continue

        return cached_dimensions

    def _load_from_cache(self) -> List[Dict[str, Any]]:
        """Load judges from cached depth-specific YAML files.

        Returns:
            List of all judge definitions (merged from all depth files)

        Raises:
            RuntimeError: If cache files don't exist or are invalid
        """
        # Check if cache exists (look for depth_0_parents.yaml as indicator)
        depth_0_file = self.dataset_judges_dir / "depth_0_parents.yaml"

        if not depth_0_file.exists():
            raise RuntimeError(
                f"Cache not found: {depth_0_file} does not exist. "
                f"Run with cache_strategy='force_create' to generate judges."
            )

        # Load all depth files
        all_judges = []
        depth_files = sorted(self.dataset_judges_dir.glob("depth_*.yaml"))

        if not depth_files:
            raise RuntimeError(f"No depth files found in {self.dataset_judges_dir}")

        for depth_file in depth_files:
            try:
                with depth_file.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if not isinstance(data, dict) or "judges" not in data:
                    raise ValueError(f"Invalid YAML format in {depth_file} - expected 'judges' key")

                judges = data["judges"]
                if not isinstance(judges, list):
                    raise ValueError(f"Invalid YAML format in {depth_file} - 'judges' must be a list")

                all_judges.extend(judges)
                print(f"  Loaded {len(judges)} judges from {depth_file.name}")

            except Exception as e:
                raise RuntimeError(f"Failed to load judges from {depth_file}: {e}") from e

        print(f"\n✅ Loaded {len(all_judges)} total judges from {len(depth_files)} cache files")
        return all_judges


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create dataset-specific judges with decomposition"
    )
    parser.add_argument("dataset_name", help="Dataset name (e.g., 'helpsteer2')")
    parser.add_argument(
        "--dimension",
        action="append",
        required=True,
        help="Dimension in format 'name:description' (can be used multiple times)",
    )
    parser.add_argument(
        "--cache-strategy",
        choices=["auto", "force_create", "load_only"],
        default="auto",
        help="Cache strategy (default: auto)",
    )
    parser.add_argument(
        "--decomposition-depth",
        type=int,
        default=1,
        help="Decomposition depth (default: 1)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4.1-nano",
        help="Martian model name (default: openai/gpt-4.1-nano)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4, help="Sampling temperature"
    )

    args = parser.parse_args()

    # Parse dimensions
    dimensions = []
    for dim_str in args.dimension:
        if ":" not in dim_str:
            print(f"Error: Dimension must be in format 'name:description', got: {dim_str}")
            sys.exit(1)
        name, description = dim_str.split(":", 1)
        dimensions.append({"name": name.strip(), "description": description.strip()})

    # Create orchestrator
    orchestrator = JudgeCreationOrchestrator(
        dataset_name=args.dataset_name,
        model=args.model,
        temperature=args.temperature,
    )

    # Create judges
    try:
        judges = orchestrator.create_judges_for_dataset(
            dimensions=dimensions,
            cache_strategy=args.cache_strategy,
            decomposition_depth=args.decomposition_depth,
        )
        print(f"\nSuccess! Created {len(judges)} judges for {args.dataset_name}")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
