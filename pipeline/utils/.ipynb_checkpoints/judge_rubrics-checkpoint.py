"""Unified judge rubrics for the multi-judge interpretability framework.

This module loads judge definitions from judges.yaml and generates rubric prompts
using a template system. Each judge evaluates a specific dimension on a 0.0-4.0 scale.

The YAML-based approach supports:
- Easy addition/modification of judges without code changes
- A/B testing of rubric variations
- Programmatic judge filtering and selection
- Version control of judge definitions
"""

import yaml
from pathlib import Path
from typing import Dict, List, Callable


def _load_judges_yaml() -> Dict:
    """Load judge definitions from YAML file."""
    yaml_path = Path(__file__).parent / "judges.yaml"
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def _load_prompt_template() -> str:
    """Load the judge prompt template."""
    template_path = Path(__file__).parent / "judge_prompt_template.txt"
    with open(template_path, 'r') as f:
        return f.read()


def _format_criteria(criteria: List[Dict]) -> str:
    """Format criteria section from YAML structure."""
    parts = []
    for i, criterion in enumerate(criteria):
        range_start, range_end = criterion['range']
        label = criterion['label']
        indicators = criterion['indicators']

        # Format range
        range_str = f"{range_start}" if range_start == range_end else f"{range_start}–{range_end}"
        parts.append(f"{range_str} = {label}")

        # Add indicators
        for indicator in indicators:
            parts.append(f"• {indicator}")

        # Blank line between tiers (except last)
        if i < len(criteria) - 1:
            parts.append("")

    return "\n".join(parts)


def _format_guidelines(guidelines: List[str]) -> str:
    """Format guidelines section from YAML structure."""
    return "\n".join(f"• {guideline}" for guideline in guidelines)


def _generate_rubric_prompt(judge: Dict) -> str:
    """Generate a rubric prompt from a judge definition.

    Args:
        judge: Dictionary containing judge metadata and rubric components

    Returns:
        Formatted rubric prompt string
    """
    template = _load_prompt_template()

    # Fill in template
    prompt = template.format(
        NAME=judge['name'].upper(),
        SCORING_DESCRIPTION=judge['scoring_description'],
        DEFINITION=judge['definition'].strip(),
        CRITERIA=_format_criteria(judge['criteria']),
        GUIDELINES=_format_guidelines(judge['guidelines'])
    )

    return prompt


# Load judges from YAML
_JUDGES_DATA = _load_judges_yaml()
_JUDGES_BY_ID = {judge['id']: judge for judge in _JUDGES_DATA['judges']}


def _create_rubric_function(judge: Dict) -> Callable[[], str]:
    """Create a rubric function for a judge that returns the generated prompt.

    Args:
        judge: Dictionary containing judge definition

    Returns:
        Function that returns the rubric prompt string
    """
    # Pre-generate the prompt (cached)
    prompt = _generate_rubric_prompt(judge)

    def rubric_function() -> str:
        return prompt

    return rubric_function


# Generate rubric functions for all judges
JUDGE_RUBRICS = {
    judge_id: _create_rubric_function(judge)
    for judge_id, judge in _JUDGES_BY_ID.items()
}


# Extract descriptions for all judges
JUDGE_DESCRIPTIONS = {
    judge['id']: judge['description']
    for judge in _JUDGES_DATA['judges']
}


# Utility functions for working with judges

def get_judge_info(judge_id: str) -> Dict:
    """Get complete information about a judge.

    Args:
        judge_id: The judge identifier (e.g., 'truthfulness-judge')

    Returns:
        Dictionary with judge metadata and rubric components

    Raises:
        KeyError: If judge_id is not found
    """
    return _JUDGES_BY_ID[judge_id]


def list_available_judges() -> List[str]:
    """Get list of all available judge IDs.

    Returns:
        List of judge identifier strings
    """
    return list(_JUDGES_BY_ID.keys())


def get_judges_by_criteria(
    min_version: str = None,
    tags: List[str] = None,
    exclude_ids: List[str] = None
) -> List[str]:
    """Filter judges based on criteria.

    Args:
        min_version: Minimum version required (e.g., "1.0")
        tags: List of tags judges must have (future feature)
        exclude_ids: List of judge IDs to exclude

    Returns:
        List of judge IDs matching criteria
    """
    judges = []
    exclude_ids = exclude_ids or []

    for judge_id, judge in _JUDGES_BY_ID.items():
        # Skip excluded judges
        if judge_id in exclude_ids:
            continue

        # Check version if specified
        if min_version and judge['version'] < min_version:
            continue

        # Check tags if specified (future feature)
        # if tags and not any(tag in judge.get('tags', []) for tag in tags):
        #     continue

        judges.append(judge_id)

    return judges


# Dataset-specific judge loading support

def load_judges_from_yaml(yaml_path: Path) -> Dict[str, Dict]:
    """Load judges from an arbitrary YAML file.

    Args:
        yaml_path: Path to YAML file with judge definitions

    Returns:
        Dictionary mapping judge IDs to judge definitions

    Raises:
        FileNotFoundError: If yaml_path doesn't exist
        ValueError: If YAML format is invalid
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Judge YAML file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or 'judges' not in data:
        raise ValueError(f"Invalid YAML format in {yaml_path} - expected 'judges' key")

    judges = data['judges']
    if not isinstance(judges, list):
        raise ValueError(f"Invalid YAML format in {yaml_path} - 'judges' must be a list")

    return {judge['id']: judge for judge in judges}


def merge_judge_sources(dataset_name: str = None) -> Dict[str, Dict]:
    """Merge generic judges with dataset-specific judges.

    Args:
        dataset_name: Optional dataset name (e.g., 'helpsteer2'). If provided,
                     loads and merges dataset-specific judges from
                     dataset_judges/{dataset_name}_judges.yaml

    Returns:
        Dictionary mapping judge IDs to judge definitions
        Dataset-specific judges override generic judges with same ID
    """
    # Start with generic judges
    merged = dict(_JUDGES_BY_ID)

    # Load dataset-specific judges if specified
    if dataset_name:
        dataset_judges_path = Path(__file__).parent / "dataset_judges" / f"{dataset_name}_judges.yaml"
        if dataset_judges_path.exists():
            dataset_judges = load_judges_from_yaml(dataset_judges_path)
            # Merge (dataset-specific overrides generic)
            merged.update(dataset_judges)

    return merged


def get_judge_info_with_dataset(judge_id: str, dataset_name: str = None) -> Dict:
    """Get judge information, searching both generic and dataset-specific judges.

    Args:
        judge_id: The judge identifier
        dataset_name: Optional dataset name to search dataset-specific judges

    Returns:
        Dictionary with judge metadata and rubric components

    Raises:
        KeyError: If judge_id is not found in either source
    """
    merged_judges = merge_judge_sources(dataset_name)

    if judge_id not in merged_judges:
        available = list(merged_judges.keys())
        raise KeyError(
            f"Judge '{judge_id}' not found. Available judges: {available}"
        )

    return merged_judges[judge_id]


def create_judge_rubrics_for_dataset(dataset_name: str) -> Dict[str, Callable[[], str]]:
    """Create rubric functions for all judges (generic + dataset-specific).

    Args:
        dataset_name: Dataset name (e.g., 'helpsteer2')

    Returns:
        Dictionary mapping judge IDs to rubric functions
    """
    merged_judges = merge_judge_sources(dataset_name)

    return {
        judge_id: _create_rubric_function(judge)
        for judge_id, judge in merged_judges.items()
    }


def list_dataset_judges(dataset_name: str) -> List[str]:
    """List all judges available for a specific dataset.

    Args:
        dataset_name: Dataset name (e.g., 'helpsteer2')

    Returns:
        List of judge IDs (generic + dataset-specific)
    """
    merged_judges = merge_judge_sources(dataset_name)
    return list(merged_judges.keys())


def get_judges_by_dataset_tag(dataset_name: str) -> List[str]:
    """Get judges specifically tagged for a dataset.

    Args:
        dataset_name: Dataset name (e.g., 'helpsteer2')

    Returns:
        List of judge IDs with matching 'dataset' metadata field
    """
    merged_judges = merge_judge_sources(dataset_name)
    return [
        judge_id
        for judge_id, judge in merged_judges.items()
        if judge.get('dataset') == dataset_name
    ]


def load_judges_from_files(file_paths: List[str]) -> Dict[str, Callable[[], str]]:
    """Load judges from one or more YAML files and create rubric functions.

    Args:
        file_paths: List of paths to YAML files with judge definitions
                   (relative to repo root or absolute paths)

    Returns:
        Dictionary mapping judge IDs to rubric functions

    Raises:
        FileNotFoundError: If any file path doesn't exist
        ValueError: If YAML format is invalid
    """
    from pathlib import Path

    # Get repo root for resolving relative paths
    repo_root = Path(__file__).resolve().parents[2]

    all_judges = {}

    for file_path_str in file_paths:
        file_path = Path(file_path_str)

        # Resolve relative paths from repo root
        if not file_path.is_absolute():
            file_path = repo_root / file_path

        if not file_path.exists():
            raise FileNotFoundError(f"Judge file not found: {file_path}")

        # Load judges from this file
        judges_dict = load_judges_from_yaml(file_path)

        # Merge into all_judges (later files override earlier ones with same ID)
        all_judges.update(judges_dict)

    # Create rubric functions for all loaded judges
    rubric_functions = {
        judge_id: _create_rubric_function(judge)
        for judge_id, judge in all_judges.items()
    }

    return rubric_functions


def get_judge_ids_from_files(file_paths: List[str]) -> List[str]:
    """Get list of judge IDs from YAML files without loading full rubrics.

    Args:
        file_paths: List of paths to YAML files

    Returns:
        List of judge IDs
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    all_judge_ids = []

    for file_path_str in file_paths:
        file_path = Path(file_path_str)

        if not file_path.is_absolute():
            file_path = repo_root / file_path

        if not file_path.exists():
            raise FileNotFoundError(f"Judge file not found: {file_path}")

        judges_dict = load_judges_from_yaml(file_path)
        all_judge_ids.extend(judges_dict.keys())

    return all_judge_ids
