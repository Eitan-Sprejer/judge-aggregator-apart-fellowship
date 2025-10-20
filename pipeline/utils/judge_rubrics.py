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
