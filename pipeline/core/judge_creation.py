"""
Judge Registry

Manages judge definitions and provides utilities for working with judges.
Judges are now simple rubric prompts - no API creation needed.
"""

import logging
from typing import Dict, List, Optional

from pipeline.utils.judge_rubrics import JUDGE_RUBRICS, JUDGE_DESCRIPTIONS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Judge IDs for easy reference
JUDGE_IDS = list(JUDGE_RUBRICS.keys())


def get_all_judges() -> Dict[str, Dict[str, str]]:
    """
    Get all available judges with their metadata.

    Returns:
        Dictionary mapping judge IDs to metadata (description, rubric)
    """
    judges = {}

    for judge_id in JUDGE_IDS:
        rubric_func = JUDGE_RUBRICS[judge_id]
        rubric = rubric_func()
        description = JUDGE_DESCRIPTIONS[judge_id]

        judges[judge_id] = {
            "id": judge_id,
            "description": description,
            "rubric": rubric
        }
        logger.info(f"✅ Loaded judge {judge_id}")

    logger.info(f"Successfully loaded {len(judges)} judges")
    return judges


def get_judge_ids() -> List[str]:
    """
    Get list of all available judge IDs.

    Returns:
        List of judge identifiers
    """
    return JUDGE_IDS


def get_judge_info(judge_id: str) -> Dict[str, str]:
    """
    Get information about a specific judge.

    Args:
        judge_id: Judge identifier

    Returns:
        Dictionary with judge metadata

    Raises:
        KeyError: If judge_id is not found
    """
    if judge_id not in JUDGE_RUBRICS:
        raise KeyError(f"Judge {judge_id} not found")

    rubric_func = JUDGE_RUBRICS[judge_id]
    rubric = rubric_func()
    description = JUDGE_DESCRIPTIONS[judge_id]

    return {
        "id": judge_id,
        "description": description,
        "rubric": rubric
    }


def main():
    """Main entry point for judge registry."""
    import argparse

    parser = argparse.ArgumentParser(description="Judge registry utilities")
    parser.add_argument('--list', action='store_true', help='List all judge IDs')
    parser.add_argument('--info', help='Get detailed info about a specific judge')
    parser.add_argument('--all', action='store_true', help='Get all judges with metadata')

    args = parser.parse_args()

    if args.list:
        print("\nAvailable judges:")
        for i, judge_id in enumerate(JUDGE_IDS, 1):
            print(f"  {i:2}. {judge_id}")
            print(f"      {JUDGE_DESCRIPTIONS[judge_id]}")
    elif args.info:
        try:
            info = get_judge_info(args.info)
            print(f"\nJudge: {info['id']}")
            print(f"Description: {info['description']}")
            print(f"\nRubric Preview (first 500 chars):")
            print(info['rubric'][:500] + "...")
        except KeyError as e:
            print(f"Error: {e}")
    elif args.all:
        judges = get_all_judges()
        print(f"\nLoaded {len(judges)} judges:")
        for judge_id, info in judges.items():
            print(f"  - {judge_id}: {info['description']}")
    else:
        print("Use --list, --info <judge_id>, or --all")
        print("Example: python judge_creation.py --list")


if __name__ == "__main__":
    main()