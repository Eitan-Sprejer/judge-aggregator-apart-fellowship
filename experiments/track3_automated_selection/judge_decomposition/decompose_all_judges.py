"""Batch decompose all judges from judges.yaml into granular sub-judges.

This script orchestrates the recursive decomposition pipeline across all judges,
creating a comprehensive hierarchical judge set ready for aggregation experiments.

Usage:

    python decompose_all_judges.py \
        --max-depth 1 \
        --output experiments/track3_automated_selection/generated_judges
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_UTILS = REPO_ROOT / "pipeline" / "utils"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(PIPELINE_UTILS) not in sys.path:
    sys.path.append(str(PIPELINE_UTILS))

from pipeline.utils import judge_rubrics  # type: ignore  # noqa: E402

# Import the decomposer function
from experiments.track3_automated_selection.judge_decomposition.llm_judge_decomposer import (  # noqa: E402
    ChatCompletionClient,
    InlineListDumper,
    LLMConfig,
    decompose_judge_recursively,
)

load_dotenv()


def batch_decompose_judges(
    judge_ids: List[str],
    *,
    max_depth: int = 1,
    output_dir: Path,
    model: str = "openai/gpt-4.1-nano",
    temperature: float = 0.4,
    max_tokens: int = 2048,
    name: Optional[str] = None,
) -> Path:
    """Decompose multiple judges and combine into single output YAML."""
    client = ChatCompletionClient(
        LLMConfig(model=model, temperature=temperature, max_tokens=max_tokens)
    )

    all_judges: List[Dict[str, Any]] = []

    for judge_id in judge_ids:
        print(f"\n{'='*60}")
        print(f"Decomposing: {judge_id}")
        print(f"{'='*60}")

        try:
            judges = decompose_judge_recursively(
                judge_id,
                client=client,
                max_depth=max_depth,
                current_depth=0,
            )
            all_judges.extend(judges)
            print(f"✓ Generated {len(judges)} sub-judges for {judge_id}")
        except Exception as e:
            print(f"✗ Failed to decompose {judge_id}: {e}")
            continue

    if not all_judges:
        raise RuntimeError("No judges were successfully decomposed.")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{name}-{timestamp}.yaml" if name else f"all-judges-decomposed-{timestamp}.yaml"
    output_path = output_dir / filename

    payload = {"judges": all_judges}
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.dump(payload, handle, Dumper=InlineListDumper, sort_keys=False, allow_unicode=False)

    print(f"\n{'='*60}")
    print(f"✓ Batch decomposition complete!")
    print(f"✓ Total judges generated: {len(all_judges)}")
    print(f"✓ YAML saved to: {output_path}")
    print(f"{'='*60}\n")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch decompose all judges from judges.yaml"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Maximum decomposition depth (default: 1)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4.1-nano",
        help="Martian model name (default: openai/gpt-4.1-nano)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Maximum tokens per completion"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/track3_automated_selection/generated_judges"),
        help="Directory to write the generated YAML file",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        help="Specific judge IDs to decompose (if not provided, decomposes all available judges)",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Custom name prefix for output file (default: all-judges-decomposed)",
    )

    args = parser.parse_args()

    # Load all available judges if not specified
    if args.judges:
        judge_ids = args.judges
    else:
        available = judge_rubrics.list_available_judges()
        judge_ids = available
        print(f"Found {len(judge_ids)} judges to decompose: {judge_ids}")

    output_path = batch_decompose_judges(
        judge_ids,
        max_depth=args.max_depth,
        output_dir=Path(args.output),
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        name=args.name,
    )

    print(f"Ready for aggregation experiments!")


if __name__ == "__main__":
    main()
