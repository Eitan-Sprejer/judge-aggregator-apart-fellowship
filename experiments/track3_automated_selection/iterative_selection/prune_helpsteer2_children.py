#!/usr/bin/env python3
"""
Prune HelpSteer2 child judges per dimension using existing judged data.

Uses the judged dataset saved from the baseline run and applies iterative selection
to each dimension's child judges, targeting half (rounded up).

Usage:
    python experiments/track3_automated_selection/iterative_selection/prune_helpsteer2_children.py \
        --run-dir results/helpsteer2-baseline_20260112_102426 \
        --baseline-dir results/helpsteer2-baseline_20251129_165200
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List

import pandas as pd

from pipeline.utils.judge_rubrics import load_judges_from_yaml, get_judge_ids_from_files
from experiments.track3_automated_selection.iterative_selection.iterative_selection import (
    IterativeJudgeSelector,
    SelectionConfig,
)


def _load_baseline_metrics(baseline_dir: Path, dimension: str) -> Dict:
    summary_path = baseline_dir / "dimensions" / dimension / "experiment_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text())


def _write_dimension_judges(judges: List[Dict], out_path: Path) -> None:
    import yaml

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"judges": judges}
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune HelpSteer2 child judges")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory containing data/data_with_judge_scores.pkl",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        required=True,
        help="Baseline results directory for comparison",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    baseline_dir = Path(args.baseline_dir)
    data_path = run_dir / "data" / "data_with_judge_scores.pkl"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing judged data: {data_path}")

    with data_path.open("rb") as f:
        df = pickle.load(f)

    judge_files = [
        "judges/helpsteer2/depth_0_parents.yaml",
        "judges/helpsteer2/depth_1_children.yaml",
    ]
    judge_ids = get_judge_ids_from_files(judge_files)
    child_judges = load_judges_from_yaml(Path(judge_files[1]))

    # Group child judges by dimension
    judges_by_dimension: Dict[str, List[Dict]] = {}
    for judge in child_judges.values():
        dim = judge.get("dimension")
        if not dim:
            continue
        judges_by_dimension.setdefault(dim, []).append(judge)

    summary_rows = []
    pruning_root = run_dir / "pruning_children"

    for dimension, judges in judges_by_dimension.items():
        child_ids = [j["id"] for j in judges]
        child_indices = [judge_ids.index(jid) for jid in child_ids if jid in judge_ids]
        if not child_indices:
            continue

        # Build dimension-specific dataset
        judge_scores = []
        targets = []
        for _, row in df.iterrows():
            scores = row["judge_scores"]
            judge_scores.append([scores[i] for i in child_indices])
            targets.append(row["target_human_aggregated"].get(dimension))

        dim_df = pd.DataFrame(
            {
                "judge_scores": judge_scores,
                "target": targets,
            }
        )

        dim_dir = pruning_root / dimension
        dim_data_path = dim_dir / "data_with_judge_scores.pkl"
        dim_dir.mkdir(parents=True, exist_ok=True)
        dim_df.to_pickle(dim_data_path)

        judges_yaml_path = dim_dir / "child_judges.yaml"
        _write_dimension_judges(judges, judges_yaml_path)

        target_judges = int(math.ceil(len(judges) / 2))
        config = SelectionConfig(
            name=f"helpsteer2-{dimension}-prune-children",
            description="Prune child judges to half (rounded up)",
            initial_judge_file=str(judges_yaml_path),
            protected_judges=[],
            data_file=str(dim_data_path),
            target_column="target",
            train_test_split=0.2,
            validation_split=0.15,
            max_iterations=10,
            min_judges=2,
            target_judges=target_judges,
            r2_improvement_threshold=0.0,
            plateau_patience=10,
            max_correlation=0.9,
            use_llm_suggestions=False,
            output_dir=str(dim_dir / "selection"),
            save_intermediate=True,
        )

        selector = IterativeJudgeSelector(config)
        selector.run()

        summary = json.loads((Path(config.output_dir) / "summary.json").read_text())
        iter0 = json.loads(
            (Path(config.output_dir) / "iteration_00" / "result.json").read_text()
        )
        final_iter = json.loads(
            (Path(config.output_dir) / f"iteration_{summary['total_iterations'] - 1:02d}" / "result.json").read_text()
        )

        baseline = _load_baseline_metrics(baseline_dir, dimension)
        baseline_metrics = baseline.get("gam_results", {}).get("test", {})

        summary_rows.append(
            {
                "dimension": dimension,
                "baseline_r2": baseline_metrics.get("r2"),
                "baseline_mae": baseline_metrics.get("mae"),
                "iter0_r2": iter0.get("test_metrics", {}).get("r2"),
                "iter0_mae": iter0.get("test_metrics", {}).get("mae"),
                "final_r2": final_iter.get("test_metrics", {}).get("r2"),
                "final_mae": final_iter.get("test_metrics", {}).get("mae"),
                "removed": ", ".join(r.get("removed") or "" for r in summary.get("iterations", []) if r.get("removed")),
                "final_judges": summary.get("final_n_judges"),
                "selection_dir": str(config.output_dir),
            }
        )

    summary_path = pruning_root / "pruning_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    print(f"Wrote pruning summary to {summary_path}")


if __name__ == "__main__":
    main()
