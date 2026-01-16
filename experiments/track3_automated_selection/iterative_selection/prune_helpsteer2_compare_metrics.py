#!/usr/bin/env python3
"""
Compare different pruning metrics for HelpSteer2 child judges.

Runs the pruning experiment using 6 different strategies and compares results:
1. importance - Remove judge with lowest GAM importance score
2. redundancy - Remove judge with highest mean pairwise score correlation
3. attribution_correlation - Remove judge from most-correlated attribution pair
4. human_correlation - Remove judge with lowest correlation to human targets
5. combined - Remove judge with lowest importance × (1 - redundancy)
6. random - Remove a random judge (baseline)

Usage:
    python experiments/track3_automated_selection/iterative_selection/prune_helpsteer2_compare_metrics.py \
        --run-dir results/helpsteer2-baseline_20260112_102426 \
        --baseline-dir results/helpsteer2-baseline_20251129_165200
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd

from pipeline.utils.judge_rubrics import load_judges_from_yaml, get_judge_ids_from_files
from experiments.track3_automated_selection.iterative_selection.iterative_selection import (
    IterativeJudgeSelector,
    SelectionConfig,
)

# All pruning strategies to compare
STRATEGIES = [
    "importance",
    "redundancy",
    "attribution_correlation",
    "human_correlation",
    "combined",
    "random",
]

# Fixed random seed for reproducibility
RANDOM_SEED = 42


def _load_baseline_metrics(baseline_dir: Path, dimension: str) -> Dict:
    """Load baseline metrics for a dimension."""
    summary_path = baseline_dir / "dimensions" / dimension / "experiment_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text())


def _write_dimension_judges(judges: List[Dict], out_path: Path) -> None:
    """Write judges to YAML file."""
    import yaml

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"judges": judges}
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def run_strategy_for_dimension(
    dimension: str,
    strategy: str,
    judges: List[Dict],
    dim_df: pd.DataFrame,
    pruning_root: Path,
    target_judges: int,
) -> Dict:
    """
    Run a single pruning strategy for one dimension.
    
    Returns:
        Dict with results for this strategy/dimension combination
    """
    # Set random seed for reproducibility (especially for random strategy)
    random.seed(RANDOM_SEED)
    
    dim_dir = pruning_root / dimension / strategy
    dim_data_path = dim_dir / "data_with_judge_scores.pkl"
    dim_dir.mkdir(parents=True, exist_ok=True)
    dim_df.to_pickle(dim_data_path)

    judges_yaml_path = dim_dir / "child_judges.yaml"
    _write_dimension_judges(judges, judges_yaml_path)

    config = SelectionConfig(
        name=f"helpsteer2-{dimension}-{strategy}",
        description=f"Prune child judges using {strategy} strategy",
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
        pruning_strategy=strategy,
        use_llm_suggestions=False,
        output_dir=str(dim_dir / "selection"),
        save_intermediate=True,
    )

    selector = IterativeJudgeSelector(config)
    selector.run()

    # Load results
    summary = json.loads((Path(config.output_dir) / "summary.json").read_text())
    iter0 = json.loads(
        (Path(config.output_dir) / "iteration_00" / "result.json").read_text()
    )
    
    final_iter_idx = summary["total_iterations"] - 1
    final_iter = json.loads(
        (Path(config.output_dir) / f"iteration_{final_iter_idx:02d}" / "result.json").read_text()
    )

    removed_judges = [
        r.get("removed") or "" 
        for r in summary.get("iterations", []) 
        if r.get("removed")
    ]

    return {
        "dimension": dimension,
        "strategy": strategy,
        "iter0_r2": iter0.get("test_metrics", {}).get("r2"),
        "iter0_mae": iter0.get("test_metrics", {}).get("mae"),
        "final_r2": final_iter.get("test_metrics", {}).get("r2"),
        "final_mae": final_iter.get("test_metrics", {}).get("mae"),
        "delta_r2": (final_iter.get("test_metrics", {}).get("r2", 0) or 0) - (iter0.get("test_metrics", {}).get("r2", 0) or 0),
        "removed_judges": ", ".join(removed_judges),
        "final_judge_count": summary.get("final_n_judges"),
        "total_iterations": summary.get("total_iterations"),
        "selection_dir": str(config.output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pruning metrics for HelpSteer2 child judges"
    )
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
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=STRATEGIES,
        choices=STRATEGIES,
        help=f"Strategies to compare (default: all). Options: {STRATEGIES}",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        nargs="+",
        default=None,
        help="Specific dimensions to process (default: all)",
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

    # Filter dimensions if specified
    if args.dimensions:
        judges_by_dimension = {
            k: v for k, v in judges_by_dimension.items() 
            if k in args.dimensions
        }

    comparison_rows = []
    pruning_root = run_dir / "pruning_comparison"

    total_runs = len(judges_by_dimension) * len(args.strategies)
    current_run = 0

    for dimension, judges in judges_by_dimension.items():
        child_ids = [j["id"] for j in judges]
        child_indices = [judge_ids.index(jid) for jid in child_ids if jid in judge_ids]
        
        if not child_indices:
            print(f"⚠️  Skipping {dimension}: no matching judge indices")
            continue

        # Build dimension-specific dataset (shared across strategies)
        judge_scores = []
        targets = []
        for _, row in df.iterrows():
            scores = row["judge_scores"]
            judge_scores.append([scores[i] for i in child_indices])
            targets.append(row["target_human_aggregated"].get(dimension))

        dim_df = pd.DataFrame({
            "judge_scores": judge_scores,
            "target": targets,
        })

        target_judges = int(math.ceil(len(judges) / 2))
        
        # Load baseline metrics for comparison
        baseline = _load_baseline_metrics(baseline_dir, dimension)
        baseline_metrics = baseline.get("gam_results", {}).get("test", {})

        for strategy in args.strategies:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] Running {strategy} for {dimension}...")
            
            try:
                result = run_strategy_for_dimension(
                    dimension=dimension,
                    strategy=strategy,
                    judges=judges,
                    dim_df=dim_df,
                    pruning_root=pruning_root,
                    target_judges=target_judges,
                )
                
                # Add baseline metrics
                result["baseline_r2"] = baseline_metrics.get("r2")
                result["baseline_mae"] = baseline_metrics.get("mae")
                
                comparison_rows.append(result)
                
                print(f"   ✓ {strategy}: R² {result['iter0_r2']:.4f} → {result['final_r2']:.4f} (Δ={result['delta_r2']:+.4f})")
                
            except Exception as e:
                print(f"   ✗ {strategy} failed: {e}")
                comparison_rows.append({
                    "dimension": dimension,
                    "strategy": strategy,
                    "error": str(e),
                })

    # Save comparison summary
    summary_path = pruning_root / "comparison_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(comparison_rows, indent=2))
    print(f"\n✅ Wrote comparison summary to {summary_path}")

    # Generate a quick comparison table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    # Group by dimension for display
    by_dim = {}
    for row in comparison_rows:
        if "error" in row:
            continue
        dim = row["dimension"]
        if dim not in by_dim:
            by_dim[dim] = []
        by_dim[dim].append(row)
    
    for dim, rows in by_dim.items():
        print(f"\n{dim}:")
        print(f"  {'Strategy':<25} {'Init R²':>10} {'Final R²':>10} {'Δ R²':>10} {'Judges':>8}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        
        # Sort by final R² descending
        rows_sorted = sorted(rows, key=lambda x: x.get("final_r2") or 0, reverse=True)
        for row in rows_sorted:
            init_r2 = row.get("iter0_r2") or 0
            final_r2 = row.get("final_r2") or 0
            delta = row.get("delta_r2") or 0
            judges = row.get("final_judge_count") or 0
            print(f"  {row['strategy']:<25} {init_r2:>10.4f} {final_r2:>10.4f} {delta:>+10.4f} {judges:>8}")
    
    # Find best strategy per dimension
    print("\n" + "=" * 80)
    print("BEST STRATEGY PER DIMENSION (by final R²)")
    print("=" * 80)
    for dim, rows in by_dim.items():
        best = max(rows, key=lambda x: x.get("final_r2") or 0)
        print(f"  {dim}: {best['strategy']} (R²={best.get('final_r2', 0):.4f})")


if __name__ == "__main__":
    main()
