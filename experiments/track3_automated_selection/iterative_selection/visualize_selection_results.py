#!/usr/bin/env python3
"""
Visualize iterative selection results.

Usage:
    python experiments/track3_automated_selection/iterative_selection/visualize_selection_results.py \
        --run-dir results/selection_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _find_latest_run(results_root: Path) -> Optional[Path]:
    candidates = [p for p in results_root.glob("selection_*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_iteration_results(run_dir: Path) -> List[Dict]:
    results = []
    for iter_dir in sorted(run_dir.glob("iteration_*")):
        result_path = iter_dir / "result.json"
        if result_path.exists():
            results.append(_load_json(result_path))
    results.sort(key=lambda r: r.get("iteration", 0))
    return results


def _extract_series(iteration_results: List[Dict]) -> Dict[str, List[float]]:
    metrics = {
        "iteration": [],
        "n_judges": [],
        "test_r2": [],
        "test_mae": [],
        "test_mse": [],
        "test_spearman": [],
        "test_kendall": [],
        "test_pearson": [],
        "composite_score": [],
    }
    for r in iteration_results:
        metrics["iteration"].append(r.get("iteration", 0))
        metrics["n_judges"].append(r.get("n_judges", 0))
        test_metrics = r.get("test_metrics", {})
        metrics["test_r2"].append(test_metrics.get("r2", 0.0))
        metrics["test_mae"].append(test_metrics.get("mae", 0.0))
        metrics["test_mse"].append(test_metrics.get("mse", 0.0))
        metrics["test_spearman"].append(test_metrics.get("spearman_rho", 0.0))
        metrics["test_kendall"].append(test_metrics.get("kendall_tau", 0.0))
        metrics["test_pearson"].append(test_metrics.get("pearson_r", 0.0))
        judge_set_metrics = r.get("judge_set_metrics", {})
        metrics["composite_score"].append(judge_set_metrics.get("composite_score", 0.0))
    return metrics


def _plot_series(run_dir: Path, series: Dict[str, List[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    iterations = series["iteration"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Iterative Selection Results: {run_dir.name}")

    ax = axes[0, 0]
    ax.plot(iterations, series["test_r2"], marker="o", label="Test R2")
    ax.set_title("Predictive Performance (R2)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("R2")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iterations, series["test_mae"], marker="o", color="tab:orange", label="Test MAE")
    ax.set_title("Prediction Error (MAE)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(iterations, series["composite_score"], marker="o", color="tab:green", label="Composite Score")
    ax.set_title("Judge Set Composite Score")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.step(iterations, series["n_judges"], where="post", marker="o", color="tab:purple", label="Judges")
    ax.set_title("Number of Judges")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = output_dir / "selection_metrics.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # Secondary plot with rank correlations
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(iterations, series["test_spearman"], marker="o", label="Spearman")
    ax.plot(iterations, series["test_kendall"], marker="o", label="Kendall")
    ax.plot(iterations, series["test_pearson"], marker="o", label="Pearson")
    ax.set_title("Rank Correlations (Test)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Correlation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = output_dir / "selection_correlations.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def _write_removals(run_dir: Path, summary: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["iteration\tn_judges\tremoved\tadded"]
    for entry in summary.get("iterations", []):
        lines.append(
            f"{entry.get('iteration')}\t{entry.get('n_judges')}\t"
            f"{entry.get('removed')}\t{entry.get('added')}"
        )
    (output_dir / "removals.tsv").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize iterative selection results")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Path to a selection run directory (default: latest results/selection_*)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output directory for plots (default: <run-dir>/plots)",
    )
    args = parser.parse_args()

    results_root = Path("results")
    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_run(results_root)
    if run_dir is None or not run_dir.exists():
        print("No selection run directory found. Provide --run-dir.", file=sys.stderr)
        sys.exit(1)

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"Missing summary.json in {run_dir}", file=sys.stderr)
        sys.exit(1)

    iteration_results = _load_iteration_results(run_dir)
    if not iteration_results:
        print(f"No iteration results found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else run_dir / "plots"
    summary = _load_json(summary_path)
    series = _extract_series(iteration_results)

    _plot_series(run_dir, series, output_dir)
    _write_removals(run_dir, summary, output_dir)

    print(f"Saved plots to {output_dir}")
    print(f"Removed judges log: {output_dir / 'removals.tsv'}")


if __name__ == "__main__":
    main()
