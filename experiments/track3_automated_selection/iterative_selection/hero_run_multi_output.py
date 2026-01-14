#!/usr/bin/env python3
"""
Hero Run: Multi-Output Judge Pruning Experiment

Starts with ALL 30 judges (5 parents + 25 children) and iteratively prunes
while simultaneously optimizing for all 5 HelpSteer2 dimensions.

Key features:
- Multi-output: Trains 5 GAMs (one per dimension) at each iteration
- Cross-dimension importance: Aggregates importance across all dimensions
- Comprehensive logging: Records metrics for ALL dimensions at each step
- Finds optimal judge subset that balances performance across all dimensions

Pruning strategy: human_correlation (best from comparison experiment)
- Removes judge with lowest AVERAGE correlation to human targets across dimensions

Stopping criteria:
- Stop when average R² drops more than 15% from peak, OR
- Minimum of 5 judges reached

Usage:
    python experiments/track3_automated_selection/iterative_selection/hero_run_multi_output.py \
        --run-dir results/helpsteer2-baseline_20260112_102426 \
        --baseline-dir results/helpsteer2-baseline_20251129_165200
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Import GAM components
from pipeline.core.aggregator_training import GAMAggregator
from pipeline.utils.judge_rubrics import load_judges_from_yaml, get_judge_ids_from_files
from experiments.track2_judge_interpretability.explainability.fetch_attributions import (
    gam_interp,
    contribution_based_importance,
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DIMENSIONS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

# Hyperparameters
GAM_N_SPLINES = 10
GAM_LAM = 0.6
TRAIN_TEST_SPLIT = 0.2
MIN_JUDGES = 5
MAX_R2_DROP_PCT = 0.15  # Stop if avg R² drops more than 15% from peak


@dataclass
class DimensionMetrics:
    """Metrics for a single dimension."""
    r2: float
    mae: float
    pearson_r: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class IterationResult:
    """Results from one pruning iteration."""
    iteration: int
    n_judges: int
    judges: List[str]
    removed_judge: Optional[str]
    removal_reason: str
    
    # Per-dimension metrics
    dimension_metrics: Dict[str, DimensionMetrics]
    
    # Aggregated metrics
    avg_r2: float
    avg_mae: float
    avg_pearson_r: float
    
    # Importance scores (averaged across dimensions)
    importance_scores: Dict[str, float]
    human_correlations: Dict[str, Dict[str, float]]  # judge -> {dim: corr}
    
    def to_dict(self) -> Dict:
        return {
            "iteration": self.iteration,
            "n_judges": self.n_judges,
            "judges": self.judges,
            "removed_judge": self.removed_judge,
            "removal_reason": self.removal_reason,
            "dimension_metrics": {
                dim: metrics.to_dict() 
                for dim, metrics in self.dimension_metrics.items()
            },
            "avg_r2": self.avg_r2,
            "avg_mae": self.avg_mae,
            "avg_pearson_r": self.avg_pearson_r,
            "importance_scores": self.importance_scores,
            "human_correlations": self.human_correlations,
        }


class MultiOutputHeroRunner:
    """
    Hero run that optimizes judge selection across all dimensions simultaneously.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        all_judges: List[Dict],
        judge_ids: List[str],
        output_dir: Path,
    ):
        self.df = df
        self.all_judges = all_judges
        self.judge_ids = judge_ids
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Current state
        self.current_judges = list(all_judges)
        self.iteration_results: List[IterationResult] = []
        self.peak_avg_r2 = float("-inf")
        
        # Prepare train/test split (fixed across all iterations)
        n_samples = len(df)
        n_test = int(n_samples * TRAIN_TEST_SPLIT)
        indices = np.random.RandomState(42).permutation(n_samples)
        self.test_indices = indices[:n_test]
        self.train_indices = indices[n_test:]
        
        logger.info(f"Initialized with {len(all_judges)} judges")
        logger.info(f"Train samples: {len(self.train_indices)}, Test samples: {len(self.test_indices)}")
    
    def _get_judge_scores(self, judge_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract judge scores for given judges."""
        judge_indices = [self.judge_ids.index(name) for name in judge_names]
        
        all_scores = []
        for _, row in self.df.iterrows():
            scores = row["judge_scores"]
            all_scores.append([scores[i] for i in judge_indices])
        
        X = np.array(all_scores)
        X_train = X[self.train_indices]
        X_test = X[self.test_indices]
        
        return X_train, X_test
    
    def _get_targets(self, dimension: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract targets for a dimension."""
        targets = np.array([
            row["target_human_aggregated"].get(dimension)
            for _, row in self.df.iterrows()
        ])
        
        y_train = targets[self.train_indices]
        y_test = targets[self.test_indices]
        
        return y_train, y_test
    
    def _train_gam(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        judge_names: List[str],
    ) -> GAMAggregator:
        """Train a GAM on the data."""
        gam = GAMAggregator(
            feature_names=judge_names,
            n_splines=GAM_N_SPLINES,
            lam=GAM_LAM,
        )
        gam.fit(X_train, y_train)
        return gam
    
    def _compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
    ) -> DimensionMetrics:
        """Compute regression metrics."""
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Pearson correlation
        pearson_r, _ = stats.pearsonr(y_true, y_pred)
        
        return DimensionMetrics(r2=r2, mae=mae, pearson_r=pearson_r)
    
    def _compute_human_correlations(
        self,
        X_test: np.ndarray,
        judge_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute correlation of each judge's scores with human targets for each dimension."""
        correlations = {name: {} for name in judge_names}
        
        for dim in DIMENSIONS:
            _, y_test = self._get_targets(dim)
            
            for i, name in enumerate(judge_names):
                judge_scores = X_test[:, i]
                r, _ = stats.pearsonr(judge_scores, y_test)
                correlations[name][dim] = r if not np.isnan(r) else 0.0
        
        return correlations
    
    def _compute_importance_scores(
        self,
        gams: Dict[str, GAMAggregator],
        X_test: np.ndarray,
        judge_names: List[str],
    ) -> Dict[str, float]:
        """Compute importance scores averaged across all dimensions."""
        all_importance = {name: [] for name in judge_names}
        
        for dim, gam in gams.items():
            try:
                # Get GAM feature importance
                importance = gam.get_feature_importance(X_test)
                
                # Get contribution-based importance
                interp_df = pd.DataFrame({"judge_scores": list(X_test)})
                attributions = gam_interp(gam.model, interp_df, {"n_splines": GAM_N_SPLINES})
                contrib_list = contribution_based_importance(attributions)
                
                # Combine 50/50
                for i, name in enumerate(judge_names):
                    base = importance.get(name, 0.0)
                    contrib = contrib_list[i] if i < len(contrib_list) else 0.0
                    combined = 0.5 * base + 0.5 * contrib
                    all_importance[name].append(combined)
                    
            except Exception as e:
                logger.warning(f"Failed importance for {dim}: {e}")
                for name in judge_names:
                    all_importance[name].append(0.0)
        
        # Average across dimensions
        avg_importance = {
            name: np.mean(scores) if scores else 0.0
            for name, scores in all_importance.items()
        }
        
        # Normalize to [0, 1]
        values = np.array(list(avg_importance.values()))
        if values.max() > values.min():
            values = (values - values.min()) / (values.max() - values.min())
        
        return {name: float(values[i]) for i, name in enumerate(avg_importance.keys())}
    
    def _select_judge_to_remove(
        self,
        human_correlations: Dict[str, Dict[str, float]],
        importance_scores: Dict[str, float],
    ) -> Tuple[str, str]:
        """
        Select judge to remove using human_correlation strategy.
        
        Returns:
            Tuple of (judge_name, reason)
        """
        # Compute average correlation across dimensions for each judge
        avg_correlations = {}
        for judge, dim_corrs in human_correlations.items():
            avg_correlations[judge] = np.mean(list(dim_corrs.values()))
        
        # Find judge with lowest average correlation
        worst_judge = min(avg_correlations, key=avg_correlations.get)
        worst_corr = avg_correlations[worst_judge]
        
        # Get per-dimension breakdown for logging
        dim_breakdown = human_correlations[worst_judge]
        breakdown_str = ", ".join(f"{d}={v:.3f}" for d, v in dim_breakdown.items())
        
        reason = f"Lowest avg human correlation: {worst_corr:.4f} ({breakdown_str})"
        
        return worst_judge, reason
    
    def _evaluate_iteration(
        self,
        iteration: int,
        judge_names: List[str],
        removed_judge: Optional[str] = None,
        removal_reason: str = "",
    ) -> IterationResult:
        """Run evaluation for current judge set across all dimensions."""
        X_train, X_test = self._get_judge_scores(judge_names)
        
        # Train GAM for each dimension
        gams = {}
        dimension_metrics = {}
        
        for dim in DIMENSIONS:
            y_train, y_test = self._get_targets(dim)
            
            gam = self._train_gam(X_train, y_train, judge_names)
            gams[dim] = gam
            
            y_pred = gam.predict(X_test)
            metrics = self._compute_metrics(y_test, y_pred)
            dimension_metrics[dim] = metrics
        
        # Compute aggregated metrics
        avg_r2 = np.mean([m.r2 for m in dimension_metrics.values()])
        avg_mae = np.mean([m.mae for m in dimension_metrics.values()])
        avg_pearson = np.mean([m.pearson_r for m in dimension_metrics.values()])
        
        # Compute importance and correlations
        human_correlations = self._compute_human_correlations(X_test, judge_names)
        importance_scores = self._compute_importance_scores(gams, X_test, judge_names)
        
        result = IterationResult(
            iteration=iteration,
            n_judges=len(judge_names),
            judges=judge_names,
            removed_judge=removed_judge,
            removal_reason=removal_reason,
            dimension_metrics=dimension_metrics,
            avg_r2=avg_r2,
            avg_mae=avg_mae,
            avg_pearson_r=avg_pearson,
            importance_scores=importance_scores,
            human_correlations=human_correlations,
        )
        
        return result
    
    def _should_stop(self, current_avg_r2: float, n_judges: int) -> Tuple[bool, str]:
        """Check if we should stop pruning."""
        if n_judges <= MIN_JUDGES:
            return True, f"Reached minimum judges ({MIN_JUDGES})"
        
        if self.peak_avg_r2 > 0:
            drop_pct = (self.peak_avg_r2 - current_avg_r2) / self.peak_avg_r2
            if drop_pct > MAX_R2_DROP_PCT:
                return True, f"Avg R² dropped {drop_pct:.1%} from peak (threshold: {MAX_R2_DROP_PCT:.0%})"
        
        return False, ""
    
    def run(self) -> List[IterationResult]:
        """Run the hero pruning experiment."""
        logger.info("=" * 70)
        logger.info("HERO RUN: Multi-Output Judge Pruning")
        logger.info("=" * 70)
        
        current_judge_names = [j["id"] for j in self.current_judges]
        iteration = 0
        
        while True:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Iteration {iteration}")
            logger.info(f"Current judges: {len(current_judge_names)}")
            logger.info(f"{'=' * 60}")
            
            # Evaluate current state
            removed = None
            reason = ""
            if iteration > 0 and self.iteration_results:
                # Get removal info from previous decision
                prev_result = self.iteration_results[-1]
                # The judge was already removed, we just log it
                pass
            
            result = self._evaluate_iteration(
                iteration, 
                current_judge_names,
                removed_judge=removed,
                removal_reason=reason,
            )
            
            self.iteration_results.append(result)
            
            # Update peak
            if result.avg_r2 > self.peak_avg_r2:
                self.peak_avg_r2 = result.avg_r2
            
            # Log results
            logger.info(f"Avg R²: {result.avg_r2:.4f} (peak: {self.peak_avg_r2:.4f})")
            logger.info(f"Avg MAE: {result.avg_mae:.4f}")
            for dim, metrics in result.dimension_metrics.items():
                logger.info(f"  {dim}: R²={metrics.r2:.4f}, MAE={metrics.mae:.4f}")
            
            # Save iteration results
            iter_dir = self.output_dir / f"iteration_{iteration:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            (iter_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
            (iter_dir / "judges.txt").write_text("\n".join(current_judge_names))
            
            # Check stopping criteria
            should_stop, stop_reason = self._should_stop(result.avg_r2, len(current_judge_names))
            if should_stop:
                logger.info(f"STOPPING: {stop_reason}")
                break
            
            # Select judge to remove
            judge_to_remove, removal_reason = self._select_judge_to_remove(
                result.human_correlations,
                result.importance_scores,
            )
            
            logger.info(f"Removing: {judge_to_remove}")
            logger.info(f"Reason: {removal_reason}")
            
            # Remove judge and update the last result with removal info
            result.removed_judge = judge_to_remove
            result.removal_reason = removal_reason
            
            current_judge_names = [j for j in current_judge_names if j != judge_to_remove]
            self.current_judges = [j for j in self.current_judges if j["id"] != judge_to_remove]
            
            iteration += 1
        
        # Save summary
        self._save_summary()
        
        return self.iteration_results
    
    def _save_summary(self) -> None:
        """Save final summary."""
        # Find best iteration (highest avg R²)
        best_iter = max(self.iteration_results, key=lambda r: r.avg_r2)
        
        summary = {
            "total_iterations": len(self.iteration_results),
            "initial_judges": self.iteration_results[0].n_judges,
            "final_judges": self.iteration_results[-1].n_judges,
            "best_iteration": best_iter.iteration,
            "best_n_judges": best_iter.n_judges,
            "best_avg_r2": best_iter.avg_r2,
            "best_judges": best_iter.judges,
            "peak_avg_r2": self.peak_avg_r2,
            "final_avg_r2": self.iteration_results[-1].avg_r2,
            "trajectory": [
                {
                    "iteration": r.iteration,
                    "n_judges": r.n_judges,
                    "avg_r2": r.avg_r2,
                    "removed": r.removed_judge,
                    "dimension_r2": {d: m.r2 for d, m in r.dimension_metrics.items()},
                }
                for r in self.iteration_results
            ],
        }
        
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info(f"\nSummary saved to {summary_path}")
        
        # Print final summary
        logger.info("\n" + "=" * 70)
        logger.info("HERO RUN COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Started with {summary['initial_judges']} judges")
        logger.info(f"Best performance at iteration {best_iter.iteration} with {best_iter.n_judges} judges")
        logger.info(f"Best avg R²: {best_iter.avg_r2:.4f}")
        logger.info(f"Best judges: {', '.join(best_iter.judges)}")
        logger.info("\nPer-dimension R² at best iteration:")
        for dim, metrics in best_iter.dimension_metrics.items():
            logger.info(f"  {dim}: {metrics.r2:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hero Run: Multi-Output Judge Pruning Experiment"
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
        help="Baseline results directory (for reference)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output directory name (default: hero_run_TIMESTAMP)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_path = run_dir / "data" / "data_with_judge_scores.pkl"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Missing judged data: {data_path}")

    # Load data
    logger.info(f"Loading data from {data_path}")
    with data_path.open("rb") as f:
        df = pickle.load(f)
    logger.info(f"Loaded {len(df)} samples")

    # Load ALL judges (parents + children)
    judge_files = [
        "judges/helpsteer2/depth_0_parents.yaml",
        "judges/helpsteer2/depth_1_children.yaml",
    ]
    judge_ids = get_judge_ids_from_files(judge_files)
    logger.info(f"Total judge IDs from files: {len(judge_ids)}")
    
    # Load judge definitions
    parent_judges = load_judges_from_yaml(Path(judge_files[0]))
    child_judges = load_judges_from_yaml(Path(judge_files[1]))
    all_judges = list(parent_judges.values()) + list(child_judges.values())
    logger.info(f"Loaded {len(parent_judges)} parents + {len(child_judges)} children = {len(all_judges)} judges")
    
    # Verify we have scores for all judges
    sample_scores = df.iloc[0]["judge_scores"]
    logger.info(f"Sample has {len(sample_scores)} judge scores")
    
    if len(sample_scores) != len(judge_ids):
        logger.warning(f"Mismatch: {len(sample_scores)} scores vs {len(judge_ids)} judge IDs")
        # Use only judges that have scores
        judge_ids = judge_ids[:len(sample_scores)]
        all_judges = [j for j in all_judges if j["id"] in judge_ids]
        logger.info(f"Using {len(all_judges)} judges with available scores")
    
    # Setup output directory
    if args.output_name:
        output_name = args.output_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"hero_run_{timestamp}"
    
    output_dir = run_dir / output_name
    
    # Run hero experiment
    runner = MultiOutputHeroRunner(
        df=df,
        all_judges=all_judges,
        judge_ids=judge_ids,
        output_dir=output_dir,
    )
    
    results = runner.run()
    
    print(f"\n✅ Hero run complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
