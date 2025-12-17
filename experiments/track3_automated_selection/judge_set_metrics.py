"""
Judge Set Metrics Module

Provides composite metrics for evaluating judge sets in the iterative selection pipeline.
Metrics include coverage, redundancy, predictive power, and diversity indices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


@dataclass
class JudgeSetMetrics:
    """Container for all judge set evaluation metrics."""
    
    # Predictive power metrics
    r2: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    spearman_rho: float = 0.0
    kendall_tau: float = 0.0
    pearson_r: float = 0.0
    
    # Coverage metrics
    variance_explained: float = 0.0  # How much target variance judges explain
    coverage_score: float = 0.0  # Composite coverage metric
    
    # Redundancy metrics
    mean_pairwise_correlation: float = 0.0
    max_pairwise_correlation: float = 0.0
    redundancy_score: float = 0.0  # 1 - mean(|r|), higher = less redundant
    highly_correlated_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    
    # Diversity metrics
    effective_dimensionality: float = 0.0  # Based on PCA
    diversity_index: float = 0.0  # Composite diversity score
    
    # Importance distribution
    importance_gini: float = 0.0  # Gini coefficient of importance scores
    importance_entropy: float = 0.0  # Entropy of importance distribution
    
    # Overall composite score
    composite_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "predictive_power": {
                "r2": self.r2,
                "mse": self.mse,
                "mae": self.mae,
                "spearman_rho": self.spearman_rho,
                "kendall_tau": self.kendall_tau,
                "pearson_r": self.pearson_r,
            },
            "coverage": {
                "variance_explained": self.variance_explained,
                "coverage_score": self.coverage_score,
            },
            "redundancy": {
                "mean_pairwise_correlation": self.mean_pairwise_correlation,
                "max_pairwise_correlation": self.max_pairwise_correlation,
                "redundancy_score": self.redundancy_score,
                "highly_correlated_pairs": [
                    {"judge_a": a, "judge_b": b, "correlation": r}
                    for a, b, r in self.highly_correlated_pairs
                ],
            },
            "diversity": {
                "effective_dimensionality": self.effective_dimensionality,
                "diversity_index": self.diversity_index,
            },
            "importance_distribution": {
                "gini_coefficient": self.importance_gini,
                "entropy": self.importance_entropy,
            },
            "composite_score": self.composite_score,
        }


class JudgeSetEvaluator:
    """Evaluates judge sets using multiple metrics."""
    
    def __init__(
        self,
        correlation_threshold: float = 0.9,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            correlation_threshold: Threshold for flagging highly correlated pairs
            weights: Weights for composite score calculation. Keys:
                     'predictive', 'coverage', 'redundancy', 'diversity'
        """
        self.correlation_threshold = correlation_threshold
        self.weights = weights or {
            "predictive": 0.4,
            "coverage": 0.2,
            "redundancy": 0.2,
            "diversity": 0.2,
        }
    
    def evaluate(
        self,
        judge_scores: np.ndarray,
        judge_names: List[str],
        predictions: np.ndarray,
        targets: np.ndarray,
        importance_scores: Optional[Dict[str, float]] = None,
    ) -> JudgeSetMetrics:
        """
        Evaluate a judge set comprehensively.
        
        Args:
            judge_scores: Judge score matrix (n_samples, n_judges)
            judge_names: Names of judges
            predictions: Aggregator predictions (n_samples,)
            targets: Ground truth targets (n_samples,)
            importance_scores: Optional dict mapping judge names to importance
            
        Returns:
            JudgeSetMetrics with all computed metrics
        """
        metrics = JudgeSetMetrics()
        
        # Compute predictive power metrics
        self._compute_predictive_metrics(metrics, predictions, targets)
        
        # Compute coverage metrics
        self._compute_coverage_metrics(metrics, judge_scores, targets)
        
        # Compute redundancy metrics
        self._compute_redundancy_metrics(metrics, judge_scores, judge_names)
        
        # Compute diversity metrics
        self._compute_diversity_metrics(metrics, judge_scores)
        
        # Compute importance distribution metrics
        if importance_scores:
            self._compute_importance_metrics(metrics, importance_scores)
        
        # Compute composite score
        self._compute_composite_score(metrics)
        
        return metrics
    
    def _compute_predictive_metrics(
        self,
        metrics: JudgeSetMetrics,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Compute predictive power metrics."""
        # Basic regression metrics
        metrics.r2 = float(r2_score(targets, predictions))
        metrics.mse = float(mean_squared_error(targets, predictions))
        metrics.mae = float(mean_absolute_error(targets, predictions))
        
        # Correlation metrics
        if len(predictions) > 2:
            spearman_result = stats.spearmanr(predictions, targets)
            metrics.spearman_rho = float(spearman_result.correlation) if not np.isnan(spearman_result.correlation) else 0.0
            
            kendall_result = stats.kendalltau(predictions, targets)
            metrics.kendall_tau = float(kendall_result.correlation) if not np.isnan(kendall_result.correlation) else 0.0
            
            pearson_result = stats.pearsonr(predictions, targets)
            metrics.pearson_r = float(pearson_result.statistic) if not np.isnan(pearson_result.statistic) else 0.0
    
    def _compute_coverage_metrics(
        self,
        metrics: JudgeSetMetrics,
        judge_scores: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Compute coverage metrics."""
        # Variance explained by judges collectively (using simple linear regression)
        if judge_scores.shape[0] > judge_scores.shape[1]:  # More samples than features
            try:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(judge_scores, targets)
                predictions = model.predict(judge_scores)
                metrics.variance_explained = float(r2_score(targets, predictions))
            except Exception as e:
                logger.warning(f"Could not compute variance explained: {e}")
                metrics.variance_explained = 0.0
        
        # Coverage score is variance explained normalized
        metrics.coverage_score = max(0.0, min(1.0, metrics.variance_explained))
    
    def _compute_redundancy_metrics(
        self,
        metrics: JudgeSetMetrics,
        judge_scores: np.ndarray,
        judge_names: List[str],
    ) -> None:
        """Compute redundancy metrics."""
        n_judges = judge_scores.shape[1]
        
        if n_judges < 2:
            metrics.mean_pairwise_correlation = 0.0
            metrics.max_pairwise_correlation = 0.0
            metrics.redundancy_score = 1.0
            return
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(judge_scores.T)
        
        # Extract upper triangle (excluding diagonal)
        upper_tri_indices = np.triu_indices(n_judges, k=1)
        pairwise_correlations = corr_matrix[upper_tri_indices]
        
        # Filter out NaN values
        valid_correlations = pairwise_correlations[~np.isnan(pairwise_correlations)]
        
        if len(valid_correlations) > 0:
            abs_correlations = np.abs(valid_correlations)
            metrics.mean_pairwise_correlation = float(np.mean(abs_correlations))
            metrics.max_pairwise_correlation = float(np.max(abs_correlations))
            
            # Redundancy score: 1 - mean(|r|), higher = less redundant
            metrics.redundancy_score = 1.0 - metrics.mean_pairwise_correlation
        
        # Find highly correlated pairs
        for i in range(n_judges):
            for j in range(i + 1, n_judges):
                corr = corr_matrix[i, j]
                if not np.isnan(corr) and abs(corr) >= self.correlation_threshold:
                    metrics.highly_correlated_pairs.append(
                        (judge_names[i], judge_names[j], float(corr))
                    )
    
    def _compute_diversity_metrics(
        self,
        metrics: JudgeSetMetrics,
        judge_scores: np.ndarray,
    ) -> None:
        """Compute diversity metrics using PCA."""
        n_samples, n_judges = judge_scores.shape
        
        if n_judges < 2 or n_samples < 2:
            metrics.effective_dimensionality = float(n_judges)
            metrics.diversity_index = 1.0
            return
        
        try:
            # Center the data
            centered = judge_scores - np.mean(judge_scores, axis=0)
            
            # Compute SVD
            _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
            
            # Compute explained variance ratios
            variance_ratios = (singular_values ** 2) / np.sum(singular_values ** 2)
            
            # Effective dimensionality (participation ratio)
            # Higher values indicate more diverse/orthogonal judges
            metrics.effective_dimensionality = float(
                1.0 / np.sum(variance_ratios ** 2)
            ) if np.sum(variance_ratios ** 2) > 0 else n_judges
            
            # Diversity index: effective_dim / n_judges, normalized to [0, 1]
            metrics.diversity_index = min(1.0, metrics.effective_dimensionality / n_judges)
            
        except Exception as e:
            logger.warning(f"Could not compute diversity metrics: {e}")
            metrics.effective_dimensionality = float(n_judges)
            metrics.diversity_index = 1.0
    
    def _compute_importance_metrics(
        self,
        metrics: JudgeSetMetrics,
        importance_scores: Dict[str, float],
    ) -> None:
        """Compute importance distribution metrics."""
        if not importance_scores:
            return
        
        values = np.array(list(importance_scores.values()))
        
        if len(values) < 2:
            metrics.importance_gini = 0.0
            metrics.importance_entropy = 0.0
            return
        
        # Normalize to positive values
        values = values - np.min(values) + 1e-10
        values = values / np.sum(values)
        
        # Gini coefficient
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        metrics.importance_gini = float(
            (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
        )
        
        # Entropy (normalized)
        entropy = -np.sum(values * np.log(values + 1e-10))
        max_entropy = np.log(len(values))
        metrics.importance_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    
    def _compute_composite_score(self, metrics: JudgeSetMetrics) -> None:
        """Compute weighted composite score."""
        # Normalize component scores to [0, 1]
        predictive_score = max(0.0, metrics.r2)  # R² can be negative for bad models
        coverage_score = metrics.coverage_score
        redundancy_score = metrics.redundancy_score
        diversity_score = metrics.diversity_index
        
        metrics.composite_score = (
            self.weights["predictive"] * predictive_score +
            self.weights["coverage"] * coverage_score +
            self.weights["redundancy"] * redundancy_score +
            self.weights["diversity"] * diversity_score
        )
    
    def compare_judge_sets(
        self,
        metrics_list: List[JudgeSetMetrics],
        names: List[str],
    ) -> Dict[str, Any]:
        """
        Compare multiple judge sets.
        
        Args:
            metrics_list: List of JudgeSetMetrics to compare
            names: Names for each judge set
            
        Returns:
            Comparison summary dict
        """
        comparison = {
            "judge_sets": names,
            "best_by_metric": {},
            "rankings": {},
        }
        
        # Find best for each key metric
        metric_extractors = {
            "r2": lambda m: m.r2,
            "spearman_rho": lambda m: m.spearman_rho,
            "redundancy_score": lambda m: m.redundancy_score,
            "diversity_index": lambda m: m.diversity_index,
            "composite_score": lambda m: m.composite_score,
        }
        
        for metric_name, extractor in metric_extractors.items():
            values = [extractor(m) for m in metrics_list]
            best_idx = int(np.argmax(values))
            comparison["best_by_metric"][metric_name] = {
                "winner": names[best_idx],
                "value": values[best_idx],
            }
            
            # Rank all judge sets for this metric
            rankings = np.argsort(values)[::-1]  # Descending
            comparison["rankings"][metric_name] = [
                {"rank": i + 1, "name": names[idx], "value": values[idx]}
                for i, idx in enumerate(rankings)
            ]
        
        return comparison


def compute_quick_redundancy(
    judge_scores: np.ndarray,
    judge_names: List[str],
    threshold: float = 0.9,
) -> Dict[str, Any]:
    """
    Quick redundancy check for judge scores.
    
    Args:
        judge_scores: (n_samples, n_judges) array
        judge_names: List of judge names
        threshold: Correlation threshold for flagging pairs
        
    Returns:
        Dict with redundancy analysis
    """
    n_judges = len(judge_names)
    corr_matrix = np.corrcoef(judge_scores.T)
    
    highly_correlated = []
    for i in range(n_judges):
        for j in range(i + 1, n_judges):
            corr = corr_matrix[i, j]
            if not np.isnan(corr) and abs(corr) >= threshold:
                highly_correlated.append({
                    "judge_a": judge_names[i],
                    "judge_b": judge_names[j],
                    "correlation": float(corr),
                })
    
    # Mean absolute correlation
    upper_tri = corr_matrix[np.triu_indices(n_judges, k=1)]
    valid_corrs = upper_tri[~np.isnan(upper_tri)]
    mean_abs_corr = float(np.mean(np.abs(valid_corrs))) if len(valid_corrs) > 0 else 0.0
    
    return {
        "n_judges": n_judges,
        "mean_abs_correlation": mean_abs_corr,
        "highly_correlated_pairs": highly_correlated,
        "n_redundant_pairs": len(highly_correlated),
        "correlation_matrix": corr_matrix.tolist(),
    }
