"""
Gap Analyzer Module

Analyzes prediction errors from aggregator models to identify systematic patterns
and suggest new judge dimensions that could improve coverage.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class GapPattern:
    """Represents an identified gap pattern in predictions."""
    
    pattern_type: str  # "systematic_over", "systematic_under", "high_variance", "cluster"
    description: str
    severity: float  # 0-1, higher = more severe
    affected_samples: int
    sample_indices: List[int] = field(default_factory=list)
    characteristics: Dict[str, Any] = field(default_factory=dict)
    suggested_dimension: Optional[str] = None


@dataclass
class GapAnalysisResult:
    """Result of gap analysis."""
    
    patterns: List[GapPattern]
    overall_error_stats: Dict[str, float]
    judge_error_correlations: Dict[str, float]  # Which judges correlate with errors
    suggested_dimensions: List[str]
    analysis_summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "patterns": [
                {
                    "type": p.pattern_type,
                    "description": p.description,
                    "severity": p.severity,
                    "affected_samples": p.affected_samples,
                    "characteristics": p.characteristics,
                    "suggested_dimension": p.suggested_dimension,
                }
                for p in self.patterns
            ],
            "overall_error_stats": self.overall_error_stats,
            "judge_error_correlations": self.judge_error_correlations,
            "suggested_dimensions": self.suggested_dimensions,
            "analysis_summary": self.analysis_summary,
        }


class GapAnalyzer:
    """Analyzes gaps in aggregator predictions to guide judge selection."""
    
    def __init__(
        self,
        error_threshold: float = 0.5,
        cluster_count: int = 3,
        use_llm_suggestions: bool = True,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize gap analyzer.
        
        Args:
            error_threshold: Threshold for flagging high-error samples
            cluster_count: Number of clusters for error pattern analysis
            use_llm_suggestions: Whether to use LLM for dimension suggestions
            llm_client: ChatCompletionClient for LLM-based suggestions
        """
        self.error_threshold = error_threshold
        self.cluster_count = cluster_count
        self.use_llm_suggestions = use_llm_suggestions
        self.llm_client = llm_client
    
    def analyze(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        judge_scores: np.ndarray,
        judge_names: List[str],
        sample_texts: Optional[List[Dict[str, str]]] = None,
    ) -> GapAnalysisResult:
        """
        Analyze prediction errors to identify gaps.
        
        Args:
            predictions: Aggregator predictions (n_samples,)
            targets: Ground truth targets (n_samples,)
            judge_scores: Judge score matrix (n_samples, n_judges)
            judge_names: Names of judges
            sample_texts: Optional list of dicts with 'prompt' and 'response' keys
            
        Returns:
            GapAnalysisResult with identified patterns and suggestions
        """
        # Calculate errors
        errors = targets - predictions  # Positive = underpredicted, Negative = overpredicted
        abs_errors = np.abs(errors)
        
        # Overall error statistics
        error_stats = self._compute_error_stats(errors, abs_errors)
        
        # Find patterns
        patterns = []
        
        # 1. Systematic over/under prediction
        patterns.extend(self._find_systematic_bias(errors, abs_errors))
        
        # 2. High variance regions
        patterns.extend(self._find_high_variance_regions(
            errors, judge_scores, judge_names
        ))
        
        # 3. Cluster-based patterns
        if judge_scores.shape[0] >= self.cluster_count * 2:
            patterns.extend(self._find_cluster_patterns(
                errors, judge_scores, judge_names
            ))
        
        # 4. Judge-error correlations (identify which judges miss certain patterns)
        judge_error_correlations = self._compute_judge_error_correlations(
            errors, judge_scores, judge_names
        )
        
        # Generate dimension suggestions
        suggested_dimensions = self._generate_suggestions(
            patterns, judge_error_correlations, judge_names
        )
        
        # Use LLM for richer suggestions if available
        if self.use_llm_suggestions and self.llm_client and sample_texts:
            llm_suggestions = self._get_llm_suggestions(
                patterns, errors, sample_texts, judge_names
            )
            suggested_dimensions.extend(llm_suggestions)
        
        # Generate summary
        summary = self._generate_summary(patterns, error_stats, suggested_dimensions)
        
        return GapAnalysisResult(
            patterns=patterns,
            overall_error_stats=error_stats,
            judge_error_correlations=judge_error_correlations,
            suggested_dimensions=suggested_dimensions,
            analysis_summary=summary,
        )
    
    def _compute_error_stats(
        self,
        errors: np.ndarray,
        abs_errors: np.ndarray,
    ) -> Dict[str, float]:
        """Compute overall error statistics."""
        return {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "mean_abs_error": float(np.mean(abs_errors)),
            "median_abs_error": float(np.median(abs_errors)),
            "max_abs_error": float(np.max(abs_errors)),
            "pct_high_error": float(np.mean(abs_errors > self.error_threshold) * 100),
            "skewness": float(stats.skew(errors)),
            "kurtosis": float(stats.kurtosis(errors)),
        }
    
    def _find_systematic_bias(
        self,
        errors: np.ndarray,
        abs_errors: np.ndarray,
    ) -> List[GapPattern]:
        """Find systematic over/under prediction patterns."""
        patterns = []
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Check for systematic underprediction
        if mean_error > 0.1:  # Model underpredicts (targets > predictions)
            high_underpredict_mask = errors > self.error_threshold
            patterns.append(GapPattern(
                pattern_type="systematic_under",
                description=f"Model systematically underpredicts by {mean_error:.3f} on average",
                severity=min(1.0, abs(mean_error) / 2.0),
                affected_samples=int(np.sum(high_underpredict_mask)),
                sample_indices=np.where(high_underpredict_mask)[0].tolist(),
                characteristics={
                    "mean_underprediction": float(mean_error),
                    "std": float(std_error),
                },
                suggested_dimension="quality-enhancement",
            ))
        
        # Check for systematic overprediction
        if mean_error < -0.1:  # Model overpredicts (predictions > targets)
            high_overpredict_mask = errors < -self.error_threshold
            patterns.append(GapPattern(
                pattern_type="systematic_over",
                description=f"Model systematically overpredicts by {abs(mean_error):.3f} on average",
                severity=min(1.0, abs(mean_error) / 2.0),
                affected_samples=int(np.sum(high_overpredict_mask)),
                sample_indices=np.where(high_overpredict_mask)[0].tolist(),
                characteristics={
                    "mean_overprediction": float(abs(mean_error)),
                    "std": float(std_error),
                },
                suggested_dimension="critical-evaluation",
            ))
        
        return patterns
    
    def _find_high_variance_regions(
        self,
        errors: np.ndarray,
        judge_scores: np.ndarray,
        judge_names: List[str],
    ) -> List[GapPattern]:
        """Find regions where error variance is high."""
        patterns = []
        
        # For each judge, check if low scores correlate with high errors
        for i, judge_name in enumerate(judge_names):
            judge_col = judge_scores[:, i]
            
            # Split into low/high score groups
            median_score = np.median(judge_col)
            low_mask = judge_col < median_score
            high_mask = judge_col >= median_score
            
            if np.sum(low_mask) > 5 and np.sum(high_mask) > 5:
                low_error_var = np.var(errors[low_mask])
                high_error_var = np.var(errors[high_mask])
                
                # Check for significant difference in variance
                if low_error_var > high_error_var * 2:
                    patterns.append(GapPattern(
                        pattern_type="high_variance",
                        description=f"High error variance when '{judge_name}' scores are low",
                        severity=min(1.0, low_error_var / (high_error_var + 0.01) / 4),
                        affected_samples=int(np.sum(low_mask)),
                        sample_indices=np.where(low_mask)[0].tolist()[:50],  # Limit
                        characteristics={
                            "judge": judge_name,
                            "low_score_error_var": float(low_error_var),
                            "high_score_error_var": float(high_error_var),
                        },
                        suggested_dimension=f"{judge_name.lower()}-specificity",
                    ))
        
        return patterns
    
    def _find_cluster_patterns(
        self,
        errors: np.ndarray,
        judge_scores: np.ndarray,
        judge_names: List[str],
    ) -> List[GapPattern]:
        """Find cluster-based error patterns."""
        patterns = []
        
        try:
            # Scale judge scores
            scaler = StandardScaler()
            scaled_scores = scaler.fit_transform(judge_scores)
            
            # Cluster based on judge scores
            kmeans = KMeans(n_clusters=self.cluster_count, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_scores)
            
            # Analyze errors per cluster
            for cluster_id in range(self.cluster_count):
                cluster_mask = clusters == cluster_id
                cluster_errors = errors[cluster_mask]
                cluster_abs_errors = np.abs(cluster_errors)
                
                mean_cluster_error = np.mean(cluster_errors)
                mean_abs_cluster_error = np.mean(cluster_abs_errors)
                overall_mean_abs_error = np.mean(np.abs(errors))
                
                # Check if this cluster has significantly higher errors
                if mean_abs_cluster_error > overall_mean_abs_error * 1.5:
                    # Find dominant characteristics of this cluster
                    cluster_scores = judge_scores[cluster_mask]
                    mean_cluster_scores = np.mean(cluster_scores, axis=0)
                    overall_mean_scores = np.mean(judge_scores, axis=0)
                    
                    # Which judges have notably different scores in this cluster?
                    score_diffs = mean_cluster_scores - overall_mean_scores
                    notable_judges = [
                        (judge_names[i], float(score_diffs[i]))
                        for i in np.argsort(np.abs(score_diffs))[-3:]
                    ]
                    
                    direction = "underpredicts" if mean_cluster_error > 0 else "overpredicts"
                    
                    patterns.append(GapPattern(
                        pattern_type="cluster",
                        description=(
                            f"Cluster {cluster_id}: Model {direction} for samples with "
                            f"distinctive judge score patterns"
                        ),
                        severity=min(1.0, mean_abs_cluster_error / overall_mean_abs_error / 2),
                        affected_samples=int(np.sum(cluster_mask)),
                        sample_indices=np.where(cluster_mask)[0].tolist()[:50],
                        characteristics={
                            "cluster_id": cluster_id,
                            "mean_error": float(mean_cluster_error),
                            "mean_abs_error": float(mean_abs_cluster_error),
                            "notable_judges": notable_judges,
                        },
                        suggested_dimension="cross-dimension-interaction",
                    ))
        
        except Exception as e:
            logger.warning(f"Cluster analysis failed: {e}")
        
        return patterns
    
    def _compute_judge_error_correlations(
        self,
        errors: np.ndarray,
        judge_scores: np.ndarray,
        judge_names: List[str],
    ) -> Dict[str, float]:
        """Compute correlation between each judge's scores and prediction errors."""
        correlations = {}
        
        for i, judge_name in enumerate(judge_names):
            judge_col = judge_scores[:, i]
            
            # Remove NaN values
            valid_mask = ~np.isnan(judge_col)
            if np.sum(valid_mask) > 2:
                corr, _ = stats.pearsonr(judge_col[valid_mask], errors[valid_mask])
                correlations[judge_name] = float(corr) if not np.isnan(corr) else 0.0
            else:
                correlations[judge_name] = 0.0
        
        return correlations
    
    def _generate_suggestions(
        self,
        patterns: List[GapPattern],
        judge_error_correlations: Dict[str, float],
        judge_names: List[str],
    ) -> List[str]:
        """Generate dimension suggestions based on patterns."""
        suggestions = []
        
        # Collect suggestions from patterns
        for pattern in patterns:
            if pattern.suggested_dimension and pattern.severity > 0.3:
                suggestions.append(pattern.suggested_dimension)
        
        # Suggest based on judge-error correlations
        # High positive correlation: judge misses issues that lower target scores
        # High negative correlation: judge overweights something that doesn't matter
        for judge, corr in judge_error_correlations.items():
            if corr > 0.3:
                # High judge score but also high positive error = underprediction
                suggestions.append(f"{judge.lower()}-complement")
            elif corr < -0.3:
                # High judge score with negative error = overprediction
                suggestions.append(f"counter-{judge.lower()}")
        
        # Deduplicate while preserving order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)
        
        return unique_suggestions
    
    def _get_llm_suggestions(
        self,
        patterns: List[GapPattern],
        errors: np.ndarray,
        sample_texts: List[Dict[str, str]],
        judge_names: List[str],
    ) -> List[str]:
        """Use LLM to suggest dimensions based on high-error samples."""
        if not self.llm_client:
            return []
        
        # Get indices of highest error samples
        high_error_indices = np.argsort(np.abs(errors))[-5:]
        
        # Build context from high-error samples
        samples_context = []
        for idx in high_error_indices:
            if idx < len(sample_texts):
                sample = sample_texts[idx]
                samples_context.append({
                    "prompt": sample.get("prompt", "N/A")[:300],
                    "response": sample.get("response", "N/A")[:300],
                    "error": float(errors[idx]),
                })
        
        if not samples_context:
            return []
        
        # Build pattern summary
        pattern_summary = "\n".join([
            f"- {p.pattern_type}: {p.description} (severity: {p.severity:.2f})"
            for p in patterns[:5]
        ])
        
        system_prompt = """You are an expert in evaluation rubric design. Given analysis of where 
a judge aggregation model makes errors, suggest new evaluation dimensions that could help.
Respond with a JSON list of 2-3 dimension suggestions."""
        
        user_prompt = f"""Current judges: {', '.join(judge_names)}

Error patterns identified:
{pattern_summary}

High-error samples:
{samples_context[:3]}

Based on these error patterns, suggest 2-3 new evaluation dimensions that could help 
reduce prediction errors. These should be orthogonal to existing judges.

Respond with JSON:
{{"suggestions": ["dimension-1", "dimension-2", "dimension-3"]}}
"""
        
        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            import json
            data = json.loads(response)
            return data.get("suggestions", [])[:3]
        except Exception as e:
            logger.warning(f"LLM suggestion failed: {e}")
            return []
    
    def _generate_summary(
        self,
        patterns: List[GapPattern],
        error_stats: Dict[str, float],
        suggested_dimensions: List[str],
    ) -> str:
        """Generate human-readable summary of analysis."""
        lines = [
            f"Gap Analysis Summary",
            f"=" * 40,
            f"Overall: {error_stats['pct_high_error']:.1f}% samples with error > {self.error_threshold}",
            f"Mean absolute error: {error_stats['mean_abs_error']:.3f}",
            f"Error skewness: {error_stats['skewness']:.3f}",
            "",
            f"Identified {len(patterns)} gap patterns:",
        ]
        
        for i, pattern in enumerate(patterns[:5], 1):
            lines.append(f"  {i}. [{pattern.pattern_type}] {pattern.description}")
            lines.append(f"     Severity: {pattern.severity:.2f}, Affected: {pattern.affected_samples} samples")
        
        if suggested_dimensions:
            lines.append("")
            lines.append(f"Suggested new dimensions: {', '.join(suggested_dimensions)}")
        
        return "\n".join(lines)


def identify_least_important_judge(
    importance_scores: Dict[str, float],
    protected_judges: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """
    Identify the least important judge from importance scores.
    
    Args:
        importance_scores: Dict mapping judge names to importance scores
        protected_judges: Judges that should not be removed
        
    Returns:
        Tuple of (judge_name, importance_score) for least important judge
    """
    protected = set(protected_judges or [])
    
    candidates = {
        name: score for name, score in importance_scores.items()
        if name not in protected
    }
    
    if not candidates:
        raise ValueError("No removable judges (all are protected)")
    
    least_important = min(candidates.items(), key=lambda x: x[1])
    return least_important
