#!/usr/bin/env python3
"""
Iterative Judge Selection Pipeline

Orchestrates the iterative loop for optimal judge set discovery:
1. Train aggregator on current judge set
2. Analyze importance and identify least valuable judge
3. Analyze gaps and propose complementary judges
4. Evaluate new judge set
5. Repeat until stopping criteria met

Usage:
    python experiments/track3_automated_selection/iterative_selection.py \
        --config config/selection_experiment.yaml \
        --max-iterations 10 \
        --output results/selection_run
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.core.aggregator_training import GAMAggregator, compute_metrics
from pipeline.core.judge_evaluation import JudgeEvaluator
from pipeline.utils.judge_rubrics import load_judges_from_yaml, JUDGE_RUBRICS

from experiments.track3_automated_selection.iterative_selection.judge_set_metrics import (
    JudgeSetMetrics,
    JudgeSetEvaluator,
    compute_quick_redundancy,
)
from experiments.track3_automated_selection.iterative_selection.gap_analyzer import (
    GapAnalyzer,
    GapAnalysisResult,
    identify_least_important_judge,
)
from experiments.track3_automated_selection.judge_decomposition.llm_judge_decomposer import (
    ChatCompletionClient,
    LLMConfig,
)

from experiments.track2_judge_interpretability.explainability.fetch_attributions import (
    gam_interp,
    contribution_based_importance,
)

logger = logging.getLogger(__name__)


@dataclass
class SelectionConfig:
    """Configuration for iterative judge selection."""
    
    # Metadata
    name: str = "iterative-selection"
    description: str = ""
    
    # Initial judge set
    initial_judge_file: str = "judges/helpsteer2/depth_0_parents.yaml"
    protected_judges: List[str] = field(default_factory=list)  # Judges not to remove
    
    # Data configuration
    data_file: str = ""  # Path to pickled DataFrame with judge_scores and target
    target_column: str = "target"
    train_test_split: float = 0.3  # Fraction for test set
    validation_split: float = 0.15  # Fraction of train for validation
    
    # Stopping criteria
    max_iterations: int = 10
    min_judges: int = 3
    target_judges: Optional[int] = None
    max_judges: int = 15
    r2_improvement_threshold: float = 0.01  # Stop if R² improves less than this
    r2_degradation_threshold: Optional[float] = None  # Stop if R² drops more than this
    plateau_patience: int = 2  # Stop after this many iterations without improvement
    selection_mode: str = "backward"  # "backward" or "forward"
    
    # Redundancy thresholds
    max_correlation: float = 0.9  # Remove if pair exceeds this
    
    # Pruning strategy
    pruning_strategy: str = "importance"  # "importance", "redundancy", "attribution_correlation", "human_correlation", "combined", "random"
    
    # Judge proposal settings
    proposal_mode: str = "decompose"  # "decompose" (children) or "create" (new parents)
    use_llm_suggestions: bool = True
    llm_model: str = "openai/gpt-5-nano"
    
    # GAM settings
    gam_n_splines: int = 10
    gam_lam: float = 0.6
    
    # Output
    output_dir: str = "results/selection"
    save_intermediate: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "SelectionConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, default_flow_style=False)


@dataclass
class IterationResult:
    """Results from a single iteration."""
    
    iteration: int
    judge_names: List[str]
    n_judges: int
    
    # Aggregator metrics
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    
    # Judge set metrics
    judge_set_metrics: Dict[str, Any]
    
    # Importance scores
    importance_scores: Dict[str, float]
    
    # Changes made
    removed_judge: Optional[str] = None
    added_judge: Optional[str] = None
    
    # Gap analysis
    gap_analysis: Optional[Dict[str, Any]] = None
    
    # Attribution correlations (for redundancy-based pruning)
    attribution_correlations: Optional[Dict[str, Dict[str, float]]] = None
    
    # Stopping info
    improvement: float = 0.0
    should_stop: bool = False
    stop_reason: Optional[str] = None


class IterativeJudgeSelector:
    """Main controller for iterative judge selection."""
    
    def __init__(self, config: SelectionConfig):
        """
        Initialize selector.
        
        Args:
            config: Selection configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config.to_yaml(str(self.output_dir / "config.yaml"))
        
        # Initialize evaluators
        self.judge_set_evaluator = JudgeSetEvaluator(
            correlation_threshold=config.max_correlation
        )
        
        # Initialize LLM client for gap analysis if needed
        llm_client = None
        if config.use_llm_suggestions:
            try:
                llm_config = LLMConfig(
                    model=config.llm_model,
                    temperature=0.4,
                    max_tokens=2048,
                )
                llm_client = ChatCompletionClient(llm_config)
            except RuntimeError as exc:
                logger.warning("LLM client unavailable, continuing without suggestions: %s", exc)
        
        self.gap_analyzer = GapAnalyzer(
            use_llm_suggestions=config.use_llm_suggestions,
            llm_client=llm_client,
        )
        
        # State tracking
        self.iteration_results: List[IterationResult] = []
        self.current_judges: List[Dict[str, Any]] = []
        self.best_r2 = -float("inf")
        self.plateau_count = 0
        
        # Load data
        self.df: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        
        logger.info(f"Initialized IterativeJudgeSelector with output_dir={self.output_dir}")
    
    def load_data(self, df: Optional[pd.DataFrame] = None) -> None:
        """
        Load experiment data.
        
        Args:
            df: Optional DataFrame to use directly. If None, loads from config.data_file
        """
        if df is not None:
            self.df = df
        elif self.config.data_file:
            with open(self.config.data_file, "rb") as f:
                self.df = pickle.load(f)
        else:
            raise ValueError("Must provide df or config.data_file")
        
        # Extract target values if needed (for workshop data with human_feedback dict)
        if self.config.target_column not in self.df.columns:
            if "human_feedback" in self.df.columns:
                logger.info("Extracting target from human_feedback column")
                self.df[self.config.target_column] = self.df["human_feedback"].apply(
                    lambda x: x.get("score", x.get("average_score", 0)) if isinstance(x, dict) else 0
                )
            else:
                raise ValueError(f"Target column '{self.config.target_column}' not found in data")
        
        logger.info(f"Loaded data with {len(self.df)} samples")
    
    def load_initial_judges(self) -> List[Dict[str, Any]]:
        """Load initial judge set from config."""
        if self.config.initial_judge_file:
            judges_dict = load_judges_from_yaml(Path(self.config.initial_judge_file))
            judges = list(judges_dict.values())
        else:
            # Use default judges from JUDGE_RUBRICS
            judges = [{"id": k, **v} for k, v in JUDGE_RUBRICS.items()]
        
        self.current_judges = judges
        logger.info(f"Loaded {len(judges)} initial judges")
        return judges
    
    def _prepare_data(self, judge_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare train/test data from DataFrame.
        
        Args:
            judge_names: Names of judges to include in features
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Extract judge scores
        if "judge_scores" in self.df.columns:
            # Scores stored as list per row - need to track original judge order
            all_scores = np.array(self.df["judge_scores"].tolist())
            
            # Get indices of judges we want to keep
            # We need the original full judge list to map indices
            if not hasattr(self, '_original_judge_names'):
                # First call - store original judge names
                self._original_judge_names = judge_names.copy()
            
            # Find column indices for the requested judges
            col_indices = []
            for name in judge_names:
                if name in self._original_judge_names:
                    col_indices.append(self._original_judge_names.index(name))
            
            if col_indices:
                X = all_scores[:, col_indices]
            else:
                X = all_scores
        else:
            # Scores in separate columns (judge_name format)
            score_cols = [f"{name}_score" for name in judge_names]
            available_cols = [c for c in score_cols if c in self.df.columns]
            if not available_cols:
                # Try without _score suffix
                available_cols = [c for c in judge_names if c in self.df.columns]
            X = self.df[available_cols].values
        
        # Extract target
        y = self.df[self.config.target_column].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.train_test_split,
            random_state=42,
        )
        
        return X_train, X_test, y_train, y_test
    
    def _train_aggregator(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        judge_names: List[str],
    ) -> GAMAggregator:
        """Train GAM aggregator on current judge set."""
        gam = GAMAggregator(
            feature_names=judge_names,
            n_splines=self.config.gam_n_splines,
            lam=self.config.gam_lam,
        )
        gam.fit(X_train, y_train)
        return gam
    
    def _evaluate_iteration(
        self,
        iteration: int,
        judge_names: List[str],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        removed_judge: Optional[str] = None,
        added_judge: Optional[str] = None,
    ) -> IterationResult:
        """
        Run a single iteration evaluation.
        
        Returns:
            IterationResult with all metrics and analysis
        """
        # Train aggregator
        gam = self._train_aggregator(X_train, y_train, judge_names)
        
        # Get predictions
        train_predictions = gam.predict(X_train)
        test_predictions = gam.predict(X_test)
        
        # Compute regression metrics
        train_metrics = compute_metrics(y_train, train_predictions)
        test_metrics = compute_metrics(y_test, test_predictions)
        
        # Get importance scores
        importance = gam.get_feature_importance(X_test)
        
        # Enhanced importance calculation using Track 2 methods
        attribution_correlations = None
        try:
            # Prepare data for attribution analysis
            interp_df = pd.DataFrame({"judge_scores": list(X_test)})
            
            attributions = gam_interp(
                gam.model,
                interp_df,
                {"n_splines": self.config.gam_n_splines},
            )
            contrib_importance_list = contribution_based_importance(attributions)
            
            # Compute attribution correlation matrix for redundancy-based pruning
            attr_matrix = np.array(attributions)
            if attr_matrix.shape[1] > 1:
                attr_df = pd.DataFrame(attr_matrix, columns=judge_names)
                corr_matrix = attr_df.corr().fillna(0)
                attribution_correlations = corr_matrix.to_dict()
            
            # Map back to judge names
            contrib_importance = {
                name: score
                for name, score in zip(judge_names, contrib_importance_list)
            }
            
            # Combine with attribution-based importance
            base_vals = np.array([importance.get(name, 0.0) for name in judge_names], dtype=float)
            contrib_vals = np.array([contrib_importance.get(name, 0.0) for name in judge_names], dtype=float)
            
            def normalize(vals: np.ndarray) -> np.ndarray:
                if vals.max() > vals.min():
                    return (vals - vals.min()) / (vals.max() - vals.min())
                return vals
            
            base_norm = normalize(base_vals)
            contrib_norm = normalize(contrib_vals)
            
            combined_importance = {}
            for i, name in enumerate(judge_names):
                combined_importance[name] = 0.5 * base_norm[i] + 0.5 * contrib_norm[i]
            
            logger.info("Combined importance scores calculated (Attribution + Contribution)")
            importance = combined_importance
            
        except Exception as e:
            logger.warning(f"Failed to calculate contribution-based importance: {e}")
            # Fallback to attribution-based importance (already in 'importance' variable)
        
        # Evaluate judge set
        judge_set_metrics = self.judge_set_evaluator.evaluate(
            judge_scores=X_test,
            judge_names=judge_names,
            predictions=test_predictions,
            targets=y_test,
            importance_scores=importance,
        )
        
        # Gap analysis
        gap_result = self.gap_analyzer.analyze(
            predictions=test_predictions,
            targets=y_test,
            judge_scores=X_test,
            judge_names=judge_names,
        )
        
        # Calculate improvement
        current_r2 = test_metrics.get("r2", 0.0)
        improvement = current_r2 - self.best_r2
        
        # Check stopping criteria
        should_stop, stop_reason = self._check_stopping_criteria(
            iteration=iteration,
            n_judges=len(judge_names),
            current_r2=current_r2,
            improvement=improvement,
        )
        
        # Update best R² and plateau counter
        if self.config.r2_degradation_threshold is not None:
            if current_r2 > self.best_r2:
                self.best_r2 = current_r2
                self.plateau_count = 0
            else:
                self.plateau_count += 1
        else:
            if improvement > self.config.r2_improvement_threshold:
                self.best_r2 = current_r2
                self.plateau_count = 0
            else:
                self.plateau_count += 1
        
        result = IterationResult(
            iteration=iteration,
            judge_names=judge_names,
            n_judges=len(judge_names),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            judge_set_metrics=judge_set_metrics.to_dict(),
            importance_scores=importance,
            removed_judge=removed_judge,
            added_judge=added_judge,
            gap_analysis=gap_result.to_dict(),
            attribution_correlations=attribution_correlations,
            improvement=improvement,
            should_stop=should_stop,
            stop_reason=stop_reason,
        )
        
        return result
    
    def _check_stopping_criteria(
        self,
        iteration: int,
        n_judges: int,
        current_r2: float,
        improvement: float,
    ) -> Tuple[bool, Optional[str]]:
        """Check if selection should stop."""
        
        # Max iterations reached
        if iteration >= self.config.max_iterations:
            return True, "max_iterations_reached"
        
        # Target number of judges reached
        if self.config.target_judges is not None and n_judges <= self.config.target_judges:
            return True, f"target_judges_reached_{self.config.target_judges}"
        
        # Minimum judges reached
        if n_judges <= self.config.min_judges:
            return True, "min_judges_reached"
        
        # Performance degradation too severe
        if (
            self.config.r2_degradation_threshold is not None
            and improvement < -self.config.r2_degradation_threshold
        ):
            return True, f"performance_degraded_by_{abs(improvement):.4f}"
        
        # Plateau detected
        if self.plateau_count >= self.config.plateau_patience:
            return True, f"plateau_detected_after_{self.plateau_count}_iterations"
        
        return False, None
    
    def _select_judge_to_remove(
        self,
        importance_scores: Dict[str, float],
        judge_scores: Optional[np.ndarray] = None,
        judge_names: Optional[List[str]] = None,
        targets: Optional[np.ndarray] = None,
        attribution_correlations: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Optional[str]:
        """
        Select which judge to remove based on configured pruning strategy.
        
        Args:
            importance_scores: Dict mapping judge names to importance scores
            judge_scores: Judge score matrix (n_samples, n_judges) for redundancy strategies
            judge_names: List of judge names corresponding to columns in judge_scores
            targets: Target values for human correlation strategy
            attribution_correlations: Dict of attribution correlation matrix for attribution strategy
            
        Returns:
            Name of judge to remove, or None if no candidate found
        """
        strategy = self.config.pruning_strategy
        
        if strategy == "importance":
            return self._remove_by_importance(importance_scores)
        elif strategy == "redundancy":
            return self._remove_by_redundancy(judge_scores, judge_names, importance_scores)
        elif strategy == "attribution_correlation":
            return self._remove_by_attribution_correlation(
                attribution_correlations, importance_scores, judge_names
            )
        elif strategy == "human_correlation":
            return self._remove_by_human_correlation(judge_scores, judge_names, targets)
        elif strategy == "combined":
            return self._remove_by_combined(importance_scores, judge_scores, judge_names)
        elif strategy == "random":
            return self._remove_random(judge_names)
        else:
            logger.warning(f"Unknown pruning strategy '{strategy}', falling back to importance")
            return self._remove_by_importance(importance_scores)
    
    def _remove_by_importance(self, importance_scores: Dict[str, float]) -> Optional[str]:
        """Remove judge with lowest importance score (original strategy)."""
        try:
            judge_name, score = identify_least_important_judge(
                importance_scores,
                protected_judges=self.config.protected_judges,
            )
            logger.info(f"[importance] Selecting {judge_name} (score={score:.4f})")
            return judge_name
        except ValueError:
            return None
    
    def _remove_by_redundancy(
        self,
        judge_scores: Optional[np.ndarray],
        judge_names: Optional[List[str]],
        importance_scores: Dict[str, float],
    ) -> Optional[str]:
        """Remove judge with highest mean pairwise score correlation."""
        if judge_scores is None or judge_names is None:
            logger.warning("[redundancy] Missing judge_scores or judge_names, falling back to importance")
            return self._remove_by_importance(importance_scores)
        
        protected = set(self.config.protected_judges)
        n_judges = len(judge_names)
        
        if n_judges < 2:
            return None
        
        # Compute correlation matrix of judge scores
        corr_matrix = np.corrcoef(judge_scores.T)
        
        # Calculate mean absolute correlation for each judge (excluding self)
        redundancy_scores = {}
        for i, name in enumerate(judge_names):
            if name in protected:
                continue
            other_corrs = []
            for j in range(n_judges):
                if i != j and not np.isnan(corr_matrix[i, j]):
                    other_corrs.append(abs(corr_matrix[i, j]))
            if other_corrs:
                redundancy_scores[name] = np.mean(other_corrs)
        
        if not redundancy_scores:
            return self._remove_by_importance(importance_scores)
        
        # Remove judge with highest redundancy (most correlated with others)
        most_redundant = max(redundancy_scores.items(), key=lambda x: x[1])
        logger.info(f"[redundancy] Selecting {most_redundant[0]} (mean_corr={most_redundant[1]:.4f})")
        return most_redundant[0]
    
    def _remove_by_attribution_correlation(
        self,
        attribution_correlations: Optional[Dict[str, Dict[str, float]]],
        importance_scores: Dict[str, float],
        judge_names: Optional[List[str]],
    ) -> Optional[str]:
        """Remove judge from most-correlated attribution pair (lower importance one)."""
        if attribution_correlations is None or not judge_names:
            logger.warning("[attribution_correlation] Missing data, falling back to importance")
            return self._remove_by_importance(importance_scores)
        
        protected = set(self.config.protected_judges)
        threshold = self.config.max_correlation
        
        # Find all pairs exceeding correlation threshold
        correlated_pairs = []
        judges = list(attribution_correlations.keys())
        for i, j1 in enumerate(judges):
            if j1 not in importance_scores:
                continue
            for j2 in judges[i + 1:]:
                if j2 not in importance_scores:
                    continue
                corr = attribution_correlations.get(j1, {}).get(j2, 0)
                if abs(corr) > threshold:
                    correlated_pairs.append((abs(corr), j1, j2))
        
        # Sort by highest correlation
        correlated_pairs.sort(key=lambda x: x[0], reverse=True)
        
        for corr, j1, j2 in correlated_pairs:
            # Find candidate to remove (not protected, lower importance)
            candidates = []
            if j1 not in protected:
                candidates.append((j1, importance_scores.get(j1, 0)))
            if j2 not in protected:
                candidates.append((j2, importance_scores.get(j2, 0)))
            
            if not candidates:
                continue
            
            # Remove the one with lower importance
            to_remove = min(candidates, key=lambda x: x[1])[0]
            logger.info(
                f"[attribution_correlation] Selecting {to_remove} "
                f"(corr={corr:.4f} with {j2 if to_remove == j1 else j1})"
            )
            return to_remove
        
        # No highly correlated pairs found, fall back to importance
        logger.info("[attribution_correlation] No pairs above threshold, falling back to importance")
        return self._remove_by_importance(importance_scores)
    
    def _remove_by_human_correlation(
        self,
        judge_scores: Optional[np.ndarray],
        judge_names: Optional[List[str]],
        targets: Optional[np.ndarray],
    ) -> Optional[str]:
        """Remove judge with lowest correlation to human targets."""
        if judge_scores is None or judge_names is None or targets is None:
            logger.warning("[human_correlation] Missing data, cannot compute correlations")
            return None
        
        protected = set(self.config.protected_judges)
        from scipy import stats
        
        correlations = {}
        for i, name in enumerate(judge_names):
            if name in protected:
                continue
            judge_col = judge_scores[:, i]
            # Filter NaN values
            mask = ~(np.isnan(judge_col) | np.isnan(targets))
            if mask.sum() < 3:
                correlations[name] = 0.0
                continue
            r, _ = stats.pearsonr(judge_col[mask], targets[mask])
            correlations[name] = r if not np.isnan(r) else 0.0
        
        if not correlations:
            return None
        
        # Remove judge with lowest correlation to human labels
        lowest = min(correlations.items(), key=lambda x: x[1])
        logger.info(f"[human_correlation] Selecting {lowest[0]} (pearson_r={lowest[1]:.4f})")
        return lowest[0]
    
    def _remove_by_combined(
        self,
        importance_scores: Dict[str, float],
        judge_scores: Optional[np.ndarray],
        judge_names: Optional[List[str]],
    ) -> Optional[str]:
        """Remove judge with lowest combined score: importance × (1 - redundancy)."""
        if judge_scores is None or judge_names is None:
            logger.warning("[combined] Missing judge_scores, falling back to importance")
            return self._remove_by_importance(importance_scores)
        
        protected = set(self.config.protected_judges)
        n_judges = len(judge_names)
        
        if n_judges < 2:
            return self._remove_by_importance(importance_scores)
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(judge_scores.T)
        
        # Calculate combined scores
        combined_scores = {}
        for i, name in enumerate(judge_names):
            if name in protected:
                continue
            
            imp = importance_scores.get(name, 0.0)
            
            # Mean correlation with other judges
            other_corrs = []
            for j in range(n_judges):
                if i != j and not np.isnan(corr_matrix[i, j]):
                    other_corrs.append(abs(corr_matrix[i, j]))
            
            redundancy = np.mean(other_corrs) if other_corrs else 0.0
            
            # Combined score: high importance + low redundancy = high value (keep)
            # We want to remove low combined scores
            combined = imp * (1.0 - redundancy)
            combined_scores[name] = combined
        
        if not combined_scores:
            return self._remove_by_importance(importance_scores)
        
        # Remove judge with lowest combined score
        lowest = min(combined_scores.items(), key=lambda x: x[1])
        logger.info(f"[combined] Selecting {lowest[0]} (combined_score={lowest[1]:.4f})")
        return lowest[0]
    
    def _remove_random(self, judge_names: Optional[List[str]]) -> Optional[str]:
        """Remove a random non-protected judge (baseline strategy)."""
        if not judge_names:
            return None
        
        protected = set(self.config.protected_judges)
        candidates = [j for j in judge_names if j not in protected]
        
        if not candidates:
            return None
        
        selected = random.choice(candidates)
        logger.info(f"[random] Selecting {selected}")
        return selected
    
    def _propose_new_judge(
        self,
        gap_analysis: GapAnalysisResult,
        current_judge_names: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Propose a new judge based on gap analysis.
        
        For now, returns None. Full implementation would integrate with
        llm_judge_decomposer.py to create new judges.
        """
        # TODO: Integrate with ParentJudgeCreatorAgent or DecompositionAgent
        # based on config.proposal_mode
        
        if not gap_analysis.suggested_dimensions:
            return None
        
        # Placeholder for LLM-based judge creation
        logger.info(f"Gap analysis suggests: {gap_analysis.suggested_dimensions}")
        return None
    
    def run(self) -> List[IterationResult]:
        """
        Run the iterative selection process.
        
        Returns:
            List of IterationResult for each iteration
        """
        logger.info("Starting iterative judge selection")
        
        # Load initial judges
        if not self.current_judges:
            self.load_initial_judges()
        
        # Load data if not already loaded
        if self.df is None:
            self.load_data()
        
        current_judge_names = [j["id"] for j in self.current_judges]
        
        for iteration in range(self.config.max_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration + 1} / {self.config.max_iterations}")
            logger.info(f"Current judges: {len(current_judge_names)}")
            logger.info(f"{'='*60}")
            
            # Prepare data for current judge set
            X_train, X_test, y_train, y_test = self._prepare_data(current_judge_names)
            
            # Evaluate current iteration
            result = self._evaluate_iteration(
                iteration=iteration,
                judge_names=current_judge_names,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )
            
            self.iteration_results.append(result)
            
            # Log results
            logger.info(f"Test R²: {result.test_metrics.get('r2', 0):.4f}")
            logger.info(f"Improvement: {result.improvement:.4f}")
            logger.info(f"Composite score: {result.judge_set_metrics.get('composite_score', 0):.4f}")
            
            # Save intermediate results
            if self.config.save_intermediate:
                self._save_iteration(result)
            
            # Check stopping criteria
            if result.should_stop:
                logger.info(f"Stopping: {result.stop_reason}")
                break
            
            # Decide on modifications
            removed = None
            added = None
            
            # Select judge to remove based on configured pruning strategy
            judge_to_remove = self._select_judge_to_remove(
                importance_scores=result.importance_scores,
                judge_scores=X_test,
                judge_names=current_judge_names,
                targets=y_test,
                attribution_correlations=result.attribution_correlations,
            )
            
            removal_floor = (
                self.config.target_judges
                if self.config.target_judges is not None
                else self.config.min_judges
            )
            
            if judge_to_remove and len(current_judge_names) > removal_floor:
                logger.info(
                    "Removing judge: %s (importance: %.4f)",
                    judge_to_remove,
                    result.importance_scores.get(judge_to_remove, 0.0),
                )
                current_judge_names = [j for j in current_judge_names if j != judge_to_remove]
                self.current_judges = [j for j in self.current_judges if j["id"] != judge_to_remove]
                removed = judge_to_remove
            elif self.config.target_judges is not None and len(current_judge_names) <= self.config.target_judges:
                logger.info(
                    "Reached target of %s judges. Stopping removal.",
                    self.config.target_judges,
                )
            
            # TODO: Propose new judge based on gap analysis
            # new_judge = self._propose_new_judge(...)
            
            # Update result with changes
            result.removed_judge = removed
            result.added_judge = added
        
        # Save final results
        self._save_final_results()
        
        return self.iteration_results
    
    def _save_iteration(self, result: IterationResult) -> None:
        """Save iteration results to disk."""
        iter_dir = self.output_dir / f"iteration_{result.iteration:02d}"
        iter_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        with open(iter_dir / "result.json", "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        # Save judges list
        with open(iter_dir / "judges.txt", "w") as f:
            f.write("\n".join(result.judge_names))
    
    def _save_final_results(self) -> None:
        """Save final summary results."""
        summary = {
            "total_iterations": len(self.iteration_results),
            "final_n_judges": self.iteration_results[-1].n_judges if self.iteration_results else 0,
            "final_r2": self.iteration_results[-1].test_metrics.get("r2", 0) if self.iteration_results else 0,
            "best_r2": self.best_r2,
            "stop_reason": self.iteration_results[-1].stop_reason if self.iteration_results else None,
            "iterations": [
                {
                    "iteration": r.iteration,
                    "n_judges": r.n_judges,
                    "test_r2": r.test_metrics.get("r2", 0),
                    "removed": r.removed_judge,
                    "added": r.added_judge,
                }
                for r in self.iteration_results
            ],
        }
        
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nFinal summary saved to {self.output_dir / 'summary.json'}")
        logger.info(f"Final judge count: {summary['final_n_judges']}")
        logger.info(f"Final test R²: {summary['final_r2']:.4f}")
        logger.info(f"Best R² achieved: {summary['best_r2']:.4f}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Iterative Judge Selection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        help="Path to pickled DataFrame with judge scores",
    )
    parser.add_argument(
        "--judges", "-j",
        type=str,
        help="Path to initial judges YAML file",
    )
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=10,
        help="Maximum iterations (default: 10)",
    )
    parser.add_argument(
        "--min-judges",
        type=int,
        default=3,
        help="Minimum number of judges to keep (default: 3)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/selection",
        help="Output directory (default: results/selection)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Build config
    if args.config:
        config = SelectionConfig.from_yaml(args.config)
    else:
        config = SelectionConfig()
    
    # Override with CLI args
    if args.data:
        config.data_file = args.data
    if args.judges:
        config.initial_judge_file = args.judges
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    if args.min_judges:
        config.min_judges = args.min_judges
    if args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output_dir = f"{args.output}_{timestamp}"
    
    # Run selection
    selector = IterativeJudgeSelector(config)
    results = selector.run()
    
    print(f"\n✅ Selection complete. Results saved to {config.output_dir}")
    print(f"   Total iterations: {len(results)}")
    if results:
        print(f"   Final R²: {results[-1].test_metrics.get('r2', 0):.4f}")


if __name__ == "__main__":
    main()
