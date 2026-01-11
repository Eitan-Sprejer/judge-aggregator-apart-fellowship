"""
Attribution-based feature importance analysis for GAM and MLP models.

This module provides functions to compute per-sample attributions and derive
feature importance scores based on the variance of contributions across samples.

Key difference from traditional feature importance:
- Traditional: Static importance based on model coefficients (aggregate)
- Attribution-based: Variance of per-sample contributions (distribution-aware)

Judges with high attribution variance are context-dependent (important for some
samples, not others). Judges with low variance are consistently important/unimportant.
"""

import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def gam_attributions(gam_model, X_data: np.ndarray, n_splines: int) -> np.ndarray:
    """
    Compute per-sample attributions for a GAM model.

    For each sample, extracts the contribution of each feature's spline basis
    functions to the final prediction.

    Args:
        gam_model: Trained PyGAM model (LinearGAM or similar)
        X_data: Input data array of shape (n_samples, n_features)
        n_splines: Number of splines per feature in the GAM

    Returns:
        Array of shape (n_samples, n_features) with per-feature attributions
    """
    n_samples, n_features = X_data.shape
    attributions = np.zeros((n_samples, n_features))

    for sample_idx in range(n_samples):
        X_sample = X_data[sample_idx:sample_idx+1]  # Keep 2D shape

        # Get the model matrix (design matrix with spline basis functions)
        modelmat = gam_model._modelmat(X_sample)

        # Get coefficients (excluding intercept term at index -1)
        coef_indices = gam_model.terms.get_coef_indices(-1)
        coefficients = gam_model.coef_[coef_indices]

        # For each feature, sum its spline basis function contributions
        for feat_idx in range(n_features):
            start_idx = feat_idx * n_splines
            end_idx = start_idx + n_splines

            # Contribution = basis functions dotted with their coefficients
            contribution = modelmat[:, start_idx:end_idx].dot(coefficients[start_idx:end_idx])
            attributions[sample_idx, feat_idx] = contribution[0]

    return attributions


def normalize_attributions(attributions: np.ndarray) -> np.ndarray:
    """
    Normalize attributions to sum to 1 (absolute value) per sample.

    Args:
        attributions: Array of shape (n_samples, n_features)

    Returns:
        Normalized attributions where abs values sum to 1 per sample
    """
    abs_sums = np.abs(attributions).sum(axis=1, keepdims=True)
    # Avoid division by zero
    abs_sums = np.where(abs_sums == 0, 1, abs_sums)
    return attributions / abs_sums


def contribution_based_importance(attributions: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute feature importance based on variance of attributions across samples.

    The key insight: features with high attribution variance are context-dependent
    (their importance varies by sample), while features with low variance are
    consistently important or unimportant.

    Args:
        attributions: Array of shape (n_samples, n_features)
        normalize: Whether to normalize attributions before computing variance

    Returns:
        Array of shape (n_features,) with importance scores
    """
    if normalize:
        attributions = normalize_attributions(attributions)

    # Compute variance of each feature's attributions across samples
    variance_importance = np.var(np.abs(attributions), axis=0)

    return variance_importance


def mean_absolute_importance(attributions: np.ndarray) -> np.ndarray:
    """
    Compute feature importance based on mean absolute attribution.

    This is a simpler alternative to variance-based importance that measures
    the average magnitude of each feature's contribution.

    Args:
        attributions: Array of shape (n_samples, n_features)

    Returns:
        Array of shape (n_features,) with importance scores
    """
    return np.abs(attributions).mean(axis=0)


def get_gam_feature_importance(
    gam_model,
    X_data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    method: str = 'variance'
) -> Dict[str, float]:
    """
    Compute feature importance for a GAM model using attribution analysis.

    This is the main entry point for GAM feature importance computation.

    Args:
        gam_model: Trained PyGAM model
        X_data: Input data array of shape (n_samples, n_features)
        feature_names: Optional list of feature names
        method: Importance method - 'variance' (default) or 'mean_absolute'

    Returns:
        Dictionary mapping feature names to importance scores
    """
    n_features = X_data.shape[1]

    # Get n_splines from the model
    # PyGAM stores this in the terms
    try:
        # Try to get n_splines from first term
        n_splines = gam_model.terms[0].n_splines
    except (AttributeError, IndexError):
        # Fallback: try to infer from coefficient count
        total_coefs = len(gam_model.coef_) - 1  # Exclude intercept
        n_splines = total_coefs // n_features
        logger.warning(f"Could not get n_splines from model, inferred {n_splines}")

    # Compute attributions
    attributions = gam_attributions(gam_model, X_data, n_splines)

    # Compute importance based on method
    if method == 'variance':
        importance_scores = contribution_based_importance(attributions)
    elif method == 'mean_absolute':
        importance_scores = mean_absolute_importance(attributions)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'variance' or 'mean_absolute'")

    # Create feature name mapping
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    return {name: float(score) for name, score in zip(feature_names, importance_scores)}


def attribution_correlation_matrix(
    gam_model,
    X_data: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Compute correlation matrix between judge attributions.

    High correlation between two judges means they contribute similarly across samples,
    suggesting redundancy. This can help identify which judges provide unique information.

    Args:
        gam_model: Trained PyGAM model
        X_data: Input data array of shape (n_samples, n_features)
        feature_names: Optional list of feature names

    Returns:
        Dictionary with:
        - 'correlation_matrix': np.ndarray of shape (n_features, n_features)
        - 'feature_names': List of feature names
        - 'redundant_pairs': List of (feature1, feature2, correlation) for |corr| > 0.7
    """
    n_features = X_data.shape[1]

    # Get n_splines from the model
    try:
        n_splines = gam_model.terms[0].n_splines
    except (AttributeError, IndexError):
        total_coefs = len(gam_model.coef_) - 1
        n_splines = total_coefs // n_features

    # Compute attributions
    attributions = gam_attributions(gam_model, X_data, n_splines)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(attributions.T)

    # Handle NaN values (can occur if a feature has zero variance)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Find redundant pairs (|correlation| > 0.7)
    redundant_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr = corr_matrix[i, j]
            if abs(corr) > 0.7:
                redundant_pairs.append((
                    feature_names[i],
                    feature_names[j],
                    float(corr)
                ))

    # Sort by absolute correlation (descending)
    redundant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    return {
        'correlation_matrix': corr_matrix,
        'feature_names': feature_names,
        'redundant_pairs': redundant_pairs
    }


def bootstrap_importance_confidence_intervals(
    gam_model,
    X_data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    method: str = 'mean_absolute',
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for feature importance scores.

    Resamples the data with replacement and computes importance for each
    bootstrap sample to estimate the uncertainty in importance scores.

    Args:
        gam_model: Trained PyGAM model
        X_data: Input data array of shape (n_samples, n_features)
        feature_names: Optional list of feature names
        method: Importance method - 'variance' or 'mean_absolute'
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping feature names to dicts with:
        - 'mean': Mean importance across bootstrap samples
        - 'std': Standard deviation
        - 'ci_lower': Lower bound of confidence interval
        - 'ci_upper': Upper bound of confidence interval
        - 'median': Median importance
    """
    np.random.seed(random_seed)

    n_samples, n_features = X_data.shape

    # Get n_splines from the model
    try:
        n_splines = gam_model.terms[0].n_splines
    except (AttributeError, IndexError):
        total_coefs = len(gam_model.coef_) - 1
        n_splines = total_coefs // n_features

    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Collect bootstrap importance scores
    bootstrap_scores = np.zeros((n_bootstrap, n_features))

    for b in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_data[indices]

        # Compute attributions for bootstrap sample
        attributions = gam_attributions(gam_model, X_boot, n_splines)

        # Compute importance
        if method == 'variance':
            bootstrap_scores[b] = contribution_based_importance(attributions)
        elif method == 'mean_absolute':
            bootstrap_scores[b] = mean_absolute_importance(attributions)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Compute statistics
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    results = {}
    for i, name in enumerate(feature_names):
        scores = bootstrap_scores[:, i]
        results[name] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'ci_lower': float(np.percentile(scores, lower_percentile)),
            'ci_upper': float(np.percentile(scores, upper_percentile)),
            'median': float(np.median(scores))
        }

    return results


def get_importance_with_confidence(
    gam_model,
    X_data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    method: str = 'mean_absolute',
    n_bootstrap: int = 100,
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to get importance scores with confidence intervals.

    This is the main entry point for importance with uncertainty quantification.

    Args:
        gam_model: Trained PyGAM model
        X_data: Input data array of shape (n_samples, n_features)
        feature_names: Optional list of feature names
        method: Importance method - 'variance' or 'mean_absolute'
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for CI

    Returns:
        Dictionary mapping feature names to importance statistics with CI
    """
    return bootstrap_importance_confidence_intervals(
        gam_model, X_data, feature_names, method, n_bootstrap, confidence_level
    )
