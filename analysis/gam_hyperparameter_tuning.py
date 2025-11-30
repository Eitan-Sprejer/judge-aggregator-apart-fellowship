#!/usr/bin/env python3
"""
GAM Hyperparameter Tuning for Multi-Judge Interpretability

Simplified tuner that accepts prepared data directly and finds optimal GAM configurations.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy import stats

try:
    from pygam import LinearGAM
    HAS_GAM = True
except ImportError:
    HAS_GAM = False
    print("❌ PyGAM not installed. Install with: pip install pygam")
    exit(1)

from pipeline.core.aggregator_training import GAMAggregator, compute_metrics


class GAMHyperparameterTuner:
    """Hyperparameter tuning for GAM aggregation models."""

    def __init__(self, output_dir: str, feature_names: List[str], random_seed: int = 42):
        """Initialize GAM tuner.

        Args:
            output_dir: Directory to save tuning results
            feature_names: List of judge names
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.random_seed = random_seed

        random.seed(random_seed)
        np.random.seed(random_seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"🔧 GAM tuner initialized with {self.n_features} features: {', '.join(feature_names[:3])}...")

    def define_search_space(self) -> Dict[str, Any]:
        """Define hyperparameter search space for GAM.

        Returns:
            Dictionary with:
                - n_splines: List of spline counts to test (10 values)
                - lam_grid: Log-spaced lambda values from 0.1 to 100 (20 values)
                - max_iter: Fixed iteration limit
                - tol: Fixed convergence tolerance
        """
        return {
            'n_splines': [5, 7, 10, 12, 15, 18, 20, 25, 30, 35],  # 10 values (n_splines=3 removed due to PyGAM constraint: n_splines must be > spline_order=3)
            'lam_grid': np.logspace(-1, 2, 20),  # Log-spaced: 0.1 to 100 (20 values)
            'max_iter': 100,
            'tol': 1e-4
        }

    def evaluate_config(
        self,
        n_splines: int,
        lam_grid: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-4,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """Evaluate GAM configuration using PyGAM's gridsearch.

        Returns:
            Dictionary with best config (after gridsearch), metrics, and model
        """
        try:
            if normalize:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
            else:
                X_train_scaled = X_train.copy()
                X_val_scaled = X_val.copy()
                scaler = None

            gam_aggregator = GAMAggregator(
                feature_names=self.feature_names,
                n_splines=n_splines,
                lam=lam_grid[0],
                max_iter=max_iter,
                tol=tol
            )
            gam_aggregator.fit(X_train_scaled, y_train)

            gam_aggregator.model.gridsearch(
                X_train_scaled, y_train,
                lam=lam_grid,
                objective='GCV',
                progress=False,
                keep_best=True
            )

            best_lam = gam_aggregator.model.lam

            train_pred = gam_aggregator.predict(X_train_scaled)
            val_pred = gam_aggregator.predict(X_val_scaled)

            train_metrics = compute_metrics(y_train, train_pred)
            val_metrics = compute_metrics(y_val, val_pred)

            gam = gam_aggregator.model
            try:
                val_loglik = gam.loglikelihood(X_val_scaled, y_val)
                null_loglik = np.sum(stats.norm.logpdf(y_val, loc=np.mean(y_val), scale=np.std(y_val)))
                deviance = -2 * (val_loglik - null_loglik)
            except:
                deviance = np.nan

            gam_metrics = {
                'aic': gam.statistics_['AIC'],
                'deviance': deviance,
                'edof': gam.statistics_['edof'],
                'gcv': gam.statistics_['GCV'],
                'n_terms': len(gam.terms)
            }

            try:
                p_values = gam.statistics_['p_values']
                feature_importance = {}
                for i, label in enumerate(self.feature_names):
                    if i < len(p_values):
                        feature_importance[label] = max(0, 1.0 - p_values[i])
                    else:
                        feature_importance[label] = 0.0
            except:
                feature_importance = {}

            config = {
                'n_splines': n_splines,
                'lam': float(best_lam[0][0]) if isinstance(best_lam, list) else float(best_lam),
                'max_iter': max_iter,
                'tol': tol
            }

            return {
                'config': config,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'gam_metrics': gam_metrics,
                'feature_importance': feature_importance,
                'model': gam_aggregator,
                'scaler': scaler,
                'normalize': normalize,
                'success': True
            }

        except Exception as e:
            return {
                'config': {'n_splines': n_splines, 'lam': 'gridsearch_failed'},
                'error': str(e),
                'success': False
            }

    def evaluate_config_cv(
        self,
        n_splines: int,
        lam_grid: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        n_folds: int = 5,
        max_iter: int = 100,
        tol: float = 1e-4,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """Evaluate GAM configuration using K-fold cross-validation.

        Args:
            n_splines: Number of splines per feature
            lam_grid: Lambda values for PyGAM gridsearch
            X_train, y_train: Training data
            X_test, y_test: Optional test data for final evaluation
            n_folds: Number of CV folds (default: 5)
            max_iter: Maximum GAM iterations
            tol: Convergence tolerance
            normalize: Whether to normalize features

        Returns:
            Dictionary with CV metrics, best config, and optionally test metrics
        """
        try:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)

            cv_results = {
                'train_r2': [],
                'val_r2': [],
                'train_mse': [],
                'val_mse': [],
                'best_lam': []
            }

            # Run K-fold CV
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                y_fold_val = y_train[val_idx]

                # Normalize if requested
                if normalize:
                    scaler = StandardScaler()
                    X_fold_train = scaler.fit_transform(X_fold_train)
                    X_fold_val = scaler.transform(X_fold_val)

                # Train GAM with gridsearch
                gam_aggregator = GAMAggregator(
                    feature_names=self.feature_names,
                    n_splines=n_splines,
                    lam=lam_grid[0],
                    max_iter=max_iter,
                    tol=tol
                )
                gam_aggregator.fit(X_fold_train, y_fold_train)

                # Gridsearch for best lambda
                gam_aggregator.model.gridsearch(
                    X_fold_train, y_fold_train,
                    lam=lam_grid,
                    objective='GCV',
                    progress=False,
                    keep_best=True
                )

                best_lam = gam_aggregator.model.lam
                cv_results['best_lam'].append(
                    float(best_lam[0][0]) if isinstance(best_lam, list) else float(best_lam)
                )

                # Evaluate on fold
                train_pred = gam_aggregator.predict(X_fold_train)
                val_pred = gam_aggregator.predict(X_fold_val)

                train_metrics = compute_metrics(y_fold_train, train_pred)
                val_metrics = compute_metrics(y_fold_val, val_pred)

                cv_results['train_r2'].append(train_metrics['r2'])
                cv_results['val_r2'].append(val_metrics['r2'])
                cv_results['train_mse'].append(train_metrics['mse'])
                cv_results['val_mse'].append(val_metrics['mse'])

            # Aggregate CV results
            cv_summary = {
                'train_r2_mean': np.mean(cv_results['train_r2']),
                'train_r2_std': np.std(cv_results['train_r2']),
                'val_r2_mean': np.mean(cv_results['val_r2']),
                'val_r2_std': np.std(cv_results['val_r2']),
                'train_mse_mean': np.mean(cv_results['train_mse']),
                'train_mse_std': np.std(cv_results['train_mse']),
                'val_mse_mean': np.mean(cv_results['val_mse']),
                'val_mse_std': np.std(cv_results['val_mse']),
                'best_lam_mean': np.mean(cv_results['best_lam']),
                'best_lam_std': np.std(cv_results['best_lam'])
            }

            # Train final model on all training data with mean lambda
            if normalize:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
            else:
                X_train_scaled = X_train.copy()
                scaler = None

            final_gam = GAMAggregator(
                feature_names=self.feature_names,
                n_splines=n_splines,
                lam=cv_summary['best_lam_mean'],
                max_iter=max_iter,
                tol=tol
            )
            final_gam.fit(X_train_scaled, y_train)

            # Get final model metrics
            train_pred = final_gam.predict(X_train_scaled)
            train_metrics = compute_metrics(y_train, train_pred)

            # Optionally evaluate on test set
            test_metrics = None
            if X_test is not None and y_test is not None:
                if normalize:
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_test_scaled = X_test.copy()
                test_pred = final_gam.predict(X_test_scaled)
                test_metrics = compute_metrics(y_test, test_pred)

            # Get GAM-specific metrics
            gam = final_gam.model
            gam_metrics = {
                'aic': gam.statistics_['AIC'],
                'edof': gam.statistics_['edof'],
                'gcv': gam.statistics_['GCV'],
                'n_terms': len(gam.terms)
            }

            # Get feature importance
            try:
                p_values = gam.statistics_['p_values']
                feature_importance = {}
                for i, label in enumerate(self.feature_names):
                    if i < len(p_values):
                        feature_importance[label] = max(0, 1.0 - p_values[i])
                    else:
                        feature_importance[label] = 0.0
            except:
                feature_importance = {}

            config = {
                'n_splines': n_splines,
                'lam': cv_summary['best_lam_mean'],
                'max_iter': max_iter,
                'tol': tol
            }

            result = {
                'config': config,
                'cv_summary': cv_summary,
                'cv_folds': cv_results,
                'train_metrics': train_metrics,
                'gam_metrics': gam_metrics,
                'feature_importance': feature_importance,
                'model': final_gam,
                'scaler': scaler,
                'normalize': normalize,
                'success': True
            }

            if test_metrics:
                result['test_metrics'] = test_metrics

            return result

        except Exception as e:
            return {
                'config': {'n_splines': n_splines, 'lam': 'cv_failed'},
                'error': str(e),
                'success': False
            }

    def run_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        use_cv: bool = True,
        n_folds: int = 5,
        normalize: bool = True
    ) -> List[Dict]:
        """Run exhaustive GAM hyperparameter search.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Optional validation data (used if use_cv=False)
            use_cv: Whether to use K-fold CV (default: True, recommended)
            n_folds: Number of CV folds if use_cv=True (default: 5)
            normalize: Whether to normalize features with StandardScaler

        Returns:
            List of results sorted by validation R² (or CV mean R² if use_cv=True)

        Note:
            GAM search is always exhaustive over n_splines values.
            PyGAM's gridsearch finds optimal lambda for each n_splines.
            Search space: 10 n_splines × 20 lambda values = 200 total evaluations.
        """

        search_space = self.define_search_space()
        n_configs = len(search_space['n_splines'])

        print(f"🔍 GAM hyperparameter search")
        print(f"   Method: {'K-fold CV' if use_cv else 'Single train/val split'}")
        if use_cv:
            print(f"   CV folds: {n_folds}")
        print(f"   Configurations: {n_configs} n_splines values")
        print(f"   Lambda gridsearch: {len(search_space['lam_grid'])} values per config")
        print(f"   Total evaluations: {n_configs} × {len(search_space['lam_grid'])} = {n_configs * len(search_space['lam_grid'])}")

        results = []
        successful = 0

        for config_idx, n_splines in enumerate(search_space['n_splines']):
            print(f"\n[{config_idx + 1}/{n_configs}] n_splines={n_splines}")
            print(f"   → PyGAM gridsearch over {len(search_space['lam_grid'])} lambdas...")

            if use_cv:
                # Use K-fold cross-validation
                result = self.evaluate_config_cv(
                    n_splines=n_splines,
                    lam_grid=search_space['lam_grid'],
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_val,
                    y_test=y_val,
                    n_folds=n_folds,
                    max_iter=search_space['max_iter'],
                    tol=search_space['tol'],
                    normalize=normalize
                )

                if result['success']:
                    results.append(result)
                    successful += 1
                    cv_r2 = result['cv_summary']['val_r2_mean']
                    cv_std = result['cv_summary']['val_r2_std']
                    best_lam = result['config']['lam']
                    print(f"   ✅ Best λ={best_lam:.2f}: CV R²={cv_r2:.4f} ± {cv_std:.4f}, "
                          f"AIC={result['gam_metrics']['aic']:.2f}")
                else:
                    print(f"   ❌ Failed: {result['error']}")

            else:
                # Use single train/val split (legacy behavior)
                if X_val is None or y_val is None:
                    raise ValueError("X_val and y_val must be provided when use_cv=False")

                result = self.evaluate_config(
                    n_splines=n_splines,
                    lam_grid=search_space['lam_grid'],
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    max_iter=search_space['max_iter'],
                    tol=search_space['tol'],
                    normalize=normalize
                )

                if result['success']:
                    results.append(result)
                    successful += 1
                    best_lam = result['config']['lam']
                    print(f"   ✅ Best λ={best_lam:.2f}: val_R²={result['val_metrics']['r2']:.4f}, "
                          f"AIC={result['gam_metrics']['aic']:.2f}")
                else:
                    print(f"   ❌ Failed: {result['error']}")

        print(f"\n📊 Completed {successful}/{n_configs} configurations")

        # Sort by appropriate metric
        if use_cv:
            results.sort(key=lambda x: x['cv_summary']['val_r2_mean'], reverse=True)
        else:
            results.sort(key=lambda x: x['val_metrics']['r2'], reverse=True)

        self._save_results(results, use_cv=use_cv)

        return results

    def _save_results(self, results: List[Dict], use_cv: bool = False):
        """Save tuning results to JSON.

        Args:
            results: List of evaluation results
            use_cv: Whether results include CV metrics
        """
        results_for_json = []
        for result in results:
            if result['success']:
                json_result = {
                    'config': result['config'],
                    'train_metrics': result['train_metrics'],
                    'gam_metrics': result['gam_metrics'],
                    'feature_importance': result['feature_importance']
                }

                if use_cv:
                    # Include CV summary and optionally test metrics
                    json_result['cv_summary'] = result['cv_summary']
                    if 'test_metrics' in result:
                        json_result['test_metrics'] = result['test_metrics']
                else:
                    # Include validation metrics (legacy format)
                    json_result['val_metrics'] = result['val_metrics']

                results_for_json.append(json_result)

        results_path = self.output_dir / 'gam_tuning_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_for_json, f, indent=2)

        print(f"💾 Results saved to {results_path}")
