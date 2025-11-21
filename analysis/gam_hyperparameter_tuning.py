#!/usr/bin/env python3
"""
GAM Hyperparameter Tuning for Multi-Judge Interpretability

Simplified tuner that accepts prepared data directly and finds optimal GAM configurations.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler
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
        """Define hyperparameter search space for GAM."""
        return {
            'n_splines': [5, 8, 10],
            'lam_grid': np.array([2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0]),
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

    def run_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        method: str = "random",
        n_trials: int = 30,
        normalize: bool = True
    ) -> List[Dict]:
        """Run hyperparameter search.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            method: "random" or "exhaustive" (both use exhaustive for GAM)
            n_trials: Ignored for GAM (kept for API compatibility)
            normalize: Whether to normalize features

        Returns:
            List of results sorted by validation R²
        """
        search_space = self.define_search_space()
        n_configs = len(search_space['n_splines'])

        print(f"🔍 GAM hyperparameter search")
        print(f"   Configurations: {n_configs} (n_splines values)")
        print(f"   Lambda gridsearch: {len(search_space['lam_grid'])} values per config")

        results = []
        successful = 0

        for config_idx, n_splines in enumerate(search_space['n_splines']):
            print(f"\n[{config_idx + 1}/{n_configs}] n_splines={n_splines}")
            print(f"   → PyGAM gridsearch over {len(search_space['lam_grid'])} lambdas...")

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

        results.sort(key=lambda x: x['val_metrics']['r2'], reverse=True)

        self._save_results(results)

        return results

    def _save_results(self, results: List[Dict]):
        """Save tuning results to JSON."""
        results_for_json = []
        for result in results:
            if result['success']:
                json_result = {
                    'config': result['config'],
                    'train_metrics': result['train_metrics'],
                    'val_metrics': result['val_metrics'],
                    'gam_metrics': result['gam_metrics'],
                    'feature_importance': result['feature_importance']
                }
                results_for_json.append(json_result)

        results_path = self.output_dir / 'gam_tuning_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_for_json, f, indent=2)

        print(f"💾 Results saved to {results_path}")
