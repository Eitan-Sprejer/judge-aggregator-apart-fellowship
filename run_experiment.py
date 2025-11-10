#!/usr/bin/env python3
"""
Simplified Experiment Runner for Multi-Judge Interpretability

This script runs experiments based on YAML configuration files with:
- Simple file-based dataset caching
- Shared judged data cache (MD5 hash-based)
- Auto persona simulation when target='target_synthetic'
- GAM training and baseline comparison

Usage:
  python run_experiment.py config.yaml
  python run_experiment.py experiments/baseline_summeval.yaml
"""

import asyncio
import hashlib
import json
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load environment variables
load_dotenv()

# Import project modules
from pipeline.config.experiment_config import ExperimentConfig
from pipeline.core.dataset_loader import DatasetLoader
from pipeline.core.persona_simulation import PersonaSimulator
from pipeline.core.judge_evaluation import JudgeEvaluator
from pipeline.core.aggregator_training import GAMAggregator, compute_metrics
from pipeline.core.baseline_models import BaselineEvaluator
from utils.logging_setup import (
    setup_universal_logging, log_experiment_start,
    log_experiment_milestone, log_experiment_complete
)


class ExperimentRunner:
    """Simplified experiment runner with caching and auto persona detection."""

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config

        # Set random seeds
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Create result directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{config.name}_{timestamp}"
        self.run_dir = Path("results") / self.run_name

        # Create subdirectories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "data").mkdir(exist_ok=True)
        (self.run_dir / "models").mkdir(exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)

        # Shared cache for judged data
        self.judged_cache_dir = Path("results") / "_judged_cache"
        self.judged_cache_dir.mkdir(parents=True, exist_ok=True)

        # Save config to run directory
        config_path = self.run_dir / "config.yaml"
        config.to_yaml(config_path)

        # Set up logging
        self.log_info = setup_universal_logging(
            experiment_name=self.run_name,
            log_dir=str(self.run_dir / "logs")
        )

        log_experiment_start({
            'name': config.name,
            'dataset': config.dataset,
            'target': config.target,
            'judges': config.judges.judge_ids,
            'run_dir': str(self.run_dir)
        })

        print(f"🚀 Starting experiment: {self.run_name}")
        print(f"📁 Run directory: {self.run_dir}")

    def _compute_judge_cache_key(self) -> str:
        """Compute MD5 hash of judge configuration for cache key.

        Returns:
            MD5 hash string
        """
        # Sort judge IDs for consistent hashing
        sorted_ids = sorted(self.config.judges.judge_ids)
        cache_str = "_".join(sorted_ids)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def load_dataset(self) -> pd.DataFrame:
        """Load dataset with caching.

        Returns:
            Dataset DataFrame
        """
        log_experiment_milestone(f"Loading dataset: {self.config.dataset}")

        loader = DatasetLoader()

        # Load with caching (uses datasets/processed/*.pkl)
        df = loader.load(
            dataset_name=self.config.dataset,
            use_cache=True,
            **self.config.dataset_kwargs
        )

        log_experiment_milestone(f"Loaded {len(df)} samples from {self.config.dataset}")

        # Validate config against loaded data
        self.config.validate_with_data(df)

        # Save copy to run directory for traceability
        dataset_path = self.run_dir / "data" / "dataset.pkl"
        df.to_pickle(dataset_path)

        return df

    async def run_persona_simulation_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run persona simulation if target is 'target_synthetic' and no synthetic scores exist.

        Args:
            df: Dataset DataFrame

        Returns:
            DataFrame with persona scores (if needed)
        """
        # Check if persona simulation is needed
        if not self.config.needs_persona_simulation:
            log_experiment_milestone(
                f"Target is {self.config.target}, skipping persona simulation"
            )
            return df

        # Check if synthetic scores already exist
        if df['target_synthetic'].notna().any():
            log_experiment_milestone("Synthetic scores already present, skipping simulation")
            return df

        log_experiment_milestone("Running persona simulation (target='target_synthetic')")

        # Initialize simulator
        simulator = PersonaSimulator()

        # Run simulation
        df_with_personas = await simulator.simulate_dataset(
            df,
            concurrency=self.config.concurrency,
            checkpoint_interval=10,
            checkpoint_dir=self.run_dir / "checkpoints"
        )

        # Save result
        personas_path = self.run_dir / "data" / "data_with_personas.pkl"
        df_with_personas.to_pickle(personas_path)

        log_experiment_milestone(f"Persona simulation complete, saved to {personas_path}")

        return df_with_personas

    def get_judged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get judge scores for dataset, using shared cache if available.

        Args:
            df: Dataset DataFrame

        Returns:
            DataFrame with judge scores
        """
        # Compute cache key
        cache_key = self._compute_judge_cache_key()
        cache_path = self.judged_cache_dir / f"{self.config.dataset}_{cache_key}.pkl"

        # Check cache
        if cache_path.exists():
            log_experiment_milestone(f"Loading judged data from shared cache: {cache_path}")
            df_with_judges = pd.read_pickle(cache_path)

            # Verify cache matches current dataset size
            if len(df_with_judges) >= len(df):
                # Use subset if cached data is larger
                df_judged = df_with_judges.iloc[:len(df)].copy()
                log_experiment_milestone(f"Using cached judge scores for {len(df_judged)} samples")

                # Save to run directory
                (self.run_dir / "data" / "data_with_judge_scores.pkl").write_bytes(
                    cache_path.read_bytes()
                )

                return df_judged

        log_experiment_milestone("Running judge evaluation (no cache found)")

        # Initialize judge evaluator
        evaluator = JudgeEvaluator(judge_ids=self.config.judges.judge_ids)

        # Run evaluation
        df_with_judges = evaluator.evaluate_dataset(
            df,
            checkpoint_dir=self.run_dir / "checkpoints",
            checkpoint_interval=10,
            max_workers=self.config.concurrency
        )

        # Save to shared cache
        df_with_judges.to_pickle(cache_path)
        log_experiment_milestone(f"Saved judged data to shared cache: {cache_path}")

        # Save to run directory
        judge_path = self.run_dir / "data" / "data_with_judge_scores.pkl"
        df_with_judges.to_pickle(judge_path)

        return df_with_judges

    def prepare_training_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare features and targets for training.

        Args:
            df: DataFrame with judge scores and target annotations

        Returns:
            Dictionary with train/val/test splits
        """
        log_experiment_milestone("Preparing training data")

        # Extract features (judge scores)
        X = np.array([row['judge_scores'] for _, row in df.iterrows()])

        # Extract targets based on config.target (always use target_dimension)
        if self.config.target == 'target_human_aggregated':
            # Use specific dimension
            y = np.array([
                row['target_human_aggregated'].get(self.config.target_dimension, np.nan)
                if row['target_human_aggregated'] is not None else np.nan
                for _, row in df.iterrows()
            ])
        elif self.config.target == 'target_human_individual':
            # TODO: Fit separate aggregators for each individual annotator
            raise NotImplementedError(
                "target_human_individual not yet implemented. "
                "This should train separate aggregators for each annotator, not average them. "
                "Use target_human_aggregated for now."
            )
        elif self.config.target == 'target_synthetic':
            # Use specific dimension
            y = np.array([
                row['target_synthetic'].get(self.config.target_dimension, np.nan)
                if row['target_synthetic'] is not None else np.nan
                for _, row in df.iterrows()
            ])
        else:
            raise ValueError(f"Invalid target: {self.config.target}")

        # Filter out NaN targets
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        log_experiment_milestone(
            f"Training data: {len(X)} samples with {X.shape[1]} judges, "
            f"target={self.config.target}, dimension={self.config.target_dimension}"
        )

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.models.test_size,
            random_state=self.config.random_seed
        )

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.models.val_size,
            random_state=self.config.random_seed
        )

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'judge_names': self.config.judges.judge_names
        }

    def train_gam(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train GAM aggregator.

        Args:
            data: Training data dictionary

        Returns:
            Dictionary with model and metrics
        """
        if not self.config.models.train_gam:
            log_experiment_milestone("GAM training disabled in config")
            return {}

        log_experiment_milestone("Training GAM aggregator")

        # Initialize GAM
        gam = GAMAggregator(
            n_splines=self.config.models.gam.n_splines,
            lam=self.config.models.gam.lam,
            max_iter=self.config.models.gam.max_iter
        )

        # Train
        gam.fit(data['X_train'], data['y_train'])

        # Evaluate
        train_metrics = compute_metrics(
            data['y_train'],
            gam.predict(data['X_train'])
        )
        val_metrics = compute_metrics(
            data['y_val'],
            gam.predict(data['X_val'])
        )
        test_metrics = compute_metrics(
            data['y_test'],
            gam.predict(data['X_test'])
        )

        log_experiment_milestone(
            f"GAM Results - Train R²: {train_metrics['r2']:.4f}, "
            f"Val R²: {val_metrics['r2']:.4f}, Test R²: {test_metrics['r2']:.4f}"
        )

        # Save model
        model_path = self.run_dir / "models" / "gam_model.pkl"
        gam.save_model(str(model_path))

        return {
            'model': gam,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

    def run_baselines(self, data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Run baseline models for comparison.

        Args:
            data: Training data dictionary

        Returns:
            Dictionary mapping baseline name to metrics
        """
        log_experiment_milestone("Running baseline models")

        evaluator = BaselineEvaluator(
            random_seed=self.config.random_seed,
            test_size=self.config.models.test_size
        )

        # Run all baseline methods
        baseline_results = {}

        # Naive mean (no scaling)
        result = evaluator.naive_mean_baseline(
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test']
        )
        baseline_results['naive_mean'] = result['metrics']
        log_experiment_milestone(f"Baseline 'naive_mean' - Test R²: {result['metrics']['r2']:.4f}")

        # Linear scaling mean (main experiment method)
        result = evaluator.linear_scaling_mean_baseline(
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test']
        )
        baseline_results['linear_scaling_mean'] = result['metrics']
        log_experiment_milestone(f"Baseline 'linear_scaling_mean' - Test R²: {result['metrics']['r2']:.4f}")

        # StandardScaler + LinearRegression mean
        result = evaluator.standardscaler_lr_mean_baseline(
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test']
        )
        baseline_results['standardscaler_lr_mean'] = result['metrics']
        log_experiment_milestone(f"Baseline 'standardscaler_lr_mean' - Test R²: {result['metrics']['r2']:.4f}")

        # Best single judge (naive)
        result = evaluator.best_single_judge_naive(
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test'],
            judge_names=data['judge_names']
        )
        baseline_results['best_single_judge_naive'] = result['metrics']
        log_experiment_milestone(f"Baseline 'best_single_judge_naive' - Test R²: {result['metrics']['r2']:.4f}")

        return baseline_results

    def save_results(
        self,
        gam_results: Dict[str, Any],
        baseline_results: Dict[str, Dict[str, float]]
    ):
        """Save experiment results to JSON.

        Args:
            gam_results: GAM model results
            baseline_results: Baseline model results
        """
        log_experiment_milestone("Saving experiment results")

        summary = {
            'experiment_name': self.config.name,
            'dataset': self.config.dataset,
            'target': self.config.target,
            'judges': self.config.judges.judge_ids,
            'n_judges': len(self.config.judges.judge_ids),
            'random_seed': self.config.random_seed,
            'gam_results': {
                'train': gam_results.get('train_metrics', {}),
                'val': gam_results.get('val_metrics', {}),
                'test': gam_results.get('test_metrics', {})
            },
            'baseline_results': baseline_results,
            'run_dir': str(self.run_dir),
            'timestamp': datetime.now().isoformat()
        }

        # Save summary
        summary_path = self.run_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        log_experiment_milestone(f"Results saved to {summary_path}")

    async def run(self):
        """Run complete experiment pipeline."""
        try:
            # 1. Load dataset
            df = self.load_dataset()

            # 2. Run persona simulation if needed
            df = await self.run_persona_simulation_if_needed(df)

            # 3. Get judge scores (with caching)
            df = self.get_judged_data(df)

            # 4. Prepare training data
            data = self.prepare_training_data(df)

            # 5. Train GAM
            gam_results = self.train_gam(data)

            # 6. Run baselines
            baseline_results = self.run_baselines(data)

            # 7. Save results
            self.save_results(gam_results, baseline_results)

            log_experiment_complete({
                'status': 'SUCCESS',
                'run_dir': str(self.run_dir)
            })

            print(f"✅ Experiment complete! Results in {self.run_dir}")

        except Exception as e:
            log_experiment_complete({
                'status': 'FAILED',
                'error': str(e)
            })
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run experiment from YAML config"
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1

    config = ExperimentConfig.from_yaml(config_path)

    # Run experiment
    runner = ExperimentRunner(config)
    asyncio.run(runner.run())

    return 0


if __name__ == "__main__":
    exit(main())
