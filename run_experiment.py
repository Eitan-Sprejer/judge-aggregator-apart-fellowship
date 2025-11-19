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
            Dictionary with train/val/test splits and test_indices in full df
        """
        log_experiment_milestone("Preparing training data")

        # Extract features (judge scores)
        X = np.array([row['judge_scores'] for _, row in df.iterrows()])

        # Extract targets based on config.target (always use target_dimension)
        if self.config.target == 'target_human_aggregated':
            y = np.array([
                row['target_human_aggregated'].get(self.config.target_dimension, np.nan)
                if row['target_human_aggregated'] is not None else np.nan
                for _, row in df.iterrows()
            ])
        elif self.config.target == 'target_human_individual':
            raise NotImplementedError(
                "target_human_individual not yet implemented. "
                "This should train separate aggregators for each annotator, not average them. "
                "Use target_human_aggregated for now."
            )
        elif self.config.target == 'target_synthetic':
            y = np.array([
                row['target_synthetic'].get(self.config.target_dimension, np.nan)
                if row['target_synthetic'] is not None else np.nan
                for _, row in df.iterrows()
            ])
        else:
            raise ValueError(f"Invalid target: {self.config.target}")

        # Filter out NaN targets
        valid_mask = ~np.isnan(y)
        valid_indices = np.where(valid_mask)[0]
        X = X[valid_mask]
        y = y[valid_mask]

        log_experiment_milestone(
            f"Training data: {len(X)} samples with {X.shape[1]} judges, "
            f"target={self.config.target}, dimension={self.config.target_dimension}"
        )

        # Train/test split
        indices = np.arange(len(X))
        indices_train, indices_test, X_train, X_test, y_train, y_test = train_test_split(
            indices, X, y,
            test_size=self.config.models.test_size,
            random_state=self.config.random_seed
        )

        # Map back to original df indices
        test_indices_in_full = valid_indices[indices_test]

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
            'judge_names': self.config.judges.judge_names,
            'test_indices_in_full': test_indices_in_full
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

    def run_baselines(self, data: Dict[str, Any],
                      human_rubric_predictions: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """Run baseline: single LLM judge with exact human annotation rubric.

        Args:
            data: Training data dictionary with test set
            human_rubric_predictions: Predictions from human-rubric judge on test set
        """
        log_experiment_milestone("Running baseline models")

        baseline_results = {}

        if human_rubric_predictions is not None:
            y_test = data['y_test']

            if len(human_rubric_predictions) != len(y_test):
                log_experiment_milestone(
                    f"Warning: Human-rubric predictions length mismatch ({len(human_rubric_predictions)} vs {len(y_test)}), skipping baseline"
                )
            else:
                metrics = compute_metrics(y_test, human_rubric_predictions)
                baseline_results['human_rubric_judge'] = metrics

                log_experiment_milestone(
                    f"Baseline 'human_rubric_judge' - R²: {metrics['r2']:.4f}, "
                    f"Spearman ρ: {metrics['spearman_rho']:.4f}, Kendall τ: {metrics['kendall_tau']:.4f}"
                )
        else:
            log_experiment_milestone("No human-rubric predictions available, skipping baseline")

        return baseline_results

    def run_human_rubric_evaluation(self, df: pd.DataFrame, test_indices: np.ndarray) -> Optional[np.ndarray]:
        """Evaluate test set using human annotation rubric via MartianClient.

        Args:
            df: Full dataset
            test_indices: Indices of test samples

        Returns:
            Predictions for test samples only
        """
        import hashlib
        import yaml
        from pathlib import Path
        from pipeline.utils.martian_client import MartianClient

        log_experiment_milestone("Running human-rubric evaluation on test set")

        # Get test subset
        df_test = df.iloc[test_indices].reset_index(drop=True)

        # Cache key: dataset + dimension + test size + seed (for reproducibility)
        cache_key_str = f"{self.config.dataset}_{self.config.target_dimension}_{len(df_test)}_{self.config.random_seed}"
        cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
        cache_dir = Path("results") / "_baseline_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"human_rubric_{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_predictions = pickle.load(f)
                if len(cached_predictions) == len(df_test):
                    log_experiment_milestone(f"Loaded human-rubric scores from cache: {cache_path}")
                    return cached_predictions
            except Exception as e:
                log_experiment_milestone(f"Cache load failed: {e}, regenerating...")
        rubric_path = Path(__file__).parent / "pipeline" / "utils" / "dataset_rubrics.yaml"
        if not rubric_path.exists():
            log_experiment_milestone(f"Rubric file not found: {rubric_path}, skipping human-rubric baseline")
            return None

        with open(rubric_path) as f:
            rubrics = yaml.safe_load(f)

        if self.config.dataset not in rubrics:
            log_experiment_milestone(
                f"Dataset '{self.config.dataset}' not in rubrics, skipping human-rubric baseline"
            )
            return None

        rubric_config = rubrics[self.config.dataset]
        rubric_text = rubric_config['rubric']
        dimensions = [d['name'] for d in rubric_config['dimensions']]

        # Validate target dimension
        if self.config.target_dimension not in dimensions:
            log_experiment_milestone(
                f"Target dimension '{self.config.target_dimension}' not in rubric dimensions {dimensions}"
            )
            return None

        # Initialize Martian client
        try:
            client = MartianClient()
        except Exception as e:
            log_experiment_milestone(f"Failed to initialize MartianClient: {e}")
            return None

        # Prepare batch evaluations for test set only
        question_col = 'question' if 'question' in df_test.columns else 'instruction'
        response_col = 'response' if 'response' in df_test.columns else 'answer'

        evaluations = [
            {
                'rubric': rubric_text,
                'question': row[question_col],
                'answer': row[response_col]
            }
            for _, row in df_test.iterrows()
        ]

        # Batch evaluate with multi-dimensional format
        log_experiment_milestone(f"Batch evaluating {len(evaluations)} test samples with human rubric...")
        results = client.evaluate_batch(
            evaluations,
            model=None,
            max_workers=self.config.concurrency,
            multi_dimensional=True
        )

        # Extract target dimension scores
        # Results format: {"dimension": {"score": X, "explanation": "..."}, ...}
        predictions = []
        for idx, result in enumerate(results):
            try:
                dim_result = result.get(self.config.target_dimension, {})
                score = dim_result.get('score', float('nan'))
                predictions.append(float(score))
            except Exception as e:
                log_experiment_milestone(f"Error parsing sample {idx}: {e}")
                predictions.append(float('nan'))

        predictions = np.array(predictions)

        # Cache results
        with open(cache_path, 'wb') as f:
            pickle.dump(predictions, f)

        log_experiment_milestone(
            f"Human-rubric evaluation complete. Cached to: {cache_path}"
        )

        return predictions

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

            # 4. Prepare training data (includes test_indices_in_full)
            data = self.prepare_training_data(df)

            # 5. Train GAM
            gam_results = self.train_gam(data)

            # 6. Run human-rubric evaluation on test set and baselines
            human_rubric_predictions = self.run_human_rubric_evaluation(
                df, data['test_indices_in_full']
            )
            baseline_results = self.run_baselines(data, human_rubric_predictions)

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
