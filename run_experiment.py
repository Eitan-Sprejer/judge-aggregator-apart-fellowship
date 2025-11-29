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
from pipeline.core.judge_creation_orchestrator import JudgeCreationOrchestrator
from pipeline.utils import judge_rubrics
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

        # Create subdirectories (models and plots are created per-dimension)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "data").mkdir(exist_ok=True)
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
            'judge_files': config.judges.file_paths if config.judges.has_files else [],
            'run_dir': str(self.run_dir)
        })

        print(f"🚀 Starting experiment: {self.run_name}")
        print(f"📁 Run directory: {self.run_dir}")

    def _compute_judge_cache_key(self, df_size: int) -> str:
        """Compute MD5 hash of judge configuration + dataset parameters for cache key.

        Cache key must include:
        - Judge IDs (to invalidate when judges change)
        - Dataset size (to invalidate when n_samples changes)
        - Random seed (to invalidate when seed changes)
        - Split name (to invalidate when split changes)

        Args:
            df_size: Number of samples in dataset

        Returns:
            MD5 hash string
        """
        # Use judge IDs if available (after evaluation), otherwise use file paths
        if hasattr(self, 'judge_ids') and self.judge_ids:
            # Cache key based on actual judge IDs (most accurate)
            judges_str = "_".join(sorted(self.judge_ids))
        else:
            # Fallback: use judge file paths (before evaluation)
            file_paths = sorted(self.config.judges.file_paths)
            judges_str = "_".join(file_paths)

        # Include dataset parameters to invalidate cache when they change
        cache_str = (
            f"{judges_str}_"
            f"n{df_size}_"  # Number of samples
            f"seed{self.config.random_seed}_"  # Random seed
            f"split{self.config.dataset_kwargs.get('split', 'train')}"  # Dataset split
        )

        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_judge_dimension(self, judge_def: Dict[str, Any]) -> Optional[str]:
        """Extract dimension from judge definition.

        Args:
            judge_def: Judge definition dictionary with 'dimension' field

        Returns:
            Dimension name
        """
        return judge_def.get('dimension')

    def _compute_judge_depth(self, judge_def: Dict[str, Any]) -> int:
        """Compute depth of a judge in the hierarchy.

        Args:
            judge_def: Judge definition dictionary

        Returns:
            Depth (0 = parent, 1 = child, 2 = grandchild, etc.)
        """
        if judge_def.get('parent_id') is None:
            return 0

        # Find parent and recurse
        parent_id = judge_def['parent_id']
        parent = next((j for j in self.judges if j['id'] == parent_id), None)

        if parent is None:
            # Parent not found - treat as depth 0
            return 0

        return 1 + self._compute_judge_depth(parent)

    def _filter_judges_by_dimension(self, dimension: str) -> tuple[list[str], list[int]]:
        """Get judge IDs and indices for a specific dimension and depth.

        Filters by both dimension and the configured depth level.
        For example, if depth=1 is configured, only depth 1 judges (children)
        will be included, excluding depth 0 judges (parents).

        Args:
            dimension: Dimension name (e.g., "helpfulness")

        Returns:
            Tuple of (judge_ids, judge_indices) for the dimension and depth
        """
        if not hasattr(self, 'judges') or not self.judges:
            return [], []

        # Get configured depth from top-level judge_decomposition_depth field
        target_depth = self.config.judge_decomposition_depth if hasattr(self.config, 'judge_decomposition_depth') else None

        dimension_judge_ids = []
        dimension_indices = []

        for idx, judge_def in enumerate(self.judges):
            judge_dim = self._get_judge_dimension(judge_def)

            # Check dimension match
            if judge_dim != dimension:
                continue

            # Check depth match if depth filtering is enabled
            if target_depth is not None:
                judge_depth = self._compute_judge_depth(judge_def)
                if judge_depth != target_depth:
                    continue

            judge_id = judge_def.get('id', '')
            dimension_judge_ids.append(judge_id)
            dimension_indices.append(idx)

        return dimension_judge_ids, dimension_indices

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

    def create_judges_if_needed(self, df: pd.DataFrame):
        """Create dataset-specific judges if target_dimensions is specified in config.

        Args:
            df: Dataset DataFrame (may be used for sample Q&A pairs in judge creation)
        """
        # Check if judge creation is needed
        if not hasattr(self.config, 'target_dimensions') or not self.config.target_dimensions:
            log_experiment_milestone("No target_dimensions specified, skipping judge creation")
            return

        target_dimensions = self.config.target_dimensions
        log_experiment_milestone(
            f"Judge creation requested for {len(target_dimensions)} dimensions: {target_dimensions}"
        )

        # Get cache strategy (default: 'auto')
        cache_strategy = getattr(self.config, 'judge_cache_strategy', 'auto')
        decomposition_depth = getattr(self.config, 'judge_decomposition_depth', 1)

        # Get judge creation config (optional)
        judge_creation_config = getattr(self.config, 'judge_creation_config', None) or {}
        model = judge_creation_config.get('model', 'openai/gpt-5.1')
        temperature = judge_creation_config.get('temperature', 0.4)
        max_tokens = judge_creation_config.get('max_tokens', 2048)

        log_experiment_milestone(
            f"Judge cache strategy: {cache_strategy}, decomposition depth: {decomposition_depth}"
        )

        # Initialize orchestrator
        orchestrator = JudgeCreationOrchestrator(
            dataset_name=self.config.dataset,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Load dimension descriptions from dataset_rubrics.yaml
        import yaml
        rubrics_path = Path(__file__).parent / "pipeline" / "utils" / "dataset_rubrics.yaml"

        if not rubrics_path.exists():
            raise FileNotFoundError(
                f"dataset_rubrics.yaml not found at {rubrics_path}. "
                f"Cannot create judges without dimension descriptions."
            )

        with open(rubrics_path) as f:
            rubrics = yaml.safe_load(f)

        if self.config.dataset not in rubrics:
            raise ValueError(
                f"Dataset '{self.config.dataset}' not found in dataset_rubrics.yaml. "
                f"Available datasets: {list(rubrics.keys())}"
            )

        rubric_config = rubrics[self.config.dataset]
        rubric_dimensions = {
            d['name']: d['description']
            for d in rubric_config.get('dimensions', [])
        }

        # Validate target dimensions
        for dim in target_dimensions:
            if dim not in rubric_dimensions:
                raise ValueError(
                    f"Target dimension '{dim}' not found in dataset rubrics. "
                    f"Available dimensions: {list(rubric_dimensions.keys())}"
                )

        # Prepare dimension list for orchestrator
        dimensions = [
            {'name': dim, 'description': rubric_dimensions[dim]}
            for dim in target_dimensions
        ]

        # Optional: Extract sample Q&A pairs for context (first 3 samples)
        sample_qa_pairs = None
        if len(df) > 0:
            question_col = 'question' if 'question' in df.columns else 'instruction'
            response_col = 'response' if 'response' in df.columns else 'answer'

            sample_qa_pairs = [
                {'question': row[question_col], 'response': row[response_col]}
                for _, row in df.head(3).iterrows()
            ]

        # Create judges
        try:
            judges = orchestrator.create_judges_for_dataset(
                dimensions=dimensions,
                cache_strategy=cache_strategy,
                decomposition_depth=decomposition_depth,
                sample_qa_pairs=sample_qa_pairs
            )

            log_experiment_milestone(
                f"Judge creation complete: {len(judges)} judges available for {self.config.dataset}"
            )

            # Update config with judge file paths if not manually specified
            if not self.config.judges.has_files:
                # Auto-set judge files to all depth files created
                dataset_judges_dir = Path(__file__).parent / "judges" / self.config.dataset
                depth_files = sorted(dataset_judges_dir.glob("depth_*.yaml"))

                if depth_files:
                    # Convert to relative paths from repo root
                    relative_paths = [
                        f"judges/{self.config.dataset}/{f.name}"
                        for f in depth_files
                    ]
                    log_experiment_milestone(
                        f"Auto-detected {len(relative_paths)} judge files: {[f.name for f in depth_files]}"
                    )
                    self.config.judges.judge_files = relative_paths
                else:
                    raise RuntimeError(
                        f"Judge creation succeeded but no depth files found in {dataset_judges_dir}"
                    )

        except Exception as e:
            log_experiment_milestone(f"Judge creation failed: {e}")
            raise

    def get_judged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get judge scores for dataset, using shared cache if available.

        Args:
            df: Dataset DataFrame

        Returns:
            DataFrame with judge scores
        """
        # Validate that judge files are specified
        if not self.config.judges.has_files:
            raise ValueError(
                "No judge files specified in config. Either:\n"
                "1. Specify 'judge_file' or 'judge_files' in config, OR\n"
                "2. Specify 'target_dimensions' to trigger automatic judge creation"
            )

        # Initialize judge evaluator to get judge IDs (doesn't run evaluation yet)
        evaluator = JudgeEvaluator(
            judge_file_paths=self.config.judges.file_paths,
            model=self.config.judge_model
        )

        # Store judge IDs and names for later use
        self.judge_ids = evaluator.judge_ids
        self.judge_names = [jid.replace('-', ' ').title() for jid in evaluator.judge_ids]

        # Load full judge definitions (including dimension field) from YAML files
        self.judges = []
        for judge_path in self.config.judges.file_paths:
            import yaml
            file_path = Path(judge_path)
            if not file_path.is_absolute():
                file_path = Path.cwd() / file_path

            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'judges' in data:
                        self.judges.extend(data['judges'])

        log_experiment_milestone(f"  Loaded {len(self.judges)} judge definitions")

        # Compute cache key based on judge IDs + dataset parameters
        cache_key = self._compute_judge_cache_key(df_size=len(df))
        cache_path = self.judged_cache_dir / f"{self.config.dataset}_{cache_key}.pkl"

        # Check cache
        if cache_path.exists():
            log_experiment_milestone(f"Loading judged data from shared cache: {cache_path}")
            df_with_judges = pd.read_pickle(cache_path)

            # Verify cache exactly matches current dataset
            if len(df_with_judges) == len(df):
                log_experiment_milestone(f"Using cached judge scores for {len(df_with_judges)} samples")

                # Save to run directory
                (self.run_dir / "data" / "data_with_judge_scores.pkl").write_bytes(
                    cache_path.read_bytes()
                )

                return df_with_judges
            else:
                log_experiment_milestone(
                    f"Cache size mismatch: cached={len(df_with_judges)}, "
                    f"requested={len(df)}. Re-evaluating."
                )

        log_experiment_milestone("Running judge evaluation (no cache found)")

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

    def _store_judge_info_from_cache(self, df: pd.DataFrame):
        """Extract judge IDs from cached data and store for later use.

        Args:
            df: DataFrame with judge_scores column
        """
        # Get judge IDs from first row's judge_scores keys
        if len(df) > 0 and 'judge_scores' in df.columns:
            first_scores = df.iloc[0]['judge_scores']
            if isinstance(first_scores, dict):
                self.judge_ids = list(first_scores.keys())
                self.judge_names = [jid.replace('-', ' ').title() for jid in self.judge_ids]
            else:
                self.judge_ids = []
                self.judge_names = []
        else:
            self.judge_ids = []
            self.judge_names = []

    def prepare_training_data(self, df: pd.DataFrame, dimension: str) -> Dict[str, Any]:
        """Prepare features and targets for training on a specific dimension.

        Args:
            df: DataFrame with judge scores and target annotations
            dimension: Dimension to train on (e.g., "helpfulness")

        Returns:
            Dictionary with train/val/test splits and dimension-specific judges
        """
        log_experiment_milestone(f"Preparing training data for dimension: {dimension}")

        # Filter judges by dimension
        dimension_judge_ids, dimension_indices = self._filter_judges_by_dimension(dimension)

        if not dimension_indices:
            raise ValueError(
                f"No judges found for dimension '{dimension}'. "
                f"Available judge IDs: {self.judge_ids}"
            )

        log_experiment_milestone(
            f"  Found {len(dimension_indices)} judges for {dimension}: {dimension_judge_ids}"
        )

        # Extract features (judge scores) - ALL judges first
        X_all = np.array([row['judge_scores'] for _, row in df.iterrows()])

        # Filter to dimension-specific judges
        X = X_all[:, dimension_indices]

        # Get dimension-specific judge names
        dimension_judge_names = [self.judge_names[i] for i in dimension_indices]

        # Extract targets based on config.target
        if self.config.target == 'target_human_aggregated':
            y = np.array([
                row['target_human_aggregated'].get(dimension, np.nan)
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
                row['target_synthetic'].get(dimension, np.nan)
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
            f"  Training data: {len(X)} samples with {X.shape[1]} judges for {dimension}"
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
            'judge_names': dimension_judge_names,  # Dimension-specific judges
            'judge_ids': dimension_judge_ids,       # For caching/traceability
            'test_indices_in_full': test_indices_in_full
        }

    def tune_hyperparameters(self, data: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Run hyperparameter tuning if enabled.

        Args:
            data: Training data dictionary (includes dimension-specific judge_names)
            output_dir: Optional directory for tuning results (defaults to run_dir/tuning_analysis)

        Returns:
            Dict with 'gam' key containing optimal params (empty if disabled)
        """
        if not self.config.models.tuning.enabled:
            log_experiment_milestone("Hyperparameter tuning disabled")
            return {}

        log_experiment_milestone(
            f"Running hyperparameter tuning."
        )

        optimal_params = {}

        if self.config.models.train_gam:
            from analysis.gam_hyperparameter_tuning import GAMHyperparameterTuner

            # Use provided output_dir or default to run_dir
            if output_dir is None:
                tuning_dir = self.run_dir / "tuning_analysis" / "gam"
            else:
                tuning_dir = output_dir / "tuning_analysis" / "gam"

            tuner = GAMHyperparameterTuner(
                output_dir=str(tuning_dir),
                feature_names=data['judge_names'],  # Use dimension-specific judges
                random_seed=self.config.random_seed
            )

            # Use K-fold CV by default (more robust)
            results = tuner.run_search(
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_val=data['X_val'],  # Used as test set in CV mode
                y_val=data['y_val'],
                use_cv=True,  # Use K-fold CV for robust hyperparameter selection
                n_folds=5     # 5-fold cross-validation
            )

            if results:
                optimal_params['gam'] = results[0]['config']

                # Log appropriate metrics based on CV vs single-split
                if 'cv_summary' in results[0]:
                    cv_r2 = results[0]['cv_summary']['val_r2_mean']
                    cv_std = results[0]['cv_summary']['val_r2_std']
                    log_msg = (
                        f"✓ Optimal GAM: n_splines={optimal_params['gam']['n_splines']}, "
                        f"lam={optimal_params['gam']['lam']:.2f}, "
                        f"CV R²={cv_r2:.4f} ± {cv_std:.4f}"
                    )
                    if 'test_metrics' in results[0]:
                        log_msg += f", test R²={results[0]['test_metrics']['r2']:.4f}"
                else:
                    log_msg = (
                        f"✓ Optimal GAM: n_splines={optimal_params['gam']['n_splines']}, "
                        f"lam={optimal_params['gam']['lam']:.2f}, "
                        f"val_R²={results[0]['val_metrics']['r2']:.4f}"
                    )
                log_experiment_milestone(log_msg)
            else:
                log_experiment_milestone("⚠ GAM tuning produced no results, using config defaults")

        return optimal_params

    def train_gam(self, data: Dict[str, Any], optimal_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Train GAM aggregator with optional optimized hyperparameters.

        Args:
            data: Training data dictionary
            optimal_params: Optional dict with tuned hyperparameters

        Returns:
            Dictionary with model and metrics
        """
        if not self.config.models.train_gam:
            log_experiment_milestone("GAM training disabled in config")
            return {}

        if optimal_params:
            n_splines = optimal_params['n_splines']
            lam = optimal_params['lam']
            max_iter = optimal_params.get('max_iter', self.config.models.gam.max_iter)
            log_experiment_milestone("Training GAM with tuned hyperparameters")
        else:
            n_splines = self.config.models.gam.n_splines
            lam = self.config.models.gam.lam
            max_iter = self.config.models.gam.max_iter
            log_experiment_milestone("Training GAM with config defaults")

        gam = GAMAggregator(
            feature_names=data['judge_names'],  # Use dimension-specific judges
            n_splines=n_splines,
            lam=lam,
            max_iter=max_iter
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

        # Model will be saved in save_results_for_dimension() to dimension-specific directory

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

    def run_human_rubric_evaluation(self, df: pd.DataFrame, test_indices: np.ndarray, dimension: str) -> Optional[np.ndarray]:
        """Evaluate test set using human annotation rubric via MartianClient.

        Args:
            df: Full dataset
            test_indices: Indices of test samples
            dimension: Dimension to evaluate (e.g., "helpfulness")

        Returns:
            Predictions for test samples only
        """
        import hashlib
        import yaml
        from pathlib import Path
        from pipeline.utils.martian_client import MartianClient

        log_experiment_milestone(f"Running human-rubric evaluation for {dimension} on test set")

        # Get test subset
        df_test = df.iloc[test_indices].reset_index(drop=True)

        # Cache key: dataset + dimension + test size + seed (for reproducibility)
        cache_key_str = f"{self.config.dataset}_{dimension}_{len(df_test)}_{self.config.random_seed}"
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
        if dimension not in dimensions:
            log_experiment_milestone(
                f"Target dimension '{dimension}' not in rubric dimensions {dimensions}"
            )
            return None

        # Initialize Martian client (uses same model as judge evaluation)
        try:
            client = MartianClient(default_model=self.config.judge_model)
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

        log_experiment_milestone(f"Batch evaluating {len(evaluations)} test samples with human rubric...")
        results = client.evaluate_batch(
            evaluations,
            dataset=self.config.dataset,
            model=self.config.judge_model,
            max_workers=self.config.concurrency
        )

        predictions = []
        for idx, result in enumerate(results):
            try:
                if "error" in result:
                    log_experiment_milestone(f"Sample {idx} evaluation failed: {result['error']}")
                    predictions.append(float('nan'))
                else:
                    dim_data = result.get(dimension)
                    if dim_data and isinstance(dim_data, dict):
                        score = dim_data.get('score', float('nan'))
                    else:
                        score = float('nan')
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
            'judge_files': self.config.judges.file_paths,
            'judges': self.judge_ids,
            'n_judges': len(self.judge_ids),
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

    def save_results_for_dimension(
        self,
        dimension: str,
        dim_dir: Path,
        gam_results: Dict[str, Any],
        baseline_results: Dict[str, Dict[str, float]],
        data: Dict[str, Any]
    ):
        """Save dimension-specific experiment results to JSON.

        Args:
            dimension: Dimension name (e.g., "helpfulness")
            dim_dir: Dimension-specific results directory
            gam_results: GAM model results
            baseline_results: Baseline model results
            data: Training data dict (includes judge_names, judge_ids)
        """
        log_experiment_milestone(f"Saving results for {dimension}")

        summary = {
            'experiment_name': self.config.name,
            'dimension': dimension,
            'dataset': self.config.dataset,
            'target': self.config.target,
            'judge_files': self.config.judges.file_paths,
            'judges': data.get('judge_ids', []),  # Dimension-specific judges
            'n_judges': len(data.get('judge_ids', [])),
            'random_seed': self.config.random_seed,
            'gam_results': {
                'train': gam_results.get('train_metrics', {}),
                'val': gam_results.get('val_metrics', {}),
                'test': gam_results.get('test_metrics', {})
            },
            'baseline_results': baseline_results,
            'run_dir': str(self.run_dir),
            'dimension_dir': str(dim_dir),
            'timestamp': datetime.now().isoformat()
        }

        # Save dimension-specific summary
        summary_path = dim_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Also save GAM model to dimension directory
        if gam_results and 'model' in gam_results:
            import joblib
            model_path = dim_dir / "models" / "gam_model.pkl"
            joblib.dump(gam_results['model'], model_path)
            log_experiment_milestone(f"  Saved GAM model to {model_path}")

        log_experiment_milestone(f"Results for {dimension} saved to {summary_path}")

    async def run(self):
        """Run complete experiment pipeline with multi-dimension support."""
        try:
            # ========== SHARED STEPS (1-3): Run once for ALL dimensions ==========
            print(f"🚀 Starting experiment with {len(self.config.target_dimensions)} dimensions: {self.config.target_dimensions}\n")

            # 1. Load dataset
            df = self.load_dataset()

            # 2. Run persona simulation if needed
            df = await self.run_persona_simulation_if_needed(df)

            # 3. Create dataset-specific judges if needed + evaluate
            self.create_judges_if_needed(df)
            df = self.get_judged_data(df)

            # ========== PER-DIMENSION STEPS (4-9): Loop over dimensions ==========
            for dimension in self.config.target_dimensions:
                print(f"\n{'='*80}")
                print(f"📊 Training models for dimension: {dimension}")
                print(f"{'='*80}\n")

                # Create dimension subdirectory inside dimensions/ folder
                dim_dir = self.run_dir / "dimensions" / dimension
                dim_dir.mkdir(parents=True, exist_ok=True)
                (dim_dir / "models").mkdir(exist_ok=True)
                (dim_dir / "plots").mkdir(exist_ok=True)

                # 4. Prepare training data for this dimension
                data = self.prepare_training_data(df, dimension)

                # 5. OPTIONAL: Tune hyperparameters (saves to dimension directory)
                optimal_params = self.tune_hyperparameters(data, output_dir=dim_dir)

                # 6. Train GAM with optimal or default params
                gam_results = self.train_gam(data, optimal_params.get('gam'))

                # 7. Run human-rubric evaluation on test set and baselines
                human_rubric_predictions = self.run_human_rubric_evaluation(
                    df, data['test_indices_in_full'], dimension
                )
                baseline_results = self.run_baselines(data, human_rubric_predictions)

                # 8. Save results (dimension-specific)
                self.save_results_for_dimension(dimension, dim_dir, gam_results, baseline_results, data)

                # 9. Generate visualizations (dimension-specific)
                if self.config.models.train_gam and gam_results:
                    from pipeline.core.visualization import ExperimentVisualizer

                    log_experiment_milestone(f"Generating visualizations for {dimension}")

                    visualizer = ExperimentVisualizer(
                        run_dir=dim_dir,  # Dimension-specific directory
                        config=self.config,
                        gam_model=gam_results.get('model'),
                        gam_results=gam_results,
                        baseline_results=baseline_results,
                        judge_names=data['judge_names'],  # Dimension-specific judges
                        dimension_name=dimension,  # Pass dimension name for plot filenames
                        data=data
                    )

                    plot_paths = visualizer.generate_all_plots()
                    log_experiment_milestone(f"Generated {len(plot_paths)} visualizations for {dimension}")

                print(f"\n✅ Completed dimension: {dimension}")

            log_experiment_complete({
                'status': 'SUCCESS',
                'run_dir': str(self.run_dir),
                'dimensions': self.config.target_dimensions
            })

            print(f"\n🎉 All dimensions complete! Results in {self.run_dir}")

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

    # Run experiment (handles multiple dimensions internally)
    runner = ExperimentRunner(config)
    asyncio.run(runner.run())

    return 0


if __name__ == "__main__":
    exit(main())
