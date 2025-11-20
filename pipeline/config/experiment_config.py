"""Experiment configuration classes for multi-judge interpretability framework.

This module provides configuration dataclasses that define experiment parameters
including judges, models, and training settings. Configs can be created programmatically
or loaded from YAML files.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class JudgeConfig:
    """Configuration for judge selection and scoring.

    Attributes:
        judge_file: Path to single YAML file with judge definitions
        judge_files: List of paths to YAML files (for multi-depth experiments)
        score_range: Score range for judges (default: 0.0-4.0)
    """
    judge_file: Optional[str] = None
    judge_files: Optional[List[str]] = None
    score_range: Tuple[float, float] = (0.0, 4.0)

    def __post_init__(self):
        """Validate judge configuration."""
        # Must specify either judge_file or judge_files (or neither for auto-detection)
        if self.judge_file is not None and self.judge_files is not None:
            raise ValueError("Cannot specify both 'judge_file' and 'judge_files' - choose one")

        if self.score_range[0] >= self.score_range[1]:
            raise ValueError(f"Invalid score_range: {self.score_range}")

    @property
    def file_paths(self) -> List[str]:
        """Get list of judge file paths to load.

        Returns:
            List of file paths (empty if no files specified)
        """
        if self.judge_file:
            return [self.judge_file]
        elif self.judge_files:
            return self.judge_files
        else:
            return []

    @property
    def has_files(self) -> bool:
        """Check if any judge files are specified."""
        return bool(self.file_paths)


@dataclass
class GAMConfig:
    """Hyperparameters for GAM (Generalized Additive Model).

    Attributes:
        n_splines: Number of splines for each feature
        lam: Lambda regularization parameter
        max_iter: Maximum iterations for fitting
    """
    n_splines: int = 10
    lam: float = 0.6
    max_iter: int = 100


@dataclass
class MLPConfig:
    """Hyperparameters for MLP (Multi-Layer Perceptron).

    Attributes:
        hidden_dim: Hidden layer dimension
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        n_epochs: Maximum number of training epochs
        dropout: Dropout probability (0.0 = no dropout)
        l2_reg: L2 regularization strength (0.0 = no regularization)
        early_stopping_patience: Epochs to wait before stopping if no improvement
        min_delta: Minimum change to qualify as improvement
    """
    hidden_dim: int = 64
    learning_rate: float = 0.005
    batch_size: int = 16
    n_epochs: int = 100
    dropout: float = 0.0
    l2_reg: float = 0.0
    early_stopping_patience: int = 15
    min_delta: float = 1e-4


@dataclass
class ModelConfig:
    """Configuration for aggregation models.

    Attributes:
        gam: GAM hyperparameters
        mlp: MLP hyperparameters
        train_gam: Whether to train GAM model
        train_mlp: Whether to train MLP model
        test_size: Fraction of data for test set
        val_size: Fraction of training data for validation set
    """
    gam: GAMConfig = field(default_factory=GAMConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    train_gam: bool = True
    train_mlp: bool = True
    test_size: float = 0.2
    val_size: float = 0.15  # Of remaining training data


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.

    Attributes:
        name: Experiment name (for logging/saving)
        dataset: Dataset name ('ultrafeedback', 'judge_bench', 'helpsteer2', etc.)
        target: Target field for training ('target_human_aggregated', 'target_human_individual', 'target_synthetic')
        target_dimension: REQUIRED dimension name to predict (e.g., 'helpfulness', 'overall')
        dataset_kwargs: Additional arguments for dataset loader
        judges: Judge configuration
        models: Model configuration and hyperparameters
        concurrency: Max concurrent API calls for judge evaluation and persona simulation
        random_seed: Random seed for reproducibility
        target_dimensions: Optional list of dimensions for automatic judge creation
        judge_cache_strategy: Strategy for judge caching ('auto', 'force_create', 'load_only')
        judge_decomposition_depth: Depth of judge decomposition (0 = no decomposition, parents only)
        judge_creation_config: Optional dict with judge creation settings (model, temperature, max_tokens)
        judge_model: Model to use for judge evaluation (default: 'gpt-5-mini')

    Note:
        Persona simulation runs automatically when target='target_synthetic' and the dataset
        has no synthetic annotations yet.
    """
    name: str
    dataset: str  # 'ultrafeedback', 'judge_bench', 'helpsteer2', etc.
    target: str  # 'target_human_aggregated', 'target_human_individual', 'target_synthetic'
    target_dimension: str  # Specific dimension name (e.g., 'helpfulness', 'overall')
    judges: JudgeConfig
    models: ModelConfig = field(default_factory=ModelConfig)
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    concurrency: int = 1  # Conservative default for API rate limiting
    random_seed: int = 42

    # Judge creation fields
    target_dimensions: Optional[List[str]] = None
    judge_cache_strategy: str = 'auto'  # 'auto' | 'force_create' | 'load_only'
    judge_decomposition_depth: int = 1  # 0 = no decomposition (parent only), 1 = one level of children, etc.
    judge_creation_config: Optional[Dict[str, Any]] = None

    # Judge evaluation model (default: gpt-5-mini)
    judge_model: str = 'gpt-5-mini'

    def __post_init__(self):
        """Validate experiment configuration."""
        valid_targets = ['target_human_aggregated', 'target_human_individual', 'target_synthetic']
        if self.target not in valid_targets:
            raise ValueError(f"target must be one of {valid_targets}, got: {self.target}")

        # Validate judge cache strategy
        valid_cache_strategies = ['auto', 'force_create', 'load_only']
        if self.judge_cache_strategy not in valid_cache_strategies:
            raise ValueError(
                f"judge_cache_strategy must be one of {valid_cache_strategies}, "
                f"got: {self.judge_cache_strategy}"
            )

        # Validate judge decomposition depth
        if self.judge_decomposition_depth < 0:
            raise ValueError(
                f"judge_decomposition_depth must be >= 0, got: {self.judge_decomposition_depth}"
            )

    @property
    def needs_persona_simulation(self) -> bool:
        """Check if persona simulation should run based on target field.

        Returns:
            True if target is 'target_synthetic', False otherwise
        """
        return self.target == 'target_synthetic'

    def validate_with_data(self, df) -> None:
        """Validate configuration against loaded dataset.

        Args:
            df: Loaded DataFrame with standardized format

        Raises:
            ValueError: If configuration is invalid for the dataset
        """
        import pandas as pd

        # Check dataset has enough samples
        n_requested = self.dataset_kwargs.get('n_samples')
        if n_requested and n_requested > len(df):
            logger.warning(
                f"Requested {n_requested} samples but dataset only has {len(df)}. "
                f"Using all {len(df)} samples."
            )

        # Check target dimension exists in dataset
        first_row = df.iloc[0]
        dimensions = first_row.get('dimensions', [])

        if not dimensions:
            raise ValueError(
                f"Dataset '{self.dataset}' has no dimensions field. "
                f"Cannot use target_dimension."
            )

        if self.target_dimension not in dimensions:
            raise ValueError(
                f"target_dimension '{self.target_dimension}' not found in dataset dimensions: {dimensions}. "
                f"Available dimensions: {', '.join(dimensions)}"
            )

        logger.info(f"✓ Using target dimension: {self.target_dimension}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved config to {path}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary.

        Supports both old and new config structures:
        - Old: target_dimensions, judge_cache_strategy, judge_decomposition_depth at top level
        - New: judges.create.{dimensions, cache, depth, model, temperature}

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            ExperimentConfig instance
        """
        judges_dict = config_dict['judges']

        # Check if new simplified structure is used
        if 'create' in judges_dict:
            # New structure: judges.create.{cache, depth, model, temperature}
            # Uses target_dimension to determine what judges to create
            create_config = judges_dict['create']
            # Use target_dimension from top level (single dimension per experiment)
            target_dimensions = [config_dict['target_dimension']]
            judge_cache_strategy = create_config.get('cache', 'auto')
            judge_decomposition_depth = create_config.get('depth', 1)
            judge_creation_config = {
                'model': create_config.get('model', 'openai/gpt-5.1'),
                'temperature': create_config.get('temperature', 0.4),
                'max_tokens': create_config.get('max_tokens', 10000)
            }
        else:
            # Old structure: top-level target_dimensions, judge_cache_strategy, etc.
            target_dimensions = config_dict.get('target_dimensions')
            judge_cache_strategy = config_dict.get('judge_cache_strategy', 'auto')
            judge_decomposition_depth = config_dict.get('judge_decomposition_depth', 1)
            judge_creation_config = config_dict.get('judge_creation_config')

        # Handle judge_model (can be in judges.judge_model or top-level)
        judge_model = judges_dict.get('judge_model') or config_dict.get('judge_model', 'gpt-5-mini')

        # Handle use_files (new) vs judge_file/judge_files (old)
        if 'use_files' in judges_dict:
            # New structure: judges.use_files
            use_files = judges_dict['use_files']
            if isinstance(use_files, list):
                judge_config_dict = {'judge_files': use_files}
            else:
                judge_config_dict = {'judge_file': use_files}
        else:
            # Old structure: judges.judge_file or judges.judge_files
            judge_config_dict = {}
            if 'judge_file' in judges_dict:
                judge_config_dict['judge_file'] = judges_dict['judge_file']
            if 'judge_files' in judges_dict:
                judge_config_dict['judge_files'] = judges_dict['judge_files']

        # Add score_range
        judge_config_dict['score_range'] = tuple(judges_dict.get('score_range', [0.0, 4.0]))

        judges = JudgeConfig(**judge_config_dict)

        # Parse model configs with defaults
        models_dict = config_dict.get('models', {})
        gam_dict = models_dict.get('gam', {})
        mlp_dict = models_dict.get('mlp', {})

        models = ModelConfig(
            gam=GAMConfig(**gam_dict),
            mlp=MLPConfig(**mlp_dict),
            train_gam=models_dict.get('train_gam', True),
            train_mlp=models_dict.get('train_mlp', True),
            test_size=models_dict.get('test_size', 0.2),
            val_size=models_dict.get('val_size', 0.15)
        )

        return cls(
            name=config_dict['name'],
            dataset=config_dict['dataset'],
            target=config_dict['target'],
            target_dimension=config_dict['target_dimension'],
            judges=judges,
            models=models,
            dataset_kwargs=config_dict.get('dataset_kwargs', {}),
            concurrency=config_dict.get('concurrency', 1),
            random_seed=config_dict.get('random_seed', 42),
            # Judge creation fields (resolved from new or old structure)
            target_dimensions=target_dimensions,
            judge_cache_strategy=judge_cache_strategy,
            judge_decomposition_depth=judge_decomposition_depth,
            judge_creation_config=judge_creation_config,
            judge_model=judge_model
        )

    @classmethod
    def from_yaml(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            ExperimentConfig instance
        """
        logger.info(f"Loading config from {path}")
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


# Default configurations for common use cases

DEFAULT_10_JUDGES = JudgeConfig(
    judge_file="judges/ultrafeedback/depth_0_parents.yaml",  # Generic 10 baseline judges
    score_range=(0.0, 4.0)
)


def create_default_config(
    name: str,
    target_dimension: str,
    dataset: str = 'ultrafeedback',
    target: str = 'target_synthetic',
    n_samples: Optional[int] = None,
    judge_config: Optional[JudgeConfig] = None,
    concurrency: int = 1
) -> ExperimentConfig:
    """Create a default experiment configuration.

    Args:
        name: Experiment name
        target_dimension: REQUIRED dimension name (e.g., 'helpfulness', 'overall')
        dataset: Dataset to use
        target: Target field ('target_human_aggregated', 'target_human_individual', 'target_synthetic')
        n_samples: Number of samples (None = all)
        judge_config: Judge configuration (None = use all 10 judges)
        concurrency: Max concurrent API calls (default: 1 for rate limiting)

    Returns:
        ExperimentConfig with sensible defaults
    """
    judges = judge_config or DEFAULT_10_JUDGES

    dataset_kwargs = {}
    if n_samples is not None:
        dataset_kwargs['n_samples'] = n_samples

    return ExperimentConfig(
        name=name,
        dataset=dataset,
        target=target,
        target_dimension=target_dimension,
        judges=judges,
        dataset_kwargs=dataset_kwargs,
        concurrency=concurrency
    )
