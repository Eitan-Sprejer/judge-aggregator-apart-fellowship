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
        judge_ids: List of Martian judge IDs to use
        judge_names: Human-readable names for judges (for plots/reports)
        score_range: Score range for judges (default: 0.0-4.0 for Martian)
    """
    judge_ids: List[str]
    judge_names: List[str]
    score_range: Tuple[float, float] = (0.0, 4.0)

    def __post_init__(self):
        """Validate judge configuration."""
        if len(self.judge_ids) != len(self.judge_names):
            raise ValueError(f"judge_ids and judge_names must have same length "
                           f"(got {len(self.judge_ids)} vs {len(self.judge_names)})")
        if len(self.judge_ids) == 0:
            raise ValueError("Must specify at least one judge")
        if self.score_range[0] >= self.score_range[1]:
            raise ValueError(f"Invalid score_range: {self.score_range}")

    @property
    def n_judges(self) -> int:
        """Number of judges in this configuration."""
        return len(self.judge_ids)


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
        dataset: Dataset name ('ultrafeedback', 'judge_bench', 'maj_eval')
        dataset_kwargs: Additional arguments for dataset loader
        judges: Judge configuration
        models: Model configuration and hyperparameters
        random_seed: Random seed for reproducibility
    """
    name: str
    dataset: str  # 'ultrafeedback', 'judge_bench', 'maj_eval', etc.
    judges: JudgeConfig
    models: ModelConfig = field(default_factory=ModelConfig)
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42

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

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            ExperimentConfig instance
        """
        # Parse nested configs
        judges = JudgeConfig(**config_dict['judges'])

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
            judges=judges,
            models=models,
            dataset_kwargs=config_dict.get('dataset_kwargs', {}),
            random_seed=config_dict.get('random_seed', 42)
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
    judge_ids=[
        "truthfulness-judge",
        "harmlessness-judge",
        "helpfulness-judge",
        "honesty-judge",
        "explanatory-depth-judge",
        "instruction-following-judge",
        "clarity-judge",
        "conciseness-judge",
        "logical-consistency-judge",
        "creativity-judge"
    ],
    judge_names=[
        "Truthfulness",
        "Harmlessness",
        "Helpfulness",
        "Honesty",
        "Explanatory Depth",
        "Instruction Following",
        "Clarity",
        "Conciseness",
        "Logical Consistency",
        "Creativity"
    ],
    score_range=(0.0, 4.0)
)


def create_default_config(
    name: str,
    dataset: str = 'ultrafeedback',
    n_samples: Optional[int] = None,
    judge_config: Optional[JudgeConfig] = None
) -> ExperimentConfig:
    """Create a default experiment configuration.

    Args:
        name: Experiment name
        dataset: Dataset to use
        n_samples: Number of samples (None = all)
        judge_config: Judge configuration (None = use all 10 judges)

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
        judges=judges,
        dataset_kwargs=dataset_kwargs
    )
