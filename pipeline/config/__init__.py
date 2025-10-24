"""Configuration module for multi-judge interpretability experiments."""

from .experiment_config import (
    JudgeConfig,
    ModelConfig,
    GAMConfig,
    MLPConfig,
    ExperimentConfig,
    DEFAULT_10_JUDGES
)

__all__ = [
    'JudgeConfig',
    'ModelConfig',
    'GAMConfig',
    'MLPConfig',
    'ExperimentConfig',
    'DEFAULT_10_JUDGES'
]
