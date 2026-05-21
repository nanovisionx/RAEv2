"""
Configuration dataclasses for all training stages.

Configs are plain dataclasses loaded via OmegaConf structured configs:

    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(path)))
    config.post_process()  # Stage2Config only
"""

from .shared import (
    DatasetConfig,
    EvalConfig,
    MiscConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from .stage1 import (
    DiscAugmentConfig,
    DiscriminatorArchConfig,
    GanConfig,
    GanLossConfig,
    Stage1Config,
)
from .stage2 import (
    ConditioningConfig,
    GuidanceConfig,
    RepaConfig,
    SamplerConfig,
    Stage2Config,
    TransportConfig,
)

__all__ = [
    # Shared
    "ModelConfig",
    "MiscConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "DatasetConfig",
    "EvalConfig",
    "TrainingConfig",
    # Stage 1
    "Stage1Config",
    "DiscriminatorArchConfig",
    "DiscAugmentConfig",
    "GanLossConfig",
    "GanConfig",
    # Stage 2
    "Stage2Config",
    "TransportConfig",
    "SamplerConfig",
    "GuidanceConfig",
    "ConditioningConfig",
    "RepaConfig",
]
