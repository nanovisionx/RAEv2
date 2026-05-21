"""Stage 1 (GAN) config dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .shared import DatasetConfig, EvalConfig, ModelConfig, OptimizerConfig, SchedulerConfig, TrainingConfig


@dataclass
class DiscriminatorArchConfig:
    """Discriminator architecture config."""
    dino_ckpt_path: str = "pretrained_models/encoders/dino/dino_vit_small_patch8_224.pth"
    ks: int = 9
    norm_type: str = "bn"
    using_spec_norm: bool = True
    recipe: str = "S_8"


@dataclass
class DiscAugmentConfig:
    """Discriminator augmentation config."""
    prob: float = 1.0
    cutout: float = 0.0


@dataclass
class GanLossConfig:
    """GAN loss configuration."""
    disc_loss: str = "hinge"
    gen_loss: str = "vanilla"
    disc_weight: float = 0.75
    perceptual_weight: float = 1.0
    disc_start: int = 8  # epoch
    disc_upd_start: int = 6  # epoch
    lpips_start: int = 0  # epoch
    max_d_weight: float = 10000.0
    disc_updates: int = 1


@dataclass
class GanConfig:
    """Full GAN configuration for Stage 1."""
    arch: DiscriminatorArchConfig = field(default_factory=DiscriminatorArchConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: Optional[SchedulerConfig] = None
    augment: DiscAugmentConfig = field(default_factory=DiscAugmentConfig)
    loss: GanLossConfig = field(default_factory=GanLossConfig)


@dataclass
class Stage1Config:
    """Top-level configuration for Stage 1 training.

    Combines all configs needed for Stage 1 (GAN-based autoencoder training).
    """
    stage_1: ModelConfig = field(default_factory=ModelConfig)  # RAE model config
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    gan: GanConfig = field(default_factory=GanConfig)
    eval: Optional[EvalConfig] = None
