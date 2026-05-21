"""Shared config dataclasses used across all training scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import MISSING


@dataclass
class ModelConfig:
    """Generic model configuration for instantiate_from_config().
    Used for stage_1 (RAE) and stage_2 (DiT) model definitions.
    The params dict is passed as kwargs to the target class constructor.
    """
    target: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    ckpt: Optional[str] = None


@dataclass
class MiscConfig:
    """Miscellaneous model-related parameters."""
    latent_size: List[int] = field(default_factory=lambda: [768, 16, 16])  # [C, H, W]
    num_classes: int = 1000
    time_dist_shift_dim: int = 196608  # 16*16*768
    time_dist_shift_base: int = 4096


@dataclass
class OptimizerConfig:
    """Optimizer configuration (shared across all training)."""
    type: str = "adamw"  # "adamw", "gmuon"
    lr: float = 2.0e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    eps: float = 1e-8
    # GMuon-specific
    momentum: float = 0.95
    nesterov: bool = True
    adamw_lr: Optional[float] = None
    ns_use_kernels: bool = False
    ns_coefficients_preset: str = "POLAR_EXPRESS_COEFFICIENTS"


@dataclass
class SchedulerConfig:
    """LR scheduler configuration."""
    type: str = "cosine"  # "cosine" or "linear"
    warmup_epochs: float = 1.0
    warmup_steps: Optional[int] = None
    warmup_from_zero: bool = True
    decay_end_epoch: float = 16.0
    decay_end_steps: Optional[int] = None
    base_lr: float = 2.0e-4
    final_lr: float = 2.0e-5


@dataclass
class DatasetConfig:
    """Dataset configuration (shared across all training)."""
    target: str = "imagenet"
    type: str = "hf"  # ["hf", "wds"]
    data_dir: str = "./data"
    split: Any = "train"
    condition_type: Optional[str] = None  # "label", "text", or "nwm"
    shared_tmpdir: str = "~/tmp"
    # WDS-specific
    shuffle_buffer: int = 10000
    seed: int = 42
    # Free-form per-task params (e.g., nwm: context_size, len_traj_pred, ...)
    params: Optional[Dict[str, Any]] = None
    # Multi-subset wds (e.g. blip3o splits, or 'root' for flat 256 pools)
    splits: Optional[List[str]] = None
    subsets: Optional[List[str]] = None
    # Multi-source mix: list of nested dataset configs each with a `weight`.
    # Entries are passed verbatim to prepare_unified_dataloader recursively.
    mix: Optional[List[Any]] = None


@dataclass
class EvalConfig:
    """Evaluation configuration.
    eval.datasets.{name}.reference_npz, eval.datasets.{name}.metrics
    """
    eval_interval: int = 5000
    eval_model: bool = False  # Eval non-EMA model too
    eval_dir: str = MISSING  # directory for eval CSVs, e.g. "experiments/<user>/evals/stage2"
    datasets: Optional[Dict[str, Any]] = None


@dataclass
class TrainingConfig:
    """Base training configuration (shared across all)."""
    epochs: int = 16
    batch_size: int = 32
    global_batch_size: Optional[int] = None
    num_workers: int = 4
    global_seed: int = 0
    ema_decay: float = 0.9995
    clip_grad: Optional[float] = None
    log_interval: int = 100
    checkpoint_interval: int = 4
    sample_every: int = 2500
    virtual_epoch_steps: Optional[int] = None
    grad_accum_steps: int = 1
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: Optional[SchedulerConfig] = None
    image_size: int = 256
