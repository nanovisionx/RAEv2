"""
Stage 2 (diffusion/flow matching) config dataclasses.

Contains:
- TransportConfig: Flow matching transport settings
- SamplerConfig: ODE/SDE sampler settings
- GuidanceConfig: Eval-time guidance settings (CFG and/or IG)
- RepaConfig: REPA loss settings
- ConditioningConfig: Conditioning settings (label vs text)
- Stage2Config: Top-level config for Stage 2 training
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .shared import DatasetConfig, EvalConfig, MiscConfig, ModelConfig, TrainingConfig


@dataclass
class TransportConfig:
    """Transport configuration for flow matching."""
    prediction: str = "velocity"  # "velocity" or "x"
    time_dist_type: str = "logit-normal_0_1"
    t_eps: float = 0.05


@dataclass
class SamplerConfig:
    """Sampler configuration for ODE Euler flow matching."""
    num_steps: int = 50


@dataclass
class CFGConfig:
    """CFG configuration for test-time guidance."""
    scale: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0


@dataclass
class IGConfig:
    """IG configuration for test-time guidance."""
    scale: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    unconditional_scale: Optional[float] = None


@dataclass
class GuidanceConfig:
    """Guidance configuration for test-time guidance."""
    cfg: Optional[CFGConfig] = field(default_factory=CFGConfig)
    ig: Optional[IGConfig] = field(default_factory=IGConfig)

    @property
    def use_cfg(self):
        return self.cfg is not None and self.cfg.scale > 1.0

    @property
    def use_ig(self):
        return self.ig is not None and (
            self.ig.scale > 1.0
            or (self.ig.unconditional_scale is not None and self.ig.unconditional_scale != 1.0)
        )

    @property
    def any_guidance_active(self):
        return self.use_cfg or self.use_ig


@dataclass
class RepaConfig:
    """REPA loss configuration with multi-layer support."""
    use_repa: bool = False
    repa_layer_depth: int = 8
    repa_coeff: float = 0.5
    target_encoder: str = "dinov2-vit-b"
    target_encoder_resolution: int = 256
    z_dim: Optional[int] = None  # initialized later in train.py


@dataclass
class ConditioningArchConfig:
    """In-context conditioning architecture configuration."""
    num_t_tokens: int = 4
    num_c_tokens: int = 8
    num_cfg_omega_tokens: int = 4
    n_action_tokens: int = 4  # nwm only: action embedding tokens (rel_time adds 1 more)


@dataclass
class TextEncoderConfig:
    """Text encoder configuration."""
    model_name: str = "Qwen/Qwen3-0.6B"
    max_length: int = 256


@dataclass
class ConditioningConfig:
    """Conditioning configuration for ImageNet and T2I"""
    type: str = "label"
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    cfg_dropout_prob: float = 0.1
    context_dim: Optional[int] = None  # initialized later in train.py
    arch: ConditioningArchConfig = field(default_factory=ConditioningArchConfig)


@dataclass
class InternalGuidanceConfig:
    """Internal guidance training-related configuration."""
    base_model_depth: Optional[int] = None
    base_model_coeff: float = 1.0


@dataclass
class Stage2Config:
    """Top-level configuration for Stage 2 training."""
    stage_1: ModelConfig = field(default_factory=ModelConfig)
    stage_2: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    repa: RepaConfig = field(default_factory=RepaConfig)
    misc: MiscConfig = field(default_factory=MiscConfig)
    internal_guidance: InternalGuidanceConfig = field(default_factory=InternalGuidanceConfig)
    eval: Optional[EvalConfig] = None

    def post_process(self):
        """Post-process the config to set certain runtime fields."""
        if self.conditioning.type == "label" and self.dataset.condition_type is not None:
            self.conditioning.type = self.dataset.condition_type

        if self.conditioning.type == "text":
            self.conditioning.arch.num_c_tokens = self.conditioning.text_encoder.max_length

        if self.conditioning.type == "nwm":
            # num_c_tokens = K * patches_per_frame + n_action_tokens + 1 (rel_time)
            ds_params = self.dataset.params or {}
            K = ds_params.get("context_size", 4)
            input_size = self.stage_2.params.get("input_size")
            patch_size = self.stage_2.params.get("patch_size", [1, 1])
            s_patch_size = patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size
            if input_size is None:
                raise ValueError("conditioning.type='nwm' requires stage_2.params.input_size to be set")
            patches_per_frame = (input_size // s_patch_size) ** 2
            n_action_tokens = self.conditioning.arch.n_action_tokens
            self.conditioning.arch.num_c_tokens = K * patches_per_frame + n_action_tokens + 1

    def prepare_model_params(self):
        """Populate stage_2.params from typed config fields for model construction.

        Call after setting runtime fields (conditioning.context_dim, repa.z_dim)
        and before instantiating the model. Uses setdefault so YAML-specified
        params are never overwritten.
        """
        params = self.stage_2.params

        # Conditioning
        params.setdefault('condition_type', self.conditioning.type)
        params.setdefault('num_classes', self.misc.num_classes)
        params.setdefault('context_dim', self.conditioning.context_dim)

        # REPA
        if self.repa.use_repa:
            params.setdefault('enable_repa', True)
            params.setdefault('repa_layer_depth', self.repa.repa_layer_depth)
            if self.repa.z_dim is not None:
                params.setdefault('z_dim', self.repa.z_dim)

        # Conditioning architecture
        params.setdefault('cond_arch', self.conditioning.arch)

        # Internal guidance
        if self.internal_guidance.base_model_depth is not None:
            params.setdefault('base_model_depth', self.internal_guidance.base_model_depth)
