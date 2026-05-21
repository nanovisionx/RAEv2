"""Optimizer and scheduler utilities using typed configs."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from configs import OptimizerConfig, SchedulerConfig


class MuonAdamW(Optimizer):
    """Composite optimizer: Muon for 2D params, AdamW for the rest."""

    def __init__(self, muon_opt: Optimizer, adamw_opt: Optimizer):
        self._muon = muon_opt
        self._adamw = adamw_opt
        self.param_groups = muon_opt.param_groups + adamw_opt.param_groups
        self.defaults: Dict[str, Any] = {}

    @property
    def state(self) -> Dict:
        merged: Dict = {}
        merged.update(self._muon.state)
        merged.update(self._adamw.state)
        return merged

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        self._muon.step(closure=closure)
        self._adamw.step(closure=closure)

    def state_dict(self) -> Dict[str, Any]:
        return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._muon.load_state_dict(state_dict["muon"])
        self._adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = self._muon.param_groups + self._adamw.param_groups


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    config: OptimizerConfig,
) -> tuple[Optimizer, str]:
    """Build optimizer from typed OptimizerConfig."""
    if config.type == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
            eps=config.eps,
            fused=True,
        )
        msg = f"AdamW(lr={config.lr}, betas={config.betas}, wd={config.weight_decay})"

    elif config.type == "gmuon":
        from gram_newton_schulz import Muon as GMuon

        params_list = list(parameters)
        muon_params = [p for p in params_list if p.ndim == 2]
        fallback_params = [p for p in params_list if p.ndim != 2]

        adamw_opt = torch.optim.AdamW(
            fallback_params if fallback_params else [torch.nn.Parameter(torch.empty(0))],
            lr=config.adamw_lr if config.adamw_lr is not None else config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
            eps=config.eps,
        )

        gmuon_opt = GMuon(
            muon_params,
            lr=config.lr,
            momentum=config.momentum,
            nesterov=config.nesterov,
            weight_decay=config.weight_decay,
            ns_coefficients_preset=config.ns_coefficients_preset,
            ns_use_kernels=config.ns_use_kernels,
            adjust_lr="rms_norm",
        )

        optimizer = MuonAdamW(gmuon_opt, adamw_opt)
        msg = (f"GMuon(lr={config.lr}, momentum={config.momentum}, "
               f"preset={config.ns_coefficients_preset}, kernels={config.ns_use_kernels}, "
               f"{len(muon_params)} 2D params, {len(fallback_params)} fallback)")

    else:
        raise ValueError(f"Unsupported optimizer '{config.type}'. Choose from ['adamw', 'gmuon'].")

    return optimizer, msg


def build_scheduler(
    optimizer: Optimizer,
    steps_per_epoch: int,
    config: SchedulerConfig,
    state_dict: Optional[Dict[str, Any]] = None,
) -> tuple[LambdaLR, str]:
    """Build LR scheduler from typed SchedulerConfig."""
    # Compute steps from epochs or use direct step values
    if config.warmup_steps is not None:
        warmup_steps = config.warmup_steps
    else:
        warmup_steps = int(config.warmup_epochs * steps_per_epoch)

    if config.decay_end_steps is not None:
        decay_end_steps = config.decay_end_steps
    else:
        decay_end_steps = int(config.decay_end_epoch * steps_per_epoch)

    warmup_steps = max(warmup_steps, 0)
    decay_end_steps = max(decay_end_steps, warmup_steps)
    total_decay_steps = max(decay_end_steps - warmup_steps, 1)

    base_lr = config.base_lr
    final_lr = config.final_lr
    final_ratio = final_lr / base_lr if base_lr > 0 else 1.0
    warmup_from_zero = config.warmup_from_zero

    # Set optimizer LR to base_lr
    for group in optimizer.param_groups:
        if group.get('name') not in ('encoder', 'decoder'):
            group["lr"] = base_lr

    if config.type == "linear":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps if warmup_from_zero else 1.0
            if step >= decay_end_steps:
                return final_ratio
            progress = (step - warmup_steps) / total_decay_steps
            return 1.0 - (1.0 - final_ratio) * progress

    elif config.type == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps if warmup_from_zero else 1.0
            if step >= decay_end_steps:
                return final_ratio
            progress = (step - warmup_steps) / total_decay_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_ratio + (1.0 - final_ratio) * cosine

    else:
        raise ValueError(f"Unsupported scheduler '{config.type}'. Choose from ['linear', 'cosine'].")

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    if state_dict is not None:
        scheduler.load_state_dict(state_dict)

    msg = f"{config.type}(warmup={warmup_steps}, decay_end={decay_end_steps}, lr={base_lr}->{final_lr})"
    return scheduler, msg
