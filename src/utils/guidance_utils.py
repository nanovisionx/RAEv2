"""Guidance utilities for eval-time model forward selection.

Supports CFG (classifier-free guidance) and IG (internal guidance).
GuidanceConfig lives in configs.stage2; this module provides
get_model_forward_fn() which selects the right forward method.
"""
from functools import partial

import torch

from configs.stage2 import GuidanceConfig


def forward_with_cfg(model, x, t, cfg_scale, cfg_interval=(0, 1), **condition_kwargs):
    """Forward pass with classifier-free guidance."""
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    model_out = model(combined, t, **condition_kwargs)
    if isinstance(model_out, tuple):
        # IG models return (full, base) tuple
        model_out = model_out[0]
    eps, rest = model_out[:, :model.in_channels], model_out[:, model.in_channels:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    guid_t_min, guid_t_max = cfg_interval
    assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
    t = t[: len(t) // 2]
    half_eps = torch.where(
        ((t >= guid_t_min) & (t <= guid_t_max)).view(-1, *[1] * (len(cond_eps.shape) - 1)),
        uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps
    )
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)


def slice_context_kwargs(condition_kwargs, batch_size):
    half_kwargs = {}
    for key in condition_kwargs.keys():
        if condition_kwargs[key] is not None:
            half_kwargs[key] = condition_kwargs[key][:batch_size]
    return half_kwargs


def forward_with_internalguidance(model, x, t, ig_scale, ig_interval=(0, 1), **condition_kwargs):
    """Pure IG forward. Math: ig_output = base + ig_scale * (full - base)"""
    half = x[: len(x) // 2]
    t_half = t[: len(t) // 2]
    half_context_kwargs = slice_context_kwargs(condition_kwargs, half.shape[0])
    full_out, base_out = model(half, t_half, **half_context_kwargs)
    eps_full = full_out[:, :model.in_channels]
    eps_base = base_out[:, :model.in_channels]
    ig_t_min, ig_t_max = ig_interval
    assert ig_t_min < ig_t_max, "ig_interval should be (min, max) with min < max"
    ig_out = torch.where(
        ((t_half >= ig_t_min) & (t_half <= ig_t_max)).view(-1, *[1] * (eps_full.ndim - 1)),
        eps_base + ig_scale * (eps_full - eps_base),
        eps_full
    )
    return torch.cat([ig_out, ig_out], dim=0)


def forward_with_ig_and_cfg(
    model, x, t, ig_scale, cfg_scale, ig_interval=(0, 1), cfg_interval=(0, 1),
    uncond_ig_scale=None, **condition_kwargs
):
    """Combined IG + CFG. Expects doubled batch [cond, uncond].

    Args:
        uncond_ig_scale: IG scale for unconditional branch. Defaults to ig_scale.
    """
    uncond_ig_scale = ig_scale if uncond_ig_scale is None else uncond_ig_scale

    full_out, base_out = model(x, t, **condition_kwargs)

    eps_full = full_out[:, :model.in_channels]
    eps_base = base_out[:, :model.in_channels]

    full_c, full_u = eps_full.chunk(2, dim=0)
    base_c, base_u = eps_base.chunk(2, dim=0)
    t_half = t[: len(t) // 2]

    # Apply IG to cond/uncond branches
    ig_t_min, ig_t_max = ig_interval
    assert ig_t_min < ig_t_max, "ig_interval should be (min, max) with min < max"
    ig_cond = torch.where(
        ((t_half >= ig_t_min) & (t_half <= ig_t_max)).view(-1, *[1] * (full_c.ndim - 1)),
        base_c + ig_scale * (full_c - base_c),
        full_c
    )
    ig_uncond = torch.where(
        ((t_half >= ig_t_min) & (t_half <= ig_t_max)).view(-1, *[1] * (full_u.ndim - 1)),
        base_u + uncond_ig_scale * (full_u - base_u),
        full_u
    )

    # Apply CFG
    cfg_t_min, cfg_t_max = cfg_interval
    assert cfg_t_min < cfg_t_max, "cfg_interval should be (min, max) with min < max"
    out = torch.where(
        ((t_half >= cfg_t_min) & (t_half <= cfg_t_max)).view(-1, *[1] * (ig_cond.ndim - 1)),
        ig_uncond + cfg_scale * (ig_cond - ig_uncond),
        ig_cond
    )
    return torch.cat([out, out], dim=0)


def get_model_forward_fn(model, guid_cfg: GuidanceConfig):
    """Get the appropriate model forward function based on guidance config.

    Args:
        model: The stage2 model
        guid_cfg: Parsed guidance configuration

    Returns:
        Tuple of (model_fn, sample_kwargs)
    """
    if guid_cfg.use_ig and guid_cfg.use_cfg:
        # Combined IG + CFG
        model_fn = partial(forward_with_ig_and_cfg, model)
        sample_kwargs = dict(
            ig_scale=guid_cfg.ig.scale,
            cfg_scale=guid_cfg.cfg.scale,
            ig_interval=(guid_cfg.ig.t_min, guid_cfg.ig.t_max),
            cfg_interval=(guid_cfg.cfg.t_min, guid_cfg.cfg.t_max),
            uncond_ig_scale=guid_cfg.ig.unconditional_scale,
        )
    elif guid_cfg.use_ig:
        # IG only
        model_fn = partial(forward_with_internalguidance, model)
        sample_kwargs = dict(
            ig_scale=guid_cfg.ig.scale,
            ig_interval=(guid_cfg.ig.t_min, guid_cfg.ig.t_max),
        )
    elif guid_cfg.use_cfg:
        # CFG only
        model_fn = partial(forward_with_cfg, model)
        sample_kwargs = dict(
            cfg_scale=guid_cfg.cfg.scale,
            cfg_interval=(guid_cfg.cfg.t_min, guid_cfg.cfg.t_max),
        )
    else:
        # No guidance
        model_fn = model.forward
        sample_kwargs = dict()

    return model_fn, sample_kwargs
