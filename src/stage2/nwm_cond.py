"""NWM-specific conditioning helpers.

All dict-handling logic for the Navigation World Model lives here so that
src/stage2/utils.py, src/stage2/engine.py, and src/eval/generation.py keep
their label/text paths identical to main and only need a single dispatch
line each to delegate to this module.

The training-time context dict has the keys:
    context_latents: (B, K, C, h, w)   -- RAE-encoded past frames
    action:          (B, 3)            -- normalized egocentric (dx, dy, dyaw)
    rel_time:        (B, 1)            -- offset / 128 (raenwm convention)
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def batch_size(ctx: Dict[str, torch.Tensor]) -> int:
    return next(iter(ctx.values())).shape[0]


def slice(ctx: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
    return {k: v[:n] for k, v in ctx.items()}


def clone_context(ctx: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in ctx.items()}


def encode_train_context(y: Dict[str, torch.Tensor], rae, device) -> Dict[str, torch.Tensor]:
    """Encode the dataloader output into the (latents, action, rel_time) dict."""
    with torch.no_grad():
        ctx_frames = y["context_frames"].to(device)
        B, K = ctx_frames.shape[:2]
        ctx_lat = rae.encode(ctx_frames.flatten(0, 1))
        ctx_lat = ctx_lat.reshape(B, K, *ctx_lat.shape[1:])
    return {
        "context_latents": ctx_lat,
        "action": y["action"].to(device),
        "rel_time": y["rel_time"].to(device),
    }


def viz_context(y: Dict[str, torch.Tensor], n: int, rae, device) -> Dict[str, torch.Tensor]:
    """Build the fixed viz context from the first batch (no_grad)."""
    with torch.no_grad():
        ctx = y["context_frames"][:n].to(device)
        B, K = ctx.shape[:2]
        ctx_lat = rae.encode(ctx.flatten(0, 1))
        ctx_lat = ctx_lat.reshape(B, K, *ctx_lat.shape[1:])
    return {
        "context_latents": ctx_lat,
        "action": y["action"][:n].clone().to(device),
        "rel_time": y["rel_time"][:n].clone().to(device),
    }


def null_context(config, batch_size: int, device) -> Dict:
    """Return a zero-valued null cond dict shaped like the real train cond.

    Output matches the get_null_cond contract: {"context": <dict>, "attn_mask": None}.
    """
    ds_params = config.dataset.params or {}
    K = ds_params.get("context_size", 4)
    C = config.stage_2.params["in_channels"]
    H = W = config.stage_2.params["input_size"]
    return {
        "context": {
            "context_latents": torch.zeros(batch_size, K, C, H, W, device=device),
            "action": torch.zeros(batch_size, 3, device=device),
            "rel_time": torch.zeros(batch_size, 1, device=device),
        },
        "attn_mask": None,
    }


def apply_cfg_dropout(model_conds, model_conds_null, cfg_dropout_prob=0.1):
    """Drop the dict-valued context to its null value per-sample (one mask, all keys)."""
    ctx = model_conds["context"]
    any_v = next(iter(ctx.values()))
    mask = torch.rand(any_v.shape[0], device=any_v.device) < cfg_dropout_prob
    null_ctx = model_conds_null["context"]
    dropped_ctx = {
        k: torch.where(mask.view(-1, *([1] * (v.ndim - 1))), null_ctx[k], v)
        for k, v in ctx.items()
    }
    out = {"context": dropped_ctx}
    for k, v in model_conds.items():
        if k == "context":
            continue
        if v is None:
            out[k] = None
        else:
            v_null = model_conds_null[k]
            out[k] = torch.where(mask.view(-1, *([1] * (v.ndim - 1))), v_null, v)
    return out, mask


def encode_eval_context(
    cond: Dict[str, torch.Tensor],
    rae,
    device,
    latent_size,
    use_guidance: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], None, int]:
    """Build z, context dict, attn_mask, n for the eval/generation path."""
    ctx_frames = cond["context_frames"].to(device)
    B, K = ctx_frames.shape[:2]
    ctx_lat = rae.encode(ctx_frames.flatten(0, 1))
    ctx_lat = ctx_lat.reshape(B, K, *ctx_lat.shape[1:])
    n = B
    z = torch.randn(n, *latent_size, device=device)
    context = {
        "context_latents": ctx_lat,
        "action": cond["action"].to(device),
        "rel_time": cond["rel_time"].to(device),
    }
    if use_guidance:
        z = torch.cat([z, z], dim=0)
        null = {k: torch.zeros_like(v) for k, v in context.items()}
        context = {k: torch.cat([context[k], null[k]], dim=0) for k in context}
    return z, context, None, n
