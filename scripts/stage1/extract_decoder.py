#!/usr/bin/env python3
"""
Extract RAE decoder weights and save to a path.

Usage examples:
  # Just instantiate from config and save (no weights loaded)
  python extract_decoder.py --config configs/stage1.yaml --out decoder_init.pt

  # Load a stage-1 checkpoint and save EMA decoder
  python extract_decoder.py --config configs/stage1.yaml --ckpt ckpts/exp/ep-0000004.pt --use-ema --out decoder_ema.pt

  # Load a stage-1 checkpoint and save training-model decoder
  python extract_decoder.py --config configs/stage1.yaml --ckpt ckpts/exp/ep-0000004.pt --out decoder_model.pt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to path for imports (scripts/stage1 -> scripts -> project root -> src)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import argparse
from typing import Any, Dict, Tuple

import torch
from omegaconf import OmegaConf

# Repo utilities (as used in your stage-1 training script)
from utils.model_utils import instantiate_from_config


def _strip_prefix(key: str, prefixes: Tuple[str, ...]) -> str:
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Handle common wrappers:
      - DDP:      "module."
      - compile:  "_orig_mod."
    Sometimes combined: "module._orig_mod."
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        k2 = k
        # strip repeatedly in case of nested prefixes
        changed = True
        while changed:
            old = k2
            k2 = _strip_prefix(k2, ("module.",))
            k2 = _strip_prefix(k2, ("_orig_mod.",))
            changed = (k2 != old)
        out[k2] = v
    return out


def _load_checkpoint(path: str) -> Any:
    return torch.load(path, map_location="cpu")


def _select_model_state(ckpt_obj: Any, use_ema: bool) -> Dict[str, torch.Tensor]:
    """
    Supports:
      - training checkpoints: {"model": ..., "ema": ..., ...}
      - raw state_dict checkpoints: {param_name: tensor, ...}
    """
    if isinstance(ckpt_obj, dict) and ("model" in ckpt_obj or "ema" in ckpt_obj):
        if use_ema:
            if "ema" not in ckpt_obj:
                raise KeyError("Checkpoint has no 'ema' key. Remove --use-ema or use a different checkpoint.")
            sd = ckpt_obj["ema"]
        else:
            if "model" not in ckpt_obj:
                raise KeyError("Checkpoint has no 'model' key. Use --use-ema or use a different checkpoint.")
            sd = ckpt_obj["model"]
        if not isinstance(sd, dict):
            raise TypeError("Checkpoint 'model'/'ema' entry is not a state_dict (dict).")
        return sd
    if isinstance(ckpt_obj, dict):
        # assume raw state_dict
        return ckpt_obj
    raise TypeError(f"Unrecognized checkpoint format: {type(ckpt_obj)}")


def _get_rae_config(full_cfg: Any) -> Any:
    """
    Best-effort to mirror your training script:
      full_cfg = OmegaConf.load(...)
      (rae_config, *_) = parse_configs(full_cfg)
    If parse_configs is not importable, we fall back to common keys.
    """
    try:
        from utils.train_utils import parse_configs  # used in your stage-1 script
        rae_config, *_ = parse_configs(full_cfg)
        return rae_config
    except Exception:
        # Fallback heuristics (keep this conservative)
        for k in ("stage_1", "stage1", "rae", "model"):
            if k in full_cfg:
                return full_cfg[k]
        raise KeyError(
            "Could not find RAE config. Expected utils.train_utils.parse_configs(full_cfg) to work, "
            "or one of top-level keys: stage_1/stage1/rae/model."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config used to instantiate RAE.")
    ap.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint to load (ep-*.pt or raw state_dict).")
    ap.add_argument("--use-ema", action="store_true", help="If set, load 'ema' from checkpoint (else 'model').")
    ap.add_argument("--out", type=str, required=True, help="Output path to save decoder weights.")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Cast decoder weights before saving.")
    args = ap.parse_args()

    full_cfg = OmegaConf.load(args.config)
    rae_cfg = _get_rae_config(full_cfg)

    # Instantiate
    rae = instantiate_from_config(rae_cfg)

    # Load weights if provided
    load_report = ""
    if args.ckpt is not None:
        ckpt_obj = _load_checkpoint(args.ckpt)
        model_sd = _select_model_state(ckpt_obj, use_ema=args.use_ema)
        model_sd = _normalize_state_dict_keys(model_sd)

        missing, unexpected = rae.load_state_dict(model_sd, strict=False)
        load_report = (
            f"Loaded checkpoint: {args.ckpt}\n"
            f"  use_ema={args.use_ema}\n"
            f"  missing_keys={len(missing)} unexpected_keys={len(unexpected)}\n"
        )

    # Extract decoder
    decoder = rae.decoder
    dec_sd = decoder.state_dict()

    # Optional cast for storage
    if args.dtype != "fp32":
        tgt = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        dec_sd = {k: (v.to(dtype=tgt) if torch.is_floating_point(v) else v) for k, v in dec_sd.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)


    torch.save(dec_sd, str(out_path))

    print(load_report.rstrip())
    print(f"Saved decoder to: {out_path}")
    print(f"Keys: {len(dec_sd)}")


if __name__ == "__main__":
    main()
