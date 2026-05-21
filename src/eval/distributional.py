"""Rank-0 wrapper around fd_evaluator.compute_metrics.

Called after distributed generation gathers per-rank shards into a single
combined uint8 NHWC array. Same callsite shape as the existing FID block in
generation.py -- this just delegates to the fd_evaluator package for
gfid/fdr6/mind6/fdr_<ext>/mind_<ext> metric strings.

Defaults follow nanogen conventions:
  - feature cache:           ~/.cache/nanogen-evals/features/
                             (overridable via NANOGEN_EVALS_CACHE_DIR)
  - MIND raw-image reference: <data_dir>/imagenet-256-val.npz
                             (overridable via NANOGEN_EVALS_REF_IMAGES)
  - mu/Sigma stats:           auto-downloaded from
                             huggingface.co/datasets/nanovisionx/nanogen-evals-stats

Yaml schema is unchanged. Add the new metric strings to
`eval.datasets.<name>.metrics` and that's all.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_DISTRIBUTIONAL_METRICS = {"fid", "inception_score", "fdr6", "mind6"}
_DISTRIBUTIONAL_PREFIXES = ("fdr_", "fd_", "mind_")


def is_distributional_metric(name: str) -> bool:
    return name in _DISTRIBUTIONAL_METRICS or name.startswith(_DISTRIBUTIONAL_PREFIXES)


def filter_distributional(metrics: list[str]) -> list[str]:
    return [m for m in metrics if is_distributional_metric(m)]


def compute_distributional_metrics(
    gen_arr: np.ndarray,
    metrics: list[str],
    *,
    reference_npz: Optional[str] = None,
    data_dir: Optional[str] = None,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float]:
    """Run fd_evaluator on a fully-gathered uint8 NHWC array (rank 0 only).

    `reference_npz` (if set) overrides the auto-resolved FID stats and is passed
    as fid_reference. Lets non-ImageNet datasets (e.g. NWM with recon_val_stats)
    plug in their own stats file.
    """
    from fd_evaluator import compute_metrics

    needs_mind = any(m.startswith("mind") for m in metrics)
    reference_images: Optional[str] = None
    if needs_mind:
        reference_images = os.environ.get("NANOGEN_EVALS_REF_IMAGES")
        if not reference_images and data_dir:
            candidate = os.path.join(data_dir, "imagenet-256-val.npz")
            if os.path.exists(candidate):
                reference_images = candidate
        if not reference_images:
            raise FileNotFoundError(
                "MIND metric requested but no raw-image reference found. "
                "Set NANOGEN_EVALS_REF_IMAGES or ensure "
                f"<data_dir>/imagenet-256-val.npz exists (data_dir={data_dir!r})."
            )

    cache_dir = os.environ.get(
        "NANOGEN_EVALS_CACHE_DIR",
        str(Path.home() / ".cache" / "nanogen-evals" / "features"),
    )

    kwargs = dict(
        images=gen_arr,
        metrics=metrics,
        reference_images=reference_images,
        device=("cuda" if device.type == "cuda" else "cpu"),
        batch_size=batch_size,
        feature_cache_dir=cache_dir,
        feature_cache_key=None,
        reference_feature_cache_key="imagenet256_val",
        verbose=True,
    )
    if reference_npz:
        kwargs["fid_reference"] = reference_npz
    return compute_metrics(**kwargs)
