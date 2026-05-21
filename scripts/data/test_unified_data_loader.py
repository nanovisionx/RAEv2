"""Sanity-test the unified dataloader for every supported dataset format.

For each dataset, fetches one batch and saves a sample grid to assets/.
Visualizes:
    - ImageNet (label conditioning)        -> assets/data_imagenet_label.png
    - ImageNet (text conditioning)         -> assets/data_imagenet_text.png
    - BLIP3O (WebDataset, text caption)    -> assets/data_blip3o.png
    - NWM (context frames + action)        -> assets/data_nwm.png
    - Combined (ImageNet + BLIP3O grid)    -> assets/data_combined.png

Usage:
    uv run python scripts/data/test_unified_data_loader.py --datasets all
    uv run python scripts/data/test_unified_data_loader.py --datasets imagenet_label blip3o
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path.home() / ".claude/skills/paper-figures/scripts"))

from data import prepare_unified_dataloader  # noqa: E402
from paper_style import setup_paper_style  # noqa: E402

ASSETS_DIR = REPO_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


# === FIGURE CONFIGURATION ===
N_SAMPLES = 8                # number of samples to visualize per dataset
GRID_NCOLS = 4               # samples per row in the grid
THUMB_SIZE = 2.4             # inches per thumbnail
CAPTION_FONTSIZE = 9         # caption text size under each thumb
CAPTION_WRAP = 36            # wrap captions to this width
DPI = 200
IMAGE_SIZE = 256
BATCH_SIZE = N_SAMPLES
NUM_WORKERS = 0              # in-process; spawn workers don't play well with `uv run -c`
# === END CONFIGURATION ===


def _to_hwc_uint8(img: torch.Tensor) -> torch.Tensor:
    """Convert (C, H, W) float in [0, 1] to (H, W, C) uint8 for imshow."""
    img = img.detach().cpu().clamp(0, 1)
    return (img.permute(1, 2, 0) * 255).to(torch.uint8)


def _save_image_grid(images: list[torch.Tensor], captions: list[str], out_path: Path, suptitle: str | None = None):
    """Save N image thumbnails as a grid with captions underneath."""
    n = len(images)
    ncols = min(GRID_NCOLS, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(THUMB_SIZE * ncols, (THUMB_SIZE + 0.4) * nrows),
        squeeze=False,
    )
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        ax.axis("off")
        if idx >= n:
            continue
        ax.imshow(_to_hwc_uint8(images[idx]).numpy())
        cap = textwrap.fill(str(captions[idx]), width=CAPTION_WRAP)
        ax.set_xlabel(cap, fontsize=CAPTION_FONTSIZE)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path.relative_to(REPO_ROOT)}")


def _imagenet_class_name(label_idx: int) -> str:
    from data import IMAGENET_CLASSES
    if 0 <= label_idx < len(IMAGENET_CLASSES):
        return IMAGENET_CLASSES[label_idx].split(",")[0]
    return f"<class {label_idx}>"


def _fetch_one_batch(target: str, condition_type: str = "label", extra_config: dict | None = None) -> tuple:
    """Build a loader for the given target and return the first batch."""
    config = {
        "target": target,
        "data_dir": str(REPO_ROOT / "data" / _default_data_subdir(target)),
        "split": "train" if target != "nwm" else "train",
    }
    if extra_config:
        config.update(extra_config)

    result = prepare_unified_dataloader(
        config=config,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        rank=0,
        world_size=1,
        condition_type=condition_type,
        shuffle=False,
    )
    result.set_epoch(0)
    return next(iter(result.loader))


def _default_data_subdir(target: str) -> str:
    return {
        "imagenet": "imagenet",
        "blip3o": "blip3o-256",
        "nwm": "recon",
    }.get(target, target)


#########################################################
# Per-dataset visualizers
#########################################################
def test_imagenet_label():
    images, labels = _fetch_one_batch("imagenet", condition_type="label")
    captions = [_imagenet_class_name(int(y)) for y in labels[:N_SAMPLES]]
    _save_image_grid(
        list(images[:N_SAMPLES]), captions,
        ASSETS_DIR / "data_imagenet_label.png",
        suptitle="ImageNet (label conditioning)",
    )


def test_imagenet_text():
    images, prompts = _fetch_one_batch("imagenet", condition_type="text")
    _save_image_grid(
        list(images[:N_SAMPLES]), list(prompts[:N_SAMPLES]),
        ASSETS_DIR / "data_imagenet_text.png",
        suptitle="ImageNet (text conditioning)",
    )


def test_blip3o():
    images, captions = _fetch_one_batch("blip3o", extra_config={"split": "short-caption"})
    _save_image_grid(
        list(images[:N_SAMPLES]), list(captions[:N_SAMPLES]),
        ASSETS_DIR / "data_blip3o.png",
        suptitle="BLIP3O (text-to-image)",
    )


def test_nwm():
    target_images, cond_dict = _fetch_one_batch("nwm")
    context = cond_dict["context_frames"]   # (B, K, 3, H, W)
    actions = cond_dict["action"]            # (B, 3)
    rel_time = cond_dict["rel_time"]         # (B, 1)
    n = min(N_SAMPLES, target_images.shape[0])

    # Show: K context frames + 1 target frame per sample, one row per sample
    K = context.shape[1]
    fig, axes = plt.subplots(n, K + 1, figsize=(THUMB_SIZE * (K + 1), THUMB_SIZE * n), squeeze=False)
    for i in range(n):
        for k in range(K):
            ax = axes[i][k]
            ax.imshow(_to_hwc_uint8(context[i, k]).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"ctx t-{K-k}", fontsize=10)
        ax = axes[i][K]
        ax.imshow(_to_hwc_uint8(target_images[i]).numpy())
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title("target", fontsize=10)
        action_str = f"a=({actions[i, 0]:+.2f}, {actions[i, 1]:+.2f}, {actions[i, 2]:+.2f}) t={rel_time[i, 0]:+.2f}"
        ax.set_xlabel(action_str, fontsize=CAPTION_FONTSIZE)

    fig.suptitle("NWM (RECON context frames + target)", fontsize=14)
    fig.tight_layout()
    out = ASSETS_DIR / "data_nwm.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def test_combined():
    """Side-by-side: top row ImageNet (label), bottom row BLIP3O (caption)."""
    in_images, in_labels = _fetch_one_batch("imagenet", condition_type="label")
    bl_images, bl_captions = _fetch_one_batch("blip3o", extra_config={"split": "short-caption"})

    n_per = N_SAMPLES // 2
    fig, axes = plt.subplots(2, n_per, figsize=(THUMB_SIZE * n_per, (THUMB_SIZE + 0.6) * 2), squeeze=False)

    for i in range(n_per):
        ax = axes[0][i]
        ax.imshow(_to_hwc_uint8(in_images[i]).numpy())
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(_imagenet_class_name(int(in_labels[i])), fontsize=CAPTION_FONTSIZE)
        if i == 0:
            ax.set_ylabel("ImageNet", fontsize=11)

        ax = axes[1][i]
        ax.imshow(_to_hwc_uint8(bl_images[i]).numpy())
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(textwrap.fill(str(bl_captions[i]), width=CAPTION_WRAP), fontsize=CAPTION_FONTSIZE)
        if i == 0:
            ax.set_ylabel("BLIP3O", fontsize=11)

    fig.suptitle("Combined (ImageNet + BLIP3O)", fontsize=14)
    fig.tight_layout()
    out = ASSETS_DIR / "data_combined.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


DATASETS = {
    "imagenet_label": test_imagenet_label,
    "imagenet_text":  test_imagenet_text,
    "blip3o":         test_blip3o,
    "nwm":            test_nwm,
    "combined":       test_combined,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["all"], choices=["all", *DATASETS.keys()])
    args = p.parse_args()

    setup_paper_style()
    to_run = list(DATASETS) if "all" in args.datasets else args.datasets

    failures = []
    for name in to_run:
        print(f"[{name}]")
        try:
            DATASETS[name]()
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            failures.append((name, e))

    print(f"\nDone: {len(to_run) - len(failures)} OK, {len(failures)} FAIL")
    if failures:
        for name, e in failures:
            print(f"  - {name}: {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
