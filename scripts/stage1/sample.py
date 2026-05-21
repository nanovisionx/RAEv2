"""Stage-1 RAE reconstruction sample.

Given an input image and a stage-1 spec (either a YAML config or explicit CLI
args), runs encode -> decode through the RAE and saves three PNGs to assets/:

    <stem>_original.png     -- original input, resized to the RAE resolution
    <stem>_recon.png        -- RAE reconstruction
    <stem>_comparison.png   -- side-by-side original | recon (paper-figures
                                style), annotated with PSNR

Two ways to specify the RAE:

(A) From a YAML config:
    uv run python scripts/stage1/sample.py \
        --config experiments/jas/jobs/stage1-combined-reference/configs/dinov3-vit-l16.yaml \
        --image path/to/image.jpg

(B) Direct CLI (use this for paper-style RAEv2 recons):
    uv run python scripts/stage1/sample.py \
        --image paper-figures/samples/imagenet/text/cls919_traffic-sign_05.jpg \
        --encoder 'dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]' \
        --decoder-config configs/decoder/ViTXL \
        --decoder-ckpt models/decoders/dinov3-mls-last7-more-data-ep5/decoder.pt \
        --stats models/stats/dinov3-mls-last7-more-data-ep5/stats.pt \
        --stem traffic-sign

CLI args override matching YAML values when both are provided.
"""

from __future__ import annotations

import argparse
import inspect
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path.home() / ".claude/skills/paper-figures/scripts"))

from utils.model_utils import get_obj_from_str  # noqa: E402
from paper_style import setup_paper_style  # noqa: E402

ASSETS_DIR = REPO_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


# === FIGURE CONFIGURATION ===
DPI = 200
THUMB_SIZE = 4.5         # inches per panel in comparison figure
TITLE_FONTSIZE = 14
# === END CONFIGURATION ===


def load_image(path: Path, resolution: int) -> torch.Tensor:
    """Load image -> (1, 3, H, W) float tensor in [0, 1], center-cropped and resized."""
    import numpy as np
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((resolution, resolution), Image.BICUBIC)
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    return arr.permute(2, 0, 1).unsqueeze(0)


def to_pil(x: torch.Tensor) -> Image.Image:
    """(C, H, W) float in [0, 1] -> PIL.Image."""
    x = x.detach().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    return Image.fromarray(arr)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1)).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def save_comparison(original: torch.Tensor, recon: torch.Tensor, out_path: Path):
    """Side-by-side original | recon (paper-figures style, no title)."""
    fig, axes = plt.subplots(1, 2, figsize=(THUMB_SIZE * 2, THUMB_SIZE + 0.4))
    for ax, img, label in zip(axes, (original, recon), ("original", "reconstruction")):
        ax.imshow(to_pil(img.squeeze(0)))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_frame_on(False)
        ax.set_xlabel(label, fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True, help="Input image path.")
    # Mode A: YAML config (training-style)
    p.add_argument("--config", type=Path, default=None, help="Optional stage-1 YAML config.")
    # Mode B: direct args (paper-style RAEv2 recon)
    p.add_argument("--target", type=str, default="stage1.RAE", help="Model target (default: stage1.RAE).")
    p.add_argument("--encoder", type=str, default=None, help="Override encoder_name (e.g. 'dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]').")
    p.add_argument("--decoder-config", type=str, default=None, help="Override decoder_config_path (e.g. configs/decoder/ViTXL).")
    p.add_argument("--decoder-ckpt", type=str, default=None, help="Override pretrained_decoder_path (.pt with trained decoder weights).")
    p.add_argument("--stats", type=str, default=None, help="Override normalization_stat_path (.pt with mean/var).")
    p.add_argument("--resolution", type=int, default=None, help="Override resolution (default: 256 or YAML value).")
    p.add_argument("--noise-tau", type=float, default=0.0, help="Encoder noise tau (default: 0.0 for inference).")
    # Output
    p.add_argument("--output-dir", type=Path, default=ASSETS_DIR, help="Where to write PNGs (default: assets/).")
    p.add_argument("--stem", type=str, default=None, help="Filename prefix for outputs (default: input image stem).")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto).")
    return p.parse_args()


def build_rae_params(args) -> tuple[str, dict]:
    """Merge YAML stage_1 section with CLI overrides. Returns (target, params)."""
    params: dict = {}
    target = args.target

    if args.config is not None:
        if not args.config.exists():
            raise FileNotFoundError(f"--config not found: {args.config}")
        yaml_cfg = OmegaConf.to_object(OmegaConf.load(args.config))
        stage_1 = yaml_cfg.get("stage_1") or yaml_cfg.get("stage1")
        if stage_1 is None or "target" not in stage_1:
            raise ValueError(f"Config missing stage_1.target: {args.config}")
        target = stage_1["target"]
        params = dict(stage_1.get("params") or {})

    # CLI overrides
    if args.encoder is not None:           params["encoder_name"] = args.encoder
    if args.decoder_config is not None:    params["decoder_config_path"] = args.decoder_config
    if args.decoder_ckpt is not None:      params["pretrained_decoder_path"] = args.decoder_ckpt
    if args.stats is not None:             params["normalization_stat_path"] = args.stats
    if args.resolution is not None:        params["resolution"] = args.resolution
    params.setdefault("noise_tau", args.noise_tau)
    params.setdefault("resolution", 256)

    return target, params


def main():
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"--image not found: {args.image}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stem = args.stem or args.image.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_paper_style()

    #########################################################
    # Build RAE
    #########################################################
    target, params = build_rae_params(args)
    cls = get_obj_from_str(target)
    valid = set(inspect.signature(cls.__init__).parameters) - {"self"}

    if "encoder_name" in valid and "encoder_name" not in params:
        raise ValueError("Missing encoder_name. Provide --encoder or use a --config that sets stage_1.params.encoder_name.")

    dropped = sorted(set(params) - valid)
    if dropped:
        print(f"[config] dropping unknown params for {target}: {dropped}")
    params = {k: v for k, v in params.items() if k in valid}

    rae = cls(**params).to(device).eval()
    resolution = params.get("resolution", 256)
    spec = params.get("encoder_name") or params.get("vae_type") or target
    print(f"[model] {target} ({spec})  resolution={resolution}  device={device}")
    if params.get("pretrained_decoder_path"):
        print(f"[decoder] {params['pretrained_decoder_path']}")
    if params.get("normalization_stat_path"):
        print(f"[stats]   {params['normalization_stat_path']}")

    #########################################################
    # Reconstruct
    #########################################################
    x = load_image(args.image, resolution).to(device)
    with torch.no_grad():
        x_rec = rae(x).clamp(0, 1)
    psnr_val = psnr(x, x_rec)
    print(f"[recon] PSNR={psnr_val:.2f} dB  shape={tuple(x_rec.shape)}")

    #########################################################
    # Save outputs
    #########################################################
    orig_path = args.output_dir / f"{stem}_original.png"
    rec_path  = args.output_dir / f"{stem}_recon.png"
    cmp_path  = args.output_dir / f"{stem}_comparison.png"

    to_pil(x.squeeze(0)).save(orig_path)
    to_pil(x_rec.squeeze(0)).save(rec_path)
    save_comparison(x, x_rec, cmp_path)

    def _display(p: Path) -> Path:
        try:
            return p.resolve().relative_to(REPO_ROOT)
        except ValueError:
            return p

    print(f"\nWrote:")
    print(f"  {_display(orig_path)}")
    print(f"  {_display(rec_path)}")
    print(f"  {_display(cmp_path)}")


if __name__ == "__main__":
    main()
