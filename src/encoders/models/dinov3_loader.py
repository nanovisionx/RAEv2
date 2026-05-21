import os
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from torchvision import transforms


@contextmanager
def _rank0_first():
    """Gate torch.hub download so only rank 0 fetches, others wait then read from cache."""
    initialized = dist.is_initialized()
    rank = dist.get_rank() if initialized else 0
    if initialized and rank != 0:
        dist.barrier()
    try:
        yield
    finally:
        if initialized and rank == 0:
            dist.barrier()


def make_dinov3_transform(resize_size: int = 224):
    to_tensor = transforms.Lambda(lambda x: x / 255.)
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])


DINOV3_HUB_REF = "facebookresearch/dinov3:94a96ac83c2446f15f9bdcfae23cad3c6a9d4988"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parents[3] / "pretrained_models" / "encoders" / "dinov3"

MODEL_NAMES = {
    'dinov3_vits16',
    "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vith16plus",
    "dinov3_vit7b16",
}
SHA_CHECKSUM = {
    "dinov3_vits16": "08c60483",
    "dinov3_vits16plus": "4057cbaa",
    "dinov3_vitb16": "73cec8be",
    "dinov3_vitl16": "8aa4cbdd",
    "dinov3_vith16plus": "7c1da9a5",
    "dinov3_vit7b16": "a955f4ea",
}


def load_dinov3(model_name):
    assert model_name in MODEL_NAMES
    ckpt_dir = os.environ.get("DINOV3_CKPT_DIR", str(DEFAULT_CKPT_DIR))
    weights = os.path.join(ckpt_dir, f"{model_name}_pretrain_lvd1689m-{SHA_CHECKSUM[model_name]}.pth")
    repo_dir = os.environ.get("DINOV3_REPO_DIR")
    with _rank0_first():
        if repo_dir and os.path.isfile(os.path.join(repo_dir, "hubconf.py")):
            return torch.hub.load(
                repo_dir,
                model_name,
                source="local",
                trust_repo=True,
                skip_validation=True,
                weights=weights,
            )
        return torch.hub.load(
            DINOV3_HUB_REF,
            model_name,
            source="github",
            trust_repo=True,
            skip_validation=True,
            weights=weights,
        )
