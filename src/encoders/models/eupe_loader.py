import os
from pathlib import Path

import torch
from torchvision import transforms


def make_eupe_transform(resize_size: int = 224):
    to_tensor = transforms.Lambda(lambda x: x / 255.)
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])


EUPE_HUB_REF = "facebookresearch/EUPE:f6cdf04d82b97009941a27e9bed3ca17ac7e0366"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parents[3] / "pretrained_models" / "encoders" / "eupe"

MODEL_NAMES = {
    'eupe_vitt16',
    'eupe_vits16',
    'eupe_vitb16',
}
CHECKPOINT_FILENAMES = {
    'eupe_vitt16': 'EUPE-ViT-T.pt',
    'eupe_vits16': 'EUPE-ViT-S.pt',
    'eupe_vitb16': 'EUPE-ViT-B.pt',
}


def load_eupe(model_name):
    assert model_name in MODEL_NAMES
    ckpt_dir = os.environ.get("EUPE_CKPT_DIR", str(DEFAULT_CKPT_DIR))
    weights = os.path.join(ckpt_dir, CHECKPOINT_FILENAMES[model_name])
    return torch.hub.load(
        EUPE_HUB_REF,
        model_name,
        source="github",
        trust_repo=True,
        skip_validation=True,
        weights=weights,
    )
