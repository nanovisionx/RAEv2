import torch

from .diffaug import DiffAug
from .discriminator import DinoDiscriminator
from .gan_loss import (
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
    select_gan_losses,
    calculate_adaptive_weight,
)
from .lpips import LPIPS

from configs import DiscriminatorArchConfig, DiscAugmentConfig


def build_discriminator(
    arch_config: DiscriminatorArchConfig,
    device: torch.device,
    augment_config: DiscAugmentConfig = None,
) -> tuple[DinoDiscriminator, DiffAug]:
    """Instantiate Dino-based discriminator and its augmentation policy."""
    if not arch_config.dino_ckpt_path:
        raise ValueError("DINO discriminator requires 'dino_ckpt_path'.")

    disc = DinoDiscriminator(
        device=device,
        dino_ckpt_path=arch_config.dino_ckpt_path,
        ks=arch_config.ks,
        norm_type=arch_config.norm_type,
        using_spec_norm=arch_config.using_spec_norm,
        recipe=arch_config.recipe,
    ).to(device)

    if augment_config is None:
        augment_config = DiscAugmentConfig()
    augment = DiffAug(prob=augment_config.prob, cutout=augment_config.cutout)

    return disc, augment


__all__ = [
    "LPIPS",
    "DiffAug",
    "DinoDiscriminator",
    "hinge_d_loss",
    "vanilla_d_loss",
    "vanilla_g_loss",
    "select_gan_losses",
    "calculate_adaptive_weight",
    "build_discriminator",
]
