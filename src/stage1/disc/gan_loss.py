"""Adapted and modified from https://github.com/CompVis/taming-transformers"""

import torch
import torch.nn.functional as F


def hinge_d_loss(logits_real, logits_fake, reduction: str = "mean") -> torch.Tensor:
    """Hinge discriminator loss used by VQGAN."""
    reduce = torch.mean if reduction == "mean" else torch.sum
    loss_real = reduce(F.relu(1.0 - logits_real))
    loss_fake = reduce(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real, logits_fake, reduction: str = "mean") -> torch.Tensor:
    """Original GAN discriminator loss."""
    reduce = torch.mean if reduction == "mean" else torch.sum
    return 0.5 * (
        reduce(F.softplus(-logits_real)) + reduce(F.softplus(logits_fake))
    )


def vanilla_g_loss(logits_fake, reduction: str = "mean") -> torch.Tensor:
    """Original GAN generator loss."""
    if reduction == "mean":
        return -torch.mean(logits_fake)
    if reduction == "sum":
        return -torch.sum(logits_fake)
    raise ValueError(f"Unsupported reduction '{reduction}'")


def select_gan_losses(disc_kind: str, gen_kind: str):
    """Select discriminator and generator loss functions by name.

    Args:
        disc_kind: Discriminator loss type ('hinge' or 'vanilla')
        gen_kind: Generator loss type ('vanilla')

    Returns:
        Tuple of (disc_loss_fn, gen_loss_fn)
    """
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'")

    if gen_kind == "vanilla":
        gen_loss_fn = vanilla_g_loss
    else:
        raise ValueError(f"Unsupported generator loss '{gen_kind}'")
    return disc_loss_fn, gen_loss_fn


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    """Calculate adaptive weight for GAN loss based on gradient norms.

    Args:
        recon_loss: Reconstruction loss tensor
        gan_loss: GAN loss tensor
        layer: The layer parameter to compute gradients with respect to
        max_d_weight: Maximum discriminator weight clamp value

    Returns:
        Adaptive weight for balancing reconstruction and GAN losses
    """
    torch._functorch.config.donated_buffer = False  # Allow retain_graph=True with torch.compile
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()
