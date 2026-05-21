from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn


@dataclass
class VAEConfig:
    """Configuration for a VAE type."""
    pretrained_path: str
    subfolder: Optional[str] = ""
    latent_channels: int = 16
    scaling_factor: Optional[float] = None  # None = read from vae.config
    shift_factor: Optional[float] = None    # None = read from vae.config
    downsample_factor: int = 8


# Pre-defined VAE configurations
VAE_CONFIGS: Dict[str, VAEConfig] = {
    "flux": VAEConfig(
        pretrained_path="black-forest-labs/FLUX.1-dev",
        subfolder="vae",
        latent_channels=16,
        downsample_factor=8,
    ),
    "flux2": VAEConfig(
        pretrained_path="black-forest-labs/FLUX.2-dev",
        subfolder="vae",
        latent_channels=128,
        downsample_factor=16,
    ),
    "e2e-flux": VAEConfig(
        pretrained_path="REPA-E/e2e-flux-vae",
        latent_channels=16,
        downsample_factor=8,
    ),
    "sd3.5": VAEConfig(
        pretrained_path="stabilityai/stable-diffusion-3.5-large",
        subfolder="vae",
        latent_channels=16,
        downsample_factor=8,
    ),
    "e2e-sd3.5": VAEConfig(
        pretrained_path="REPA-E/e2e-sd3.5-vae",
        latent_channels=16,
        downsample_factor=8,
    ),
    "sdvae-ema": VAEConfig(
        pretrained_path="stabilityai/sd-vae-ft-ema",
        latent_channels=4,
        scaling_factor=0.18215,
        shift_factor=0.0,
        downsample_factor=8,
    ),
    "sdvae-mse": VAEConfig(
        pretrained_path="stabilityai/sd-vae-ft-mse",
        latent_channels=4,
        scaling_factor=0.18215,
        shift_factor=0.0,
        downsample_factor=8,
    ),
    "e2e-vavae": VAEConfig(
        pretrained_path="REPA-E/e2e-vavae-hf",
        latent_channels=32,
        downsample_factor=16,
    ),
    "vavae": VAEConfig(
        pretrained_path="REPA-E/vavae-hf",
        latent_channels=32,
        downsample_factor=16,
    ),
    "sdxl-vae": VAEConfig(
        pretrained_path="madebyollin/sdxl-vae-fp16-fix",
        latent_channels=4,
        downsample_factor=8,
        scaling_factor=0.13025,
        shift_factor=0.0,
    ),
    "qwen-vae": VAEConfig(
        pretrained_path="Qwen/Qwen-Image",
        latent_channels=16,
        downsample_factor=8,
        subfolder="vae",
    ),
    "e2e-qwen-vae": VAEConfig(
        pretrained_path="REPA-E/e2e-qwenimage-vae",
        latent_channels=16,
        downsample_factor=8,
    ),
    "e2e-sdvae-mse": VAEConfig(
        pretrained_path="REPA-E/e2e-sdvae-hf",
        latent_channels=4,
        downsample_factor=8,
    ),
}


class VAE(nn.Module):
    """
    VAE wrapper for Diffusers AutoencoderKL that matches RAE interface.

    Supports Flux VAE, SD3.5 VAE etc. through config presets
    or any custom Diffusers-based VAE via pretrained_path.

    Example usage in config:
        stage_1:
          target: stage1.VAE
          params:
            vae_type: "flux"
            resolution: 256
            sample_mode: "mode"
    """

    def __init__(
        self,
        vae_type: str,
        resolution: int = 256,
        eps: float = 1e-5,
        sample_mode: Literal["sample", "mode"] = "mode",
    ):
        super().__init__()

        self.resolution = resolution
        self.eps = eps
        self.sample_mode = sample_mode

        config = VAE_CONFIGS[vae_type]
        self._pretrained_path = config.pretrained_path
        self._subfolder = config.subfolder
        self._latent_channels = config.latent_channels
        self._config_scaling_factor = config.scaling_factor
        self._config_shift_factor = config.shift_factor
        self._downsample_factor = config.downsample_factor

        self._load_vae()

        if hasattr(self.vae.config, 'latents_mean') and self.vae.config.latents_mean is not None:
            self.register_buffer('shift_factor', torch.tensor(self.vae.config.latents_mean).reshape(1, -1, 1, 1))
        elif self._config_shift_factor is not None:
            self.shift_factor = self._config_shift_factor
        else:
            self.shift_factor = getattr(self.vae.config, 'shift_factor', 0.0)

        if hasattr(self.vae.config, 'latents_std') and self.vae.config.latents_std is not None:
            self.register_buffer('scaling_factor', 1 / torch.tensor(self.vae.config.latents_std).reshape(1, -1, 1, 1))
        elif self._config_scaling_factor is not None:
            self.scaling_factor = self._config_scaling_factor
        else:
            self.scaling_factor = getattr(self.vae.config, 'scaling_factor', 1.0)

    def _load_vae(self):
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(self._pretrained_path, subfolder=self._subfolder).eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    @property
    def latent_dim(self) -> int:
        """Return latent channels for compatibility with RAE interface."""
        return self._latent_channels

    @property
    def patch_size(self) -> int:
        """Return effective patch size (downsample factor) for compatibility."""
        return self._downsample_factor

    @property
    def hidden_size(self) -> int:
        """Alias for latent_dim for compatibility."""
        return self._latent_channels

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input images for VAE.

        Args:
            x: Images in [0, 1] range, shape (B, 3, H, W)

        Returns:
            Images in [-1, 1] range, resized to target resolution
        """
        # Resize if needed
        _, _, h, w = x.shape
        if h != self.resolution or w != self.resolution:
            x = nn.functional.interpolate(
                x,
                size=(self.resolution, self.resolution),
                mode='bilinear',
                align_corners=False
            )

        # Convert from [0, 1] to [-1, 1]
        x = x * 2.0 - 1.0
        return x

    def _vae_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x).latent_dist

    def _vae_decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z).sample

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latents.

        Args:
            x: Images in [0, 1] range, shape (B, 3, H, W)

        Returns:
            Latents in shape (B, C, H, W) where C=latent_channels, H=W=latent_size
        """
        x = self._preprocess(x)
        posterior = self._vae_encode(x)
        if self.sample_mode == "sample":
            z = posterior.sample()
        else:
            z = posterior.mode()

        z = (z - self.shift_factor) * self.scaling_factor

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images.

        Args:
            z: Latents in shape (B, C, H, W)

        Returns:
            Images in [0, 1] range, shape (B, 3, H, W)
        """
        z = z / self.scaling_factor + self.shift_factor
        x = self._vae_decode(z)
        x = ((x + 1.0) / 2.0).clamp(0, 1)  # Convert from [-1, 1] to [0, 1]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode (for reconstruction)."""
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec


class Flux2VAE(VAE):
    """
    Flux2 VAE wrapper with special patchify + BatchNorm normalization.

    Flux2 differs from other VAEs:
    1. VAE outputs 32 channels at 32x32 for 256x256 input
    2. Latents are patchified (2x2) before normalization
    3. Uses BatchNorm (bn.running_mean/std) instead of simple scale/shift
    4. After patchify+BN: (B, 32, 32, 32) -> (B, 128, 16, 16)

    Example usage in config:
        stage_1:
          target: stage1.Flux2VAE
          params:
            resolution: 256
    """

    def __init__(
        self,
        resolution: int = 256,
        eps: float = 1e-5,
        sample_mode: Literal["sample", "mode"] = "mode",
    ):
        super().__init__(
            vae_type="flux2",
            resolution=resolution,
            eps=eps,
            sample_mode=sample_mode,
        )
        self.register_buffer('bn_mean', self.vae.bn.running_mean.clone())
        self.register_buffer('bn_std', torch.sqrt(self.vae.bn.running_var.clone() + eps))
        print(f"Flux2VAE: Loaded BN stats, mean shape={self.bn_mean.shape}")

    def _load_vae(self):
        """Load Flux2 VAE with BatchNorm support using AutoencoderKLFlux2."""
        from diffusers import AutoencoderKLFlux2
        self.vae = AutoencoderKLFlux2.from_pretrained(self._pretrained_path, subfolder=self._subfolder).eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    def _patchify(self, z: torch.Tensor) -> torch.Tensor:
        """
        Patchify latents with 2x2 patches.

        Args:
            z: (B, C, H, W) e.g. (B, 32, 32, 32)

        Returns:
            (B, C*4, H//2, W//2) e.g. (B, 128, 16, 16)
        """
        B, C, H, W = z.shape
        # Reshape to (B, C, H//2, 2, W//2, 2)
        z = z.view(B, C, H // 2, 2, W // 2, 2)
        # Permute to (B, C, 2, 2, H//2, W//2)
        z = z.permute(0, 1, 3, 5, 2, 4)
        # Reshape to (B, C*4, H//2, W//2)
        z = z.reshape(B, C * 4, H // 2, W // 2)
        return z

    def _unpatchify(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reverse patchification.

        Args:
            z: (B, C*4, H//2, W//2) e.g. (B, 128, 16, 16)

        Returns:
            (B, C, H, W) e.g. (B, 32, 32, 32)
        """
        B, C4, H2, W2 = z.shape
        C = C4 // 4
        # Reshape to (B, C, 2, 2, H//2, W//2)
        z = z.view(B, C, 2, 2, H2, W2)
        # Permute to (B, C, H//2, 2, W//2, 2)
        z = z.permute(0, 1, 4, 2, 5, 3)
        # Reshape to (B, C, H, W)
        z = z.reshape(B, C, H2 * 2, W2 * 2)
        return z

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to Flux2 latents with patchify + BN.

        Args:
            x: Images in [0, 1] range, shape (B, 3, H, W)

        Returns:
            Latents in shape (B, 128, 16, 16) for 256x256 input
        """
        x = self._preprocess(x)
        posterior = self.vae.encode(x).latent_dist
        if self.sample_mode == "sample":
            z = posterior.sample()
        else:
            z = posterior.mode()

        z = self._patchify(z)
        bn_mean = self.bn_mean.to(z.device, z.dtype).view(1, -1, 1, 1)
        bn_std = self.bn_std.to(z.device, z.dtype).view(1, -1, 1, 1)
        z = (z - bn_mean) / bn_std

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode Flux2 latents to images.

        Args:
            z: Latents in shape (B, 128, 16, 16)

        Returns:
            Images in [0, 1] range, shape (B, 3, H, W)
        """
        bn_mean = self.bn_mean.to(z.device, z.dtype).view(1, -1, 1, 1)
        bn_std = self.bn_std.to(z.device, z.dtype).view(1, -1, 1, 1)
        z = z * bn_std + bn_mean

        z = self._unpatchify(z)
        x = self.vae.decode(z).sample
        x = ((x + 1.0) / 2.0).clamp(0, 1)  # Convert from [-1, 1] to [0, 1]

        return x


class QwenVAE(VAE):
    def __init__(
        self,
        vae_type: str = "qwen-vae",
        resolution: int = 256,
        eps: float = 1e-5,
        sample_mode: Literal["sample", "mode"] = "mode",
    ):
        super().__init__(
            vae_type=vae_type,
            resolution=resolution,
            eps=eps,
            sample_mode=sample_mode,
        )

    def _load_vae(self):
        from diffusers import AutoencoderKLQwenImage
        self.vae = AutoencoderKLQwenImage.from_pretrained(self._pretrained_path, subfolder=self._subfolder).eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    def _vae_encode(self, x: torch.Tensor):
        x = x.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
        posterior = self.vae.encode(x).latent_dist
        posterior.mean = posterior.mean.squeeze(2)
        posterior.logvar = posterior.logvar.squeeze(2)
        return posterior

    def _vae_decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
        x = self.vae.decode(z).sample
        return x.squeeze(2)  # (B, 3, 1, H, W) -> (B, 3, H, W)
