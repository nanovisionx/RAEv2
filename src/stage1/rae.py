from math import sqrt
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoConfig
from encoders.vision_encoder import create_encoder
from .decoders import GeneralDecoder

def _load_decoder(config_path, hidden_size, patch_size, num_patches, pretrained_path=None):
    config = AutoConfig.from_pretrained(config_path)
    config.hidden_size = hidden_size
    config.patch_size = patch_size
    config.image_size = int(patch_size * sqrt(num_patches))
    decoder = GeneralDecoder(config, num_patches=num_patches)
    if pretrained_path is not None:
        print(f"Loading pretrained decoder from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        keys = decoder.load_state_dict(state_dict, strict=False)
        if keys.missing_keys:
            print(f"Missing keys: {keys.missing_keys}")
    return decoder

def _load_normalization_stats(path):
    if path is None:
        return None, None, False
    stats = torch.load(path, map_location='cpu', weights_only=False)
    print(f"Loaded normalization stats from {path}")
    return stats.get('mean', None), stats.get('var', None), True


class RAE(nn.Module):
    def __init__(self,
        encoder_name: str,
        resolution: int = 256,
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        noise_tau: float = 0.8,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.encoder = create_encoder(encoder_name, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), resolution=resolution)
        self.resolution = resolution
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        self.base_patches = (resolution // 16) ** 2
        self.eps = eps
        self.noise_tau = noise_tau

        self.decoder = _load_decoder(
            decoder_config_path, self.latent_dim, decoder_patch_size,
            self.base_patches, pretrained_decoder_path,
        )
        self.latent_mean, self.latent_var, self.do_normalization = _load_normalization_stats(normalization_stat_path)
        print(f"RAE: encoder={encoder_name}, resolution={resolution}, "
              f"patch_size={self.encoder_patch_size}, hidden_size={self.latent_dim}")

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        return x + noise_sigma * torch.randn_like(x)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.max() <= 1.0:
            x = x * 255.0
        _, _, h, w = x.shape
        if h != self.resolution or w != self.resolution:
            x = nn.functional.interpolate(x, size=(self.resolution, self.resolution), mode='bicubic', align_corners=False)

        z = self.encoder(x)

        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        # Reshape [B, N, C] -> [B, C, H, W]
        b, n, c = z.shape
        h = w = int(sqrt(n))
        z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        # Reshape [B, C, H, W] -> [B, N, C]
        b, c, h, w = z.shape
        z = z.view(b, c, h * w).transpose(1, 2)
        output = self.decoder(z, drop_cls_token=False).logits
        return self.decoder.unpatchify(output)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encode(x)
        x_rec = self.decode(z)
        if return_latent:
            return x_rec, z
        return x_rec
