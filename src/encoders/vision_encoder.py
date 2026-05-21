from typing import Dict, List, Optional

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


class VisionEncoder(nn.Module):
    """Base class for all vision encoders"""

    def __init__(self, encoder_type: str, architecture: str, model_config: str,
                 device: torch.device, resolution: int = 256, accelerator=None):
        super().__init__()  # Initialize nn.Module
        self.encoder_type = encoder_type
        self.architecture = architecture
        self.model_config = model_config
        self.device = device
        self.resolution = resolution
        self.accelerator = accelerator
        self._embed_dim = None
        self.model = None
        self.patch_size = None  # Subclasses should set this

    def load_model(self):
        """Load and initialize the encoder model - subclasses should override"""
        raise NotImplementedError("Subclasses must implement load_model()")

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess raw images - subclasses should override
        Args:
            x: Raw images tensor (B, C, H, W) in range [0, 255]
        Returns:
            Preprocessed tensor ready for encoder
        """
        raise NotImplementedError("Subclasses must implement preprocess()")

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass through encoder
        Args:
            x: Preprocessed images
        Returns:
            Dictionary with:
                - 'x_norm_clstoken': (B, D) CLS token or None if not available
                - 'x_norm_patchtokens': (B, T, D) patch tokens
        """
        # Default implementation - subclasses should override if needed
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            return out
        else:
            # Assume it's just patch tokens
            return {
                'x_norm_clstoken': None,
                'x_norm_patchtokens': out
            }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RAE-compatible forward pass returning only patch tokens.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Patch tokens (B, T, D)
        """
        x = self.preprocess(x)
        features = self.forward_features(x)
        return features['x_norm_patchtokens']

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def hidden_size(self) -> int:
        return self._embed_dim

    def eval(self):
        """Set model to eval mode"""
        if self.model is not None:
            self.model.eval()
        return self

    def to(self, device):
        """Move model to device"""
        if self.model is not None:
            self.model = self.model.to(device)
        self.device = device
        return self


class DINOv2Encoder(VisionEncoder):
    """DINOv2 encoder implementation.

    Supports optional flags in model_config: e.g., 'b[norm,woreg]'
      - Default (no flags): registers=True, norm_affine=False (matches legacy Dinov2withNorm)
      - [norm]: keep layernorm affine params
      - [woreg]: without register tokens
    """

    # Known flags that can appear in model_config after the base size letter
    _KNOWN_FLAGS = {'norm', 'woreg'}

    def _parse_config(self):
        """Parse model_config for base config and flags.

        Supports multiple formats:
            'b'              -> base='b', flags=set()
            'b[norm,woreg]'  -> base='b', flags={'norm','woreg'}  (bracket syntax)
            'bnormworeg'     -> base='b', flags={'norm','woreg'}  (concatenated suffix)
        """
        import re
        # Try bracket syntax first: e.g. 'b[norm,woreg]'
        match = re.match(r'^([a-z])(?:\[([^\]]+)\])?$', self.model_config)
        if match and match.group(2) is not None:
            base = match.group(1)
            flags = set(f.strip() for f in match.group(2).split(','))
            return base, flags

        # Try concatenated suffix syntax: e.g. 'bnorm', 'bworeg', 'bnormworeg'
        # First character is the size, rest is parsed for known flags
        cfg = self.model_config
        if len(cfg) >= 1 and cfg[0].isalpha():
            base = cfg[0]
            suffix = cfg[1:]
            if not suffix:
                return base, set()
            # Greedily match known flags from the suffix
            flags = set()
            remaining = suffix
            while remaining:
                matched = False
                for flag in self._KNOWN_FLAGS:
                    if remaining.startswith(flag):
                        flags.add(flag)
                        remaining = remaining[len(flag):]
                        matched = True
                        break
                if not matched:
                    # Unknown suffix — return raw config as base
                    return self.model_config, set()
            return base, flags

        return self.model_config, set()

    def load_model(self):
        import timm

        # Parse config and flags
        base_config, flags = self._parse_config()

        # Default: registers=True, norm_affine=False (legacy behavior)
        use_reg = 'woreg' not in flags
        use_norm_affine = 'norm' in flags

        # Load model from torch hub
        model_name = f'dinov2_vit{base_config}14{"_reg" if use_reg else ""}'

        if self.accelerator is not None:
            with self.accelerator.main_process_first():
                self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        else:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)

        # Remove head
        del self.model.head
        self.model.head = torch.nn.Identity()

        # Resample position embeddings if needed
        patch_resolution = 16 * (self.resolution // 256)
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [patch_resolution, patch_resolution],
        )

        # Set embed dim and patch size
        self._embed_dim = self.model.embed_dim
        self.patch_size = 14  # DINOv2 models use patch size 14

        # Remove layernorm affine params by default (matches legacy normalize=True)
        if not use_norm_affine:
            # Replace with LayerNorm without affine params
            self.model.norm = nn.LayerNorm(self._embed_dim, elementwise_affine=False)

        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Apply ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        # Interpolate if needed
        x = torch.nn.functional.interpolate(x, 224 * (self.resolution // 256), mode='bicubic')
        return x

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # DINOv2 returns a dictionary with cls and patch tokens
        out = self.model.forward_features(x)
        return {
            'x_norm_clstoken': out.get('x_norm_clstoken'),
            'x_norm_patchtokens': out.get('x_norm_patchtokens')
        }


class DINOv3Encoder(VisionEncoder):
    """DINOv3 encoder implementation.

    Supports optional flags in model_config: e.g., 'b16[norm]'
      - Default (no flags): norm_affine=False (matches DINOv2 default)
      - [norm]: keep layernorm affine params
    """

    _KNOWN_FLAGS = {'norm'}
    _KNOWN_BASES = {'s16', 's16plus', 'b16', 'l16', 'h16plus', '7b16'}

    def _parse_config(self):
        """Parse model_config for base config and flags.

        DINOv3 base configs are multi-character (s16, b16, l16, etc.).
        Supports:
            'b16'           -> base='b16', flags=set()
            'b16[norm]'     -> base='b16', flags={'norm'}
            'b16norm'       -> base='b16', flags={'norm'}
        """
        import re
        # Bracket syntax: e.g. 'b16[norm]'
        match = re.match(r'^(.+?)\[([^\]]+)\]$', self.model_config)
        if match:
            base = match.group(1)
            flags = set(f.strip() for f in match.group(2).split(','))
            return base, flags

        # Concatenated suffix: match longest known base, parse flags from remainder
        cfg = self.model_config
        best_base = None
        for known_base in sorted(self._KNOWN_BASES, key=len, reverse=True):
            if cfg.startswith(known_base):
                best_base = known_base
                break

        if best_base:
            suffix = cfg[len(best_base):]
            if not suffix:
                return best_base, set()
            flags = set()
            remaining = suffix
            while remaining:
                matched = False
                for flag in self._KNOWN_FLAGS:
                    if remaining.startswith(flag):
                        flags.add(flag)
                        remaining = remaining[len(flag):]
                        matched = True
                        break
                if not matched:
                    return self.model_config, set()
            return best_base, flags

        return self.model_config, set()

    def load_model(self):
        from .models.dinov3_loader import load_dinov3

        base_config, flags = self._parse_config()
        use_norm_affine = 'norm' in flags

        self.model = load_dinov3(f"dinov3_vit{base_config}")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Set embed dim and patch size
        self._embed_dim = self.model.embed_dim
        self.patch_size = 16

        # Strip norm affine by default (matches DINOv2 default)
        if not use_norm_affine:
            self.model.norm = nn.LayerNorm(self._embed_dim, elementwise_affine=False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        from .models.dinov3_loader import make_dinov3_transform
        transform_func = make_dinov3_transform(resize_size=self.resolution)
        return transform_func(x)

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward_features(x)
        return {
            'x_norm_clstoken': out.get('x_norm_clstoken'),
            'x_norm_patchtokens': out.get('x_norm_patchtokens')
        }


class DINOv3MultiLayerSimpleAddEncoder(DINOv3Encoder):
    """DINOv3 encoder that averages patch tokens from multiple layers.

    Same approach as DINOv2MultiLayerSimpleAddEncoder but for DINOv3 models.
    Config format: 'l16[layers=21.23]', 'b16[layers=7.9.11]'
    Default layers per model: l16=[5,11,17,23], b16=[2,5,8,11]
    """

    DEFAULT_LAYERS = {
        's16': [2, 5, 8, 11],
        'b16': [2, 5, 8, 11],
        'l16': [5, 11, 17, 23],
        'h16plus': [8, 16, 24, 31],
    }

    def load_model(self):
        super().load_model()
        base_config, flags = self._parse_config()
        layers_flag = [f for f in flags if f.startswith('layers=')]
        if layers_flag:
            self.layer_indices = [int(i) for i in layers_flag[0].split('=')[1].split('.')]
        else:
            self.layer_indices = self.DEFAULT_LAYERS.get(base_config, [2, 5, 8, 11])

    def _parse_config(self):
        import re
        match = re.match(r'^(.+?)\[([^\]]+)\]$', self.model_config)
        if match:
            base = match.group(1)
            flags = [f.strip() for f in match.group(2).split(',')]
            return base, flags
        return self.model_config, []

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        outputs = self.model.get_intermediate_layers(
            x, n=self.layer_indices, reshape=False,
            return_class_token=False, norm=True
        )
        patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
        final_mean = outputs[-1].mean(dim=1, keepdim=True)
        patch_tokens = patch_tokens + final_mean
        return {
            'x_norm_clstoken': final_mean.squeeze(1),
            'x_norm_patchtokens': patch_tokens,
        }


class DINOv2MultiLayerSimpleAddEncoder(DINOv2Encoder):
    """DINOv2 encoder that averages patch tokens from multiple layers.

    Config format: 'b[layers=2.11]', 'b[layers=2.5.8.11]'
    Default layers: b=[2,5,8,11]
    """

    DEFAULT_LAYERS = {'s': [2, 5, 8, 11], 'b': [2, 5, 8, 11], 'l': [5, 11, 17, 23], 'g': [10, 20, 30, 39]}

    def load_model(self):
        super().load_model()
        base_config, flags = self._parse_config()
        for f in flags:
            if f.startswith('layers='):
                self.layer_indices = [int(i) for i in f.split('=')[1].split('.')]
                return
        self.layer_indices = self.DEFAULT_LAYERS.get(base_config, [2, 5, 8, 11])

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        outputs = self.model.get_intermediate_layers(
            x, n=self.layer_indices, reshape=False,
            return_class_token=False, norm=True
        )
        patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
        return {
            'x_norm_clstoken': patch_tokens.mean(dim=1),
            'x_norm_patchtokens': patch_tokens,
        }


class SigLIP2Encoder(VisionEncoder):
    """SigLIP2 encoder implementation"""

    def load_model(self):
        from transformers import SiglipVisionModel

        # Map model config to full model name
        model_map = {
            'b': 'google/siglip2-base-patch16-256',
            'l': 'google/siglip2-large-patch16-256',
            'so400m': 'google/siglip2-so400m-patch16-256',
            'g': 'google/siglip2-giant-opt-patch16-256'
        }

        if self.model_config not in model_map:
            raise ValueError(f"Unknown SigLIP2 model config: {self.model_config}")

        self.model = SiglipVisionModel.from_pretrained(model_map[self.model_config])
        self.model.to(self.device)
        self.model.eval()

        # patch size
        self.patch_size = 16
        self._embed_dim = self.model.config.hidden_size

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Apply ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, self.resolution, mode='bicubic')
        return x

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model(x).last_hidden_state
        return {
            'x_norm_clstoken': None,  # SigLIP has no CLS token
            'x_norm_patchtokens': out
        }


class SigLIP2MultiLayerSimpleAddEncoder(SigLIP2Encoder):
    """SigLIP2 encoder that averages patch tokens from multiple layers.

    Mirrors DINOv3MultiLayerSimpleAddEncoder. Layer indices are 0-based block
    indices (matches DINOv3 convention), so e.g. layers=[5,11,17,23] selects
    blocks 5, 11, 17, 23 of a 24-block ViT-L.

    The model's final LayerNorm (vision_model.post_layernorm) is applied to
    each selected block output, matching DINOv3-mls's norm=True semantics.
    A broadcast mean of the final selected layer is added to the average,
    acting as a global pooled signal in lieu of a CLS token.

    Config format: 'l[layers=11.13.15.17.19.21.23]', 'b[layers=2.5.8.11]'.
    """

    DEFAULT_LAYERS = {
        'b': [2, 5, 8, 11],
        'l': [5, 11, 17, 23],
        'so400m': [5, 12, 19, 26],
        'g': [10, 20, 30, 39],
    }

    def _parse_config(self):
        import re
        match = re.match(r'^(.+?)\[([^\]]+)\]$', self.model_config)
        if match:
            base = match.group(1)
            flags = [f.strip() for f in match.group(2).split(',')]
            return base, flags
        return self.model_config, []

    def load_model(self):
        from transformers import SiglipVisionModel

        base_config, flags = self._parse_config()
        model_map = {
            'b': 'google/siglip2-base-patch16-256',
            'l': 'google/siglip2-large-patch16-256',
            'so400m': 'google/siglip2-so400m-patch16-256',
            'g': 'google/siglip2-giant-opt-patch16-256',
        }
        if base_config not in model_map:
            raise ValueError(f"Unknown SigLIP2 model config: {base_config}")

        self.model = SiglipVisionModel.from_pretrained(model_map[base_config])
        self.model.to(self.device)
        self.model.eval()

        self.patch_size = 16
        self._embed_dim = self.model.config.hidden_size
        self._num_hidden_layers = self.model.config.num_hidden_layers

        layers_flag = [f for f in flags if f.startswith('layers=')]
        if layers_flag:
            self.layer_indices = [int(i) for i in layers_flag[0].split('=')[1].split('.')]
        else:
            self.layer_indices = self.DEFAULT_LAYERS.get(base_config, [2, 5, 8, 11])

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # hidden_states layout (HF SigLIP2): tuple of length N+1
        #   hs[0]   = post-embedding (input to block 0)
        #   hs[k]   for k in 1..N-1 = raw output of block k-1 (pre-post_layernorm)
        #   hs[N]   = post_layernorm(output of block N-1) == last_hidden_state
        # Apply post_layernorm to non-final selected blocks to match DINOv3-mls
        # (norm=True) semantics; for the final block, hs[N] is already normed.
        hs = self.model(x, output_hidden_states=True).hidden_states
        post_ln = self.model.vision_model.post_layernorm
        N = self._num_hidden_layers

        outputs = []
        for li in self.layer_indices:
            if li == N - 1:
                outputs.append(hs[N])
            else:
                outputs.append(post_ln(hs[li + 1]))

        patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
        final_mean = outputs[-1].mean(dim=1, keepdim=True)
        patch_tokens = patch_tokens + final_mean
        return {
            'x_norm_clstoken': None,
            'x_norm_patchtokens': patch_tokens,
        }


class MAEEncoder(VisionEncoder):
    """MAE (Masked Autoencoder) encoder implementation.

    Matches legacy MAEwNorm behavior: no layernorm affine, mask_ratio=0, removes CLS token.
    """

    def load_model(self):
        from transformers import ViTMAEForPreTraining

        model_map = {
            'b': 'facebook/vit-mae-base',
            'l': 'facebook/vit-mae-large',
            'h': 'facebook/vit-mae-huge',
        }

        if self.model_config not in model_map:
            raise ValueError(f"Unknown MAE model config: {self.model_config}")

        self.model = ViTMAEForPreTraining.from_pretrained(model_map[self.model_config]).vit
        # Remove layernorm affine (matches legacy MAEwNorm)
        self.model.layernorm.elementwise_affine = False
        self.model.layernorm.weight = None
        self.model.layernorm.bias = None
        # No masking
        self.model.config.mask_ratio = 0.

        self._embed_dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size

        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, self.resolution, mode='bicubic')
        return x

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        h, w = x.shape[2], x.shape[3]
        patch_num = int(h * w // self.patch_size ** 2)
        noise = torch.arange(patch_num).unsqueeze(0).expand(x.shape[0], -1).to(x.device).to(x.dtype)
        outputs = self.model(x, noise, interpolate_pos_encoding=True)
        # Remove CLS token (first token)
        patch_tokens = outputs.last_hidden_state[:, 1:]
        return {
            'x_norm_clstoken': None,
            'x_norm_patchtokens': patch_tokens
        }


class WebSSLEncoder(VisionEncoder):
    """WebSSL encoder implementation"""

    def load_model(self):
        from transformers import AutoImageProcessor, Dinov2Model

        model_name = f"facebook/webssl-{self.model_config.replace('_', '-')}"
        self.model = Dinov2Model.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self._embed_dim = self.model.config.hidden_size
        self.patch_size = 14

        # Also load processor for preprocessing
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (self.resolution // 256), mode='bicubic')
        return x

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # Skip CLS token (index 0)
        out = self.model.forward(x).last_hidden_state
        cls_token = out[:, 0]
        patch_tokens = out[:, 1:]
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens
        }


class PEEncoder(VisionEncoder):
    """PE (Perceptual Encoder) implementation"""

    def load_model(self):
        from encoders.models import pe

        # Check if using normalization
        self.use_norm = self.model_config.endswith("norm")
        if self.use_norm:
            config_name = self.model_config[:-4]
        else:
            config_name = self.model_config

        # Map config to model name
        if self.encoder_type == "pe":
            config_map = {
                "t": "PE-Core-T16-384",
                "s": "PE-Core-S16-384",
                "b": "PE-Core-B16-224",
                "l": "PE-Core-L14-336",
                "g": "PE-Core-G14-448"
            }
        elif self.encoder_type == "spatialpe":
            config_map = {
                "b": "PE-Spatial-B16-512",
                "l": "PE-Spatial-L14-448",
                "g": "PE-Spatial-G14-448"
            }
        elif self.encoder_type == "langpe":
            config_map = {
                "l": "PE-Lang-L14-448",
                "g": "PE-Lang-G14-448"
            }
        else:
            raise ValueError(f"Unknown PE encoder type: {self.encoder_type}")

        if config_name not in config_map:
            raise ValueError(f"Unknown PE model config: {config_name}")

        self.model = pe.VisionTransformer.from_config(config_map[config_name], pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._embed_dim = self.model.width

        # Get patch size for preprocessing
        if config_name in {"t", "s", "b", "tnorm", "snorm", "bnorm"}:
            self.patch_size = 16
        elif config_name in {"l", "g", "lnorm", "gnorm"}:
            self.patch_size = 14
        else:
            raise NotImplementedError()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = torch.nn.functional.interpolate(
            x, self.patch_size * (self.resolution // 16), mode='bilinear'
        )
        x = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(x)
        return x

    def forward_features(self, x: torch.Tensor, layer_idx: int = -1) -> Dict[str, Optional[torch.Tensor]]:
        # PE returns patch tokens without CLS
        out = self.model.forward_features(x, norm=self.use_norm, layer_idx=layer_idx, strip_cls_token=False)
        if self.model.use_cls_token:
            cls_token = out[:, 0]
            patch_tokens = out[:, 1:]
        else:
            cls_token = None
            patch_tokens = out
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens
        }

class EUPEEncoder(VisionEncoder):
    """EUPE (Efficient Universal Perception Encoder) from Meta AI."""

    def load_model(self):
        from .models.eupe_loader import load_eupe
        model_name = f"eupe_vit{self.model_config}"
        self.model = load_eupe(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._embed_dim = self.model.embed_dim
        self.patch_size = 16

        # Strip norm affine by default (matches DINOv2/DINOv3 default)
        self.model.norm = nn.LayerNorm(self._embed_dim, elementwise_affine=False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        from .models.eupe_loader import make_eupe_transform
        return make_eupe_transform(self.resolution)(x)

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward_features(x)
        return {
            'x_norm_clstoken': out.get('x_norm_clstoken'),
            'x_norm_patchtokens': out.get('x_norm_patchtokens'),
        }


class EUPEMultiLayerSimpleAddEncoder(EUPEEncoder):
    """EUPE encoder that sums patch tokens from multiple layers.

    Config format: 'b16[layers=9.10.11]'
    Default layers per model: t16/s16/b16=[2,5,8,11]
    """

    DEFAULT_LAYERS = {
        't16': [2, 5, 8, 11],
        's16': [2, 5, 8, 11],
        'b16': [2, 5, 8, 11],
    }

    def load_model(self):
        from .models.eupe_loader import load_eupe
        base_config, flags = self._parse_config()
        model_name = f"eupe_vit{base_config}"
        self.model = load_eupe(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._embed_dim = self.model.embed_dim
        self.patch_size = 16
        self.model.norm = nn.LayerNorm(self._embed_dim, elementwise_affine=False)
        # parse layer indices
        layers_flag = [f for f in flags if f.startswith('layers=')]
        if layers_flag:
            self.layer_indices = [int(i) for i in layers_flag[0].split('=')[1].split('.')]
        else:
            self.layer_indices = self.DEFAULT_LAYERS.get(base_config, [2, 5, 8, 11])

    def _parse_config(self):
        import re
        match = re.match(r'^(.+?)\[([^\]]+)\]$', self.model_config)
        if match:
            base = match.group(1)
            flags = [f.strip() for f in match.group(2).split(',')]
            return base, flags
        return self.model_config, []

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        outputs = self.model.get_intermediate_layers(
            x, n=self.layer_indices, reshape=False,
            return_class_token=False, norm=True
        )
        patch_tokens = torch.stack(outputs, dim=0).sum(dim=0)
        return {
            'x_norm_clstoken': patch_tokens.mean(dim=1),
            'x_norm_patchtokens': patch_tokens,
        }


class TIPSEncoder(VisionEncoder):
    """TIPSv2 vision encoder from Google DeepMind (loaded from HuggingFace)."""

    def load_model(self):
        from .models.tips_loader import load_tipsv2
        self.model = load_tipsv2(self.model_config)

        self._embed_dim = self.model.embed_dim
        self.patch_size = 14
        self.model.norm = nn.LayerNorm(self._embed_dim, elementwise_affine=False)
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        target = 14 * (self.resolution // 16)
        x = torch.nn.functional.interpolate(x, target, mode='bicubic')
        return x

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward_features(x)
        cls = out.get('x_norm_1st_clstoken')
        if cls is not None and cls.dim() == 3:
            cls = cls.squeeze(1)
        return {
            'x_norm_clstoken': cls,
            'x_norm_patchtokens': out['x_norm_patchtokens'],
        }


class CLIPEncoder(VisionEncoder):
    """CLIP encoder; matches RAEv2 implementation exactly (clip.load + UpdatedVisionTransformer)."""

    def load_model(self):
        import clip
        from .models.clip_vit import UpdatedVisionTransformer

        encoder_ = clip.load(f"ViT-{self.model_config}/14", device='cpu')[0].visual
        self.model = UpdatedVisionTransformer(encoder_).to(self.device)
        self._embed_dim = self.model.model.transformer.width
        self.patch_size = 14
        self.model.eval()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        resolution = x.shape[-1]
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
        return x

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward(x)
        cls_token = out[:, 0]
        patch_tokens = out[:, 1:]
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens,
        }


class MoCoV3Encoder(VisionEncoder):
    """MoCoV3 ViT encoder; loads local pretrained checkpoint."""

    def load_model(self):
        from .encoder_utils import fix_mocov3_state_dict
        from .models import mocov3_vit

        if self.model_config == 's':
            self.model = mocov3_vit.vit_small()
        elif self.model_config == 'b':
            self.model = mocov3_vit.vit_base()
        elif self.model_config == 'l':
            self.model = mocov3_vit.vit_large()
        else:
            raise ValueError(f"Unknown MoCoV3 model config: {self.model_config}")

        ckpt = torch.load(f'./pretrained_models/encoders/mocov3/mocov3_vit{self.model_config}.pth',
                          map_location='cpu', weights_only=False)
        state_dict = fix_mocov3_state_dict(ckpt['state_dict'])
        del self.model.head
        self.model.load_state_dict(state_dict, strict=True)
        self.model.head = torch.nn.Identity()

        self.model = self.model.to(self.device)
        self.model.eval()
        self.patch_size = 16
        self._embed_dim = self.model.embed_dim

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 256 * (self.resolution // 256), mode='bicubic')
        return x

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward_features(x)
        cls_token = out[:, 0]
        patch_tokens = out[:, 1:]
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens,
        }


class JEPAEncoder(VisionEncoder):
    """I-JEPA ViT-H encoder; loads local pretrained checkpoint."""

    def load_model(self):
        from .models.jepa import vit_huge

        if self.model_config != 'h':
            raise ValueError(f"Only JEPA ViT-H is supported (got {self.model_config})")

        self.model = vit_huge(img_size=[224, 224], patch_size=14).to(self.device)

        with open(f"pretrained_models/encoders/ijepa/ijepa_vit{self.model_config}.pth", "rb") as f:
            state_dict = torch.load(f, map_location=self.device, weights_only=False)

        new_state_dict = {k[7:]: v for k, v in state_dict['encoder'].items()}
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        self._embed_dim = self.model.embed_dim
        self.patch_size = 14

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (self.resolution // 256), mode='bicubic')
        return x

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward(x)
        return {
            'x_norm_clstoken': None,
            'x_norm_patchtokens': out,
        }


# Registry mapping encoder types to classes
ENCODER_REGISTRY = {
    # dinov2 and dinov3 encoders
    'dinov2': DINOv2Encoder,
    'dinov2mls': DINOv2MultiLayerSimpleAddEncoder,
    'dinov3': DINOv3Encoder,
    'dinov3mls': DINOv3MultiLayerSimpleAddEncoder,
    'siglip2': SigLIP2Encoder,
    'siglip2mls': SigLIP2MultiLayerSimpleAddEncoder,
    'mae': MAEEncoder,
    # webssl encoder
    'webssl': WebSSLEncoder,
    # PE encoders
    'pe': PEEncoder,
    'spatialpe': PEEncoder,
    'langpe': PEEncoder,
    # EUPE encoder
    'eupe': EUPEEncoder,
    'eupemls': EUPEMultiLayerSimpleAddEncoder,
    # TIPS encoders
    'tipsv2': TIPSEncoder,
    # supervised / contrastive encoders
    'clip': CLIPEncoder,
    'mocov3': MoCoV3Encoder,
    'jepa': JEPAEncoder,
}


def create_encoder(encoder_string: str, device: torch.device,
                   resolution: int = 256, accelerator=None) -> VisionEncoder:
    """
    Factory function to create encoder from string specification

    Args:
        encoder_string: Format "encoder_type-architecture-model_config"
        device: torch device
        resolution: Input image resolution
        accelerator: Optional accelerator for distributed training

    Returns:
        VisionEncoder instance
    """
    parts = encoder_string.split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid encoder string format: {encoder_string}. "
                        f"Expected format: encoder_type-architecture-model_config")

    encoder_type, architecture, model_config = parts

    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Available types: {list(ENCODER_REGISTRY.keys())}")

    encoder_class = ENCODER_REGISTRY[encoder_type]
    encoder = encoder_class(encoder_type, architecture, model_config,
                            device, resolution, accelerator)
    encoder.load_model()

    return encoder


@torch.no_grad()
def load_encoders(enc_type: str, device: torch.device, resolution: int = 256,
                  accelerator=None) -> List[VisionEncoder]:
    """
    Load multiple encoders from comma-separated string

    Args:
        enc_type: Comma-separated encoder specifications
        device: torch device
        resolution: Input image resolution
        accelerator: Optional accelerator for distributed training

    Returns:
        List of VisionEncoder instances
    """
    enc_names = enc_type.split(',')
    encoders = []

    for enc_name in enc_names:
        # Parse encoder specification
        parts = enc_name.split('-')
        if len(parts) != 3:
            raise ValueError(f"Invalid encoder format: {enc_name}")

        encoder = create_encoder(enc_name, device, resolution, accelerator)
        encoder.eval()
        encoders.append(encoder)

    return encoders
