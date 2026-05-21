from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm

from .utils import RandomWindowCrop

fused_mlp_func = None
flash_attn_qkvpacked_func = None


def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):

    attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL

    if attn_mask is not None: attn.add_(attn_mask)

    return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value


class MLPNoDrop(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func # None for TPU
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.fused_mlp_func(
                x=x,
                weight1=self.fc1.weight,
                weight2=self.fc2.weight,
                bias1=self.fc1.bias,
                bias2=self.fc2.bias,
                activation='gelu_approx',
                save_pre_act=self.training,
                return_residual=False,
                checkpoint_lvl=0,
                heuristic=0,
                process_group=None,
            )
        else:
            return self.fc2(self.act(self.fc1(x)))

    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class SelfAttentionNoDrop(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12, flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads  # =64
        self.scale = 1 / math.sqrt(self.head_dim)
        self.qkv, self.proj = nn.Linear(embed_dim, embed_dim * 3, bias=True), nn.Linear(embed_dim, embed_dim, bias=True)
        self.using_flash_attn = False

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        if self.using_flash_attn and qkv.dtype != torch.float32:
            oup = flash_attn_qkvpacked_func(qkv, softmax_scale=self.scale).view(B, L, C)
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # BHLc
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
        return self.proj(oup)

    def extra_repr(self) -> str:
        return f'using_flash_attn={self.using_flash_attn}'

class SABlockNoDrop(nn.Module):
    def __init__(self, block_idx, embed_dim, num_heads, mlp_ratio, norm_eps):
        super(SABlockNoDrop, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = SelfAttentionNoDrop(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, flash_if_available=True)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = MLPNoDrop(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), fused_if_available=True)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.ratio = 1 / np.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.float()
        return (self.fn(x).add(x)).mul_(self.ratio)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 1, eps: float = 1e-6):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.float()

        # Reshape batch into groups.
        G = np.ceil(x.size(0) / self.virtual_bs).astype(int) # disbale ghostnorm , for there will be division problem when bs % virtual_bs != 0
        x = x.view(G, -1, x.size(-2), x.size(-1))
        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


recipes = {
    "S_16": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "norm_eps": 1e-6,
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 384,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "S_8": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "norm_eps": 1e-6,
        "patch_size": 8,
        "in_chans": 3,
        "embed_dim": 384,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "B_16": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "norm_eps": 1e-6,
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 768,
        "num_heads": 12,
        "mlp_ratio": 4.0,
    },
}


def make_block(channels: int, kernel_size: int, norm_type: str, norm_eps: float, using_spec_norm: bool) -> nn.Module:
    if norm_type == "bn":
        norm = BatchNormLocal(channels, eps=norm_eps)
    elif norm_type == "gn":
        norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=norm_eps, affine=True)
    else:
        raise NotImplementedError(f"Unknown norm_type '{norm_type}'")

    conv = SpectralConv1d if using_spec_norm else nn.Conv1d
    return nn.Sequential(
        conv(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode="circular"),
        norm,
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class DinoDisc(nn.Module):
    def __init__(
        self,
        device: torch.device,
        dino_ckpt_path: str,
        ks: int,
        key_depths=(2, 5, 8, 11),
        norm_type="bn",
        using_spec_norm=True,
        norm_eps=1e-6,
        recipe: str = "S_16",
    ):
        super().__init__()
        state = torch.load(dino_ckpt_path, map_location="cpu")
        for key in sorted(state.keys()):
            if ".attn.qkv.bias" in key:
                bias = state[key]
                C = bias.numel() // 3
                bias[C : 2 * C].zero_()

        recipe_cfg = dict(recipes[recipe])
        key_depths = tuple(d for d in key_depths if d < recipe_cfg["depth"])
        recipe_cfg.update({"key_depths": key_depths, "norm_eps": norm_eps})
        dino = FrozenDINONoDrop(**recipe_cfg)
        missing, unexpected = dino.load_state_dict(state, strict=False)
        missing = [m for m in missing if all(x not in m for x in {"x_scale", "x_shift"})]
        if missing:
            raise RuntimeError(f"DINO checkpoint missing keys: {missing}")
        if unexpected:
            raise RuntimeError(f"DINO checkpoint has unexpected keys: {unexpected}")
        dino.eval()
        self.dino_proxy: Tuple[FrozenDINONoDrop, ...] = (dino.to(device=device),)
        dino_C = self.dino_proxy[0].embed_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    make_block(dino_C, kernel_size=1, norm_type=norm_type, norm_eps=norm_eps, using_spec_norm=using_spec_norm),
                    ResidualBlock(
                        make_block(dino_C, kernel_size=ks, norm_type=norm_type, norm_eps=norm_eps, using_spec_norm=using_spec_norm)
                    ),
                    (SpectralConv1d if using_spec_norm else nn.Conv1d)(dino_C, 1, kernel_size=1, padding=0),
                )
                for _ in range(len(key_depths) + 1)
            ]
        )
        # train heads
        for p in self.heads:
            p.requires_grad_(True)
        self.dino_proxy[0].requires_grad_(False)

    def forward(self, x_in_pm1: torch.Tensor, grad_ckpt: bool = False) -> torch.Tensor:
        if grad_ckpt and x_in_pm1.requires_grad:
            raise RuntimeError("DINO discriminator does not support grad checkpointing.")
        activations: List[torch.Tensor] = self.dino_proxy[0](x_in_pm1, grad_ckpt=False)
        batch = x_in_pm1.shape[0]
        outputs = []
        for head, act in zip(self.heads, activations):
            out = head(act).view(batch, -1)
            outputs.append(out)
        return torch.cat(outputs, dim=1)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class FrozenDINONoDrop(nn.Module):
    def __init__(
        self,
        depth=12,
        key_depths=(2, 5, 8, 11),
        norm_eps=1e-6,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4.0,
        crop_prob: float = -0.5,
        no_resize: bool = False,
        original_input_size: int | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.img_size = 224
        self.original_input_size = original_input_size if original_input_size is not None else self.img_size
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_size = patch_size
        self.patch_nums = self.img_size // patch_size

        # x \in [-1, 1]
        # x = ((x+1)/2 - m) / s = 0.5x/s + 0.5/s - m/s = (0.5/s) x + (0.5-m)/s
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        self.register_buffer("x_scale", (0.5 / std).reshape(1, 3, 1, 1))
        self.register_buffer("x_shift", ((0.5 - mean) / std).reshape(1, 3, 1, 1))
        self.crop = RandomWindowCrop(self.original_input_size, self.img_size, 9, False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_nums * self.patch_nums + 1, embed_dim))

        self.key_depths = set(d for d in key_depths if d < depth)
        self.blocks = nn.Sequential(
            *[
                SABlockNoDrop(block_idx=i, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, norm_eps=norm_eps)
                for i in range(max(depth, 1 + max(self.key_depths, default=0)))
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.crop_prob = crop_prob
        self.no_resize = no_resize
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def inter_pos_embed(self, patch_nums=(14, 14)):
        if patch_nums[0] == self.patch_nums and patch_nums[1] == self.patch_nums:
            return self.pos_embed
        pe_cls, pe_grid = self.pos_embed[:, :1], self.pos_embed[0, 1:]
        pe_grid = pe_grid.reshape(1, self.patch_nums, self.patch_nums, -1).permute(0, 3, 1, 2)
        pe_grid = F.interpolate(pe_grid, size=patch_nums, mode="bilinear", align_corners=False)
        pe_grid = pe_grid.permute(0, 2, 3, 1).reshape(1, patch_nums[0] * patch_nums[1], -1)
        return torch.cat([pe_cls, pe_grid], dim=1)

    def forward(self, x, grad_ckpt=False):
        if not self.no_resize:
            x = F.interpolate(x, size=(self.original_input_size, self.original_input_size), mode="bilinear", align_corners=False)
            if self.crop_prob > 0 and torch.rand(()) < self.crop_prob:
                x = self.crop(x)
        else:
            if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
                x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = x * self.x_scale + self.x_shift
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if x.shape[1] != self.pos_embed.shape[1]:
            h = w = int(math.sqrt(x.shape[1] - 1))
            pos_embed = self.inter_pos_embed((h, w))
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed

        activations = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.key_depths:
                activations.append(x[:, 1:, :].transpose(1, 2))
        activations.insert(0, x[:, 1:, :].transpose(1, 2))
        return activations
