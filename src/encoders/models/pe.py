from collections import OrderedDict
from dataclasses import asdict
from functools import partial
from logging import getLogger
from typing import Callable, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import DropPath
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.utils.checkpoint import checkpoint

from .pe_config import PE_VISION_CONFIG, PEConfig, fetch_pe_checkpoint
from .rope import Rope2D

logger = getLogger()



class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.dim = dim
        self.init_values = init_values

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

    def init_tensors(self):
        self.gamma = nn.Parameter(self.init_values * torch.ones(self.dim))


class AttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert (
            self.embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.probe = nn.Parameter(torch.randn(1, num_probe, self.embed_dim))
        self.attn = nn.MultiheadAttention(
            self.embed_dim, self.num_heads, batch_first=True
        )

        self.layernorm = norm_layer(embed_dim)
        self.mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(self.embed_dim, self.mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(self.mlp_width, self.embed_dim)),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        batch, _, _ = x.shape

        q = self.probe.repeat((batch, 1, 1)).to(x.dtype)
        x = self.attn(q, x, x, need_weights=False)[0]
        x = x + self.mlp(self.layernorm(x))

        return x


class SelfAttention(nn.Module):
    r"""
    Implements sequence packed attention and RoPe
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope: Optional[nn.Module] = None,
    ):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # To make this compatibile with nn.MultiHeadAttention
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.rope = rope
        self.scale = self.head_dim ** (-0.5)

    def init_tensors(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(self, x, attn_mask=None):
        batch, seq, embed_dim = x.shape
        proj = F.linear(x, self.in_proj_weight, self.in_proj_bias)

        # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
        proj = (
            proj.unflatten(-1, (3, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        q, k, v = proj[0], proj[1], proj[2]

        # Use "q_" so that we don't accidentally quit in pdb :)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if self.rope:
            q, k = self.rope(q, k)

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
        )
        attn = rearrange(attn, "b h s d -> b s (h d)")

        return F.linear(attn, self.out_proj.weight, self.out_proj.bias)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()

        if rope:
            self.attn = SelfAttention(d_model, n_head, rope=rope)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )

    def _call_attn(
        self,
        q_x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):

        if attn_mask is not None:
            # Leave boolean masks as is
            if not attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(q_x.dtype)

        if isinstance(self.attn, SelfAttention):
            return self.attn(q_x, attn_mask=attn_mask)
        else:
            return self.attn(q_x, q_x, q_x, attn_mask=attn_mask, need_weights=False)[0]

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        x = x + self.drop_path1(
            self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask))
        )
        x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                    rope=rope,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
    ):
        stop_idx = (self.layers + layer_idx) % self.layers

        for i, r in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)

            if i == stop_idx:
                break

        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5),
        use_ln_pre: bool = True,
        use_ln_post: bool = True,
        ls_init_value: float = None,
        drop_path: float = 0.0,
        image_size: int = 448,  # Pretrain image size only; you can pass in any image size
        use_abs_posemb: bool = True,
        use_rope2d: bool = True,
        use_cls_token: bool = False,
        output_dim: Optional[int] = 1280,
        attn_pooler_heads: int = 8,
        pool_type: Literal["attn", "tok", "avg", "none"] = "attn",
    ):
        super().__init__()
        assert pool_type in ("attn", "tok", "avg", "none")
        self.pool_type = pool_type
        self.patch_size = patch_size

        self.output_dim = output_dim or width
        self.proj_dim = output_dim
        self.heads = heads
        self.width = width
        self.layers = layers

        self.use_abs_posemb = use_abs_posemb
        self.use_cls_token = use_cls_token
        self.use_rope2d = use_rope2d
        self.image_size = image_size

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.rope = (
            Rope2D(
                dim=width // heads,
                use_cls_token=self.use_cls_token,
            )
            if self.use_rope2d
            else None
        )

        self.ln_pre = norm_layer(width) if use_ln_pre else nn.Identity()
        self.ln_post = norm_layer(self.width) if use_ln_post else nn.Identity()

        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path,
            rope=self.rope,
        )

        if pool_type == "attn":
            self.attn_pool = AttentionPooling(
                embed_dim=width,
                num_heads=attn_pooler_heads,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None

        self.init_tensors()


    def init_tensors(self):
        def init_submodule_tensors(module):
            for name, child in module.named_children():
                if hasattr(child, "init_tensors"):
                    logger.debug(f"Initializing tensors for submodule: {name}")
                    child.init_tensors()
                init_submodule_tensors(child)

        init_submodule_tensors(self)
        self.rope.init_tensors()

        # class embeddings and positional embeddings
        init_scale = self.width**-0.5

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(init_scale * torch.randn(self.width))

        if self.use_abs_posemb:
            self.posemb_grid_size = self.image_size // self.patch_size
            self.positional_embedding = nn.Parameter(
                init_scale
                * torch.randn(
                    int(self.use_cls_token) + self.posemb_grid_size**2, self.width
                )
            )

        if self.proj_dim is not None:
            self.proj = nn.Parameter(
                init_scale * torch.randn(self.width, self.proj_dim)
            )


    def load_ckpt(self, ckpt_path: str, verbose: bool = True):
        _sd = torch.load(ckpt_path, weights_only=True)
        if "state_dict" in _sd:
            _sd = _sd["state_dict"]
        elif "weights" in _sd:
            _sd = _sd["weights"]

        # for backwards compatibility
        _sd = {k.replace("module.", ""): v for k, v in _sd.items()}
        if any(k.startswith("visual.") for k in _sd):
            _sd = {k.replace("visual.", ""): v for k, v in _sd.items() if "visual" in k}

        m, u = self.load_state_dict(_sd, strict=False)

        if verbose or (m or u):
            logger.info(f"Missing keys for loading vision encoder: {m}")
            logger.info(f"Unexpected keys for loading vision encoder: {u}")
            print(f"Missing keys for loading vision encoder: {m}")
            print(f"Unexpected keys for loading vision encoder: {u}")


    @classmethod
    def from_config(
        cls,
        name: str,
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None,
        **kwdargs
    ):
        if name not in PE_VISION_CONFIG:
            raise RuntimeError(f"{name} not found in configs.")

        args = asdict(PE_VISION_CONFIG[name])
        args.update(kwdargs)

        model = cls(**args)
        if pretrained:
            model.load_ckpt(fetch_pe_checkpoint(name, checkpoint_path))

        return model

    def _sample_abs_posemb(self, grid_h: int, grid_w: int):
        """Interpolates the absolute position embedding if necessary."""
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]

        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pos_embed = F.interpolate(
            pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.width).contiguous()

        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)

        return pos_embed[None, ...]

    def _pool(self, x: torch.Tensor):
        if self.pool_type == "tok":
            return x[:, 0]
        elif self.pool_type == "avg":
            return x.mean(dim=1)
        elif self.pool_type == "attn":
            return self.attn_pool(x).squeeze(1)
        elif self.pool_type == "none":
            return x
        else:
            raise NotImplementedError

    def forward_features(
        self,
        x: torch.Tensor,
        norm: bool = False,
        layer_idx: int = -1,
        strip_cls_token: bool = False
    ):
        batch, _, h, w = x.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, self.width)

        if self.use_cls_token:
            x = torch.cat(
                [self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x],
                dim=1,
            )

        if self.use_abs_posemb:
            x = x + self._sample_abs_posemb(grid_h, grid_w)

        if self.use_rope2d:
            self.rope.update_grid(x.device, grid_h, grid_w)

        x = self.ln_pre(x)
        x = self.transformer(x, layer_idx=layer_idx)

        if norm:
            x = self.ln_post(x)

        if strip_cls_token and self.use_cls_token:
            x = x[:, 1:, :]

        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.forward_features(x, norm=True, **kwargs)
        x = self._pool(x)

        if self.proj_dim is not None:
            x = x @ self.proj

        return x




