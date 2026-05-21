import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

from .model_utils import ConditionEmbedder, GaussianFourierEmbedding, NormAttention, RMSNorm, RoPE, SwiGLUFFN


class LightningDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = NormAttention(hidden_size, num_heads)
        self.mlp = SwiGLUFFN(hidden_size, int(2/3 * hidden_size * mlp_ratio))

    def forward(self, x, rope, attn_mask=None):
        x = x + self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class LightningFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x):
        return self.linear(self.norm_final(x))


class LightningDiT(nn.Module):
    def __init__(
        self,
        input_size=16,
        in_channels=768,
        patch_size=1,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        enable_repa=False,
        repa_layer_depth=8,
        z_dim=None,
        num_classes=1000,
        condition_type="label",
        context_dim=768,
        cond_arch=None,
        use_cfg_conds=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.repa_layer_depth = repa_layer_depth
        self.use_cfg_conds = use_cfg_conds

        self.x_embedder = PatchEmbed(input_size, self.patch_size, in_channels, hidden_size)

        self.num_cond_tokens = cond_arch.num_t_tokens + cond_arch.num_c_tokens
        self.t_embedder = GaussianFourierEmbedding(self.hidden_size, cond_arch.num_t_tokens)
        self.ctx_embedder = ConditionEmbedder(
            self.hidden_size, num_classes, context_dim, condition_type, cond_arch.num_c_tokens
        )
        if self.use_cfg_conds:
            self.num_cond_tokens += cond_arch.num_cfg_omega_tokens
            self.cfg_w_embedder = GaussianFourierEmbedding(self.hidden_size, cond_arch.num_cfg_omega_tokens)

        self.blocks = nn.ModuleList([
            LightningDiTBlock(self.hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.final_layer = LightningFinalLayer(self.hidden_size, self.patch_size, self.in_channels)
        self.rope = RoPE(self.hidden_size // num_heads, self.x_embedder.num_patches, self.num_cond_tokens)
        if enable_repa:
            self.repa_projector = nn.Linear(self.hidden_size, z_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Patch embedders
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Condition embedders
        if hasattr(self.ctx_embedder, "mlp"):
            nn.init.normal_(self.ctx_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.ctx_embedder.mlp[2].weight, std=0.02)
        if hasattr(self.ctx_embedder, "embedding_table"):
            nn.init.normal_(self.ctx_embedder.embedding_table.weight, std=0.02)

        # Timestep embedding MLP
        t_embedders = ["t_embedder", "cfg_w_embedder"]
        for t_embedder in t_embedders:
            if hasattr(self, t_embedder):
                nn.init.normal_(getattr(self, t_embedder).mlp[0].weight, std=0.02)
                nn.init.normal_(getattr(self, t_embedder).mlp[2].weight, std=0.02)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """[N, T, patch_size**2 * C] -> [N, C, H, W]"""
        h, c, p = int(x.shape[1] ** 0.5), self.in_channels, self.patch_size
        x = x.reshape(x.shape[0], h, h, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, h*p, h*p)
        return x

    def _build_sequence(self, x, t, condition_kwargs):
        seq = []
        seq.append(self.x_embedder(x))
        seq.append(self.t_embedder(t))
        if self.use_cfg_conds:
            seq.append(self.cfg_w_embedder(condition_kwargs["omega"]))
        seq.append(self.ctx_embedder(condition_kwargs["context"]))
        seq = torch.cat(seq, dim=1)
        return seq

    def _build_attn_mask(self, seq, condition_kwargs):
        # Create multiplicative mask template
        attn_mask = torch.ones((seq.shape[0], seq.shape[1]), device=seq.device)
        cond_mask = condition_kwargs.get("attn_mask")
        if cond_mask is not None:
            attn_mask[:, -cond_mask.shape[1]:] = cond_mask
        # Convert to additive mask
        attn_mask = (1.0 - attn_mask[:, None, None, :]) * torch.finfo(seq.dtype).min
        return attn_mask

    def forward(self, x, t, return_intermediate=False, **condition_kwargs):
        zt_intermediate = None
        x = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(x, condition_kwargs)
        for i, block in enumerate(self.blocks):
            x = block(x, self.rope, attn_mask)
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                zt_intermediate = self.repa_projector(x[:, :self.x_embedder.num_patches, :])

        x = x[:, :self.x_embedder.num_patches, :]
        x = self.final_layer(x)
        x = self.unpatchify(x)

        if return_intermediate:
            return x, zt_intermediate
        return x


class LightningDiTIG(LightningDiT):
    def __init__(self, base_model_depth=8, **kwargs):
        super().__init__(**kwargs)
        self.base_model_depth = base_model_depth

        self.base_final_layer = LightningFinalLayer(self.hidden_size, self.patch_size, self.in_channels)
        nn.init.constant_(self.base_final_layer.linear.weight, 0)
        nn.init.constant_(self.base_final_layer.linear.bias, 0)

    def forward(self, x, t, return_intermediate=False, **condition_kwargs):
        zt_intermediate = None
        x_base = None
        x = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(x, condition_kwargs)
        for i, block in enumerate(self.blocks):
            x = block(x, self.rope, attn_mask)
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                zt_intermediate = self.repa_projector(x[:, :self.x_embedder.num_patches, :])
            if (i + 1) == self.base_model_depth:
                x_base = x[:, :self.x_embedder.num_patches, :]

        x = x[:, :self.x_embedder.num_patches, :]
        x = self.final_layer(x)
        x = self.unpatchify(x)

        x_base = self.base_final_layer(x_base)
        x_base = self.unpatchify(x_base)

        if return_intermediate:
            return (x, x_base), zt_intermediate
        return x, x_base
