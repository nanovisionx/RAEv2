import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from .model_utils import ConditionEmbedder, GaussianFourierEmbedding, NormAttention, RMSNorm, RoPE, SwiGLUFFN


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DDTEncoderBlock(nn.Module):
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


class DDTDecoderBlock(DDTEncoderBlock):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__(hidden_size, num_heads, mlp_ratio)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6*hidden_size)
        )

    def forward(self, x, c, rope, attn_mask=None):
        modulation = self.adaln_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=rope, attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaln_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class DiTwDDTHead(nn.Module):
    def __init__(
        self,
        input_size=16,
        in_channels=768,
        patch_size=[1, 1],
        hidden_size=[1152, 2048],
        depth=[28, 2],
        num_heads=[16, 16],
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
        self.enc_hidden_size, dec_hidden_size = hidden_size
        self.num_enc_blocks, self.num_dec_blocks = depth
        self.s_patch_size, self.x_patch_size = patch_size
        enc_num_heads, dec_num_heads = num_heads

        self.repa_layer_depth = repa_layer_depth
        self.use_cfg_conds = use_cfg_conds

        self.s_embedder = PatchEmbed(input_size, self.s_patch_size, in_channels, self.enc_hidden_size)
        self.x_embedder = PatchEmbed(input_size, self.x_patch_size, in_channels, dec_hidden_size)
        self.s_projector = nn.Linear(self.enc_hidden_size, dec_hidden_size) if self.enc_hidden_size != dec_hidden_size else nn.Identity()

        self.num_cond_tokens = cond_arch.num_t_tokens + cond_arch.num_c_tokens
        self.t_embedder = GaussianFourierEmbedding(self.enc_hidden_size, cond_arch.num_t_tokens)
        self.ctx_embedder = ConditionEmbedder(
            self.enc_hidden_size, num_classes, context_dim, condition_type, cond_arch.num_c_tokens,
            latent_in_channels=in_channels,
            latent_patch_size=self.s_patch_size,
            n_action_tokens=getattr(cond_arch, "n_action_tokens", 4),
        )
        if self.use_cfg_conds:
            self.num_cond_tokens += cond_arch.num_cfg_omega_tokens
            self.cfg_w_embedder = GaussianFourierEmbedding(self.enc_hidden_size, cond_arch.num_cfg_omega_tokens)

        self.blocks = []
        for _ in range(self.num_enc_blocks):
            self.blocks.append(DDTEncoderBlock(self.enc_hidden_size, enc_num_heads, mlp_ratio))
        for _ in range(self.num_dec_blocks):
            self.blocks.append(DDTDecoderBlock(dec_hidden_size, dec_num_heads, mlp_ratio))
        self.blocks = nn.ModuleList(self.blocks)

        self.final_layer = DDTFinalLayer(dec_hidden_size, self.x_patch_size, in_channels)
        self.enc_rope = RoPE(self.enc_hidden_size // enc_num_heads, self.s_embedder.num_patches, self.num_cond_tokens)
        self.dec_rope = RoPE(dec_hidden_size // dec_num_heads, self.x_embedder.num_patches)
        if enable_repa:
            self.repa_projector = nn.Linear(self.enc_hidden_size, z_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Patch embedders
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Condition embedders
        if hasattr(self.ctx_embedder, "mlp"):
            nn.init.normal_(self.ctx_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.ctx_embedder.mlp[2].weight, std=0.02)
        if hasattr(self.ctx_embedder, "embedding_table"):
            nn.init.normal_(self.ctx_embedder.embedding_table.weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            if hasattr(block, "adaln_modulation"):
                nn.init.constant_(block.adaln_modulation[-1].weight, 0)
                nn.init.constant_(block.adaln_modulation[-1].bias, 0)

        # Timestep embedding MLP
        t_embedders = ["t_embedder", "cfg_w_embedder"]
        for t_embedder in t_embedders:
            if hasattr(self, t_embedder):
                nn.init.normal_(getattr(self, t_embedder).mlp[0].weight, std=0.02)
                nn.init.normal_(getattr(self, t_embedder).mlp[2].weight, std=0.02)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """[N, T, patch_size**2 * C] -> [N, C, H, W]"""
        h, c = int(x.shape[1] ** 0.5), self.in_channels
        x = x.reshape(x.shape[0], h, h, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, h*p, h*p)
        return x

    def _build_sequence(self, x, t, condition_kwargs):
        """Returns sequence concatenated with all condition tokens, and the base timestep embedding (no learnable tokens)"""
        seq = []
        seq.append(self.s_embedder(x))
        t_emb_base, t_emb = self.t_embedder(t, return_base_embed=True)
        seq.append(t_emb)
        if self.use_cfg_conds:
            seq.append(self.cfg_w_embedder(condition_kwargs["omega"]))
        seq.append(self.ctx_embedder(condition_kwargs["context"]))
        seq = torch.cat(seq, dim=1)
        return seq, t_emb_base

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
        seq, t_emb_base = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(seq, condition_kwargs)
        for i in range(self.num_enc_blocks):
            seq = self.blocks[i](seq, self.enc_rope, attn_mask)
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                zt_intermediate = self.repa_projector(seq[:, :self.s_embedder.num_patches, :])
        seq = self.s_projector(F.silu(t_emb_base + seq[:, :self.s_embedder.num_patches, :]))

        x = self.x_embedder(x)
        for i in range(self.num_dec_blocks):
            x = self.blocks[self.num_enc_blocks + i](x, seq, self.dec_rope)

        x = self.final_layer(x, seq)
        x = self.unpatchify(x, self.x_patch_size)

        if return_intermediate:
            return x, zt_intermediate
        return x


class DiTwDDTHeadIG(DiTwDDTHead):
    def __init__(self, base_model_depth=8, **kwargs):
        super().__init__(**kwargs)
        self.base_model_depth = base_model_depth

        self.base_final_layer = DDTFinalLayer(self.enc_hidden_size, self.s_patch_size, self.in_channels)
        nn.init.constant_(self.base_final_layer.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.base_final_layer.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.base_final_layer.linear.weight, 0)
        nn.init.constant_(self.base_final_layer.linear.bias, 0)

    def forward(self, x, t, return_intermediate=False, **condition_kwargs):
        zt_intermediate = None
        x_base = None
        seq, t_emb_base = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(seq, condition_kwargs)
        for i in range(self.num_enc_blocks):
            seq = self.blocks[i](seq, self.enc_rope, attn_mask)
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                zt_intermediate = self.repa_projector(seq[:, :self.s_embedder.num_patches, :])
            if (i + 1) == self.base_model_depth:
                x_base = seq[:, :self.s_embedder.num_patches, :]
        seq = self.s_projector(F.silu(t_emb_base + seq[:, :self.s_embedder.num_patches, :]))

        x = self.x_embedder(x)
        for i in range(self.num_dec_blocks):
            x = self.blocks[self.num_enc_blocks + i](x, seq, self.dec_rope)

        x = self.final_layer(x, seq)
        x = self.unpatchify(x, self.x_patch_size)

        x_base = F.silu(t_emb_base + x_base)
        x_base = self.base_final_layer(x_base, x_base)
        x_base = self.unpatchify(x_base, self.s_patch_size)

        if return_intermediate:
            return (x, x_base), zt_intermediate
        return x, x_base
