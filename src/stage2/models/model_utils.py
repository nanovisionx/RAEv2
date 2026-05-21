import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class RoPE(nn.Module):
    def __init__(self, dim, vis_len, cond_len=0, theta=10000.,):
        super().__init__()
        # 2D RoPE for vision
        d, T = dim // 2, int(vis_len ** 0.5)
        vis_freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))  # [D//4]
        vis_base_angles = torch.outer(torch.arange(T).float(), vis_freqs)  # [T, D//4]
        vis_angles = torch.cat([
            vis_base_angles[:, None].expand(-1, T, -1),
            vis_base_angles[None, :].expand(T, -1, -1)
        ], dim=-1).reshape(vis_len, d)  # [T, T, D//2] -> [L', D//2]
        # no PE for conds
        cond_angles = torch.zeros(cond_len, dim // 2)  # [cond_len, D//2]
        angles = torch.cat([vis_angles, cond_angles], dim=0).repeat_interleave(2, dim=-1)  # [L, D]
        self.register_buffer("freqs_cos", angles.cos())
        self.register_buffer("freqs_sin", angles.sin())

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class NormAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads, self.dim, self.head_dim = num_heads, dim, dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope, attn_mask=None):
        B, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = rope(q), rope(k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        return self.proj(out)


class GaussianFourierEmbedding(nn.Module):
    def __init__(self, hidden_size, n_tokens=4, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.normal(0, scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.learnable_tokens = nn.Parameter(torch.normal(0, 1 / hidden_size**0.5, (n_tokens, hidden_size)))

    def forward(self, t, return_base_embed=False):
        t = t[:, None] * self.W[None, :] * 2 * torch.pi
        t_embed = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        t_embed = self.mlp(t_embed)
        if return_base_embed:
            t_embed = t_embed.unsqueeze(1)
            return t_embed, self.learnable_tokens + t_embed
        else:
            return self.learnable_tokens + t_embed.unsqueeze(1)


class ConditionEmbedder(nn.Module):
    def __init__(self, hidden_size, num_classes=1000, context_dim=768, condition_type="label", n_tokens=8,
                 latent_in_channels=768, latent_patch_size=1, n_action_tokens=4):
        super().__init__()
        self.condition_type = condition_type
        self.hidden_size = hidden_size

        if condition_type == "label":
            self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
            self.learnable_tokens = nn.Parameter(torch.normal(0, 1 / hidden_size**0.5, (n_tokens, hidden_size)))
        elif condition_type == "text":
            self.norm = RMSNorm(context_dim)
            self.proj = nn.Linear(context_dim, hidden_size)
        elif condition_type == "nwm":
            # Each (already RAE-encoded) context latent frame -> hidden_size patch tokens.
            self.context_patch_embed = nn.Conv2d(
                latent_in_channels, hidden_size,
                kernel_size=latent_patch_size, stride=latent_patch_size,
            )
            self.n_action_tokens = n_action_tokens
            self.action_proj = nn.Linear(3, hidden_size)
            self.action_tokens = nn.Parameter(
                torch.normal(0, 1 / hidden_size**0.5, (n_action_tokens, hidden_size))
            )
            self.time_emb = GaussianFourierEmbedding(hidden_size, n_tokens=1)
        else:
            raise ValueError(f"Unknown condition_type: {condition_type}")

    def forward(self, y) -> torch.Tensor:
        if self.condition_type == "nwm":
            ctx = y["context_latents"]                              # (B, K, C, H, W)
            B, K = ctx.shape[:2]
            patches = self.context_patch_embed(ctx.flatten(0, 1))   # (B*K, hidden, h', w')
            patches = patches.flatten(2).transpose(1, 2)            # (B*K, P, hidden)
            patches = patches.reshape(B, K * patches.shape[1], -1)  # (B, K*P, hidden)
            act_tokens = self.action_tokens + self.action_proj(y["action"]).unsqueeze(1)
            time_token = self.time_emb(y["rel_time"].squeeze(-1))   # (B, 1, hidden)
            return torch.cat([patches, act_tokens, time_token], dim=1)
        if self.condition_type == "label":
            return self.learnable_tokens + self.embedding_table(y).unsqueeze(1)
        else:
            return self.proj(self.norm(y))
