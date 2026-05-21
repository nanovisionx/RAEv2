from math import pi
from typing import Literal, Optional, Union

import torch
from einops import rearrange, repeat
from torch import Tensor, einsum, nn
from torch.amp import autocast
from torch.nn import Module


def exists(val):
    return val is not None


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t_left, t, t_right), dim=-1)

    return out.type(dtype)


class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs: Optional[Tensor] = None,
        freqs_for: Union[Literal["lang"], Literal["pixel"], Literal["constant"]] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        cache_if_possible=True,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible

        self.tmp_store("cached_freqs", None)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.tmp_store("dummy", torch.tensor(0))

        # interpolation factors

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent=False)

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        should_cache = (
            self.cache_if_possible
            and not self.learned_freq
            and exists(seq_len)
            and self.freqs_for != "pixel"
        )

        if (
            should_cache
            and exists(self.cached_freqs)
            and (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            return self.cached_freqs[offset : (offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if should_cache:
            self.tmp_store("cached_freqs", freqs.detach())

        return freqs


class Rope2D:
    """Helper class to apply RoPE2D as well as interpolate on the fly."""

    def __init__(self, dim, use_cls_token=False):
        self.dim = dim
        self.use_cls_token = use_cls_token
        self.grid_size = None
        self.freq = None

    def init_tensors(self):
        self.rope = RotaryEmbedding(self.dim // 2)

    def update_grid(self, device, grid_h, grid_w):
        if self.grid_size != (grid_h, grid_w):
            self.grid_size = (grid_h, grid_w)

            self.rope = self.rope.to(device)

            if self.use_cls_token:
                # +1 to leave space for the cls token to be (0, 0)
                grid_y_range = torch.arange(grid_h, device=device) + 1
                grid_x_range = torch.arange(grid_w, device=device) + 1
            else:
                grid_y_range = torch.arange(grid_h, device=device)
                grid_x_range = torch.arange(grid_w, device=device)

            freqs_y = self.rope(grid_y_range)[:, None].expand(grid_h, grid_w, -1)
            freqs_x = self.rope(grid_x_range)[None, :].expand(grid_h, grid_w, -1)
            freq = torch.cat([freqs_x, freqs_y], dim=-1).reshape(grid_h * grid_w, -1)

            if self.use_cls_token:
                freq = torch.cat(
                    [torch.zeros(1, freq.shape[-1], device=device), freq], dim=0
                )

            self.freq = freq[None, ...]

        self.freq = self.freq.to(device)

    def __call__(self, q, k):
        # batch, heads, seq, dim = q.shape
        q = apply_rotary_emb(self.freq[:, None, :, :], q)
        k = apply_rotary_emb(self.freq[:, None, :, :], k)

        return q, k
