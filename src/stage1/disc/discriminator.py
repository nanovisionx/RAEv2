from __future__ import annotations

import torch

from .dinodisc import DinoDisc


class DinoDiscriminator(DinoDisc):
    """Thin wrapper aligning with legacy API expecting (fake, real) outputs."""

    def __init__(self, device: torch.device, **kwargs):
        super().__init__(device=device, **kwargs)

    def classify(self, img: torch.Tensor) -> torch.Tensor:
        return super().forward(img)

    def forward(self, fake: torch.Tensor, real: torch.Tensor | None = None):
        logits_fake = self.classify(fake)
        logits_real = self.classify(real) if real is not None else None
        return logits_fake, logits_real
