"""LPIPS evaluator for paired (generated, ground-truth) images. Used for nav eval."""

import numpy as np
import torch


class LPIPSEvaluator:
    """Lazy-loaded LPIPS model for image-pair perceptual distance.

    Uses the `lpips` PyPI package (AlexNet backbone by default).
    """

    def __init__(self, device: str = "cuda", net: str = "alex"):
        self.model = None
        self.device = device
        self.net = net

    def _ensure_loaded(self):
        if self.model is None:
            import lpips  # noqa: WPS433
            self.model = lpips.LPIPS(net=self.net).to(self.device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def compute_batch_scores(self, gen_np: np.ndarray, gt_np: np.ndarray) -> torch.Tensor:
        """LPIPS distance per pair.

        Args:
            gen_np: [B, H, W, C] uint8
            gt_np:  [B, H, W, C] uint8

        Returns:
            Tensor of shape [B] with LPIPS distances.
        """
        self._ensure_loaded()
        gen = torch.from_numpy(gen_np).permute(0, 3, 1, 2).float().div(255.0).mul(2).sub(1).to(self.device)
        gt = torch.from_numpy(gt_np).permute(0, 3, 1, 2).float().div(255.0).mul(2).sub(1).to(self.device)
        return self.model(gen, gt).flatten().cpu()
