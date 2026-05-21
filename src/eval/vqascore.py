"""VQAScore evaluator for T2I evaluation."""

import torch
import numpy as np
from PIL import Image
from typing import List
from t2v_metrics import VQAScore


class VQAScoreEvaluator:
    """Lazy-loaded VQAScore model for computing image-text similarity."""

    def __init__(self, model_name: str = "clip-flant5-xl", device: str = "cuda"):
        self.scorer = None
        self.model_name = model_name
        self.device = device

    def _ensure_loaded(self):
        if self.scorer is None:
            self.scorer = VQAScore(model=self.model_name, device=self.device)

    @torch.no_grad()
    def compute_batch_scores(self, images_np: np.ndarray, prompts: List[str]) -> torch.Tensor:
        """
        Compute VQAScore scores for a batch.

        Args:
            images_np: [B, H, W, C] uint8 numpy array
            prompts: List[str] of length B

        Returns:
            Tensor of shape [B] with VQAScore similarities
        """
        self._ensure_loaded()

        # Validate images and filter out invalid ones (zero dimensions)
        valid_indices = []
        valid_images = []
        valid_prompts = []
        for i, img in enumerate(images_np):
            if img.shape[0] > 0 and img.shape[1] > 0:
                valid_indices.append(i)
                valid_images.append(Image.fromarray(img))
                valid_prompts.append(prompts[i])
            else:
                print(f"[VQAScore] Warning: Skipping image {i} with invalid shape {img.shape}")

        # If no valid images, return all zeros
        if not valid_images:
            return torch.zeros(len(images_np))

        valid_scores = self.scorer(images=valid_images, texts=valid_prompts)

        # Build scores array with 0.0 for invalid images
        scores = torch.zeros(len(images_np))
        for i, idx in enumerate(valid_indices):
            scores[idx] = valid_scores[i]
        return scores
