"""CLIP Score evaluator for T2I evaluation."""

import torch
import numpy as np
from PIL import Image
from typing import List


class CLIPScoreEvaluator:
    """Lazy-loaded CLIP model for computing image-text similarity."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        self.model = None
        self.processor = None
        self.model_name = model_name
        self.device = device

    def _ensure_loaded(self):
        if self.model is None:
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            for p in self.model.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def compute_batch_scores(self, images_np: np.ndarray, prompts: List[str]) -> torch.Tensor:
        """
        Compute CLIP scores for a batch.

        Args:
            images_np: [B, H, W, C] uint8 numpy array
            prompts: List[str] of length B

        Returns:
            Tensor of shape [B] with cosine similarities (range -1 to 1)
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
                print(f"[CLIPScore] Warning: Skipping image {i} with invalid shape {img.shape}")

        # If no valid images, return all zeros
        if not valid_images:
            return torch.zeros(len(images_np))

        inputs = self.processor(
            text=valid_prompts,
            images=valid_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        outputs = self.model(**inputs)
        # CLIP outputs are L2-normalized, so dot product = cosine similarity
        valid_scores = (outputs.image_embeds * outputs.text_embeds).sum(dim=-1)

        # Build scores array with 0.0 for invalid images
        scores = torch.zeros(len(images_np), device=valid_scores.device)
        for i, idx in enumerate(valid_indices):
            scores[idx] = valid_scores[i]
        return scores
