"""DPGBench evaluator for T2I evaluation."""

import torch
import numpy as np
from PIL import Image
from typing import List

from dpg_evaluator import MPLUG, load_prompt2id, load_dpg_metadata, evaluate_batch


class DPGEvaluator:
    """Lazy-loaded DPGBench model for computing image-text alignment scores."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.prompt2id = None
        self.question_dict = None

    def _ensure_loaded(self):
        if self.model is None:
            self.model = MPLUG(device=self.device)
        if self.prompt2id is None:
            self.prompt2id = load_prompt2id()
        if self.question_dict is None:
            self.question_dict = load_dpg_metadata()

    @torch.no_grad()
    def compute_batch_scores(self, images_np: np.ndarray, prompts: List[str]) -> torch.Tensor:
        """
        Compute DPGBench scores for a batch of image-prompt pairs.

        Args:
            images_np: numpy array of images with shape (B, H, W, C), values in [0, 255]
            prompts: list of prompt strings, length B

        Returns:
            torch.Tensor of shape (B,) containing DPGBench score per sample
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
                print(f"[DPGBench] Warning: Skipping image {i} with invalid shape {img.shape}")

        # If no valid images, return all zeros
        if not valid_images:
            return torch.zeros(len(images_np))

        results = evaluate_batch(
            images=valid_images,
            prompts=valid_prompts,
            prompt2id=self.prompt2id,
            question_dict=self.question_dict,
            vqa_fn=self.model.batch_vqa,
        )

        # Build scores array with 0.0 for invalid images
        scores = torch.zeros(len(images_np))
        for idx, result in zip(valid_indices, results):
            scores[idx] = result['score']
        return scores
