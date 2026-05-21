"""GenEval evaluator for T2I evaluation."""

from typing import List

import numpy as np
import torch
from geneval_evaluator import evaluate_pairs, fetch_metadata, load_models
from PIL import Image


class GenEvalEvaluator:
    """Lazy-loaded GenEval model for computing image-text similarity."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.models = None
        self.metadata = None

    def _ensure_loaded(self):
        if self.models is None:
            self.models = load_models(device=self.device)
        if self.metadata is None:
            self.metadata = fetch_metadata()

    @torch.no_grad()
    def compute_batch_scores(self, images_np: np.ndarray, prompts: List[str]) -> torch.Tensor:
        self._ensure_loaded()

        # Validate images and filter out invalid ones (zero dimensions)
        valid_indices = []
        valid_images = []
        valid_prompts = []
        for i, img in enumerate(images_np):
            # Check for zero dimensions which cause ZeroDivisionError in CLIP preprocessing
            if img.shape[0] > 0 and img.shape[1] > 0:
                valid_indices.append(i)
                valid_images.append(Image.fromarray(img))
                valid_prompts.append(prompts[i])
            else:
                print(f"[GenEval] Warning: Skipping image {i} with invalid shape {img.shape}")

        # If no valid images, return all zeros
        if not valid_images:
            return torch.zeros(len(images_np))

        metadatas = [self.metadata[prompt] for prompt in valid_prompts]
        results = evaluate_pairs(images=valid_images, metadata_list=metadatas, models=self.models, device=self.device, show_progress=False)

        # Build scores array with 0.0 for invalid images
        scores = torch.zeros(len(images_np))
        for idx, result in zip(valid_indices, results):
            scores[idx] = float(result['correct'])
        return scores
