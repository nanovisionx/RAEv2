"""
ImageNet dataset loader using HuggingFace Arrow format.

This module provides a PyTorch Dataset wrapper for ImageNet data stored in
Apache Arrow format, as preprocessed by the repa-baseline repository.
"""
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from .imagenet_classes import IMAGENET_CLASSES


class ImageNetHFDataset(Dataset):
    """
    PyTorch Dataset for ImageNet using HuggingFace Arrow format.

    This dataset loads ImageNet images and labels from pre-processed Arrow files,
    which provide efficient memory-mapped access to the data without requiring
    the full dataset to be loaded into memory.

    Supports both label conditioning (returns int) and text conditioning (returns str).

    Args:
        data_dir: Path to directory containing the arrow dataset.
                 Should contain 'imagenet-latents-images' folder.
        split: Dataset split, either "train" or "val". Default: "train".
        transform: Optional transform to apply to images.
        condition_type: Type of conditioning - "label" (int) or "text" (string prompts). Default: "label".
        prompt_template: Template for generating text prompts. Default: "a photo of a {class_name}".
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[object] = None,
        condition_type: str = "label",
        prompt_template: str = "a photo of a {class_name}",
    ):
        """Initialize the ImageNet HF dataset."""
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.condition_type = condition_type
        self.prompt_template = prompt_template

        # Determine the path to the arrow dataset
        arrow_path = self.data_dir / "imagenet-latents-images"
        split_str = "val" if split == "val" else ""
        dataset_path = arrow_path / split_str if split_str else arrow_path
        if not dataset_path.exists():
            raise FileNotFoundError(f"Split '{split}' not found at {dataset_path}")

        # Load the dataset using HuggingFace datasets
        self.dataset = load_from_disk(str(dataset_path))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, str]]:
        # Get sample from arrow dataset
        sample = self.dataset[idx]
        image = sample["image"]  # PIL Image
        label = sample["label"]  # int32

        # Convert PIL image to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)

        # Return based on conditioning type
        if self.condition_type == "text":
            class_name = IMAGENET_CLASSES[label]
            return image, self.prompt_template.format(class_name=class_name)
        else:
            return image, label

    @property
    def num_classes(self) -> int:
        """Return the number of classes in ImageNet."""
        return 1000
