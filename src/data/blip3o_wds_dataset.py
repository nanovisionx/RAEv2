"""BLIP3O WebDataset loader for T2I training."""
from pathlib import Path
from typing import List, Optional, Union
import webdataset as wds
from torchvision import transforms


# Dataset metadata for size estimation
# Note: num_shards are actual counts; num_samples are estimates (used for epoch length calculation)
BLIP3O_METADATA = {
    "journeydb": {"num_samples": 4_280_000, "num_shards": 419},
    "short-caption": {"num_samples": 4_770_000, "num_shards": 1831},
    "long-caption": {"num_samples": 27_200_000, "num_shards": 2891},
    "60k": {"num_samples": 57_553, "num_shards": 11},
    "60k-128shards": {"num_samples": 57_553, "num_shards": 128},  # Resharded for better worker utilization
}


def _filter_valid_samples(sample):
    """Filter function for .select() - must be module-level to be pickleable."""
    return sample[0] is not None


class BLIP3OWebDataset:
    """
    WebDataset wrapper for BLIP3O splits.
    Supports combining multiple splits and uses wds.split_by_node for DDP.

    Returns (image_tensor, caption_string) pairs compatible with existing
    text conditioning interface.
    """

    def __init__(
        self,
        data_dir: str,
        splits: Union[str, List[str]],
        transform: Optional[transforms.Compose] = None,
        image_size: int = 256,
        shuffle_buffer: int = 20000,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Base path to BLIP3O data (e.g., 'data/blip3o')
            splits: Single split name or list of splits to combine
                   Options: 'journeydb', 'short-caption', 'long-caption', '60k'
            transform: Optional custom transform. If None, uses default augmentation.
            image_size: Target image resolution
            shuffle_buffer: Size of shuffle buffer for sample-level shuffling
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.splits = [splits] if isinstance(splits, str) else list(splits)
        self.transform = transform
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        # Collect shard URLs and calculate total samples
        self._total_samples = 0
        self._shard_urls = []

        for split in self.splits:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                raise ValueError(f"Split directory not found: {split_dir}")

            tar_files = sorted(split_dir.glob("*.tar"))
            self._shard_urls.extend([str(f) for f in tar_files])

            if split in BLIP3O_METADATA:
                self._total_samples += BLIP3O_METADATA[split]["num_samples"]
            else:
                # Fallback estimate: ~3500 samples per shard (BLIP3O average)
                self._total_samples += len(tar_files) * 3500

        if not self._shard_urls:
            raise ValueError(f"No tar shards found for splits {self.splits} in {data_dir}")

        self._num_shards = len(self._shard_urls)

        # Default transform with augmentation
        if self.transform is None:
            # use as few augmentations as possible
            self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])

    @property
    def estimated_size(self) -> int:
        """Return estimated total samples (for epoch calculation)."""
        return self._total_samples

    @property
    def num_shards(self) -> int:
        """Return number of shards."""
        return self._num_shards

    def _decode_sample(self, sample):
        """Decode WebDataset sample (image + txt pair)."""
        # Get image - handle different extensions
        image = sample.get("jpg") or sample.get("png") or sample.get("jpeg") or sample.get("webp")

        # Get caption - handle bytes or string
        caption_raw = sample.get("txt", b"")
        if isinstance(caption_raw, bytes):
            caption = caption_raw.decode("utf-8").strip()
        else:
            caption = str(caption_raw).strip()

        # Apply transform if image exists
        if self.transform is not None and image is not None:
            image = self.transform(image)

        return image, caption

    def create_pipeline(self, epoch: int = 0) -> wds.WebDataset:
        """
        Create WebDataset pipeline for a given epoch.

        Call at start of each epoch to ensure proper shuffling with deterministic seed.
        """
        dataset = (
            wds.WebDataset(
                self._shard_urls,
                nodesplitter=wds.split_by_node,
                shardshuffle=1000,  # Shuffle buffer size for shards
                seed=self.seed + epoch,
            )
            .shuffle(self.shuffle_buffer, initial=self.shuffle_buffer // 2)
            .decode("pil", handler=wds.ignore_and_continue)  # Skip corrupt images
            .map(self._decode_sample, handler=wds.ignore_and_continue)
            .select(_filter_valid_samples)  # Filter failed decodes (lambda function is not pickleable for spawn)
        )
        return dataset
