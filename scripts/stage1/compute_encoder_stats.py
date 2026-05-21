#!/usr/bin/env python3
"""
Compute encoder statistics (mean and variance) for a RAE stage1 model.

This script processes a dataset through the encoder and computes the mean and variance
of the latent representations. These statistics are used for normalizing latents
during training and inference.

Supports multi-GPU processing with torchrun for faster computation.

Example usage:
    # Single GPU
    python scripts/stats/compute_encoder_stats.py \
        --config configs/stage1/pretrained/DINOv2-B.yaml \
        --use-hf-dataset \
        --hf-data-dir data \
        --output-path models/stats/dinov2/stat.pt

    # Multi-GPU with torchrun (recommended for full dataset)
    torchrun --nproc_per_node=4 scripts/stats/compute_encoder_stats.py \
        --config configs/stage1/pretrained/DINOv2-B.yaml \
        --use-hf-dataset \
        --hf-data-dir data \
        --output-path models/stats/dinov2/stat.pt

    # With encoder_name config (new VisionEncoder path)
    torchrun --nproc_per_node=4 scripts/stats/compute_encoder_stats.py \
        --config sjobs/sjobs-02-01-2026/stats-encoder-var/configs/dinov2-vit-b.yaml \
        --use-hf-dataset \
        --hf-data-dir data \
        --output-path models/stats/custom-encoders-v1/dinov2-vit-b/stat.pt
"""

import argparse
import os
import sys
from math import sqrt
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Add src to path for imports (scripts/stage1 -> scripts -> project root -> src)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.model_utils import instantiate_from_config
from data import ImageNetHFDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute encoder statistics for RAE stage1 model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to stage1 config YAML (e.g., configs/stage1/pretrained/DINOv2-B.yaml)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for stat.pt. If not provided, auto-generates from config.",
    )

    # Dataset arguments
    parser.add_argument(
        "--use-hf-dataset",
        action="store_true",
        help="Use HuggingFace arrow dataset instead of ImageFolder.",
    )
    parser.add_argument(
        "--hf-data-dir",
        type=str,
        default="../repa-baseline/data",
        help="Directory containing arrow datasets.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to use for HF dataset.",
    )
    parser.add_argument(
        "--pre-center-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, load pre-center-cropped images (HF dataset only).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to ImageFolder dataset (alternative to HF dataset).",
    )

    # Processing arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size per GPU for processing.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers per GPU.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use. Default: all samples.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size for processing (will be resized to encoder_input_size).",
    )

    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

        device = torch.device(f"cuda:{local_rank}")
        return rank, world_size, device, True
    else:
        # Single GPU or CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device, False


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class WelfordAggregator:
    """
    Welford's online algorithm for computing running mean and variance.

    This is numerically stable for large datasets and doesn't require
    storing all values in memory. Supports distributed aggregation.
    """

    def __init__(self, shape, device="cpu"):
        self.n = 0
        self.mean = torch.zeros(shape, device=device, dtype=torch.float64)
        self.M2 = torch.zeros(shape, device=device, dtype=torch.float64)
        self.device = device

    def update_batch(self, batch):
        """
        Batch update for efficiency (Chan's parallel algorithm).

        Args:
            batch: Tensor of shape [B, C, H, W]
        """
        batch = batch.to(self.device, dtype=torch.float64)
        batch_size = batch.shape[0]

        if batch_size == 0:
            return

        # Compute batch statistics
        batch_mean = batch.mean(dim=0)  # [C, H, W]
        batch_var = batch.var(dim=0, unbiased=False)  # [C, H, W]

        # Combine with running statistics using parallel algorithm
        if self.n == 0:
            self.mean = batch_mean
            self.M2 = batch_var * batch_size
            self.n = batch_size
        else:
            n_a = self.n
            n_b = batch_size
            n_total = n_a + n_b

            delta = batch_mean - self.mean
            self.mean = (n_a * self.mean + n_b * batch_mean) / n_total
            self.M2 = self.M2 + batch_var * n_b + delta ** 2 * n_a * n_b / n_total
            self.n = n_total

    def merge(self, other):
        """
        Merge statistics from another aggregator (for distributed reduction).

        Args:
            other: Another WelfordAggregator with n, mean, M2
        """
        if other.n == 0:
            return

        if self.n == 0:
            self.n = other.n
            self.mean = other.mean.clone()
            self.M2 = other.M2.clone()
            return

        n_a = self.n
        n_b = other.n
        n_total = n_a + n_b

        delta = other.mean - self.mean
        self.mean = (n_a * self.mean + n_b * other.mean) / n_total
        self.M2 = self.M2 + other.M2 + delta ** 2 * n_a * n_b / n_total
        self.n = n_total

    def all_reduce(self):
        """
        Aggregate statistics across all distributed processes.
        Uses Chan's parallel algorithm for combining statistics.
        """
        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        if world_size == 1:
            return

        # Gather n, mean, M2 from all ranks
        n_tensor = torch.tensor([self.n], device=self.device, dtype=torch.float64)
        n_list = [torch.zeros_like(n_tensor) for _ in range(world_size)]
        dist.all_gather(n_list, n_tensor)

        mean_list = [torch.zeros_like(self.mean) for _ in range(world_size)]
        dist.all_gather(mean_list, self.mean)

        M2_list = [torch.zeros_like(self.M2) for _ in range(world_size)]
        dist.all_gather(M2_list, self.M2)

        # Reset and merge all
        self.n = 0
        self.mean = torch.zeros_like(self.mean)
        self.M2 = torch.zeros_like(self.M2)

        for i in range(world_size):
            other_n = int(n_list[i].item())
            other_mean = mean_list[i]
            other_M2 = M2_list[i]

            if other_n == 0:
                continue

            if self.n == 0:
                self.n = other_n
                self.mean = other_mean.clone()
                self.M2 = other_M2.clone()
            else:
                n_a = self.n
                n_b = other_n
                n_total = n_a + n_b

                delta = other_mean - self.mean
                self.mean = (n_a * self.mean + n_b * other_mean) / n_total
                self.M2 = self.M2 + other_M2 + delta ** 2 * n_a * n_b / n_total
                self.n = n_total

    def finalize(self):
        """
        Compute final mean and variance.

        Returns:
            mean: Tensor of shape [C, H, W]
            var: Tensor of shape [C, H, W]
        """
        if self.n < 2:
            raise ValueError("Need at least 2 samples to compute variance")

        mean = self.mean.float()
        var = (self.M2 / self.n).float()  # Population variance

        return mean, var


def create_dataloader(args, image_size, rank, world_size, is_distributed):
    """Create dataloader based on arguments."""

    # Create transform
    if args.use_hf_dataset and args.pre_center_crop:
        # Pre-centered images just need resize
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
    else:
        # Raw images need resize + center crop
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    if args.use_hf_dataset:
        if args.hf_data_dir is None:
            raise ValueError("--hf-data-dir must be specified when using --use-hf-dataset")
        dataset = ImageNetHFDataset(
            data_dir=args.hf_data_dir,
            split=args.hf_split,
            transform=transform,
        )
    else:
        if args.data_path is None:
            raise ValueError("--data-path must be specified when not using --use-hf-dataset")
        dataset = ImageFolder(str(args.data_path), transform=transform)

    # Limit samples if requested
    if args.num_samples is not None and args.num_samples < len(dataset):
        indices = list(range(args.num_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    # Create sampler for distributed training
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # No need to shuffle for statistics computation
            drop_last=False,
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader, len(dataset)


def encode_batch(model, images, device):
    """
    Encode a batch of images to latents.

    This replicates the encoding logic from RAE.encode() but without
    normalization stats (since we're computing those).

    Supports both:
    - VisionEncoder path (encoder_name): expects [0,255] range, handles own preprocessing
    - Legacy path (encoder_cls): expects [0,1] range, uses encoder_mean/std

    Args:
        model: RAE model
        images: Tensor of shape [B, 3, H, W] in range [0, 1]
        device: Device to use

    Returns:
        latents: Tensor of shape [B, C, H, W]
    """
    images = images.to(device)

    with torch.no_grad():
        # Check if encoder is a VisionEncoder (has preprocess method)
        is_vision_encoder = hasattr(model.encoder, 'preprocess')

        if is_vision_encoder:
            # VisionEncoder path: expects [0,255] range, handles own preprocessing
            if images.max() <= 1.0:
                images = images * 255.0

            # Resize to resolution if needed (encoder preprocess handles resolution->input_size)
            _, _, h, w = images.shape
            if h != model.resolution or w != model.resolution:
                images = nn.functional.interpolate(
                    images,
                    size=(model.resolution, model.resolution),
                    mode='bicubic',
                    align_corners=False
                )

            # VisionEncoder.forward() handles preprocess + forward_features
            z = model.encoder(images)  # [B, N, C]
        else:
            # Legacy path: expects [0,1] range, uses encoder_mean/std
            _, _, h, w = images.shape
            if h != model.encoder_input_size or w != model.encoder_input_size:
                images = nn.functional.interpolate(
                    images,
                    size=(model.encoder_input_size, model.encoder_input_size),
                    mode='bicubic',
                    align_corners=False
                )

            # Normalize input with encoder mean/std
            images = (images - model.encoder_mean.to(device)) / model.encoder_std.to(device)

            # Encode
            z = model.encoder(images)  # [B, N, C]

        # Reshape to 2D (as done in RAE.encode when reshape_to_2d=True)
        b, n, c = z.shape
        h = w = int(sqrt(n))
        z = z.transpose(1, 2).view(b, c, h, w)  # [B, C, H, W]

    return z


def auto_generate_output_path(config_path):
    """Generate output path from config path."""
    config_path = Path(config_path)
    config_name = config_path.stem  # e.g., "DINOv2-B"

    # Create a reasonable output path
    output_dir = Path("pretrained_models/stats/computed") / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / "stat.pt"


def main():
    args = parse_args()

    # Setup distributed
    rank, world_size, device, is_distributed = setup_distributed()

    if rank == 0:
        print(f"Running on {world_size} GPU(s)")
        print(f"Loading config from {args.config}")

    config = OmegaConf.load(args.config)
    rae_config = config.get("stage_1", None)

    if rae_config is None:
        raise ValueError("Config must contain 'stage_1' section")

    # Detect encoder path: new (encoder_name) vs legacy (encoder_cls)
    encoder_name = rae_config.params.get("encoder_name", None)
    encoder_cls = rae_config.params.get("encoder_cls", None)
    is_vision_encoder = encoder_name is not None

    # Get image size from config based on encoder type
    if is_vision_encoder:
        # New path: use resolution
        image_size_from_config = rae_config.params.get("resolution", 256)
    else:
        # Legacy path: use encoder_input_size
        image_size_from_config = rae_config.params.get("encoder_input_size", 224)

    if rank == 0:
        print(f"Instantiating model...")

    # Temporarily remove normalization_stat_path to avoid loading stats
    original_stat_path = rae_config.params.get("normalization_stat_path", None)
    rae_config.params.normalization_stat_path = None

    model = instantiate_from_config(rae_config).to(device)
    model.eval()

    if rank == 0:
        encoder_id = encoder_name or encoder_cls or "unknown"
        print(f"Encoder: {encoder_id}")
        print(f"Encoder type: {'VisionEncoder (new)' if is_vision_encoder else 'Legacy'}")
        print(f"Image size from config: {image_size_from_config}")
        print(f"Latent dim: {model.latent_dim}")
        print(f"Base patches: {model.base_patches}")

    # Calculate expected output shape
    h = w = int(sqrt(model.base_patches))
    expected_shape = (model.latent_dim, h, w)

    if rank == 0:
        print(f"Expected stat shape: {expected_shape}")
        print(f"\nLoading dataset...")

    # Create dataloader
    loader, total_samples = create_dataloader(args, args.image_size, rank, world_size, is_distributed)

    if rank == 0:
        print(f"Total samples: {total_samples}")
        print(f"Batches per GPU: {len(loader)}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")

    # Synchronize before starting
    if is_distributed:
        dist.barrier()

    # Initialize aggregator
    aggregator = WelfordAggregator(expected_shape, device=device)

    # Process all batches
    if rank == 0:
        print(f"\nComputing statistics...")
        pbar = tqdm(total=len(loader), desc="Processing")
    else:
        pbar = None

    for images, _ in loader:
        z = encode_batch(model, images, device)
        aggregator.update_batch(z)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Aggregate statistics across all GPUs
    if is_distributed:
        if rank == 0:
            print(f"\nAggregating statistics across {world_size} GPUs...")
        aggregator.all_reduce()

    # Finalize statistics (all ranks compute this for verification)
    mean, var = aggregator.finalize()

    # Only rank 0 saves and prints
    if rank == 0:
        print(f"\nStatistics computed:")
        print(f"  Mean shape: {mean.shape}")
        print(f"  Mean range: [{mean.min():.6f}, {mean.max():.6f}]")
        print(f"  Var shape: {var.shape}")
        print(f"  Var range: [{var.min():.6f}, {var.max():.6f}]")
        print(f"  Total samples processed: {aggregator.n}")

        # Determine output path
        if args.output_path is None:
            output_path = auto_generate_output_path(args.config)
        else:
            output_path = Path(args.output_path)

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save statistics
        stats = {
            'mean': mean.cpu(),
            'var': var.cpu(),
        }
        torch.save(stats, output_path)
        print(f"\nSaved statistics to {output_path}")

        # Compare with existing stats if available
        if original_stat_path is not None and Path(original_stat_path).exists():
            print(f"\nComparing with existing stats at {original_stat_path}:")
            existing = torch.load(original_stat_path, map_location='cpu')

            if existing.get('mean') is not None:
                mean_diff = (mean.cpu() - existing['mean']).abs()
                print(f"  Mean difference: max={mean_diff.max():.6f}, mean={mean_diff.mean():.6f}")
            else:
                print(f"  Existing mean is None (will use 0)")

            if existing.get('var') is not None:
                var_diff = (var.cpu() - existing['var']).abs()
                print(f"  Var difference: max={var_diff.max():.6f}, mean={var_diff.mean():.6f}")

        print("\nDone!")

    # Clean up
    cleanup_distributed()


if __name__ == "__main__":
    main()
