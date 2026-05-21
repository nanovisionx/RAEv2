"""Distributed reconstruction evaluation — PSNR, SSIM, rFID."""

import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from tqdm import tqdm

from .ref_iqa import calculate_psnr, calculate_ssim, calculate_lpips
from .fid import calculate_rfid
from .distributed import setup_eval_tmpdir, create_eval_dataloader, gather_and_cleanup_shards


def compute_reconstruction_metrics(
    ref_arr: np.ndarray,
    rec_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    metrics_to_compute=("psnr", "ssim", "rfid"),
    disable_bar: bool = True,
) -> Dict[str, float]:
    """
    Compute reconstruction metrics between reference and reconstructed images.

    Args:
        ref_arr: Reference images [N, H, W, C] uint8
        rec_arr: Reconstructed images [N, H, W, C] uint8
        device: Device for computation
        batch_size: Batch size for metric computation
        metrics_to_compute: Which metrics to compute
        disable_bar: Whether to disable progress bars

    Returns:
        Dictionary with metrics: psnr, ssim, rfid
    """
    device_str = "cuda" if device.type == "cuda" else "cpu"
    results_dict = {}
    if 'psnr' in metrics_to_compute:
        psnr = calculate_psnr(ref_arr, rec_arr, batch_size, device_str, disable_bar=disable_bar)
        results_dict["psnr"] = psnr
    if 'ssim' in metrics_to_compute:
        ssim = calculate_ssim(ref_arr, rec_arr, batch_size, device_str, disable_bar=disable_bar)
        results_dict["ssim"] = ssim
    if 'lpips' in metrics_to_compute:
        lpips = calculate_lpips(ref_arr, rec_arr, batch_size, device_str, disable_bar=disable_bar)
        results_dict["lpips"] = lpips
    if 'rfid' in metrics_to_compute:
        rfid = calculate_rfid(ref_arr, rec_arr, batch_size, device_str)
        results_dict["rfid"] = rfid
    assert len(results_dict) > 0, "No metrics were computed."
    return results_dict


@torch.no_grad()
def evaluate_reconstruction_distributed(
    model,
    val_dataset,
    num_samples: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    experiment_dir: str,
    global_step: int,
    autocast_kwargs: dict,
    metric_batch_size: int = 128,
    reference_npz_path: Optional[str] = None,
    metrics_to_compute: Optional[list] = ("psnr", "ssim", "rfid"),
    shared_tmpdir: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    Evaluate reconstruction metrics using all GPUs in a distributed manner.

    Args:
        model: Model to evaluate (should be in eval mode)
        val_dataset: Validation dataset
        num_samples: Number of samples to reconstruct
        batch_size: Batch size per GPU for reconstruction
        rank: Current GPU rank
        world_size: Total number of GPUs
        device: Device to use
        experiment_dir: Experiment directory
        global_step: Current training step
        autocast_kwargs: Autocast configuration
        metric_batch_size: Batch size for metric computation (on rank 0)
        reference_npz_path: Optional path to existing reference NPZ file
        metrics_to_compute: Which metrics to compute
        shared_tmpdir: Optional shared directory for multi-node eval

    Returns:
        Dictionary of metrics (only on rank 0, None on other ranks)
    """
    temp_dir = setup_eval_tmpdir(experiment_dir, global_step, rank,
                                  shared_tmpdir=shared_tmpdir, eval_type="reconstruction")
    loader = create_eval_dataloader(val_dataset, rank, world_size, num_samples, batch_size)

    # Reconstruct images on this rank
    reconstructions = []
    iterator = tqdm(loader, desc=f"[Rank {rank}] Reconstructing", file=sys.stdout) if rank == 0 else loader

    with torch.inference_mode():
        for images, _ in iterator:
            images = images.to(device, non_blocking=True)
            with autocast(**autocast_kwargs):
                recon = model(images)

            recon = recon.clamp(0, 1)
            recon_np = recon.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for img in recon_np:
                reconstructions.append(img)

    reconstructions = np.stack(reconstructions)
    shard_path = os.path.join(temp_dir, f"recon_{global_step:07d}_{rank:02d}.npz")
    np.savez(shard_path, arr_0=reconstructions)

    if rank == 0:
        print(f"[Rank {rank}] Saved {len(reconstructions)} reconstructions to {shard_path}")

    # Wait for all ranks to finish reconstruction
    dist.barrier()

    # Rank 0 computes metrics
    metrics = None
    if rank == 0:
        combined_recons = gather_and_cleanup_shards(temp_dir, "recon", global_step, world_size, num_samples)
        print(f"[Eval] Combined reconstruction NPZ shape: {combined_recons.shape}")

        ref_npz_path = reference_npz_path
        if not os.path.exists(ref_npz_path):
            raise FileNotFoundError(f"Reference NPZ not found at {ref_npz_path}")

        ref_images = np.load(ref_npz_path)["arr_0"]
        print(f"[Eval] Loaded reference NPZ from {ref_npz_path}, shape: {ref_images.shape}")
        if ref_images.shape[0] != combined_recons.shape[0]:
            print(f"[Eval] Aligning ref to recon size: {ref_images.shape[0]} -> {combined_recons.shape[0]}")
            ref_images = ref_images[: combined_recons.shape[0]]

        print("[Eval] Computing metrics...")
        metrics = compute_reconstruction_metrics(
            ref_images,
            combined_recons,
            device,
            metric_batch_size,
            metrics_to_compute=metrics_to_compute,
            disable_bar=True,
        )

        print(f"[Eval] Step {global_step} Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

    dist.barrier()
    return metrics
