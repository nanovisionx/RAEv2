"""Shared distributed eval infrastructure (temp dirs, sharding, shard gather)."""

import os
from typing import Optional

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset


def setup_eval_tmpdir(experiment_dir: str, global_step: int, rank: int,
                      *, shared_tmpdir: Optional[str] = None, eval_type: str = "sampling") -> str:
    """Create temp directory for NPZ shards. Returns the temp_dir path."""
    if shared_tmpdir:
        results_dir = os.path.basename(os.path.dirname(experiment_dir))
        experiment_name = os.path.basename(experiment_dir)
        temp_dir = os.path.join(
            os.path.expanduser(shared_tmpdir),
            results_dir,
            experiment_name,
            "eval_npzs",
        )
    else:
        temp_dir = os.path.join(experiment_dir, "eval_npzs")

    if rank == 0:
        print(f"\n[Eval] Starting distributed {eval_type} evaluation at step {global_step}")
        os.makedirs(temp_dir, exist_ok=True)

    # Wait for rank 0 to create the directory before other ranks try to save
    dist.barrier()
    return temp_dir


def create_eval_dataloader(dataset, rank: int, world_size: int,
                           num_samples: int, batch_size: int) -> DataLoader:
    """Shard dataset across ranks and return a DataLoader for this rank's subset."""
    N = min(len(dataset), num_samples)
    chunk = N // world_size

    if rank < world_size - 1:
        start = rank * chunk
        end = (rank + 1) * chunk
    else:
        start = rank * chunk
        end = N

    rank_indices = list(range(start, end))
    subset = Subset(dataset, rank_indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        multiprocessing_context="spawn",
    )


def gather_and_cleanup_shards(temp_dir: str, prefix: str, global_step: int,
                              world_size: int, num_samples: int) -> np.ndarray:
    """Rank-0 only: load all NPZ shards, concatenate, truncate, shuffle, delete shards.

    The shuffle is fixed-seed (0) and removes the class-ordered bias from
    label-conditioned sampling. Without it, splitting `combined` into N chunks
    for Inception Score deflates the score (each chunk covers only ~100 classes
    so per-chunk `p(y)` is too peaked). FID / FDR / MIND are shuffle-invariant.
    """
    all_arrays = []
    for r in range(world_size):
        shard_file = os.path.join(temp_dir, f"{prefix}_{global_step:07d}_{r:02d}.npz")
        shard_data = np.load(shard_file)["arr_0"]
        all_arrays.append(shard_data)

    combined = np.concatenate(all_arrays, axis=0)[:num_samples]

    rng = np.random.default_rng(0)
    perm = rng.permutation(combined.shape[0])
    combined = combined[perm]

    # Cleanup shards
    for r in range(world_size):
        shard_file = os.path.join(temp_dir, f"{prefix}_{global_step:07d}_{r:02d}.npz")
        if os.path.exists(shard_file):
            os.remove(shard_file)

    return combined
