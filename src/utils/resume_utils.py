import dataclasses
import logging
import os
import re
import shutil
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from .wandb_utils import create_logger, initialize


def configure_experiment_dirs(args, rank) -> Tuple[str, str, logging.Logger]:
    experiment_name = os.environ.get("EXPERIMENT_NAME")
    assert experiment_name is not None, "Please set the EXPERIMENT_NAME environment variable."
    experiment_dir = os.path.join(args.results_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, 'rae')
        logger.info(f"Experiment directory created at {experiment_dir}")
        if args.wandb:
            entity = os.environ["WANDB_ENTITY"]
            project = os.environ["WANDB_PROJECT"]
            initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None, 'rae')

    # Multi-node support: each node's local_rank 0 creates dirs on its local filesystem
    dist.barrier()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    if local_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    return experiment_dir, checkpoint_dir, logger


def get_checkpoint_epoch(ckpt_path: str) -> int:
    """Load checkpoint and return its epoch value.

    Args:
        ckpt_path: Path to the checkpoint file.
    Returns:
        The epoch value stored in the checkpoint, or -1 if it cannot be read.
    """
    # Extract epoch number from checkpoint filename
    epoch = re.search(r'ep-(\d+)', ckpt_path)
    if epoch:
        return int(epoch.group(1))
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt.get("epoch", -1)
    except Exception:
        return -1


def find_resume_checkpoint(resume_dir: str, candidate_ckpt: Optional[str] = None) -> Optional[str]:
    """
    Find the checkpoint with the highest epoch from experiment dir and optional candidate.

    Args:
        resume_dir: Path to the experiment directory (contains checkpoints/ subdir).
        candidate_ckpt: Optional external checkpoint path to include in comparison.
    Returns:
        Path to the checkpoint with highest epoch, or None if no checkpoints found.
    """
    candidates = []

    # Add candidate ckpt if provided and exists
    if candidate_ckpt and os.path.isfile(candidate_ckpt):
        candidates.append(candidate_ckpt)

    # Gather checkpoints from experiment dir
    checkpoint_dir = os.path.join(resume_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith(".pt") or f.endswith(".ckpt") or f.endswith(".safetensor"):
                candidates.append(os.path.join(checkpoint_dir, f))

    if not candidates:
        return None

    # Return checkpoint with highest epoch
    return max(candidates, key=get_checkpoint_epoch)

def save_worktree(
    path: str,
    config,
    extra_metadata: dict = None,
) -> None:
    """Save config and source code to experiment directory.

    Args:
        path: Experiment directory path
        config: Config object (typed dataclass with to_dict(), or OmegaConf)
        extra_metadata: Optional dict to merge into saved config (e.g., cmd_args)
    """
    config_dict = dataclasses.asdict(config)

    if extra_metadata:
        config_dict.update(extra_metadata)

    OmegaConf.save(OmegaConf.create(config_dict), os.path.join(path, "config.yaml"))
    worktree_path = os.path.join(os.getcwd(), "src")
    shutil.copytree(worktree_path, os.path.join(path, "src/"), dirs_exist_ok=True)
    print(f'Worktree {worktree_path} saved to {os.path.join(path, "src/")}')
