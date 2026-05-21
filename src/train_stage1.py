"""Stage-1 RAE training script with reconstruction, LPIPS, and GAN losses."""

from __future__ import annotations

import argparse
import dataclasses
from copy import deepcopy

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from configs import Stage1Config
from data import prepare_unified_dataloader
from stage1.disc import LPIPS, build_discriminator
from eval.datasets import normalize_eval_datasets, prepare_eval_datasets
from stage1.engine import train_one_epoch
from stage1.utils import validate_stage1_config
from utils.checkpoint import load_stage1_checkpoint, save_stage1_checkpoint
from utils.dist_utils import cleanup_distributed, setup_distributed
from utils.model_utils import instantiate_from_config
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import configure_experiment_dirs, find_resume_checkpoint, save_worktree
from utils.sync_utils import sync_checkpoint_blocking, sync_evals_blocking
from utils.train_utils import get_autocast_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 RAE with GAN and LPIPS losses.")
    parser.add_argument("--config", type=str, required=True, help="YAML config.")
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store outputs.")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument('--wandb', action='store_true', help='Use W&B for logging.')
    parser.add_argument("--compile", action="store_true", help="Use torch.compile.")
    parser.add_argument("--sync-checkpoints", action="store_true", help="Sync checkpoints to S3.")
    return parser.parse_args()


def main():
    args = parse_args()

    #########################################################
    # Distributed + Config setup
    #########################################################
    rank, world_size, device = setup_distributed()
    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage1Config), OmegaConf.load(args.config)))
    validate_stage1_config(config)

    seed = config.training.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)

    #########################################################
    # Dataset and dataloader (unified)
    #########################################################
    batch_size = config.training.global_batch_size // world_size if config.training.global_batch_size else config.training.batch_size
    dataloader_result = prepare_unified_dataloader(
        config=dataclasses.asdict(config.dataset),
        image_size=config.training.image_size,
        batch_size=batch_size,
        num_workers=config.training.num_workers,
        rank=rank,
        world_size=world_size,
        shuffle=True,
    )
    dataloader = dataloader_result.loader

    steps_per_epoch = config.training.virtual_epoch_steps if config.training.virtual_epoch_steps else len(dataloader_result)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches.")

    # Eval datasets (unified format)
    eval_datasets = None
    if config.eval is not None:
        eval_datasets_config = normalize_eval_datasets(config.eval.datasets)
        if eval_datasets_config:
            eval_datasets = prepare_eval_datasets(
                eval_datasets_config,
                image_size=config.training.image_size,
                batch_size=batch_size,
                num_workers=config.training.num_workers,
                rank=rank,
                world_size=world_size,
            )

    #########################################################
    # Model and DDP setup
    #########################################################
    rae = instantiate_from_config(config.stage_1).to(device)
    rae.encoder.eval()
    rae.decoder.train()
    ema_model = deepcopy(rae).to(device).eval()
    ema_model.requires_grad_(False)
    rae.encoder.requires_grad_(False)
    rae.decoder.requires_grad_(True)

    ddp_model = DDP(rae, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    if args.compile:
        ddp_model = torch.compile(ddp_model)

    # Discriminator
    discriminator, disc_aug = build_discriminator(config.gan.arch, device, config.gan.augment)
    ddp_disc = DDP(discriminator, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    discriminator.train()

    lpips_model = LPIPS().to(device).eval()

    #########################################################
    # Optimizer and scheduler
    #########################################################
    optimizer, _ = build_optimizer(rae.decoder.parameters(), config.training.optimizer)
    disc_params = [p for p in discriminator.parameters() if p.requires_grad]
    disc_optimizer, _ = build_optimizer(disc_params, config.gan.optimizer)

    scheduler = None
    disc_scheduler = None
    if config.training.scheduler is not None:
        scheduler, _ = build_scheduler(optimizer, steps_per_epoch, config.training.scheduler)
    if config.gan.scheduler is not None:
        disc_scheduler, _ = build_scheduler(disc_optimizer, steps_per_epoch, config.gan.scheduler)

    autocast_kwargs = get_autocast_kwargs(args)

    #########################################################
    # Resume
    #########################################################
    start_epoch, global_step = 0, 0
    maybe_ckpt = find_resume_checkpoint(experiment_dir)
    if maybe_ckpt:
        logger.info(f"Resuming from {maybe_ckpt}...")
        start_epoch, global_step = load_stage1_checkpoint(
            maybe_ckpt, ddp_model, ema_model, optimizer, scheduler,
            discriminator, disc_optimizer, disc_scheduler,
        )
        logger.info(f"Resumed epoch={start_epoch}, step={global_step}.")
    else:
        if rank == 0:
            save_worktree(experiment_dir, config, {"cmd_args": vars(args)})

    # Viz samples from first eval dataset
    viz_samples = None
    if eval_datasets:
        first_ds = next(iter(eval_datasets.values()))
        viz_rng = torch.Generator().manual_seed(42)
        num_viz = min(64, len(first_ds.dataset))
        viz_indices = torch.randperm(len(first_ds.dataset), generator=viz_rng)[:num_viz].tolist()
        viz_samples = torch.stack([first_ds.dataset[i][0] for i in viz_indices]).to(device)

    # Progress bar
    total_steps = config.training.epochs * steps_per_epoch
    progress_bar = tqdm(total=total_steps, initial=global_step, desc="Training", disable=rank != 0)

    #########################################################
    # Train loop
    #########################################################
    dist.barrier()
    for epoch in range(start_epoch, config.training.epochs):
        dataloader_result.set_epoch(epoch)
        global_step = train_one_epoch(
            ddp_model, ema_model, ddp_disc, disc_aug, lpips_model,
            dataloader, optimizer, disc_optimizer, scheduler, disc_scheduler,
            autocast_kwargs, device, epoch, global_step, batch_size,
            config, args, logger, rank, world_size, checkpoint_dir, experiment_dir,
            progress_bar, eval_datasets, viz_samples,
        )
    progress_bar.close()

    #########################################################
    # Final checkpoint and cleanup
    #########################################################
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {config.training.epochs}...")
        save_stage1_checkpoint(
            f"{checkpoint_dir}/ep-{config.training.epochs:07d}.pt", global_step, config.training.epochs,
            ddp_model, ema_model, optimizer, scheduler, discriminator, disc_optimizer, disc_scheduler,
        )
        if args.sync_checkpoints:
            sync_checkpoint_blocking(checkpoint_dir, logger)
            sync_evals_blocking("evals/stage1", logger)

    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
