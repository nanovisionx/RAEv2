"""Stage 1 training engine."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Optional

import torch
from torch.cuda.amp import autocast
from torchvision.utils import make_grid

from .disc import select_gan_losses, calculate_adaptive_weight
from eval import evaluate_reconstruction_distributed
from utils import wandb_utils
from utils.checkpoint import save_stage1_checkpoint
from utils.logging import save_eval_to_csv
from utils.sync_utils import sync_checkpoint_async, sync_evals_async
from utils.train_utils import update_ema


def train_one_epoch(
    ddp_model,
    ema_model,
    ddp_disc,
    disc_aug,
    lpips_model,
    dataloader,
    optimizer,
    disc_optimizer,
    scheduler,
    disc_scheduler,
    autocast_kwargs: dict,
    device: torch.device,
    epoch: int,
    global_step: int,
    batch_size: int,
    config,
    args,
    logger,
    rank: int,
    world_size: int,
    checkpoint_dir: str,
    experiment_dir: str,
    progress_bar,
    eval_datasets: Optional[Dict] = None,
    viz_samples: Optional[torch.Tensor] = None,
) -> int:
    """Train one epoch. Returns updated global_step.

    Args:
        eval_datasets: Dict of {name: EvalDatasetInfo} for unified eval, or None to skip eval.
    """
    #########################################################
    # Setup
    #########################################################
    ddp_model.train()

    disc = ddp_disc.module
    decoder = ddp_model.module.decoder
    last_layer = decoder.decoder_pred.weight

    steps_per_epoch = config.training.virtual_epoch_steps if config.training.virtual_epoch_steps else len(dataloader)
    disc_loss_fn, gen_loss_fn = select_gan_losses(config.gan.loss.disc_loss, config.gan.loss.gen_loss)

    gan_start_step = config.gan.loss.disc_start * steps_per_epoch
    disc_update_step = config.gan.loss.disc_upd_start * steps_per_epoch
    lpips_start_step = config.gan.loss.lpips_start * steps_per_epoch

    do_eval = config.eval is not None and eval_datasets is not None
    epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
    num_batches = 0

    # Checkpoint at epoch start
    if config.training.checkpoint_interval > 0 and epoch % config.training.checkpoint_interval == 0 and rank == 0:
        logger.info(f"Saving checkpoint at epoch {epoch}...")
        ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt"
        save_stage1_checkpoint(
            ckpt_path, global_step, epoch, ddp_model, ema_model,
            optimizer, scheduler, disc, disc_optimizer, disc_scheduler,
        )
        if args.sync_checkpoints:
            sync_checkpoint_async(checkpoint_dir, logger)
            sync_evals_async("evals/stage1", logger)

    #########################################################
    # Train loop
    #########################################################
    for images, _ in dataloader:
        use_gan = global_step >= gan_start_step and config.gan.loss.disc_weight > 0.0
        train_disc = global_step >= disc_update_step and config.gan.loss.disc_weight > 0.0
        use_lpips = global_step >= lpips_start_step and config.gan.loss.perceptual_weight > 0.0

        images = images.to(device, non_blocking=True)
        real_normed = images * 2.0 - 1.0

        #########################################################
        # Train generator
        #########################################################
        optimizer.zero_grad(set_to_none=True)
        disc.eval()

        with autocast(**autocast_kwargs):
            recon = ddp_model(images)
            recon_normed = recon * 2.0 - 1.0
            rec_loss = (recon - images).abs().mean()
            lpips_loss = lpips_model(real_normed, recon_normed) if use_lpips else rec_loss.new_zeros(())
            recon_total = rec_loss + config.gan.loss.perceptual_weight * lpips_loss

            if use_gan:
                fake_aug = disc_aug.aug(recon_normed)
                logits_fake, _ = ddp_disc(fake_aug, None)
                gan_loss = gen_loss_fn(logits_fake)
            else:
                gan_loss = torch.zeros_like(recon_total)

        if use_gan:
            adaptive_weight = calculate_adaptive_weight(recon_total, gan_loss, last_layer, config.gan.loss.max_d_weight)
            total_loss = recon_total + config.gan.loss.disc_weight * adaptive_weight * gan_loss
        else:
            adaptive_weight = torch.zeros_like(recon_total)
            total_loss = recon_total

        total_loss.backward()
        if config.training.clip_grad:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), config.training.clip_grad)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        update_ema(ema_model, ddp_model.module, config.training.ema_decay)

        #########################################################
        # Train discriminator
        #########################################################
        disc_metrics: Dict[str, torch.Tensor] = {}
        if train_disc:
            ddp_model.eval()
            ddp_disc.train()
            for _ in range(config.gan.loss.disc_updates):
                disc_optimizer.zero_grad(set_to_none=True)
                with autocast(**autocast_kwargs):
                    with torch.no_grad():
                        recon_disc = ddp_model(images)
                        recon_disc_normed = recon_disc * 2.0 - 1.0
                    fake_detached = recon_disc_normed.clamp(-1.0, 1.0)
                    fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                    fake_input = disc_aug.aug(fake_detached)
                    real_input = disc_aug.aug(real_normed)
                    logits_fake, logits_real = ddp_disc(fake_input, real_input)
                    d_loss = disc_loss_fn(logits_real, logits_fake)
                    accuracy = (logits_real > logits_fake).float().mean()

                d_loss.backward()
                disc_optimizer.step()

                disc_metrics = {
                    "disc_loss": d_loss.detach(),
                    "logits_real": logits_real.detach().mean(),
                    "logits_fake": logits_fake.detach().mean(),
                    "disc_accuracy": accuracy.detach(),
                }
                epoch_metrics["disc_loss"] += d_loss.detach()
                epoch_metrics["disc_accuracy"] += accuracy.detach()
                if disc_scheduler is not None:
                    disc_scheduler.step()

            ddp_disc.eval()
            ddp_model.train()

        epoch_metrics["recon"] += rec_loss.detach()
        epoch_metrics["lpips"] += lpips_loss.detach()
        epoch_metrics["gan"] += gan_loss.detach()
        epoch_metrics["total"] += total_loss.detach()
        num_batches += 1
        progress_bar.update(1)

        #########################################################
        # Logging and visualization
        #########################################################
        if config.training.log_interval > 0 and global_step % config.training.log_interval == 0 and rank == 0:
            stats = {
                "loss/total": total_loss.detach().item(),
                "loss/recon": rec_loss.detach().item(),
                "loss/lpips": lpips_loss.detach().item(),
                "loss/gan": gan_loss.detach().item(),
                "lr/generator": optimizer.param_groups[0]["lr"],
            }
            if disc_metrics:
                stats.update({
                    "loss/disc": disc_metrics["disc_loss"].item(),
                    "disc/logits_real": disc_metrics["logits_real"].item(),
                    "disc/logits_fake": disc_metrics["logits_fake"].item(),
                    "lr/discriminator": disc_optimizer.param_groups[0]["lr"],
                    "disc/accuracy": disc_metrics["disc_accuracy"].item(),
                    "disc/weight": adaptive_weight.item(),
                })
            logger.info(f"[Epoch {epoch} | Step {global_step}] " + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items()))
            if args.wandb:
                wandb_utils.log(stats, step=global_step)
            progress_bar.set_postfix(loss=total_loss.detach().item(), lr=optimizer.param_groups[0]["lr"])

        # Visualization
        if global_step % config.training.sample_every == 0 and do_eval and viz_samples is not None:
            logger.info("Generating EMA samples...")
            with torch.no_grad():
                samples = ema_model.decode(ema_model.encode(viz_samples))
                original_grid = make_grid(viz_samples.cpu().float(), nrow=8)
                recon_grid = make_grid(samples.cpu().float(), nrow=8)
                if args.wandb:
                    wandb_utils.log_images({"viz/original": original_grid, "viz/reconstructed": recon_grid}, step=global_step)
            logger.info("Generating EMA samples done.")

        #########################################################
        # Evaluation (unified: iterate over all eval datasets)
        #########################################################
        if do_eval and config.eval.eval_interval > 0 and global_step > 0 and global_step % config.eval.eval_interval == 0:
            logger.info("Starting evaluation...")
            eval_models = [(ema_model, "ema")]
            if config.eval.eval_model:
                eval_models.append((ddp_model.module, "model"))
            experiment_name = os.environ.get("EXPERIMENT_NAME", "unknown")

            for ds_name, ds_info in eval_datasets.items():
                eval_n = min(ds_info.num_samples or len(ds_info.dataset), len(ds_info.dataset))
                for eval_mod, mod_name in eval_models:
                    eval_stats = evaluate_reconstruction_distributed(
                        eval_mod, ds_info.dataset, eval_n,
                        rank=rank, world_size=world_size, device=device, batch_size=batch_size,
                        metrics_to_compute=ds_info.metrics, experiment_dir=experiment_dir,
                        global_step=global_step, autocast_kwargs=autocast_kwargs,
                        reference_npz_path=ds_info.reference_npz, shared_tmpdir=config.dataset.shared_tmpdir,
                    )
                    if rank == 0 and eval_stats is not None:
                        save_eval_to_csv(experiment_name, f"{mod_name}_{ds_name}", global_step, eval_stats)
                    if eval_stats:
                        wandb_stats = {f"eval_{mod_name}/{k}_{ds_name}": v for k, v in eval_stats.items()}
                        if args.wandb:
                            wandb_utils.log(wandb_stats, step=global_step)
            logger.info("Evaluation done.")

        # update global step
        global_step += 1

    #########################################################
    # Epoch summary
    #########################################################
    if rank == 0 and num_batches > 0:
        epoch_stats = {
            "epoch/loss_total": (epoch_metrics["total"] / num_batches).item(),
            "epoch/loss_recon": (epoch_metrics["recon"] / num_batches).item(),
            "epoch/loss_lpips": (epoch_metrics["lpips"] / num_batches).item(),
            "epoch/loss_gan": (epoch_metrics["gan"] / num_batches).item(),
        }
        if disc_metrics:
            epoch_stats.update({
                "epoch/loss_disc": (epoch_metrics["disc_loss"] / num_batches).item(),
                "epoch/disc_logits_real": disc_metrics["logits_real"].item(),
                "epoch/disc_logits_fake": disc_metrics["logits_fake"].item(),
                "epoch/disc_accuracy": (epoch_metrics["disc_accuracy"] / num_batches).item(),
            })
        logger.info(f"[Epoch {epoch}] " + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items()))
        if args.wandb:
            wandb_utils.log(epoch_stats, step=global_step)

    return global_step
