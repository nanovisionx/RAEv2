"""Distributed generation evaluation — FID, CLIPScore, VQAScore, GenEval, DPG-Bench."""

import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from tqdm import tqdm

from .clipscore import CLIPScoreEvaluator
from .distributed import create_eval_dataloader, gather_and_cleanup_shards, setup_eval_tmpdir
from .distributional import compute_distributional_metrics, filter_distributional
from .dpgbench import DPGEvaluator
from .geneval import GenEvalEvaluator
from .lpips import LPIPSEvaluator
from .vqascore import VQAScoreEvaluator


def evaluate_image_set(
    images: np.ndarray,
    *,
    metrics_to_compute: List[str],
    reference_npz_path: Optional[Union[str, List[str]]] = None,
    data_dir: Optional[str] = None,
    device: torch.device,
    metric_batch_size: int = 128,
) -> Dict[str, float]:
    """Run rank-0 generation metrics via fd_evaluator on a uint8 NHWC array.

    Called both from the distributed-generation path (after shard gather) and
    the offline_eval `--npz` short-circuit. `reference_npz_path` overrides the
    auto-resolved FID stats; pass None to use the catalogue default.
    """
    distributional = filter_distributional(metrics_to_compute)
    if not distributional:
        return {}
    ref = reference_npz_path[0] if isinstance(reference_npz_path, list) else reference_npz_path
    return compute_distributional_metrics(
        images,
        distributional,
        reference_npz=ref,
        data_dir=data_dir,
        device=device,
        batch_size=metric_batch_size,
    )


def _init_evaluators(metrics_to_compute: List[str], condition_type: str, device: torch.device):
    """Initialize metric evaluators and local score accumulators."""
    evaluators = {}
    local_scores = {}

    if 'clipscore' in metrics_to_compute and condition_type == 'text':
        evaluators['clipscore'] = CLIPScoreEvaluator(device=str(device))
        local_scores['clipscore'] = {'sum': 0.0, 'count': 0}

    if any(elem.startswith('vqascore') for elem in metrics_to_compute) and condition_type == 'text':
        vqascore_models = [elem for elem in metrics_to_compute if elem.startswith('vqascore')]
        vqascore_evaluators = {}
        for model_name in vqascore_models:
            model_name_ = model_name.split('_')[-1] if '_' in model_name else 'clip-flant5-xl'
            vqascore_evaluators[model_name] = VQAScoreEvaluator(model_name=model_name_, device=str(device))
            local_scores[model_name] = {'sum': 0.0, 'count': 0}
        evaluators['vqascore'] = vqascore_evaluators

    if 'geneval' in metrics_to_compute and condition_type == 'text':
        evaluators['geneval'] = GenEvalEvaluator(device=str(device))
        local_scores['geneval'] = {'sum': 0.0, 'count': 0}

    if 'dpgbench' in metrics_to_compute and condition_type == 'text':
        evaluators['dpgbench'] = DPGEvaluator(device=str(device))
        local_scores['dpgbench'] = {'sum': 0.0, 'count': 0}

    if 'lpips' in metrics_to_compute and condition_type == 'nwm':
        evaluators['lpips'] = LPIPSEvaluator(device=str(device))
        local_scores['lpips'] = {'sum': 0.0, 'count': 0}

    return evaluators, local_scores


def _aggregate_distributed_metrics(local_scores: dict, device: torch.device) -> Dict[str, float]:
    """All-reduce local score sums/counts across ranks and return averaged metrics."""
    metrics = {}
    for metric_name, scores in local_scores.items():
        sum_tensor = torch.tensor([scores['sum']], device=device, dtype=torch.float64)
        count_tensor = torch.tensor([scores['count']], device=device, dtype=torch.float64)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        if count_tensor.item() > 0:
            metrics[metric_name] = sum_tensor.item() / count_tensor.item()
    return metrics


@torch.no_grad()
def evaluate_generation_distributed(
    model_fn,
    sample_fn,
    latent_size,
    additional_model_kwargs,
    use_guidance: bool,
    rae,
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
    reference_npz_path: Optional[Union[str, List[str]]] = None,
    shared_tmpdir: Optional[str] = None,
    condition_type: str = "label",
    null_label: int = 1000,
    text_encoder=None,
    metrics_to_compute: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    Evaluate generation metrics using all GPUs in a distributed manner.

    Args:
        model_fn: Model forward function
        sample_fn: Sampling function
        latent_size: Shape of latent noise
        additional_model_kwargs: Additional kwargs for model forward
        use_guidance: Whether to use classifier-free guidance
        rae: RAE model for decoding latents to images
        val_dataset: Validation dataset (returns (image, label) or (image, text))
        num_samples: Number of samples to generate
        batch_size: Batch size per GPU for generation
        rank: Current GPU rank
        world_size: Total number of GPUs
        device: Device to use
        experiment_dir: Experiment directory
        global_step: Current training step
        autocast_kwargs: Autocast configuration
        metric_batch_size: Batch size for metric computation (on rank 0)
        reference_npz_path: Optional path (or list of paths) to existing reference NPZ
            files. If a list, FID is computed once per reference and emitted as
            fid_<tag> (tag derived from filename: jit, adm, or stem); fid is set
            to the first reference's value for backwards compatibility.
        shared_tmpdir: Optional shared directory for multi-node eval
        condition_type: Type of conditioning - "label" or "text"
        null_label: Null label index for CFG (label conditioning only)
        text_encoder: Text encoder for text conditioning (required if condition_type="text")
        metrics_to_compute: List of metrics to compute (default: ['fid'])

    Returns:
        Dictionary of metrics (only on rank 0, None on other ranks)
    """
    temp_dir = setup_eval_tmpdir(experiment_dir, global_step, rank,
                                  shared_tmpdir=shared_tmpdir, eval_type="sampling")
    loader = create_eval_dataloader(val_dataset, rank, world_size, num_samples, batch_size)

    # Initialize evaluators
    if metrics_to_compute is None:
        metrics_to_compute = ['fid']
    evaluators, local_scores = _init_evaluators(metrics_to_compute, condition_type, device)

    # Generate images on this rank
    generations = []
    iterator = tqdm(loader, desc=f"[Rank {rank}] Sampling", file=sys.stdout) if rank == 0 else loader

    with torch.inference_mode():
        for gt_img, cond in iterator:
            # Handle conditioning based on type
            if condition_type == "text":
                n = len(cond)
                z = torch.randn(n, *latent_size, device=device)
                enc_out = text_encoder(list(cond))
                context = enc_out["tokens"]
                context_attn_mask = enc_out["attention_mask"]
                if use_guidance:
                    z = torch.cat([z, z], dim=0)
                    enc_null = text_encoder([""] * n)
                    context_null = enc_null["tokens"]
                    context_attn_mask_null = enc_null["attention_mask"]
                    context = torch.cat([context, context_null], dim=0)
                    context_attn_mask = torch.cat([context_attn_mask, context_attn_mask_null], dim=0)
            elif condition_type == "nwm":
                from stage2 import nwm_cond
                z, context, context_attn_mask, n = nwm_cond.encode_eval_context(
                    cond, rae, device, latent_size, use_guidance,
                )
            else:
                n = cond.size(0)
                z = torch.randn(n, *latent_size, device=device)
                context = cond.to(device)
                context_attn_mask = None
                if use_guidance:
                    z = torch.cat([z, z], dim=0)
                    context_null = torch.full((n,), null_label, device=device)
                    context = torch.cat([context, context_null], dim=0)

            model_kwargs = dict(context=context, attn_mask=context_attn_mask, **additional_model_kwargs)
            with autocast(**autocast_kwargs):
                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                if use_guidance:
                    samples = samples.chunk(2, dim=0)[0]
                samples = rae.decode(samples).clamp(0, 1)
            gen_np = samples.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Compute distributed metrics during generation
            if 'clipscore' in evaluators:
                batch_scores = evaluators['clipscore'].compute_batch_scores(gen_np, list(cond))
                local_scores['clipscore']['sum'] += batch_scores.sum().item()
                local_scores['clipscore']['count'] += len(cond)
            if 'vqascore' in evaluators:
                for model_name, evaluator in evaluators['vqascore'].items():
                    batch_scores = evaluator.compute_batch_scores(gen_np, list(cond))
                    local_scores[model_name]['sum'] += batch_scores.sum().item()
                    local_scores[model_name]['count'] += len(cond)
            if 'geneval' in evaluators:
                batch_scores = evaluators['geneval'].compute_batch_scores(gen_np, list(cond))
                local_scores['geneval']['sum'] += batch_scores.sum().item()
                local_scores['geneval']['count'] += len(cond)
            if 'dpgbench' in evaluators:
                batch_scores = evaluators['dpgbench'].compute_batch_scores(gen_np, list(cond))
                local_scores['dpgbench']['sum'] += batch_scores.sum().item()
                local_scores['dpgbench']['count'] += len(cond)
            if 'lpips' in evaluators:
                gt_np = gt_img.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                batch_scores = evaluators['lpips'].compute_batch_scores(gen_np, gt_np)
                local_scores['lpips']['sum'] += batch_scores.sum().item()
                local_scores['lpips']['count'] += gen_np.shape[0]

            for img in gen_np:
                generations.append(img)

    generations = np.stack(generations)
    shard_path = os.path.join(temp_dir, f"gen_{global_step:07d}_{rank:02d}.npz")
    np.savez(shard_path, arr_0=generations)

    if rank == 0:
        print(f"[Rank {rank}] Saved {len(generations)} generation to {shard_path}")

    # Wait for all ranks to finish generation
    dist.barrier()

    # Distributed metrics: all_reduce sum and count across all ranks
    metrics = _aggregate_distributed_metrics(local_scores, device)

    # Rank 0 computes FID (requires gathering all samples)
    save_gen_npz = os.environ.get("SAVE_GEN_NPZ")
    if rank == 0:
        need_combined = (
            'fid' in metrics_to_compute
            or 'inception_score' in metrics_to_compute
            or bool(filter_distributional(metrics_to_compute))
            or save_gen_npz
        )
        if need_combined:
            combined_recons = gather_and_cleanup_shards(temp_dir, "gen", global_step, world_size, num_samples)
            print(f"[Eval] Combined generation NPZ shape: {combined_recons.shape}")

            if save_gen_npz:
                os.makedirs(os.path.dirname(save_gen_npz), exist_ok=True)
                np.savez(save_gen_npz, arr_0=combined_recons)
                print(f"[Eval] Saved gen NPZ to {save_gen_npz}")

            metrics.update(
                evaluate_image_set(
                    combined_recons,
                    metrics_to_compute=metrics_to_compute,
                    reference_npz_path=reference_npz_path,
                    data_dir=data_dir,
                    device=device,
                    metric_batch_size=metric_batch_size,
                )
            )
        else:
            for r in range(world_size):
                shard_file = os.path.join(temp_dir, f"gen_{global_step:07d}_{r:02d}.npz")
                if os.path.exists(shard_file):
                    os.remove(shard_file)

        # Print results
        print(f"[Eval] Step {global_step} Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

    dist.barrier()
    return metrics if metrics else None
