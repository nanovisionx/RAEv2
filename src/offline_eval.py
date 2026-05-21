#!/usr/bin/env python
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Offline evaluation script for generation models.

Generates samples using a pre-trained stage-2 model and computes metrics.
Follows the same initialization patterns as train.py for consistent behavior.

Supports:
- Multiple eval datasets through unified dataloader (like train.py)
- Multiple metrics per dataset: fid, clipscore, vqascore, geneval
- Text conditioning (CLIP, T5, etc.) and label conditioning
- Internal Guidance and CFG
"""

import argparse
import dataclasses
import logging
import math
import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from configs.stage2 import Stage2Config
from encoders.vision_encoder import load_encoders
from eval import evaluate_generation_distributed, evaluate_image_set
from eval.datasets import normalize_eval_datasets, prepare_eval_datasets
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_sampler, create_transport
from stage2.utils import setup_text_encoder, validate_stage2_config
from utils.dist_utils import main_process_first
from utils.guidance_utils import get_model_forward_fn
from utils.logging import save_eval_to_csv
from utils.model_utils import instantiate_from_config
from utils.train_utils import get_autocast_kwargs

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    """Run offline evaluation with distributed execution.

    If `--npz <path>` is provided, skip distributed generation entirely and
    run metrics on the existing NPZ. Useful for validating eval-side changes
    without re-sampling. Single-process; no torchrun needed.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires at least one GPU.")

    # Enable TF32 for faster computation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    if args.npz is not None:
        _run_npz_only(args)
        return

    # Initialize distributed
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    # Setup autocast
    autocast_kwargs = get_autocast_kwargs(args)

    config: Stage2Config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config)))
    config.post_process()
    validate_stage2_config(config)

    # Set seed (per-rank stride configurable via EVAL_SEED_STRIDE env var; default 1)
    stride = int(os.environ.get("EVAL_SEED_STRIDE", "1"))
    seed = config.training.global_seed * world_size * stride + rank * stride
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #########################################################
    # Models setup
    #########################################################
    latent_size = tuple(config.misc.latent_size)

    rae: RAE = instantiate_from_config(config.stage_1).to(device)
    rae.eval()

    # repa target encoder
    repa_target_encoder = None
    if config.repa.use_repa:
        with main_process_first(rank):
            repa_target_encoder = load_encoders(config.repa.target_encoder, device, config.repa.target_encoder_resolution)[0]
        repa_target_encoder.eval()
        repa_target_encoder.model.requires_grad_(False)
        config.repa.z_dim = repa_target_encoder.embed_dim
        logger.info(f"REPA target encoder: {config.repa.target_encoder}, embed_dim={repa_target_encoder.embed_dim}")

    # text encoder for text conditioning; None if not using text conditioning
    text_encoder = setup_text_encoder(config, rank, device)

    # prepare model params (must be called before model instantiation so that
    # condition_type, context_dim, repa z_dim etc. are set)
    config.prepare_model_params()

    model: Stage2ModelProtocol = instantiate_from_config(config.stage_2).to(device)
    model.eval()
    model_fn, sample_model_kwargs = get_model_forward_fn(model, config.guidance)
    use_guidance = config.guidance.any_guidance_active
    if rank == 0:
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    #########################################################
    # Transport + Sampler setup
    #########################################################
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / config.misc.time_dist_shift_base
    )
    transport = create_transport(
        config=config.transport,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = create_sampler(transport, guidance_config=config.guidance)
    eval_sampler = transport_sampler.sample_ode(**dataclasses.asdict(config.sampler))

    # ============================================================
    # Eval datasets setup
    # ============================================================
    global_step = 0
    global_batch_size = config.training.global_batch_size or (config.training.batch_size * world_size * config.training.grad_accum_steps)
    assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
    micro_batch_size = global_batch_size // (world_size * config.training.grad_accum_steps)
    assert config.eval is not None, "eval section is required in config"
    eval_datasets_config = normalize_eval_datasets(config.eval.datasets)
    eval_datasets = prepare_eval_datasets(
        eval_datasets_config,
        image_size=config.training.image_size,
        batch_size=micro_batch_size,
        num_workers=config.training.num_workers,
        rank=rank,
        world_size=world_size,
    )
    eval_dir = config.eval.eval_dir or os.path.join("evals", "stage2")

    experiment_name = os.environ.get("EXPERIMENT_NAME")
    assert experiment_name is not None, "Please set the EXPERIMENT_NAME environment variable."

    # ============================================================
    # Run evaluation for each dataset
    # ============================================================
    for ds_name, ds_info in eval_datasets.items():
        if rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating on {ds_name}...")
            logger.info(f"  Samples: {len(ds_info.dataset)}")
            logger.info(f"  Condition type: {ds_info.condition_type}")
            logger.info(f"  Metrics: {ds_info.metrics}")
            logger.info(f"  Reference: {ds_info.reference_npz}")
            logger.info(f"{'='*60}")

        eval_n = min(ds_info.num_samples or len(ds_info.dataset), len(ds_info.dataset))
        eval_stats = evaluate_generation_distributed(
            model_fn, eval_sampler, tuple(config.misc.latent_size), sample_model_kwargs,
            use_guidance, rae, ds_info.dataset, eval_n,
            rank=rank, world_size=world_size, device=device,
            batch_size=micro_batch_size, experiment_dir=experiment_name,
            global_step=global_step, autocast_kwargs=autocast_kwargs,
            reference_npz_path=ds_info.reference_npz,
            shared_tmpdir=config.dataset.shared_tmpdir,
            condition_type=ds_info.condition_type,
            null_label=config.misc.num_classes,
            text_encoder=text_encoder if ds_info.condition_type == "text" else None,
            metrics_to_compute=ds_info.metrics,
            data_dir=ds_info.data_dir,
        )
        if eval_stats is not None and rank == 0:
            save_eval_to_csv(experiment_name, "ema", global_step, {'dataset': ds_name, **eval_stats}, eval_dir)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        logger.info("\nOffline evaluation complete.")


def _run_npz_only(args):
    """Short-circuit: run metrics on an existing NPZ; skip all model setup.

    Reads `eval.datasets` from the config: for each dataset, runs the configured
    metrics on the provided NPZ, writes one CSV row per dataset. Lets you
    validate Phase-3 wiring against the 4 already-saved NPZs without burning
    compute generating new samples.
    """
    device = torch.device("cuda", 0)

    config_root = OmegaConf.load(args.config)
    eval_cfg = config_root.eval
    eval_dir = eval_cfg.get("eval_dir") or os.path.join("evals", "stage2")

    experiment_name = os.environ.get("EXPERIMENT_NAME")
    assert experiment_name is not None, "Please set EXPERIMENT_NAME."

    logger.info(f"[npz mode] config: {args.config}")
    logger.info(f"[npz mode] npz: {args.npz}")
    logger.info(f"[npz mode] loading NPZ ...")

    import numpy as np
    npz = np.load(args.npz, mmap_mode="r")
    key = "arr_0" if "arr_0" in npz else list(npz.keys())[0]
    gen = np.ascontiguousarray(npz[key])
    rng = np.random.default_rng(0)
    gen = gen[rng.permutation(gen.shape[0])]
    logger.info(f"[npz mode] shape={gen.shape} dtype={gen.dtype} (shuffled, seed=0)")

    for ds_name, ds_cfg in eval_cfg.datasets.items():
        ds_cfg = OmegaConf.to_container(ds_cfg, resolve=True) if hasattr(ds_cfg, "_metadata") else dict(ds_cfg)
        metrics = list(ds_cfg.get("metrics") or ["fid"])
        ref = ds_cfg.get("reference_npz")
        data_dir = ds_cfg.get("data_dir")
        num_samples = ds_cfg.get("num_samples")
        if num_samples is not None and gen.shape[0] > num_samples:
            logger.info(f"  truncating gen array from {gen.shape[0]} to num_samples={num_samples}")
            gen_for_ds = gen[:num_samples]
        else:
            gen_for_ds = gen
        logger.info(f"\n[npz mode] === {ds_name} ===")
        logger.info(f"  metrics: {metrics}")
        logger.info(f"  reference_npz: {ref}")
        logger.info(f"  data_dir: {data_dir}")

        stats = evaluate_image_set(
            gen_for_ds,
            metrics_to_compute=metrics,
            reference_npz_path=ref,
            data_dir=data_dir,
            device=device,
        )
        from fd_evaluator import format_results
        logger.info("\n" + format_results(stats))
        save_eval_to_csv(experiment_name, "ema", 0, {"dataset": ds_name, **stats}, eval_dir)

    logger.info("\n[npz mode] done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation for generation models")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="bf16",
                        help="Compute precision")
    parser.add_argument("--npz", type=str, default=None,
                        help="Optional: path to an existing uint8 NHWC gen NPZ. If set, "
                             "skip distributed sampling and just compute metrics on it. "
                             "Single-process; no torchrun needed.")
    args = parser.parse_args()
    main(args)
