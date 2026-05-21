"""Evaluation dataset utilities for multi-dataset support."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from torch.utils.data import Dataset

from data.unified_dataloader import prepare_unified_dataloader

logger = logging.getLogger(__name__)


@dataclass
class EvalDatasetInfo:
    """Container for eval dataset info."""
    dataset: Dataset
    reference_npz: Optional[Union[str, List[str]]]  # Only required when 'fid' in metrics
    condition_type: str
    metrics: List[str] = field(default_factory=lambda: ['fid'])
    num_samples: Optional[int] = None  # cap eval at this many samples; None -> full val set
    data_dir: Optional[str] = None     # passed to fd_evaluator for MIND raw-image lookup


def normalize_eval_datasets(datasets_cfg):
    """
    Normalize eval.datasets config to dict of {name: dataset_config}.

    Supported format:
        eval.datasets = {mscoco: {...}, mjhq: {...}}

    Returns dict of {name: dataset_config}. If no datasets are configured, returns an empty dict.
    """
    result = {}
    for name, cfg in datasets_cfg.items():
        result[name] = cfg.copy()
        # set target to name if not explicitly provided (for simpleeval different versions)
        if 'target' not in result[name]:
            result[name]['target'] = name
    return result


def prepare_eval_datasets(
    eval_datasets_config: Dict[str, dict],
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
) -> Dict[str, EvalDatasetInfo]:
    """
    Prepare eval datasets from normalized config.

    Returns dict of {name: EvalDatasetInfo}.
    """
    eval_datasets = {}

    for ds_name, ds_cfg in eval_datasets_config.items():
        ds_cond_type = ds_cfg.get('condition_type', 'text')

        result = prepare_unified_dataloader(
            config=ds_cfg,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            condition_type=ds_cond_type,
            shuffle=False,
        )

        eval_datasets[ds_name] = EvalDatasetInfo(
            dataset=result.loader.dataset,
            reference_npz=ds_cfg.get('reference_npz'),
            condition_type=ds_cond_type,
            metrics=ds_cfg.get('metrics', ['fid']),
            num_samples=ds_cfg.get('num_samples'),
            data_dir=ds_cfg.get('data_dir'),
        )
        logger.info(f"Eval dataset loaded: {ds_name}, {len(result.loader.dataset)} samples")

    return eval_datasets
