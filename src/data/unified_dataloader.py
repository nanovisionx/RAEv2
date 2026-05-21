"""Unified dataloader interface for RAEv2 training."""
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import webdataset as wds
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from .imagenet_hf_dataset import ImageNetHFDataset

T2I_HF_DATASETS = {'mscoco', 'mjhq', 'geneval', 'dpgbench', 'genaibench', 'simpleeval', 'sft_hack_datasets'}
GENERIC_WDS_DATASETS = {'flux-synthetic-256', 'rendertext-256'}
ARROW_EVAL_TARGETS = {'arrow-eval'}  # generic map-style HF Arrow eval source; needs `data_dir` pointing at the Arrow dir


@dataclass
class DataloaderResult:
    """
    Unified result from prepare_unified_dataloader.
    Provides consistent interface for map-style and iterable datasets.
    """
    loader: Union[DataLoader, wds.WebLoader]
    sampler: Optional[DistributedSampler]
    dataset_size: int
    is_iterable: bool = False
    _wds_pipeline: Optional[object] = field(default=None, repr=False)
    _batch_size: int = 1
    _num_workers: int = 4
    _world_size: int = 1
    virtual_epoch_steps: Optional[int] = None

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling. Works for both dataset types.

        For map-style: calls sampler.set_epoch()
        For WebDataset: recreates pipeline with new seed (uses virtual_epoch_steps if set)
        """
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        elif self._wds_pipeline is not None:
            self._recreate_wds_loader(epoch)

    def _recreate_wds_loader(self, epoch: int):
        """Recreate WebDataset loader for new epoch."""
        dataset = self._wds_pipeline.create_pipeline(epoch=epoch)
        steps = self.virtual_epoch_steps or (self.dataset_size // (self._batch_size * self._world_size))
        loader = wds.WebLoader(
            dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        self.loader = loader.with_epoch(steps)

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        if self.virtual_epoch_steps is not None:
            return self.virtual_epoch_steps
        if self.is_iterable:
            return self.dataset_size // (self._batch_size * self._world_size)
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)


class _ArrowEvalDataset(Dataset):
    """Map-style dataset that loads an HF Arrow dir directly.

    Used for online stage-1 eval on pre-baked Arrow validation sets such as
    `data/rendertext-256/val/` and `data/scale-rae-flux-synthetic-256/val/`.
    Returns (image_tensor, text_string) for parity with other eval datasets;
    the trainer drops the second tuple element.
    """

    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None):
        from datasets import load_from_disk
        self.dataset = load_from_disk(str(data_dir))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        image = sample['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.get('text', '')


def _prepare_arrow_eval_loader(
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    transform: Optional[transforms.Compose],
    shuffle: bool,
) -> "DataloaderResult":
    """Prepare an eval-only loader from an HF Arrow directory."""
    data_dir = config.get('data_dir')
    if not data_dir:
        raise ValueError("arrow-eval target requires `data_dir` pointing at the Arrow directory")

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    dataset = _ArrowEvalDataset(data_dir=data_dir, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    return DataloaderResult(
        loader=loader,
        sampler=sampler,
        dataset_size=len(dataset),
        is_iterable=False,
    )


class _T2IHFDataset(Dataset):
    """Internal HuggingFace dataset wrapper for MSCOCO/MJHQ T2I datasets."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "val",
        transform: Optional[transforms.Compose] = None,
        data_dir: Optional[str] = "./data",
    ):
        from datasets import load_from_disk

        # Load from local Arrow format (e.g., data/mscoco/val)
        local_path = Path(data_dir) / dataset_name / split
        self.hf_dataset = load_from_disk(str(local_path))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        sample = self.hf_dataset[idx]
        text = sample['text']

        if 'image' in sample:
            image = sample['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        else:
            image = torch.empty(0)

        return image, text


def prepare_unified_dataloader(
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    transform: Optional[transforms.Compose] = None,
    condition_type: str = "text",
    shuffle: bool = True,
    virtual_epoch_steps: Optional[int] = None,
) -> DataloaderResult:
    """
    Unified dataloader factory for ImageNet, BLIP3O, MSCOCO, MJHQ, NWM,
    rendertext-256, flux-synthetic-256, and multi-source `mix:` configs.
    """
    # Multi-source mix: dataset.mix = [{target: ..., weight: ..., ...}, ...]
    if config.get("mix"):
        return _prepare_mixed_loader(
            config, image_size, batch_size, num_workers, rank, world_size,
            transform, condition_type, shuffle, virtual_epoch_steps,
        )

    target = config.get("target", "imagenet")

    if target in T2I_HF_DATASETS:
        result = _prepare_t2i_hf_loader(
            target, config, image_size, batch_size, num_workers, rank, world_size, transform, shuffle
        )
    elif target == "blip3o":
        result = _prepare_blip3o_loader(
            config, image_size, batch_size, num_workers, world_size, transform
        )
    elif target in GENERIC_WDS_DATASETS:
        result = _prepare_generic_wds_loader(
            target, config, image_size, batch_size, num_workers, world_size, transform
        )
    elif target in ARROW_EVAL_TARGETS:
        result = _prepare_arrow_eval_loader(
            config, image_size, batch_size, num_workers, rank, world_size, transform, shuffle
        )
    elif target == "imagenet":
        result = _prepare_imagenet_loader(
            config, image_size, batch_size, num_workers, rank, world_size, transform, condition_type, shuffle
        )
    elif target == "nwm":
        result = _prepare_nwm_loader(
            config, image_size, batch_size, num_workers, rank, world_size, transform, shuffle
        )
    else:
        raise ValueError(f"Unknown dataset target: {target!r}")
    result.virtual_epoch_steps = virtual_epoch_steps
    return result


def _prepare_generic_wds_loader(
    dataset_name: str,
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    world_size: int,
    transform: Optional[transforms.Compose],
) -> DataloaderResult:
    """Prepare a generic image-only WebDataset loader (rendertext-256, flux-synthetic-256)."""
    from .wds_image_dataset import GenericWebDataset

    data_dir = config.get("data_dir", f"./data/{dataset_name}")
    subsets = config.get("subsets", config.get("subset", []))
    if not subsets:
        raise ValueError(f"{dataset_name}: dataset config must specify 'subsets' (list of subset folders)")
    shuffle_buffer = config.get("shuffle_buffer", 10000)
    seed = config.get("seed", 42)

    wds_pipeline = GenericWebDataset(
        data_dir=data_dir,
        subsets=subsets,
        dataset_name=dataset_name,
        transform=transform,
        image_size=image_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    dataset = wds_pipeline.create_pipeline(epoch=0)
    total_samples = wds_pipeline.estimated_size
    steps = max(1, total_samples // (batch_size * world_size))

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    loader = loader.with_epoch(steps)

    return DataloaderResult(
        loader=loader,
        sampler=None,
        dataset_size=total_samples,
        is_iterable=True,
        _wds_pipeline=wds_pipeline,
        _batch_size=batch_size,
        _num_workers=num_workers,
        _world_size=world_size,
    )


def _prepare_blip3o_loader(
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    world_size: int,
    transform: Optional[transforms.Compose],
) -> DataloaderResult:
    """Prepare BLIP3O WebDataset loader."""
    from .blip3o_wds_dataset import BLIP3OWebDataset

    data_dir = config.get("data_dir", "./data/blip3o")
    # Support both 'splits' (list) and 'split' (single) keys.
    # `or` (not `dict.get` default) so a present-but-None `splits` still falls back to `split`.
    splits = config.get("splits") or config.get("split") or "short-caption"
    shuffle_buffer = config.get("shuffle_buffer", 10000)
    seed = config.get("seed", 42)

    wds_pipeline = BLIP3OWebDataset(
        data_dir=data_dir,
        splits=splits,
        transform=transform,
        image_size=image_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    dataset = wds_pipeline.create_pipeline(epoch=0)
    total_samples = wds_pipeline.estimated_size
    steps = total_samples // (batch_size * world_size)

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    # Bound epoch to exactly `steps` batches (with_epoch stops iteration, with_length only sets __len__)
    loader = loader.with_epoch(steps)

    return DataloaderResult(
        loader=loader,
        sampler=None,
        dataset_size=total_samples,
        is_iterable=True,
        _wds_pipeline=wds_pipeline,
        _batch_size=batch_size,
        _num_workers=num_workers,
        _world_size=world_size,
    )


def _prepare_t2i_hf_loader(
    dataset_name: str,
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    transform: Optional[transforms.Compose],
    shuffle: bool,
) -> DataloaderResult:
    """Prepare MSCOCO/MJHQ HuggingFace loader."""
    split = config.get("split", "val")
    data_dir = config.get("data_dir", "./data")  # Local data directory

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    dataset = _T2IHFDataset(
        dataset_name=dataset_name,
        split=split,
        transform=transform,
        data_dir=data_dir,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,  # drop_last=True for train, False for eval
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    return DataloaderResult(
        loader=loader,
        sampler=sampler,
        dataset_size=len(dataset),
        is_iterable=False,
    )


def _prepare_imagenet_loader(
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    transform: Optional[transforms.Compose],
    condition_type: str,
    shuffle: bool = True,
) -> DataloaderResult:
    """Prepare ImageNet-style loader using existing dataset classes."""
    data_dir = config.get("data_dir", "./data/imagenet-256")
    split = config.get("split", "train")

    # Build transform if not provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    # Create dataset based on type
    dataset = ImageNetHFDataset(
        data_dir=data_dir,
        split=split,
        transform=transform,
        condition_type=condition_type,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,  # drop_last=True for train, False for eval
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    return DataloaderResult(
        loader=loader,
        sampler=sampler,
        dataset_size=len(dataset),
        is_iterable=False,
    )


def _prepare_nwm_loader(
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    transform: Optional[transforms.Compose],
    shuffle: bool = True,
) -> DataloaderResult:
    """Prepare RECON nwm loader. Returns (target_image, nwm_cond_dict) batches."""
    from .nwm_dataset import NWMHFDataset, nwm_collate_fn

    params = config.get("params") or config
    data_dir = params.get("data_dir", config.get("data_dir", "./data/recon"))
    split = params.get("split", config.get("split", "train"))

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    dataset = NWMHFDataset(
        data_dir=data_dir,
        split=split,
        transform=transform,
        context_size=params.get("context_size", 4),
        len_traj_pred=params.get("len_traj_pred", 8),
        metric_waypoint_spacing=params.get("metric_waypoint_spacing", None),
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        collate_fn=nwm_collate_fn,
    )

    return DataloaderResult(
        loader=loader,
        sampler=sampler,
        dataset_size=len(dataset),
        is_iterable=False,
    )


class MixedDataloader:
    """Per-step weighted mixture of multiple child DataloaderResults.

    Each step picks one child via `random.choices(weights=...)` (seeded per epoch)
    and yields that child's next batch as-is. Per-step (not per-batch) selection
    keeps each child's batch homogeneous, preserves their DDP sharding, and
    avoids GPU-side stitching cost. The trainer drops the conditioning slot via
    `for images, _ in dataloader`, so int-label and string-caption children mix
    safely.

    Exposes the same surface as DataloaderResult: `.loader` (self),
    `.set_epoch(e)`, `__len__`, `__iter__`. Also assignable `virtual_epoch_steps`
    so the dispatch can override the auto-computed length.
    """

    def __init__(
        self,
        children: List["DataloaderResult"],
        weights: List[float],
        batch_size: int,
        world_size: int,
        seed: int = 42,
    ):
        if len(children) != len(weights):
            raise ValueError("children and weights must have equal length")
        if any(w <= 0 for w in weights):
            raise ValueError("weights must be positive")
        self.children = children
        self.weights = list(weights)
        self.batch_size = batch_size
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self._virtual_epoch_steps: Optional[int] = None

        sum_w = sum(self.weights)
        weighted_samples = sum(c.dataset_size * w for c, w in zip(children, self.weights)) / sum_w
        self._auto_steps = max(1, int(weighted_samples // (batch_size * world_size)))

        # Trainer compat: act as our own loader.
        self.loader = self
        self.sampler = None
        self.is_iterable = True
        self.dataset_size = sum(c.dataset_size for c in children)
        # Names of each child source (set by _prepare_mixed_loader); used by
        # the sanity helper to report per-source draw counts.
        self.child_names: List[str] = [f"child_{i}" for i in range(len(children))]
        # Index of the most recently drawn child (read by the sanity helper).
        self.last_child_idx: Optional[int] = None

    @property
    def virtual_epoch_steps(self) -> Optional[int]:
        return self._virtual_epoch_steps

    @virtual_epoch_steps.setter
    def virtual_epoch_steps(self, value: Optional[int]) -> None:
        self._virtual_epoch_steps = value

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        for child in self.children:
            child.set_epoch(epoch)

    def __len__(self) -> int:
        return self._virtual_epoch_steps or self._auto_steps

    def __iter__(self):
        iters = [iter(c.loader) for c in self.children]
        rng = random.Random(self.seed + self.epoch)
        n_steps = len(self)
        for _ in range(n_steps):
            i = rng.choices(range(len(self.children)), weights=self.weights, k=1)[0]
            try:
                batch = next(iters[i])
            except StopIteration:
                # Refresh exhausted child; for wds children this re-creates
                # the pipeline with a new epoch seed via DataloaderResult.set_epoch.
                self.children[i].set_epoch(self.epoch + 1)
                iters[i] = iter(self.children[i].loader)
                batch = next(iters[i])
            self.last_child_idx = i
            yield batch


def _prepare_mixed_loader(
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    transform: Optional[transforms.Compose],
    condition_type: str,
    shuffle: bool,
    virtual_epoch_steps: Optional[int],
) -> "MixedDataloader":
    """Build a MixedDataloader from a `dataset.mix: [...]` config block."""
    mix_entries = config["mix"]
    if not mix_entries:
        raise ValueError("dataset.mix must be a non-empty list")
    seed = config.get("seed", 42)

    children: List[DataloaderResult] = []
    weights: List[float] = []
    names: List[str] = []
    for entry in mix_entries:
        weight = float(entry.get("weight", 1.0))
        sub_cfg = {k: v for k, v in entry.items() if k != "weight"}
        names.append(str(entry.get("name") or entry.get("target", "unknown")))
        child = prepare_unified_dataloader(
            config=sub_cfg,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            transform=transform,
            condition_type=condition_type,
            shuffle=shuffle,
            virtual_epoch_steps=None,  # children stay un-bounded; mix sets outer length
        )
        children.append(child)
        weights.append(weight)

    mixed = MixedDataloader(
        children=children,
        weights=weights,
        batch_size=batch_size,
        world_size=world_size,
        seed=seed,
    )
    mixed.child_names = names
    mixed.virtual_epoch_steps = virtual_epoch_steps
    return mixed
