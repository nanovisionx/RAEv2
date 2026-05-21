"""Navigation World Model (nwm) dataset (RECON).

Loads from the per-frame HF Arrow built by helper_scripts/build_recon_arrow.py.
Each __getitem__ samples a (context, target, action, rel_time) tuple from a
randomly chosen trajectory, where the conditioning is K past frames + an
egocentric action delta `(dx, dy, dyaw)` (normalized) + a scalar rel_time.

Returns:
    target_image: Tensor[3, H, W]                                -- the future frame to predict
    nwm_cond:     dict {"context_frames": Tensor[K, 3, H, W],
                        "action":         Tensor[3],
                        "rel_time":       Tensor[1]}
"""
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

# raenwm convention: rel_time is normalized by 128 (max trajectory horizon used in raenwm).
RAENWM_MAX_TIMESTEP = 128.0


def _to_local_coords_2d(delta_xy: np.ndarray, curr_yaw: float) -> np.ndarray:
    """Rotate (dx, dy) into a frame oriented along curr_yaw (raenwm/misc.py:to_local_coords)."""
    c, s = math.cos(curr_yaw), math.sin(curr_yaw)
    rotmat = np.array([[c, -s], [s, c]], dtype=np.float32)
    return delta_xy @ rotmat


def _angle_difference(theta1: float, theta2: float) -> float:
    d = theta2 - theta1
    return float(d - 2 * math.pi * math.floor((d + math.pi) / (2 * math.pi)))


class NWMHFDataset(Dataset):
    """Sample (context, target, action) tuples from a per-frame RECON Arrow dataset."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[object] = None,
        context_size: int = 4,
        len_traj_pred: int = 8,
        metric_waypoint_spacing: Optional[float] = None,
        action_stats_path: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.context_size = int(context_size)
        self.len_traj_pred = int(len_traj_pred)

        ds_path = self.data_dir / split
        if not ds_path.exists():
            raise FileNotFoundError(f"Split '{split}' not found at {ds_path}")
        self.dataset = load_from_disk(str(ds_path))

        stats_path = Path(action_stats_path) if action_stats_path else self.data_dir / "recon_action_stats.json"
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.action_min = np.asarray(stats["min"], dtype=np.float32)  # (2,) -- (min_dx, min_dy)
        self.action_max = np.asarray(stats["max"], dtype=np.float32)  # (2,)
        self.metric_waypoint_spacing = float(
            metric_waypoint_spacing if metric_waypoint_spacing is not None else stats["metric_waypoint_spacing"]
        )

        # Build traj_id -> sorted [row_idx] index. The Arrow rows are out of order
        # (multiprocessing during build), so we need a scan + sort.
        traj_col = np.asarray(self.dataset["traj_id"], dtype=np.int64)
        frame_col = np.asarray(self.dataset["frame_idx"], dtype=np.int64)
        order = np.lexsort((frame_col, traj_col))  # primary: traj_id, secondary: frame_idx
        traj_sorted = traj_col[order]
        # Group contiguous runs of the same traj_id into row-index lists (sorted by frame_idx).
        self.traj_to_rows: Dict[int, np.ndarray] = {}
        if len(order) > 0:
            boundaries = np.flatnonzero(np.diff(traj_sorted)) + 1
            chunks = np.split(order, boundaries)
            for chunk in chunks:
                tid = int(traj_col[chunk[0]])
                self.traj_to_rows[tid] = chunk

        # Drop trajectories too short for a (context + target) tuple.
        min_len = self.context_size + 1
        self.valid_traj_ids = [tid for tid, rows in self.traj_to_rows.items() if len(rows) >= min_len]
        if not self.valid_traj_ids:
            raise RuntimeError(f"No trajectories with >= {min_len} frames in {ds_path}")

        # __len__ -> use number of frames as a proxy "epoch size", matching raenwm's
        # samples-per-trajectory expansion roughly. This is just a knob for sampler
        # length; the actual sampling is random.
        self._epoch_len = sum(max(0, len(rows) - self.context_size) for rows in self.traj_to_rows.values())

    def __len__(self) -> int:
        return self._epoch_len

    @property
    def num_classes(self) -> int:
        return 0  # nwm has no class labels

    def _normalize_action(self, dxy: np.ndarray, dyaw: float) -> np.ndarray:
        """RECON action normalization: local egocentric -> /spacing -> min-max to [-1, 1].

        dxy: (2,) already-rotated egocentric delta (meters)
        Yaw is left in radians (NOT min-max normalized, matching raenwm's _compute_actions).
        """
        dxy = dxy / self.metric_waypoint_spacing
        # min-max normalize to [-1, 1]
        dxy01 = (dxy - self.action_min) / (self.action_max - self.action_min)
        dxy_norm = dxy01 * 2.0 - 1.0
        return np.concatenate([dxy_norm, [dyaw]], axis=0).astype(np.float32)

    def _load_frame(self, row_idx: int) -> torch.Tensor:
        sample = self.dataset[int(row_idx)]
        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            return self.transform(img)
        # Default: just convert to tensor in [0, 1]
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # idx is a sampler index into _epoch_len; we use it only as a deterministic
        # seed proxy if needed. Tuple sampling itself is random per call.
        rng = np.random
        traj_id = self.valid_traj_ids[rng.randint(len(self.valid_traj_ids))]
        rows = self.traj_to_rows[traj_id]
        T = len(rows)

        # Sample current time t in [context_size - 1, T - 2] (need at least 1 future frame).
        # Sample offset in [1, len_traj_pred] but clipped so t + offset < T.
        max_t = T - 2
        min_t = self.context_size - 1
        if max_t < min_t:
            t = min_t
        else:
            t = rng.randint(min_t, max_t + 1)
        max_off = min(self.len_traj_pred, T - 1 - t)
        offset = rng.randint(1, max_off + 1) if max_off >= 1 else 1
        target_t = t + offset

        # Gather row indices for context (t - K + 1 .. t) and target (t + offset).
        ctx_rows = rows[t - self.context_size + 1 : t + 1]
        tgt_row = rows[target_t]

        # Pull metadata in one batched access.
        ctx_meta = self.dataset[[int(r) for r in ctx_rows]]
        tgt_meta = self.dataset[int(tgt_row)]

        # Action: target position - context[-1] position, rotated into context[-1] frame.
        curr_pos = np.array([ctx_meta["position_x"][-1], ctx_meta["position_y"][-1]], dtype=np.float32)
        curr_yaw = float(ctx_meta["yaw"][-1])
        tgt_pos = np.array([tgt_meta["position_x"], tgt_meta["position_y"]], dtype=np.float32)
        tgt_yaw = float(tgt_meta["yaw"])

        dxy_local = _to_local_coords_2d((tgt_pos - curr_pos)[None, :], curr_yaw)[0]
        dyaw_local = _angle_difference(curr_yaw, tgt_yaw)
        action = self._normalize_action(dxy_local, dyaw_local)

        # Stack frames.
        ctx_imgs = torch.stack(
            [self._load_frame(int(r)) for r in ctx_rows], dim=0
        )  # (K, 3, H, W)
        target_img = self._load_frame(int(tgt_row))  # (3, H, W)

        nwm_cond = {
            "context_frames": ctx_imgs,
            "action": torch.from_numpy(action),  # (3,)
            "rel_time": torch.tensor([offset / RAENWM_MAX_TIMESTEP], dtype=torch.float32),
        }
        return target_img, nwm_cond


def nwm_collate_fn(batch):
    """Default collate stacks the dict fields per key. Used by the DataLoader."""
    target_imgs = torch.stack([b[0] for b in batch], dim=0)
    keys = batch[0][1].keys()
    nwm_cond = {k: torch.stack([b[1][k] for b in batch], dim=0) for k in keys}
    return target_imgs, nwm_cond
