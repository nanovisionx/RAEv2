import math
from typing import List, Tuple

import torch
from torch import Tensor


def _linspace_indices(limit: int, count: int) -> List[int]:
    if count <= 1:
        return [0]
    return sorted({int(round(i * (limit / (count - 1)))) for i in range(count)})


def _gen_positions_1d(length: int, crop: int, slots: int) -> List[int]:
    limit = max(length - crop, 0)
    pos = _linspace_indices(limit, max(slots, 1))
    pos = [max(0, min(p, limit)) for p in pos]
    if slots > 1:
        pos[0] = 0
        pos[-1] = limit
    return pos


class RandomWindowCrop:
    """Random crop with a fixed catalog of windows (XLA-friendly variant)."""

    def __init__(
        self,
        input_size: int | Tuple[int, int],
        crop: int,
        num_windows: int,
        per_sample: bool = False,
    ):
        if isinstance(input_size, int):
            self.H = self.W = int(input_size)
        else:
            self.H, self.W = map(int, input_size)
        self.crop = int(crop)
        self.per_sample = bool(per_sample)

        if self.crop <= 0:
            raise ValueError("crop must be > 0")
        if self.crop > self.H or self.crop > self.W:
            raise ValueError(f"crop={self.crop} exceeds input {(self.H, self.W)}")
        if num_windows <= 0:
            raise ValueError("num_windows must be > 0")

        rows_min = math.ceil(self.H / self.crop)
        cols_min = math.ceil(self.W / self.crop)
        n_min = rows_min * cols_min
        if num_windows < n_min:
            raise ValueError(
                f"num_windows={num_windows} too small to cover {(self.H, self.W)} with crop {self.crop}"
            )

        t_rows = _gen_positions_1d(self.H, self.crop, rows_min)
        l_cols = _gen_positions_1d(self.W, self.crop, cols_min)
        base_offsets = [(t, l) for t in t_rows for l in l_cols]

        offsets = list(base_offsets)
        if num_windows > len(offsets):
            rows_t = max(rows_min, int(math.floor(math.sqrt(num_windows * self.H / self.W))))
            cols_t = max(cols_min, int(math.ceil(num_windows / rows_t)))
            while rows_t * cols_t < num_windows:
                cols_t += 1

            t_more = _gen_positions_1d(self.H, self.crop, rows_t)
            l_more = _gen_positions_1d(self.W, self.crop, cols_t)
            dense = [(t, l) for t in t_more for l in l_more]

            seen = set(offsets)
            for off in dense:
                if len(offsets) >= num_windows:
                    break
                if off not in seen:
                    offsets.append(off)
                    seen.add(off)

            idx = 0
            while len(offsets) < num_windows and idx < len(dense):
                offsets.append(dense[idx])
                idx += 1

        self.offsets: List[Tuple[int, int]] = offsets[:num_windows]
        self.num_windows = len(self.offsets)

    def __repr__(self) -> str:
        return (
            f"RandomWindowCrop(input={(self.H, self.W)}, crop={self.crop}, "
            f"windows={self.num_windows}, per_sample={self.per_sample})"
        )

    def _rand_idx(self) -> int:
        return torch.randint(0, self.num_windows, (1,)).item()

    def __call__(self, tensor: Tensor) -> Tensor:
        H, W = tensor.shape[-2], tensor.shape[-1]
        if (H, W) != (self.H, self.W):
            raise ValueError(f"Expected input {(self.H, self.W)}, got {(H, W)}")

        crop = self.crop
        if self.per_sample and tensor.dim() >= 4 and tensor.shape[0] > 1:
            outputs = []
            for i in range(tensor.shape[0]):
                top, left = self.offsets[self._rand_idx()]
                outputs.append(tensor[i, ..., top:top + crop, left:left + crop])
            return torch.stack(outputs, dim=0)

        top, left = self.offsets[self._rand_idx()]
        return tensor[..., top:top + crop, left:left + crop]
