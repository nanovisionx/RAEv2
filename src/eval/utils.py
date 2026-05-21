"""Lightweight eval data utilities. LPIPS lives in disc/lpips.py."""

import numpy as np
import torch
from torch.utils.data import Dataset


class ImgArrDataset(Dataset):
    """Wrapper for torch-fidelity FID calculation — expects [B, H, W, C] uint8 arrays."""

    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        return torch.from_numpy(self.arr[idx]).permute(2, 0, 1)


def to_torch_tensor(np_array: np.ndarray) -> torch.Tensor:
    """Convert (B, H, W, C) NumPy array to (B, C, H, W) float32 tensor in [0, 1]."""
    tensor = torch.from_numpy(np_array).permute(0, 3, 1, 2).float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor
