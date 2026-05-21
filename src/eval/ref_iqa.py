import math
import sys

import numpy as np
import torch
from tqdm import trange
from torchmetrics.functional.image.ssim import structural_similarity_index_measure

from stage1.disc.lpips import LPIPS
from .utils import to_torch_tensor


@torch.no_grad()
def calculate_psnr(arr1, arr2, bs, device="cuda", disable_bar: bool = True) -> float:
    """
    Computes PSNR between two single images or batches of images.
    PSNR is averaged over the batch if given as (B, C, H, W).
    """
    B = arr1.shape[0]
    n_batches = math.ceil(B / bs)
    psnr = torch.zeros(B, device=device)

    for i in trange(n_batches, desc="Calculating PSNR", file=sys.stdout, disable=disable_bar):
        start_idx = i * bs
        end_idx = min((i + 1) * bs, B)
        # PSNR expects input in [0, 1], (B, C, H, W)
        batch_arr1 = to_torch_tensor(arr1[start_idx:end_idx]).to(device)
        batch_arr2 = to_torch_tensor(arr2[start_idx:end_idx]).to(device)
        batch_mse = torch.mean((batch_arr1 - batch_arr2) ** 2, dim=[1,2,3])  # shape [bs]
        batch_mse = torch.clamp(batch_mse, min=1e-10)
        batch_psnr = 20.0 * torch.log10(1 / torch.sqrt(batch_mse))  # shape [bs]
        psnr[start_idx:end_idx] = batch_psnr
    return psnr.mean().item()


@torch.no_grad()
def calculate_lpips(arr1, arr2, bs, device="cuda", disable_bar: bool = True) -> float:
    """
    Computes LPIPS between two single images or batches of images.
    LPIPS is averaged over the batch if given as (B, C, H, W).
    """
    B = arr1.shape[0]
    n_batches = math.ceil(B / bs)
    loss_fn = LPIPS().eval().to(device)
    lpips = torch.zeros(B, device=device)

    for i in trange(n_batches, desc="Calculating LPIPS", file=sys.stdout, disable=disable_bar):
        start_idx = i * bs
        end_idx = min((i + 1) * bs, B)
        # LPIPS expects input in [-1, 1], (B, C, H, W)
        batch_arr1 = (to_torch_tensor(arr1[start_idx:end_idx]).to(device) - 0.5) * 2.
        batch_arr2 = (to_torch_tensor(arr2[start_idx:end_idx]).to(device) - 0.5) * 2.
        batch_lpips = loss_fn(batch_arr1, batch_arr2).squeeze()  # shape [bs]
        lpips[start_idx:end_idx] = batch_lpips
    return lpips.mean().item()


######################################################
# 4. SSIM Calculation
######################################################
@torch.no_grad()
def calculate_ssim(arr1, arr2, bs, device="cuda", disable_bar: bool = True) -> float:
    """
    Computes SSIM between two single images or batches of images.
    SSIM is averaged over the batch if given as (B, C, H, W).
    """
    B = arr1.shape[0]
    n_batches = math.ceil(B / bs)
    ssim_val = torch.zeros(B, device=device)

    for i in trange(n_batches, desc="Calculating SSIM", file=sys.stdout, disable=disable_bar):
        start_idx = i * bs
        end_idx = min((i + 1) * bs, B)
        # SSIM expects input in [0, 1], (B, C, H, W)
        batch_arr1 = to_torch_tensor(arr1[start_idx:end_idx]).to(device)
        batch_arr2 = to_torch_tensor(arr2[start_idx:end_idx]).to(device)
        # SSIM expects input in [0, 1], (B, C, H, W)
        batch_ssim = structural_similarity_index_measure(
            target=batch_arr1,
            preds=batch_arr2,
            data_range=1.0,
            reduction="none"
        )
        ssim_val[start_idx:end_idx] = batch_ssim
    return ssim_val.mean().item()
