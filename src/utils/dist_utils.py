
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from typing import Tuple


def setup_distributed() -> Tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


from contextlib import contextmanager

@contextmanager
def main_process_first(rank: int):
    """
    Context manager that ensures main process (rank 0) runs first.
    Useful for downloading models/data where only one process should download.

    Usage:
        with main_process_first(rank):
            model = load_model()  # rank 0 downloads, others wait then load from cache
    """
    is_main = rank == 0
    if not is_main and dist.is_initialized():
        dist.barrier()
    yield
    if is_main and dist.is_initialized():
        dist.barrier()


def synchronize_gradients(model: torch.nn.Module):
    """
    In a distributed setting, to enable jvp, we need to call model.module instead of model directly.
    If so, we synchronize gradients across all processes.
    """
    if not isinstance(model, DistributedDataParallel):
        return

    torch.cuda.synchronize()
    for param in model.module.parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()
