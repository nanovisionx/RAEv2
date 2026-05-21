"""Model instantiation utilities."""

import importlib

import torch

from configs import ModelConfig


def get_obj_from_str(string: str, reload: bool = False):
    """Import and return a class/function from a dotted string path."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: ModelConfig) -> object:
    """Instantiate a model from ModelConfig.

    Args:
        config: ModelConfig with target, params, and optional ckpt path

    Returns:
        Instantiated model, optionally with loaded checkpoint
    """
    if not config.target:
        raise KeyError("Expected 'target' to instantiate.")

    model = get_obj_from_str(config.target)(**config.params)

    if getattr(config, "ckpt", None) is not None:
        state_dict = torch.load(config.ckpt, map_location="cpu")
        if "ema" in state_dict:
            state_dict = state_dict["ema"]
        elif "model" in state_dict:
            raise NotImplementedError("Loading from 'model' key not implemented yet.")
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded {config.target} from {config.ckpt}")

    return model
