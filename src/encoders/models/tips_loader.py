import importlib
import json
import os
import sys


# model_config -> (factory_fn_name, ffn_layer)
MODEL_CONFIGS = {
    's14':  ('vit_small',  'mlp'),
    'b14':  ('vit_base',   'mlp'),
    'l14':  ('vit_large',  'mlp'),
    'so14': ('vit_so400m', 'mlp'),
    'g14':  ('vit_giant2', 'swiglu'),
}

HF_MODEL_MAP = {
    'b14': 'google/tipsv2-b14',
    'l14': 'google/tipsv2-l14',
    'so14': 'google/tipsv2-so400m14',
    'g14': 'google/tipsv2-g14',
}


_ie_module = None

def _get_image_encoder_module(source_dir):
    """Import image_encoder from directory via sys.path (torch.compile compatible)."""
    global _ie_module
    if _ie_module is not None:
        return _ie_module
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    _ie_module = importlib.import_module('image_encoder')
    return _ie_module


def _create_model(ie_module, model_config, img_size):
    """Create TIPS vision encoder from factory function."""
    factory_name, ffn_layer = MODEL_CONFIGS[model_config]
    factory_fn = getattr(ie_module, factory_name)
    return factory_fn(
        img_size=img_size,
        patch_size=14,
        ffn_layer=ffn_layer,
        block_chunks=0,
        init_values=1.0,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )


def load_tipsv2(model_config):
    """Load TIPSv2 vision encoder from HuggingFace.

    Downloads image_encoder.py + safetensors directly (no trust_remote_code),
    so torch.compile works.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    hf_name = HF_MODEL_MAP[model_config]

    ie_path = hf_hub_download(hf_name, "image_encoder.py")
    config_path = hf_hub_download(hf_name, "config.json")
    weights_path = hf_hub_download(hf_name, "model.safetensors")

    ie = _get_image_encoder_module(os.path.dirname(ie_path))

    with open(config_path) as f:
        config = json.load(f)

    model = _create_model(ie, model_config, img_size=config['img_size'])

    # Load only vision encoder weights from safetensors
    all_weights = load_file(weights_path)
    vision_weights = {k.replace('vision_encoder.', ''): v
                      for k, v in all_weights.items()
                      if k.startswith('vision_encoder.')}
    model.load_state_dict(vision_weights)
    return model
