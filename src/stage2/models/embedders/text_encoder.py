from typing import List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-0.6B", max_length=256):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModelForCausalLM.from_pretrained(model_name).eval()

        self.feature_dim = self.text_model.config.hidden_size

    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        device = next(self.text_model.parameters()).device

        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = self.text_model(
            **tokens,
            use_cache=False,
            output_hidden_states=True,
            return_dict=False,
        )
        return {
            "tokens": outputs[1][-1],
            "attention_mask": tokens["attention_mask"],
        }
