from dataclasses import dataclass

from arctic_platform.model.implementations.glm52.models.fp8 import make_linear

import torch
from torch import nn
from transformers.activations import ACT2FN


@dataclass
class MLPConfig:
    hidden_size: int
    intermediate_size: int
    gate_act: str
    bias: bool
    fp8_block_size: int | None = None


class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = make_linear(
            self.hidden_size, self.intermediate_size, bias=False, fp8_block_size=config.fp8_block_size
        )
        self.up_proj = make_linear(
            self.hidden_size, self.intermediate_size, bias=False, fp8_block_size=config.fp8_block_size
        )
        self.down_proj = make_linear(
            self.intermediate_size, self.hidden_size, bias=False, fp8_block_size=config.fp8_block_size
        )
        self.gate_act_fn = ACT2FN[config.gate_act]

    def forward(self, x, routed_experts: torch.Tensor | None = None):
        down_proj = self.down_proj(self.gate_act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
