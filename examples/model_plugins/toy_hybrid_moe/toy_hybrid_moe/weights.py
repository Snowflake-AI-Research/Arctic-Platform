# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer <-> Hugging Face checkpoint names for one decoder layer.

The trainer stacks routed experts (``mlp.experts.w1`` is ``[E, I, D]``) and stores the convolution as
``[heads, head_dim, kernel]``. The checkpoint keeps one tensor per expert and a ``Conv1d`` weight
``[heads * head_dim, 1, kernel]``, with the router named ``mlp.gate``.
"""

from __future__ import annotations

import re

import torch

from .config import ToyHybridMoEConfig

_EXPERT_HF = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
_EXPERT_TRAINER = {hf: trainer for trainer, hf in _EXPERT_HF.items()}
_HF_EXPERT = re.compile(r"^(?P<prefix>.*\.mlp\.experts)\.(?P<expert>\d+)\.(?P<proj>\w+)\.weight$")


def convert_layer_to_hf(
    state: dict[str, torch.Tensor], layer_idx: int, config: ToyHybridMoEConfig
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name.endswith(".mlp.router.weight"):
            out[name.replace(".mlp.router.", ".mlp.gate.")] = tensor
        elif re.search(r"\.mlp\.experts\.w[123]$", name):
            prefix, trainer_name = name.rsplit(".", 1)
            for expert, weight in enumerate(tensor.unbind(0)):
                out[f"{prefix}.{expert}.{_EXPERT_HF[trainer_name]}.weight"] = weight
        elif name.endswith(".recurrence.conv_weight"):
            heads, head_dim, kernel = tensor.shape
            out[name.replace(".conv_weight", ".conv1d.weight")] = tensor.reshape(heads * head_dim, 1, kernel)
        elif name.endswith(".conv.conv_weight"):
            out[name.replace(".conv_weight", ".conv1d.weight")] = tensor.unsqueeze(1)
        else:
            out[name] = tensor
    return out


def convert_layer_to_prime(
    state: dict[str, torch.Tensor], layer_idx: int, config: ToyHybridMoEConfig
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    experts: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
    for name, tensor in state.items():
        match = _HF_EXPERT.match(name)
        if match:
            key = (match["prefix"], _EXPERT_TRAINER[match["proj"]])
            experts.setdefault(key, {})[int(match["expert"])] = tensor
        elif name.endswith(".mlp.gate.weight"):
            out[name.replace(".mlp.gate.", ".mlp.router.")] = tensor
        elif name.endswith(".recurrence.conv1d.weight"):
            shape = (config.recurrence_heads, config.recurrence_head_dim, config.conv_kernel)
            if tuple(tensor.shape) != (shape[0] * shape[1], 1, shape[2]):
                raise ValueError(
                    f"{name}: expected Conv1d weight {(shape[0] * shape[1], 1, shape[2])}, got {tuple(tensor.shape)}"
                )
            out[name.replace(".conv1d.weight", ".conv_weight")] = tensor.reshape(shape)
        elif name.endswith(".conv.conv1d.weight"):
            if tuple(tensor.shape) != (config.conv_channels, 1, config.conv_kernel):
                raise ValueError(
                    f"{name}: expected Conv1d weight {(config.conv_channels, 1, config.conv_kernel)}, "
                    f"got {tuple(tensor.shape)}"
                )
            out[name.replace(".conv1d.weight", ".conv_weight")] = tensor.squeeze(1)
        else:
            out[name] = tensor
    for (prefix, trainer_name), by_index in experts.items():
        if sorted(by_index) != list(range(config.num_experts)):
            raise ValueError(f"{prefix}: expert indices {sorted(by_index)} are not 0..{config.num_experts - 1}")
        out[f"{prefix}.{trainer_name}"] = torch.stack([by_index[i] for i in range(config.num_experts)])
    return out
