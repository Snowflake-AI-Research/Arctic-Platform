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

"""Trainer <-> Hugging Face checkpoint names: the trainer stacks each layer's experts, the checkpoint does not.

Trainer ``model.layers.<i>.mlp.experts.w1`` ``[E, I, D]`` is checkpoint ``model.layers.<i>.mlp.experts.<e>.gate_proj
.weight`` ``[I, D]`` for every expert ``e`` (``w3`` is ``up_proj``, ``w2`` is ``down_proj``). Every other tensor
has the same name in both, and vLLM's built-in ``OlmoeForCausalLM`` loads the checkpoint names the weight sync streams.
"""

from __future__ import annotations

import re

import torch
from transformers import OlmoeConfig

_TRAINER_TO_HF = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
_HF_TO_TRAINER = {hf: trainer for trainer, hf in _TRAINER_TO_HF.items()}
_TRAINER_EXPERTS = re.compile(r"^(?P<prefix>.*\.mlp\.experts)\.(?P<proj>w[123])$")
_HF_EXPERT = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


def convert_layer_to_hf(
    state: dict[str, torch.Tensor], layer_idx: int, config: OlmoeConfig
) -> dict[str, torch.Tensor]:
    out = {}
    for name, tensor in state.items():
        match = _TRAINER_EXPERTS.match(name)
        if match is None:
            out[name] = tensor
            continue
        for expert, weight in enumerate(tensor.unbind(0)):
            out[f"{match['prefix']}.{expert}.{_TRAINER_TO_HF[match['proj']]}.weight"] = weight
    return out


def convert_layer_to_prime(
    state: dict[str, torch.Tensor], layer_idx: int, config: OlmoeConfig
) -> dict[str, torch.Tensor]:
    out = {}
    experts: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
    for name, tensor in state.items():
        match = _HF_EXPERT.match(name)
        if match is None:
            out[name] = tensor
            continue
        experts.setdefault((match["prefix"], _HF_TO_TRAINER[match["proj"]]), {})[int(match["expert"])] = tensor
    for (prefix, proj), by_index in experts.items():
        if sorted(by_index) != list(range(config.num_experts)):
            raise ValueError(f"{prefix}: expert indices {sorted(by_index)} are not 0..{config.num_experts - 1}")
        out[f"{prefix}.{proj}"] = torch.stack([by_index[e] for e in range(config.num_experts)])
    return out
