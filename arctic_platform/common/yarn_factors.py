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

"""Shared native/YaRN frequencies and float32 factor slots."""

import copy
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


@dataclass
class YarnFactors:
    inv_freq: torch.Tensor
    attention_scaling: torch.Tensor
    factors: tuple[float, ...]
    slots: dict[float, int]

    def slot(self, factor: float) -> int:
        value = torch.tensor(factor, dtype=torch.float32).item()
        if value not in self.slots:
            raise ValueError(f"yarn_factor {factor} is not in declared yarn_factors {self.factors}")
        return self.slots[value]


def build_yarn_factors(text_config: PretrainedConfig, declared_factors: list[float]) -> YarnFactors:
    config = copy.deepcopy(text_config)
    rope = config.rope_parameters
    for key in ("attention_factor", "mscale", "mscale_all_dim"):
        if key in rope:
            raise ValueError(f"yarn_factors does not support explicit {key}")
    if rope["rope_type"] != "yarn" or "original_max_position_embeddings" not in rope:
        raise ValueError("yarn_factors requires yarn rope_parameters with explicit original_max_position_embeddings")
    declared = torch.tensor(declared_factors, dtype=torch.float32)
    if not torch.all(torch.isfinite(declared) & (declared >= 1)):
        raise ValueError("yarn_factors must be finite and >= 1")
    default = torch.tensor(rope["factor"], dtype=torch.float32).item()
    values = declared.tolist()
    if default not in values:
        raise ValueError("yarn_factors must contain the model default factor")
    factors = tuple(dict.fromkeys([default, *values]))
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * rope.get("partial_rotary_factor", 1.0))
    native = 1.0 / (rope["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    inv_freq, scales = [], []
    for factor in factors:
        rope["factor"] = factor
        freq, scale = (native, 1.0) if factor == 1.0 else ROPE_INIT_FUNCTIONS["yarn"](config)
        inv_freq.append(freq)
        scales.append(scale)
    return YarnFactors(
        torch.stack(inv_freq),
        torch.tensor(scales, dtype=torch.float32),
        factors,
        {factor: slot for slot, factor in enumerate(factors)},
    )
