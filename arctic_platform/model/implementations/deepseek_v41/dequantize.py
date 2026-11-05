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

"""Turn a quantized DeepSeek V4.1 snapshot into the trained dtypes.

Dense FP8 linears are dequantized to bf16. Routed experts are dequantized too, unless the
config keeps them packed (see :func:`packs_experts`); then the int8 MXFP4 codes pass through and
only their UE8M0 scales are decoded to float32. Engram rows stay float8 with their UE8M0 scale
either way; only fetched rows are dequantized at lookup.
"""

from __future__ import annotations

import torch

from arctic_platform.model.implementations.deepseek_v4.conversion_ops import StateDict
from arctic_platform.model.implementations.deepseek_v4.dequantize import dequantize_weight

_ENGRAM_TABLE = ".engram.embed.weight"
_ROUTED_EXPERTS = ".ffn.experts."


def packs_experts(config) -> bool:
    """Whether the config still carries the official checkpoint's MXFP4 expert quantization."""
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return False
    expert_dtype = qc.get("expert_dtype") if isinstance(qc, dict) else getattr(qc, "expert_dtype", None)
    return expert_dtype == "fp4"


def quantized_pairs(keys, *, keep_experts: bool = False) -> dict[str, str]:
    """``*.weight`` -> ``*.scale`` for every pair that dequantizes. Engram tables are not listed."""
    present = set(keys)
    pairs = {}
    for key in present:
        if not key.endswith(".weight") or key.endswith(_ENGRAM_TABLE):
            continue
        if keep_experts and _ROUTED_EXPERTS in key:
            continue
        scale_key = key.removesuffix(".weight") + ".scale"
        if scale_key in present:
            pairs[key] = scale_key
    return pairs


def is_packed_expert_scale(key: str) -> bool:
    return _ROUTED_EXPERTS in key and key.endswith(".scale")


def dequantize_state_dict_(state_dict: StateDict, *, keep_experts: bool = False) -> None:
    """Dequantize on-disk ``*.weight`` / ``*.scale`` pairs in place, except Engram tables.

    With ``keep_experts`` the routed experts stay packed and their scales become float32.
    """
    for key, scale_key in quantized_pairs(state_dict, keep_experts=keep_experts).items():
        state_dict[key] = dequantize_weight(state_dict[key], state_dict.pop(scale_key))
    if keep_experts:
        for key, value in state_dict.items():
            if is_packed_expert_scale(key) and value.dtype == torch.float8_e8m0fnu:
                state_dict[key] = value.float()
