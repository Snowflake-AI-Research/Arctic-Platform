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

"""Turn a quantized DeepSeek V4.1 snapshot into bf16.

The training checkpoint is ``PrimeIntellect/DeepSeek-V4.1-Flash-bf16``: dense linears and
routed experts are already exact bf16 and have no ``.scale`` sibling, so they pass through.
An upstream FP8 or packed-MXFP4 pair is still dequantized. Engram rows stay float8 with their
UE8M0 scale either way; only fetched rows are dequantized at lookup.
"""

from __future__ import annotations

from arctic_platform.model.implementations.deepseek_v4.conversion_ops import StateDict
from arctic_platform.model.implementations.deepseek_v4.dequantize import dequantize_weight

_ENGRAM_TABLE = ".engram.embed.weight"


def quantized_pairs(keys) -> dict[str, str]:
    """``*.weight`` -> ``*.scale`` for every pair that dequantizes. Engram tables are not listed."""
    present = set(keys)
    pairs = {}
    for key in present:
        if not key.endswith(".weight") or key.endswith(_ENGRAM_TABLE):
            continue
        scale_key = key.removesuffix(".weight") + ".scale"
        if scale_key in present:
            pairs[key] = scale_key
    return pairs


def dequantize_state_dict_(state_dict: StateDict) -> None:
    """Dequantize on-disk ``*.weight`` / ``*.scale`` pairs in place, except Engram tables."""
    for key, scale_key in quantized_pairs(state_dict).items():
        state_dict[key] = dequantize_weight(state_dict[key], state_dict.pop(scale_key))
