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

from __future__ import annotations

import re

import torch
from torch import Tensor

_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
_PLE_SHARD_RE = re.compile(r"^(.*\.ple\.ple_embedding\.ngram_embedding)\.shard_(\d+)\.weight$")


def _layer_prefix(layer_idx: int) -> str:
    return f"model.language_model.layers.{layer_idx}"


def _layer_indices(state_dict: dict[str, Tensor]) -> list[int]:
    return sorted({int(match.group(1)) for name in state_dict if (match := _LAYER_RE.match(name)) is not None})


def _rename(state_dict: dict[str, Tensor], old: str, new: str) -> None:
    if old in state_dict:
        state_dict[new] = state_dict.pop(old)


def _concatenate_ple_shards(state_dict: dict[str, Tensor]) -> None:
    groups: dict[str, list[tuple[int, str]]] = {}
    for name in state_dict:
        match = _PLE_SHARD_RE.match(name)
        if match is not None:
            groups.setdefault(match.group(1), []).append((int(match.group(2)), name))

    for base, shards in groups.items():
        shards.sort()
        indices = [index for index, _ in shards]
        if indices != list(range(len(indices))):
            raise KeyError(f"Non-contiguous Qwen3.8 PLE shards for {base}: {indices}")
        state_dict[f"{base}.weight"] = torch.cat(
            [state_dict.pop(name) for _, name in shards],
            dim=0,
        )


def convert_hf_layer_to_prime(
    state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    _concatenate_ple_shards(state_dict)
    if layer_idx < 0:
        return state_dict

    prefix = _layer_prefix(layer_idx)
    fp8_keys = [name for name in state_dict if name.startswith(f"{prefix}.") and "weight_scale_inv" in name]
    if fp8_keys:
        raise NotImplementedError(
            "Qwen3.8-Flash-Next native-FP8 training is not implemented; use Qwen/Qwen3.8-Flash-Next. Found "
            + ", ".join(fp8_keys[:3])
        )

    gate_up_key = f"{prefix}.mlp.experts.gate_up_proj"
    down_key = f"{prefix}.mlp.experts.down_proj"
    if gate_up_key in state_dict:
        gate, up = state_dict.pop(gate_up_key).chunk(2, dim=1)
        state_dict[f"{prefix}.mlp.experts.w1"] = gate
        state_dict[f"{prefix}.mlp.experts.w3"] = up
    _rename(state_dict, down_key, f"{prefix}.mlp.experts.w2")
    _rename(
        state_dict,
        f"{prefix}.mlp.gate.weight",
        f"{prefix}.mlp.router.gate.weight",
    )

    for hf_name, prime_name in (
        ("gate_proj.weight", "w1.weight"),
        ("down_proj.weight", "w2.weight"),
        ("up_proj.weight", "w3.weight"),
    ):
        _rename(
            state_dict,
            f"{prefix}.mlp.shared_expert.{hf_name}",
            f"{prefix}.mlp.qwen_shared_expert.{prime_name}",
        )
    return state_dict


def convert_prime_layer_to_hf(
    state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    if layer_idx < 0:
        return state_dict

    prefix = _layer_prefix(layer_idx)
    w1_key = f"{prefix}.mlp.experts.w1"
    w2_key = f"{prefix}.mlp.experts.w2"
    w3_key = f"{prefix}.mlp.experts.w3"
    if w1_key in state_dict and w3_key in state_dict:
        state_dict[f"{prefix}.mlp.experts.gate_up_proj"] = torch.cat(
            [state_dict.pop(w1_key), state_dict.pop(w3_key)],
            dim=1,
        )
    _rename(state_dict, w2_key, f"{prefix}.mlp.experts.down_proj")
    _rename(
        state_dict,
        f"{prefix}.mlp.router.gate.weight",
        f"{prefix}.mlp.gate.weight",
    )
    state_dict.pop(f"{prefix}.mlp.tokens_per_expert", None)

    for prime_name, hf_name in (
        ("w1.weight", "gate_proj.weight"),
        ("w2.weight", "down_proj.weight"),
        ("w3.weight", "up_proj.weight"),
    ):
        _rename(
            state_dict,
            f"{prefix}.mlp.qwen_shared_expert.{prime_name}",
            f"{prefix}.mlp.shared_expert.{hf_name}",
        )
    return state_dict


def convert_hf_to_prime(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    _concatenate_ple_shards(state_dict)
    for layer_idx in _layer_indices(state_dict):
        convert_hf_layer_to_prime(state_dict, layer_idx)
    return state_dict


def convert_prime_to_hf(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    for layer_idx in _layer_indices(state_dict):
        convert_prime_layer_to_hf(state_dict, layer_idx)
    return state_dict
