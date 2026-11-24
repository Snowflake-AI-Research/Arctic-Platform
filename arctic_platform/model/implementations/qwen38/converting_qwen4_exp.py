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
from pathlib import Path

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


def _indexed_expert_keys(state_dict: dict[str, Tensor], prefix: str) -> dict[int, dict[str, str]]:
    pattern = re.compile(rf"^{re.escape(prefix)}\.mlp\.experts\.(\d+)\.(gate_proj|down_proj|up_proj)\.weight$")
    experts: dict[int, dict[str, str]] = {}
    for name in state_dict:
        match = pattern.match(name)
        if match is not None:
            projections = experts.setdefault(int(match.group(1)), {})
            projection = match.group(2)
            if projection in projections:
                raise ValueError(f"Duplicate Qwen3.8 expert projection for {prefix}: {name}")
            projections[projection] = name
    if not experts:
        return experts
    if any(
        f"{prefix}.mlp.experts.{projection}" in state_dict
        for projection in ("gate_up_proj", "down_proj", "w1", "w2", "w3")
    ):
        raise ValueError(f"Mixed Qwen3.8 expert layouts for {prefix}")
    indices = sorted(experts)
    if indices != list(range(len(indices))):
        raise KeyError(f"Non-contiguous Qwen3.8 experts for {prefix}: {indices}")
    for index, projections in experts.items():
        missing = {"gate_proj", "down_proj", "up_proj"} - projections.keys()
        if missing:
            raise KeyError(f"Incomplete Qwen3.8 expert {prefix}.{index}: missing {sorted(missing)}")
    return experts


def convert_hf_layer_to_prime(
    state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    if layer_idx < 0:
        _concatenate_ple_shards(state_dict)
        return state_dict

    prefix = _layer_prefix(layer_idx)
    experts = _indexed_expert_keys(state_dict, prefix)
    fp8_keys = [name for name in state_dict if name.startswith(f"{prefix}.") and "weight_scale_inv" in name]
    if fp8_keys:
        raise NotImplementedError(
            "Qwen3.8-Flash-Next native-FP8 training is not implemented; use Qwen/Qwen3.8-Flash-Next. Found "
            + ", ".join(fp8_keys[:3])
        )

    if experts:
        stacked = {
            f"{prefix}.mlp.experts.{prime_name}": torch.stack(
                [state_dict[experts[index][hf_name]] for index in range(len(experts))]
            )
            for hf_name, prime_name in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3"))
        }
        for projections in experts.values():
            for name in projections.values():
                state_dict.pop(name)
        state_dict.update(stacked)
    _concatenate_ple_shards(state_dict)

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
    layers = _layer_indices(state_dict)
    for layer_idx in layers:
        _indexed_expert_keys(state_dict, _layer_prefix(layer_idx))
    _concatenate_ple_shards(state_dict)
    for layer_idx in layers:
        convert_hf_layer_to_prime(state_dict, layer_idx)
    return state_dict


def _checkpoint_groups(names: tuple[str, ...]) -> list[list[str]]:
    keys = dict.fromkeys(names)
    for layer_idx in _layer_indices(keys):
        _indexed_expert_keys(keys, _layer_prefix(layer_idx))
    ple_groups: dict[str, list[int]] = {}
    groups: dict[tuple[str, str], list[str]] = {}
    for name in names:
        layer = _LAYER_RE.match(name)
        ple = _PLE_SHARD_RE.match(name)
        if ple is not None:
            ple_groups.setdefault(ple.group(1), []).append(int(ple.group(2)))
        if layer is not None:
            group = ("layer", layer.group(1))
        elif ple is not None:
            group = ("ple", ple.group(1))
        else:
            group = ("tensor", name)
        groups.setdefault(group, []).append(name)
    for base, indices in ple_groups.items():
        indices.sort()
        if indices != list(range(len(indices))):
            raise KeyError(f"Non-contiguous Qwen3.8 PLE shards for {base}: {indices}")
    return list(groups.values())


def convert_hf_checkpoint_to_prime(source_dir: Path, save_dir: Path) -> None:
    from ..moe.weights import convert_safetensor_checkpoint

    convert_safetensor_checkpoint(source_dir, save_dir, convert_hf_to_prime, _checkpoint_groups)


def convert_prime_to_hf(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    for layer_idx in _layer_indices(state_dict):
        convert_prime_layer_to_hf(state_dict, layer_idx)
    return state_dict
