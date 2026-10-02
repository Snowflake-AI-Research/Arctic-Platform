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


def _layer_prefix(layer_idx: int) -> str:
    return f"model.language_model.layers.{layer_idx}"


def _layer_indices(state_dict: dict[str, Tensor]) -> list[int]:
    return sorted({int(match.group(1)) for name in state_dict if (match := _LAYER_RE.match(name)) is not None})


def _rename(state_dict: dict[str, Tensor], old: str, new: str) -> None:
    if old in state_dict:
        state_dict[new] = state_dict.pop(old)


def _expert_indices(state_dict: dict[str, Tensor], prefix: str) -> list[int]:
    expert_prefix = f"{prefix}.mlp.experts."
    result = {
        int(part)
        for name in state_dict
        if name.startswith(expert_prefix) and (part := name[len(expert_prefix) :].split(".", 1)[0]).isdigit()
    }
    return sorted(result)


def convert_hf_layer_to_prime(
    state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    if layer_idx < 0:
        return state_dict
    prefix = _layer_prefix(layer_idx)
    fp8_scales = [name for name in state_dict if name.startswith(f"{prefix}.") and "weight_scale_inv" in name]
    if fp8_scales:
        raise NotImplementedError(
            "GLM-5.3-Flash native-FP8 training is not implemented; use zai-org/GLM-5.3-Flash-BF16. Found "
            + ", ".join(fp8_scales[:3])
        )

    for source, target in (
        ("hc_attn_fn", "attn_hc.fn"),
        ("hc_attn_base", "attn_hc.base"),
        ("hc_attn_scale", "attn_hc.scale"),
        ("hc_ffn_fn", "ffn_hc.fn"),
        ("hc_ffn_base", "ffn_hc.base"),
        ("hc_ffn_scale", "ffn_hc.scale"),
    ):
        _rename(state_dict, f"{prefix}.{source}", f"{prefix}.{target}")

    attn = f"{prefix}.self_attn"
    conv_keys = [f"{attn}.{name}_conv1d.weight" for name in ("q", "k", "v")]
    if all(name in state_dict for name in conv_keys):
        state_dict[f"{attn}.conv1d.weight"] = torch.cat(
            [state_dict.pop(name) for name in conv_keys],
            dim=0,
        )
    for source, target in (
        ("A_log", "forget_gate.A_log"),
        ("dt_bias", "forget_gate.dt_bias"),
        ("f_a_proj.weight", "forget_gate.f_a_proj.weight"),
        ("f_b_proj.weight", "forget_gate.f_b_proj.weight"),
    ):
        _rename(state_dict, f"{attn}.{source}", f"{attn}.{target}")

    expert_ids = _expert_indices(state_dict, prefix)
    if expert_ids:
        if expert_ids != list(range(len(expert_ids))):
            raise KeyError(f"Non-contiguous GLM-5.3 experts on layer {layer_idx}: {expert_ids}")
        projections = {
            "w1": "gate_proj",
            "w2": "down_proj",
            "w3": "up_proj",
        }
        for prime_name, hf_name in projections.items():
            state_dict[f"{prefix}.mlp.experts.{prime_name}"] = torch.stack(
                [state_dict.pop(f"{prefix}.mlp.experts.{expert_idx}.{hf_name}.weight") for expert_idx in expert_ids]
            )

        _rename(
            state_dict,
            f"{prefix}.mlp.gate.weight",
            f"{prefix}.mlp.router.gate.weight",
        )
        _rename(
            state_dict,
            f"{prefix}.mlp.gate.e_score_correction_bias",
            f"{prefix}.mlp.expert_bias",
        )
        for prime_name, hf_name in (
            ("w1", "gate_proj.weight"),
            ("w2", "down_proj.weight"),
            ("w3", "up_proj.weight"),
        ):
            _rename(
                state_dict,
                f"{prefix}.mlp.shared_experts.{hf_name}",
                f"{prefix}.mlp.shared_expert.{prime_name}",
            )
    return state_dict


def convert_prime_layer_to_hf(
    state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    if layer_idx < 0:
        return state_dict
    prefix = _layer_prefix(layer_idx)

    for source, target in (
        ("attn_hc.fn", "hc_attn_fn"),
        ("attn_hc.base", "hc_attn_base"),
        ("attn_hc.scale", "hc_attn_scale"),
        ("ffn_hc.fn", "hc_ffn_fn"),
        ("ffn_hc.base", "hc_ffn_base"),
        ("ffn_hc.scale", "hc_ffn_scale"),
    ):
        _rename(state_dict, f"{prefix}.{source}", f"{prefix}.{target}")

    attn = f"{prefix}.self_attn"
    conv_key = f"{attn}.conv1d.weight"
    if conv_key in state_dict:
        q_conv, k_conv, v_conv = state_dict.pop(conv_key).chunk(3, dim=0)
        state_dict[f"{attn}.q_conv1d.weight"] = q_conv
        state_dict[f"{attn}.k_conv1d.weight"] = k_conv
        state_dict[f"{attn}.v_conv1d.weight"] = v_conv
    for source, target in (
        ("forget_gate.A_log", "A_log"),
        ("forget_gate.dt_bias", "dt_bias"),
        ("forget_gate.f_a_proj.weight", "f_a_proj.weight"),
        ("forget_gate.f_b_proj.weight", "f_b_proj.weight"),
    ):
        _rename(state_dict, f"{attn}.{source}", f"{attn}.{target}")

    w1_key = f"{prefix}.mlp.experts.w1"
    w2_key = f"{prefix}.mlp.experts.w2"
    w3_key = f"{prefix}.mlp.experts.w3"
    if all(name in state_dict for name in (w1_key, w2_key, w3_key)):
        w1 = state_dict.pop(w1_key)
        w2 = state_dict.pop(w2_key)
        w3 = state_dict.pop(w3_key)
        for expert_idx in range(w1.shape[0]):
            state_dict[f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight"] = w1[expert_idx]
            state_dict[f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight"] = w2[expert_idx]
            state_dict[f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight"] = w3[expert_idx]

        _rename(
            state_dict,
            f"{prefix}.mlp.router.gate.weight",
            f"{prefix}.mlp.gate.weight",
        )
        _rename(
            state_dict,
            f"{prefix}.mlp.expert_bias",
            f"{prefix}.mlp.gate.e_score_correction_bias",
        )
        state_dict.pop(f"{prefix}.mlp.tokens_per_expert", None)
        for prime_name, hf_name in (
            ("w1", "gate_proj.weight"),
            ("w2", "down_proj.weight"),
            ("w3", "up_proj.weight"),
        ):
            _rename(
                state_dict,
                f"{prefix}.mlp.shared_expert.{prime_name}",
                f"{prefix}.mlp.shared_experts.{hf_name}",
            )
    return state_dict


def convert_hf_to_prime(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    for layer_idx in _layer_indices(state_dict):
        convert_hf_layer_to_prime(state_dict, layer_idx)
    return state_dict


def convert_prime_to_hf(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    for layer_idx in _layer_indices(state_dict):
        convert_prime_layer_to_hf(state_dict, layer_idx)
    return state_dict
