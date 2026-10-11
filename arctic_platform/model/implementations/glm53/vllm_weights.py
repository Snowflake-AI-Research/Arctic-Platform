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

import torch

from arctic_platform.model.implementations.moe.vllm_weights import pack_routed_experts_to_vllm
from arctic_platform.model.implementations.moe.vllm_weights import pack_shared_expert_to_vllm


def _rename(layer_sd: dict, old: str, new: str) -> None:
    if old in layer_sd:
        layer_sd[new] = layer_sd.pop(old)


def _fuse(layer_sd: dict, sources: list[str], target: str) -> None:
    present = [name in layer_sd for name in sources]
    if any(present) and not all(present):
        missing = [name for name, found in zip(sources, present) if not found]
        raise RuntimeError(f"cannot build {target}; missing {missing}")
    if all(present):
        layer_sd[target] = torch.cat([layer_sd.pop(name) for name in sources], dim=0)


def _pack_hyper_connections(layer_sd: dict, prefix: str) -> None:
    for source, target in (
        ("attn_hc.fn", "hc_attn_fn"),
        ("attn_hc.base", "hc_attn_base"),
        ("attn_hc.scale", "hc_attn_scale"),
        ("ffn_hc.fn", "hc_ffn_fn"),
        ("ffn_hc.base", "hc_ffn_base"),
        ("ffn_hc.scale", "hc_ffn_scale"),
    ):
        _rename(layer_sd, f"{prefix}.{source}", f"{prefix}.{target}")


def _pack_kda(layer_sd: dict, prefix: str) -> bool:
    attn = f"{prefix}.self_attn"
    conv_key = f"{attn}.conv1d.weight"
    if conv_key not in layer_sd:
        return False
    _fuse(
        layer_sd,
        [
            f"{attn}.q_proj.weight",
            f"{attn}.k_proj.weight",
            f"{attn}.v_proj.weight",
            f"{attn}.b_proj.weight",
            f"{attn}.forget_gate.f_a_proj.weight",
            f"{attn}.g_a_proj.weight",
        ],
        f"{attn}.in_proj_qkvbfg_a.weight",
    )
    q_conv, k_conv, v_conv = layer_sd.pop(conv_key).chunk(3, dim=0)
    layer_sd[f"{attn}.q_conv1d.weight"] = q_conv
    layer_sd[f"{attn}.k_conv1d.weight"] = k_conv
    layer_sd[f"{attn}.v_conv1d.weight"] = v_conv
    for source, target in (
        ("forget_gate.A_log", "A_log"),
        ("forget_gate.dt_bias", "dt_bias"),
        ("forget_gate.f_b_proj.weight", "f_b_proj.weight"),
    ):
        _rename(layer_sd, f"{attn}.{source}", f"{attn}.{target}")
    return True


def _pack_sparse_mla(layer_sd: dict, prefix: str) -> None:
    attn = f"{prefix}.self_attn"
    _fuse(
        layer_sd,
        [
            f"{attn}.q_a_proj.weight",
            f"{attn}.kv_a_proj_with_mqa.weight",
        ],
        f"{attn}.fused_qkv_a_proj.weight",
    )
    _fuse(
        layer_sd,
        [
            f"{attn}.indexer.wk.weight",
            f"{attn}.indexer.weights_proj.weight",
        ],
        f"{attn}.indexer.wk_weights_proj.weight",
    )


def _pack_mlp(layer_sd: dict, prefix: str) -> None:
    expert_key = f"{prefix}.mlp.experts.w1"
    if expert_key in layer_sd:
        expert_bias = layer_sd.get(f"{prefix}.mlp.expert_bias")
        pack_routed_experts_to_vllm(layer_sd, prefix)
        pack_shared_expert_to_vllm(layer_sd, prefix)
        if expert_bias is not None:
            layer_sd[f"{prefix}.mlp.gate.e_score_correction_bias"] = expert_bias
        _fuse(
            layer_sd,
            [
                f"{prefix}.mlp.shared_experts.gate_proj.weight",
                f"{prefix}.mlp.shared_experts.up_proj.weight",
            ],
            f"{prefix}.mlp.shared_experts.gate_up_proj.weight",
        )
        return
    _fuse(
        layer_sd,
        [
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight",
        ],
        f"{prefix}.mlp.gate_up_proj.weight",
    )


def convert_glm5_next_layer_to_vllm(
    layer_sd: dict,
    layer_idx: int,
    *,
    layer_prefix: str | None = None,
) -> dict:
    if layer_idx < 0:
        return layer_sd
    prefix = layer_prefix if layer_prefix is not None else f"model.language_model.layers.{layer_idx}"
    _pack_hyper_connections(layer_sd, prefix)
    if not _pack_kda(layer_sd, prefix):
        _pack_sparse_mla(layer_sd, prefix)
    _pack_mlp(layer_sd, prefix)
    return layer_sd
