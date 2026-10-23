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

"""``PrimeIntellect/DeepSeek-V4.1-Flash-bf16`` names to the PrimeRL module tree.

Bare keys, no ``model.`` prefix. Vision, the aligner, image embeddings, and MTP stay out of
this text load. Every other key has to map; leftovers are rejected.
"""

from __future__ import annotations

import logging

from arctic_platform.model.implementations.deepseek_v4.conversion_ops import ConvOp
from arctic_platform.model.implementations.deepseek_v4.conversion_ops import Drop
from arctic_platform.model.implementations.deepseek_v4.conversion_ops import PrefixRename
from arctic_platform.model.implementations.deepseek_v4.conversion_ops import Rename
from arctic_platform.model.implementations.deepseek_v4.conversion_ops import StateDict
from arctic_platform.model.implementations.deepseek_v4.conversion_ops import routed_experts_op

logger = logging.getLogger(__name__)

DEFERRED_CHECKPOINT_NAMESPACES = ("vision.", "aligner.", "image_", "mtp.")

_TOP_LEVEL = {"embed.weight", "norm.weight", "lm_head.weight"}
_LAYER_MARKERS = (
    ".self_attn.",
    ".mlp.",
    ".engram.",
    ".attn_hc.",
    ".ffn_hc.",
    ".input_layernorm.",
    ".post_attention_layernorm.",
)


def log_deferred_checkpoint_keys(state_dict: StateDict) -> tuple[str, ...]:
    """Log the namespaces this text load leaves behind, and which of them this checkpoint has."""
    present = tuple(
        prefix for prefix in DEFERRED_CHECKPOINT_NAMESPACES if any(key.startswith(prefix) for key in state_dict)
    )
    found = ", ".join(present) if present else "none"
    logger.info(
        "DeepSeek V4.1 text conversion defers namespaces %s; present in this checkpoint: %s",
        ", ".join(DEFERRED_CHECKPOINT_NAMESPACES),
        found,
    )
    return present


def _unexplained(state_dict: StateDict) -> list[str]:
    bad = []
    for key in state_dict:
        if key in _TOP_LEVEL:
            continue
        if key.startswith("layers.") and any(marker in key for marker in _LAYER_MARKERS):
            continue
        bad.append(key)
    return bad


class RejectUnexplained(ConvOp):
    """Fail when a key was neither renamed onto the text module tree nor listed as deferred."""

    def hf_to_prime(self, sd: StateDict) -> None:
        bad = _unexplained(sd)
        if bad:
            raise ValueError("DeepSeek V4.1 checkpoint has unexplained keys: " + ", ".join(sorted(bad)))

    def prime_to_hf(self, sd: StateDict) -> None:
        self.hf_to_prime(sd)


def _layer_ops(config, layer_idx: int) -> list[ConvOp]:
    p = f"layers.{layer_idx}"
    attn, moe = f"{p}.attn", f"{p}.ffn"
    prime_attn, prime_moe = f"{p}.self_attn", f"{p}.mlp"
    ops: list[ConvOp] = [
        Rename(f"{p}.attn_norm.weight", f"{p}.input_layernorm.weight"),
        Rename(f"{p}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight"),
        Rename(f"{p}.hc_attn_fn", f"{p}.attn_hc.fn"),
        Rename(f"{p}.hc_attn_base", f"{p}.attn_hc.base"),
        Rename(f"{p}.hc_attn_scale", f"{p}.attn_hc.scale"),
        Rename(f"{p}.hc_ffn_fn", f"{p}.ffn_hc.fn"),
        Rename(f"{p}.hc_ffn_base", f"{p}.ffn_hc.base"),
        Rename(f"{p}.hc_ffn_scale", f"{p}.ffn_hc.scale"),
        Rename(f"{attn}.wq_a.weight", f"{prime_attn}.q_a_proj.weight"),
        Rename(f"{attn}.q_norm.weight", f"{prime_attn}.q_a_norm.weight"),
        Rename(f"{attn}.wq_b.weight", f"{prime_attn}.q_b_proj.weight"),
        Rename(f"{attn}.wkv.weight", f"{prime_attn}.kv_proj.weight"),
        Rename(f"{attn}.kv_norm.weight", f"{prime_attn}.kv_norm.weight"),
        Rename(f"{attn}.wo_a.weight", f"{prime_attn}.o_a_proj.weight"),
        Rename(f"{attn}.wo_b.weight", f"{prime_attn}.o_b_proj.weight"),
        Rename(f"{attn}.attn_sink", f"{prime_attn}.sinks"),
        Rename(f"{moe}.gate.weight", f"{prime_moe}.router.gate.weight"),
        Rename(f"{moe}.gate.bias", f"{prime_moe}.expert_bias"),
        Rename(f"{moe}.gate.bias_vl", f"{prime_moe}.bias_vl"),
        Rename(f"{moe}.shared_experts.w1.weight", f"{prime_moe}.shared_expert.w1"),
        Rename(f"{moe}.shared_experts.w2.weight", f"{prime_moe}.shared_expert.w2"),
        Rename(f"{moe}.shared_experts.w3.weight", f"{prime_moe}.shared_expert.w3"),
        routed_experts_op(
            p,
            hf_experts="ffn.experts",
            prime_experts="mlp.experts",
            proj_order=(("w1", "w1"), ("w2", "w2"), ("w3", "w3")),
        ),
    ]
    if layer_idx in config.kv_source_layer_ids:
        comp, prime_comp = f"{attn}.compressor", f"{prime_attn}.compressor"
        ops += [
            Rename(f"{comp}.wkv.weight", f"{prime_comp}.wkv.weight"),
            Rename(f"{comp}.norm.weight", f"{prime_comp}.norm.weight"),
        ]
        if config.compress_ratios[layer_idx] > 1:
            ops.append(Rename(f"{comp}.wgate.weight", f"{prime_comp}.wgate.weight"))
    if layer_idx in config.index_source_layer_ids:
        idx, prime_idx = f"{attn}.indexer", f"{prime_attn}.indexer"
        ops += [
            Rename(f"{idx}.wq_b.weight", f"{prime_idx}.wq_b.weight"),
            Rename(f"{idx}.weights_proj.weight", f"{prime_idx}.weights_proj.weight"),
        ]
        if layer_idx in config.kv_source_layer_ids:
            ops += [
                Rename(f"{idx}.wk.weight", f"{prime_idx}.wk.weight"),
                Rename(f"{idx}.k_norm.weight", f"{prime_idx}.k_norm.weight"),
            ]
    if layer_idx in config.engram_layer_ids:
        eng = f"{p}.engram"
        ops += [
            Rename(f"{eng}.embed.weight", f"{eng}.embed.weight"),
            Rename(f"{eng}.embed.scale", f"{eng}.embed.scale"),
            Rename(f"{eng}.wkv.weight", f"{eng}.wkv.weight"),
            Rename(f"{eng}.q_weight", f"{eng}.q_weight"),
            Rename(f"{eng}.k_weight", f"{eng}.k_weight"),
        ]
    return ops


def conversion_chain(config) -> list[ConvOp]:
    ops: list[ConvOp] = [Drop(prefix, is_prefix=True) for prefix in DEFERRED_CHECKPOINT_NAMESPACES]
    ops.append(Rename("head.weight", "lm_head.weight"))
    for layer_idx in range(config.num_hidden_layers):
        ops.extend(_layer_ops(config, layer_idx))
    ops.append(RejectUnexplained())
    ops += [
        Rename("embed.weight", "model.embed_tokens.weight"),
        Rename("norm.weight", "model.norm.weight"),
        PrefixRename("layers.", "model.layers."),
    ]
    return ops
