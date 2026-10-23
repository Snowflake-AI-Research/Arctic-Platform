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

# Adapted from ArcticTraining's trainer/flops_counter.py, which adapts
# https://github.com/volcengine/verl/blob/main/verl/utils/flops_counter.py
# DeepSeek-V4 and GLM-5.3-Flash are counted with the checkpointed per-token
# convention used for the H200 step reports: 8 FLOPs per activated parameter,
# plus the sparse attention and indexer terms.

from __future__ import annotations

from typing import Any

# Model types the upstream counter names. Types without a specific estimator
# fall back to the dense decoder formula.
VALID_CONFIG_TYPE = frozenset(
    {
        "llama",
        "qwen2",
        "qwen2_moe",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3",
        "qwen3_moe",
        "qwen3_next",
        "qwen3_vl",
        "qwen3_vl_moe",
        "deepseek_v3",
        "deepseek_v4",
        "glm5_next",
        "glm5_next_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "minicpmv",
        "minicpmo",
        "mistral",
        "gemma3_text",
        "seed_oss",
        "apertus",
        "glm4v",
    }
)


def estimate_decoder_transformer_tflos(
    hf_model_config,
    model_size,
    batch_of_seqlens,
    enable_gradient_checkpointing,
    forward_only=False,
):
    """Total teraFLOPs and token count for a batch of packed or unpacked samples.

    ``batch_of_seqlens`` is one list of sequence lengths per batch row. A single
    unpacked sample is ``[[seqlen]]``. Unknown ``model_type`` values use the
    dense decoder formula.
    """
    flops_counter = FlopsCounter(hf_model_config, model_size, enable_gradient_checkpointing)
    seqlen = 0
    tflos = 0.0
    for seqlens in batch_of_seqlens:
        tflos += flops_counter.estimate_tflos(batch_seqlens=seqlens, delta_time=1, forward_only=forward_only)
        seqlen += sum(seqlens)
    return tflos, seqlen


def tflops_per_token(
    config, seq_len: int, model_size: int = 0, *, enable_gradient_checkpointing: bool = True
) -> float:
    """TeraFLOPs per token at ``seq_len``. Multiply by tokens/second to get TFLOP/s."""
    counter = FlopsCounter(config, model_size, enable_gradient_checkpointing)
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    return counter.estimate_tflos([seq_len]) / seq_len


class FlopsCounter:
    """Training-step FLOPs from an HF config.

    Dense GEMMs use 6 FLOPs per parameter-token, or 8 when activation
    checkpointing recomputes the forward. Attention follows the flash or
    eager backward and is halved for causal masks. ``forward_only`` counts
    the forward matmuls only.
    """

    def __init__(
        self,
        config: Any,
        model_size,
        enable_gradient_checkpointing: bool = False,
        causal: bool = True,
        attn_implementation: str | None = None,
    ):
        self.estimate_func = {
            "qwen2": self._estimate_qwen2_flops,
            "llama": self._estimate_qwen2_flops,
            "qwen2_moe": self._estimate_qwen2_moe_flops,
            "qwen2_vl": self._estimate_qwen2_flops,
            "qwen2_5_vl": self._estimate_qwen2_flops,
            "qwen3": self._estimate_qwen2_flops,
            "qwen3_moe": self._estimate_qwen2_moe_flops,
            "qwen3_next": self._estimate_qwen2_moe_flops,
            "qwen3_5_moe": self._estimate_qwen2_moe_flops,
            "qwen3_5_moe_text": self._estimate_qwen2_moe_flops,
            "deepseek_v3": self._estimate_deepseek_v3_flops,
            "deepseek_v4": self._estimate_deepseek_v4_flops,
            "glm5_next": self._estimate_glm5_next_flops,
            "glm5_next_text": self._estimate_glm5_next_flops,
            "minicpmv": self._estimate_qwen2_flops,
            "minicpmo": self._estimate_qwen2_flops,
        }
        self.config = getattr(config, "text_config", config)
        self.model_size = model_size
        # GEMM multipliers on N*tokens: forward 2, backward 4. Checkpointing
        # recomputes one forward, which adds 2.
        self.dense_flops_multiplier = 8 if enable_gradient_checkpointing else 6

        # Attention is counted in matmuls of an S x S x d product (2 FLOPs each).
        # Flash recomputes the score matrix in the backward (5 matmuls); eager
        # stores it (4). Checkpointing adds the forward again (2). Causal
        # attention only forms the lower triangle, so the S^2 term is halved.
        # Flash plus checkpointing plus causal is the 9 used for DeepSeek-V4.
        if attn_implementation is None:
            attn_implementation = getattr(self.config, "_attn_implementation", None) or "flash_attention_2"
        self.attn_implementation = attn_implementation
        recompute = attn_implementation not in ("eager", "math")
        matmuls = 2 + (5 if recompute else 4)
        if enable_gradient_checkpointing:
            matmuls += 2
        self.attn_flops_multiplier = 2 * matmuls
        self.causal_factor = 0.5 if causal else 1.0

    def _dense_flops_multiplier(self, forward_only: bool) -> int:
        return 2 if forward_only else self.dense_flops_multiplier

    def _attn_flops_multiplier(self, forward_only: bool) -> float:
        return (4 if forward_only else self.attn_flops_multiplier) * self.causal_factor

    def _estimate_dense_decoder_transformer_tflos(self, tokens_sum, batch_seqlens, delta_time=1, forward_only=False):
        del tokens_sum, delta_time

        def _inner(config, model_size, seq_len):
            dense = self._dense_flops_multiplier(forward_only) * model_size * seq_len
            attn = (
                self._attn_flops_multiplier(forward_only) * config.num_hidden_layers * config.hidden_size * seq_len**2
            )
            return (dense + attn) / 1e12

        return sum(_inner(self.config, self.model_size, seqlen) for seqlen in batch_seqlens)

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time, forward_only):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size
        head_dim = getattr(self.config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        mlp_n = hidden_size * intermediate_size * 3
        attn_linear_n = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        embed_and_lm_head_n = vocab_size * hidden_size * 2
        dense_n = (mlp_n + attn_linear_n) * num_hidden_layers + embed_and_lm_head_n
        dense_n_flops = self._dense_flops_multiplier(forward_only) * dense_n * tokens_sum

        seqlen_square_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)
        attn_qkv_flops = (
            self._attn_flops_multiplier(forward_only)
            * seqlen_square_sum
            * head_dim
            * num_attention_heads
            * num_hidden_layers
        )
        flops_all_token = dense_n_flops + attn_qkv_flops
        return flops_all_token * (1.0 / delta_time) / 1e12

    def _estimate_deepseek_v3_flops(self, tokens_sum, batch_seqlens, delta_time, forward_only):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace
        num_query_heads = self.config.num_attention_heads
        moe_num_expert = self.config.n_routed_experts
        moe_topk = self.config.num_experts_per_tok
        share_expert_num = self.config.n_shared_experts

        moe_gate_n = hidden_size * moe_num_expert
        moe_expert_n = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3
        attn_linear_n = 0
        q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        if self.config.q_lora_rank is None:
            attn_linear_n += hidden_size * num_query_heads * q_head_dim
        else:
            attn_linear_n += hidden_size * self.config.q_lora_rank
            attn_linear_n += num_query_heads * q_head_dim * self.config.q_lora_rank
        attn_linear_n += hidden_size * (self.config.kv_lora_rank + self.config.qk_rope_head_dim)
        attn_linear_n += (
            num_query_heads
            * (q_head_dim - self.config.qk_rope_head_dim + self.config.v_head_dim)
            * self.config.kv_lora_rank
        )
        attn_linear_n += num_query_heads * self.config.v_head_dim * hidden_size
        embed_and_lm_head_n = vocab_size * hidden_size * 2
        moe_n = (
            (moe_gate_n + moe_expert_n + attn_linear_n) * (num_hidden_layers - first_k_dense_replace)
            + (hidden_size * self.config.intermediate_size * 3 + attn_linear_n) * first_k_dense_replace
            + embed_and_lm_head_n
        )
        dense_n_flops = self._dense_flops_multiplier(forward_only) * moe_n * tokens_sum
        seqlen_square_sum = sum(seqlen * seqlen * num_hidden_layers for seqlen in batch_seqlens)
        attn_qkv_flops = self._attn_flops_multiplier(forward_only) * seqlen_square_sum * q_head_dim * num_query_heads
        flops_all_token = dense_n_flops + attn_qkv_flops
        return flops_all_token * (1.0 / delta_time) / 1e12

    def _estimate_qwen2_moe_flops(self, tokens_sum, batch_seqlens, delta_time, forward_only):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        moe_intermediate_size = self.config.moe_intermediate_size
        moe_topk = self.config.num_experts_per_tok
        num_experts = self.config.num_experts
        head_dim = getattr(self.config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        full_attention_interval = getattr(self.config, "full_attention_interval", 1)
        num_full_attention_layers = num_hidden_layers // full_attention_interval
        num_linear_attention_layers = num_hidden_layers - num_full_attention_layers

        moe_mlp_n = hidden_size * moe_topk * moe_intermediate_size * 3 + hidden_size * num_experts
        embed_and_lm_head_n = vocab_size * hidden_size
        dense_n = moe_mlp_n * num_hidden_layers + embed_and_lm_head_n
        dense_n_flops = self._dense_flops_multiplier(forward_only) * dense_n * tokens_sum

        seqlen_square_sum = sum(seqlen**2 for seqlen in batch_seqlens)
        full_attn_qkv_flops = (
            self._attn_flops_multiplier(forward_only)
            * seqlen_square_sum
            * head_dim
            * num_attention_heads
            * num_full_attention_layers
        )
        full_attn_linear_fwd_per_layer = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        full_attn_linear_flops = (
            self._dense_flops_multiplier(forward_only)
            * tokens_sum
            * full_attn_linear_fwd_per_layer
            * num_full_attention_layers
        )
        full_attn_flops = full_attn_qkv_flops + full_attn_linear_flops

        linear_attn_flops = 0
        if num_linear_attention_layers > 0:
            linear_key_head_dim = self.config.linear_key_head_dim
            linear_value_head_dim = self.config.linear_value_head_dim
            linear_num_key_heads = self.config.linear_num_key_heads
            linear_num_value_heads = self.config.linear_num_value_heads
            linear_conv_kernel_dim = self.config.linear_conv_kernel_dim
            qk_dim = linear_key_head_dim * linear_num_key_heads
            v_dim = linear_value_head_dim * linear_num_value_heads
            linear_attn_flops_fwd_per_layer = (
                hidden_size * (2 * qk_dim + 2 * v_dim + 2 * linear_num_value_heads)
                + linear_conv_kernel_dim * (2 * qk_dim + v_dim)
                + linear_num_value_heads * (linear_value_head_dim**2) * 4
                + hidden_size * v_dim
            )
            linear_attn_flops = (
                self._dense_flops_multiplier(forward_only)
                * tokens_sum
                * linear_attn_flops_fwd_per_layer
                * num_linear_attention_layers
            )

        flops_all_token = dense_n_flops + full_attn_flops + linear_attn_flops
        return flops_all_token * (1.0 / delta_time) / 1e12

    def _estimate_deepseek_v4_flops(self, tokens_sum, batch_seqlens, delta_time, forward_only):
        """Checkpointed DeepSeek-V4 count, scaled when recompute or the backward is off.

        The 8 / 9 / 18 factors are the training-step convention: recompute on,
        causal compressed attention, and fixed-span attention (sliding window
        and indexer top-k). Other modes scale that count by the dense
        multiplier over 8.
        """
        del tokens_sum
        per_token = _deepseek_v4_checkpointed_flops_per_token(self.config, batch_seqlens)
        total = sum(flops * seqlen for flops, seqlen in zip(per_token, batch_seqlens))
        scale = self._dense_flops_multiplier(forward_only) / 8
        return total * scale * (1.0 / delta_time) / 1e12

    def _estimate_glm5_next_flops(self, tokens_sum, batch_seqlens, delta_time, forward_only):
        """Checkpointed GLM-5.3-Flash count, scaled like DeepSeek-V4."""
        del tokens_sum
        per_token = _glm5_next_checkpointed_flops_per_token(self.config, batch_seqlens)
        total = sum(flops * seqlen for flops, seqlen in zip(per_token, batch_seqlens))
        scale = self._dense_flops_multiplier(forward_only) / 8
        return total * scale * (1.0 / delta_time) / 1e12

    def estimate_tflos(self, batch_seqlens, delta_time=1, forward_only=False):
        """TeraFLOPs for one row of sequence lengths. ``delta_time`` is ignored."""
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_dense_decoder_transformer_tflos)
        return func(tokens_sum, batch_seqlens, delta_time=delta_time, forward_only=forward_only)


def _deepseek_v4_checkpointed_flops_per_token(config, batch_seqlens) -> list[float]:
    hidden = config.hidden_size
    heads = config.num_attention_heads
    head_dim = config.head_dim
    layers = list(config.layer_types)
    num_layers = len(layers)
    compress = dict(config.compress_rates)
    csa = "compressed_sparse_attention"
    hca = "heavily_compressed_attention"
    n_csa = sum(layer == csa for layer in layers)
    n_hca = sum(layer == hca for layer in layers)
    csa_rate = compress[csa]
    hca_rate = compress[hca]
    topk = config.num_experts_per_tok
    shared = config.n_shared_experts
    routed = config.n_routed_experts
    hc = config.hc_mult
    q_lora = config.q_lora_rank
    o_groups = config.o_groups
    o_lora = config.o_lora_rank
    index_heads = config.index_n_heads
    index_dim = config.index_head_dim
    window = config.sliding_window

    attn = hidden * q_lora + q_lora * heads * head_dim + hidden * head_dim
    attn += heads * head_dim * o_lora + o_groups * o_lora * hidden
    moe = (topk + shared) * 3 * hidden * config.moe_intermediate_size + hidden * routed
    # mHC mixes ``hc`` residual streams. ``head`` is the single hyper-head.
    mix = (2 + hc) * hc
    mhc = 2 * mix * (hc * hidden)
    hyper_head = hc * (hc * hidden)
    csa_extra = 2 * (hidden * 2 * head_dim) + 2 * (hidden * 2 * index_dim)
    csa_extra += q_lora * index_heads * index_dim + hidden * index_heads
    hca_extra = 2 * (hidden * head_dim)
    lm_head = config.vocab_size * hidden
    activated = (
        num_layers * attn
        + num_layers * moe
        + num_layers * mhc
        + hyper_head
        + n_csa * csa_extra
        + n_hca * hca_extra
        + lm_head
    )

    fixed = 8 * activated + 18 * num_layers * heads * head_dim * window
    fixed += 18 * n_csa * heads * head_dim * config.index_topk
    return [
        fixed
        + 9 * n_hca * heads * head_dim * (seqlen / hca_rate)
        + 9 * n_csa * index_heads * index_dim * (seqlen / csa_rate)
        for seqlen in batch_seqlens
    ]


def _glm5_next_checkpointed_flops_per_token(config, batch_seqlens) -> list[float]:
    """GLM-5.3-Flash FLOPs per token at each sequence length.

    Dense GEMMs are 8 times the activated parameters. Each sparse layer adds
    the causal indexer over the sequence (9) and the selected top-k attention
    (18). Linear-attention layers contribute their projections only.
    """
    hidden = config.hidden_size
    heads = config.num_attention_heads
    layers = list(config.layer_types)
    sparse = "deepseek_sparse_attention"
    n_sparse = sum(layer == sparse for layer in layers)
    n_linear = sum(layer == "linear_attention" for layer in layers)
    mlp_types = list(getattr(config, "mlp_layer_types", None) or [])
    if mlp_types:
        n_dense = sum(kind == "dense" for kind in mlp_types)
        n_moe = sum(kind == "sparse" for kind in mlp_types)
    else:
        n_dense = min(config.first_k_dense_replace, len(layers))
        n_moe = len(layers) - n_dense
    indexer_types = list(getattr(config, "indexer_types", None) or [])
    n_indexer = 0
    for index, layer in enumerate(layers):
        if layer != sparse:
            continue
        kind = indexer_types[index] if index < len(indexer_types) else "full"
        if kind != "shared":
            n_indexer += 1

    q_lora = config.q_lora_rank or 0
    kv_rank = config.kv_lora_rank
    nope = config.qk_nope_head_dim
    rope = config.qk_rope_head_dim
    value = config.v_head_dim
    qk = nope + rope
    if q_lora:
        mla = hidden * q_lora + q_lora * heads * qk
    else:
        mla = hidden * heads * qk
    mla += hidden * (kv_rank + rope) + kv_rank * heads * (nope + value) + heads * value * hidden

    index_heads = config.index_n_heads
    index_dim = config.index_head_dim
    kpool = getattr(config, "index_kpool", 0) or 0
    indexer = q_lora * index_heads * index_dim + hidden * index_dim + hidden * index_heads
    indexer += index_dim * hidden + kpool * index_dim

    linear_heads = config.linear_num_heads
    linear_dim = config.linear_head_dim
    qkv = linear_dim * linear_heads
    kernel = config.linear_conv_kernel_dim
    linear = 3 * hidden * qkv + kernel * 3 * qkv + hidden * linear_heads + qkv * hidden
    linear += 2 * (hidden * linear_dim + linear_dim * qkv)

    topk = config.num_experts_per_tok
    shared = config.n_shared_experts
    moe = hidden * config.n_routed_experts + (topk + shared) * 3 * hidden * config.moe_intermediate_size
    dense = 3 * hidden * config.intermediate_size
    hc = getattr(config, "hc_mult", 0) or 0
    mhc = 2 * ((2 + hc) * hc) * (hc * hidden) if hc else 0
    activated = (
        n_sparse * mla
        + n_indexer * indexer
        + n_linear * linear
        + n_moe * moe
        + n_dense * dense
        + len(layers) * mhc
        + config.vocab_size * hidden
    )

    fixed = 8 * activated + 18 * n_sparse * heads * value * config.index_topk
    indexer_per_token = 9 * n_indexer * index_heads * index_dim
    return [fixed + indexer_per_token * seqlen for seqlen in batch_seqlens]
