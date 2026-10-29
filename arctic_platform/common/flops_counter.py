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
# DeepSeek-V4 and GLM-5.3-Flash use the per-token convention of the H200 step
# reports: 8 FLOPs per activated parameter when checkpointed, plus the sparse
# attention and indexer terms.

from __future__ import annotations

from typing import Any

from arctic_platform.common.peft import PEFT_LORA_DEFAULT_R

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
        "qwen3_5",
        "qwen3_5_text",
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
    peft_config=None,
):
    """Total teraFLOPs and token count for a batch of packed or unpacked samples.

    ``batch_of_seqlens`` is one list of sequence lengths per batch row. A single
    unpacked sample is ``[[seqlen]]``. Unknown ``model_type`` values use the
    dense decoder formula.
    """
    flops_counter = FlopsCounter(hf_model_config, model_size, enable_gradient_checkpointing, peft_config=peft_config)
    seqlen = 0
    tflos = 0.0
    for seqlens in batch_of_seqlens:
        tflos += flops_counter.estimate_tflos(batch_seqlens=seqlens, delta_time=1, forward_only=forward_only)
        seqlen += sum(seqlens)
    return tflos, seqlen


def tflops_per_token(
    config,
    seq_len: int,
    model_size: int = 0,
    *,
    enable_gradient_checkpointing: bool | float = True,
    peft_config: dict[str, Any] | None = None,
) -> float:
    """TeraFLOPs per token at ``seq_len``. Multiply by tokens/second to get TFLOP/s."""
    counter = FlopsCounter(config, model_size, enable_gradient_checkpointing, peft_config=peft_config)
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    return counter.estimate_tflos([seq_len]) / seq_len


class FlopsCounter:
    """Training-step FLOPs from an HF config.

    Dense GEMMs use 6 FLOPs per parameter-token, or 8 when activation
    checkpointing recomputes the forward. Attention follows the flash or
    eager backward and is halved for causal masks. ``forward_only`` counts
    the forward matmuls only.

    ``enable_gradient_checkpointing`` is a bool or the fraction of layers
    that recompute their forward. With a LoRA ``peft_config`` the frozen base
    weights skip their weight gradients (2 of the 6), and each adapted
    Linear adds ``r * (in + out)`` trainable parameters at the full count.
    """

    def __init__(
        self,
        config: Any,
        model_size,
        enable_gradient_checkpointing: bool | float = False,
        causal: bool = True,
        attn_implementation: str | None = None,
        peft_config: dict[str, Any] | None = None,
    ):
        if not 0 <= enable_gradient_checkpointing <= 1:
            raise ValueError(
                "enable_gradient_checkpointing must be a bool or a fraction in [0, 1], "
                f"got {enable_gradient_checkpointing}"
            )
        self.estimate_func = {
            "qwen2": self._estimate_qwen2_flops,
            "llama": self._estimate_qwen2_flops,
            "qwen2_moe": self._estimate_qwen2_moe_flops,
            "qwen2_vl": self._estimate_qwen2_flops,
            "qwen2_5_vl": self._estimate_qwen2_flops,
            "qwen3": self._estimate_qwen2_flops,
            "qwen3_moe": self._estimate_qwen2_moe_flops,
            "qwen3_next": self._estimate_qwen2_moe_flops,
            "qwen3_5": self._estimate_qwen3_5_flops,
            "qwen3_5_text": self._estimate_qwen3_5_flops,
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
        # GEMM multipliers on N*tokens: forward 2, backward 4 (input and weight
        # gradients, 2 each). Checkpointing recomputes one forward, which adds 2.
        self.trainable_flops_multiplier = 6 + 2 * enable_gradient_checkpointing
        self.dense_flops_multiplier = self.trainable_flops_multiplier - (2 if peft_config else 0)
        self.lora_params = _lora_adapter_params(self.config, peft_config) if peft_config else 0

        # Attention is counted in matmuls of an S x S x d product (2 FLOPs each).
        # Flash recomputes the score matrix in the backward (5 matmuls); eager
        # stores it (4). Checkpointing adds the forward again (2). Causal
        # attention only forms the lower triangle, so the S^2 term is halved.
        # Flash plus checkpointing plus causal is the 9 used for DeepSeek-V4.
        if attn_implementation is None:
            attn_implementation = getattr(self.config, "_attn_implementation", None) or "flash_attention_2"
        self.attn_implementation = attn_implementation
        recompute = attn_implementation not in ("eager", "math")
        matmuls = 2 + (5 if recompute else 4) + 2 * enable_gradient_checkpointing
        self.attn_flops_multiplier = 2 * matmuls
        self.causal_factor = 0.5 if causal else 1.0

    def _dense_flops_multiplier(self, forward_only: bool) -> float:
        return 2 if forward_only else self.dense_flops_multiplier

    def _attn_span_flops_multiplier(self, forward_only: bool) -> float:
        """Attention FLOPs per (query, key, head dim) when every query sees the whole span."""
        return 4 if forward_only else self.attn_flops_multiplier

    def _attn_flops_multiplier(self, forward_only: bool) -> float:
        return self._attn_span_flops_multiplier(forward_only) * self.causal_factor

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
        moe_intermediate_size = self.config.moe_intermediate_size
        moe_topk = self.config.num_experts_per_tok
        num_experts = self.config.num_experts
        moe_mlp_n = hidden_size * moe_topk * moe_intermediate_size * 3 + hidden_size * num_experts
        return self._estimate_hybrid_attention_flops(moe_mlp_n, tokens_sum, batch_seqlens, delta_time, forward_only)

    def _estimate_qwen3_5_flops(self, tokens_sum, batch_seqlens, delta_time, forward_only):
        mlp_n = self.config.hidden_size * self.config.intermediate_size * 3
        return self._estimate_hybrid_attention_flops(mlp_n, tokens_sum, batch_seqlens, delta_time, forward_only)

    def _estimate_hybrid_attention_flops(self, mlp_n, tokens_sum, batch_seqlens, delta_time, forward_only):
        """Every ``full_attention_interval``-th layer is full attention; the rest are gated delta-net."""
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        head_dim = getattr(self.config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        full_attention_interval = getattr(self.config, "full_attention_interval", 1)
        num_full_attention_layers = num_hidden_layers // full_attention_interval
        num_linear_attention_layers = num_hidden_layers - num_full_attention_layers

        embed_and_lm_head_n = vocab_size * hidden_size
        dense_n = mlp_n * num_hidden_layers + embed_and_lm_head_n
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

    def _estimate_sparse_attention_flops(self, terms, batch_seqlens, delta_time, forward_only):
        """Per-token dense, fixed-span, and causal terms times each sequence length.

        Checkpointed, that is 8 per activated parameter, 18 per fixed-span
        (query, key, head dim) such as a sliding window or indexer top-k, and
        9 per causal one, the convention of the H200 step reports.
        """
        activated, span, causal = terms
        per_token = (
            self._dense_flops_multiplier(forward_only) * activated
            + self._attn_span_flops_multiplier(forward_only) * span
        )
        causal_per_key = self._attn_flops_multiplier(forward_only) * causal
        total = sum((per_token + causal_per_key * seqlen) * seqlen for seqlen in batch_seqlens)
        return total * (1.0 / delta_time) / 1e12

    def _estimate_deepseek_v4_flops(self, tokens_sum, batch_seqlens, delta_time, forward_only):
        del tokens_sum
        terms = _deepseek_v4_flops_terms(self.config)
        return self._estimate_sparse_attention_flops(terms, batch_seqlens, delta_time, forward_only)

    def _estimate_glm5_next_flops(self, tokens_sum, batch_seqlens, delta_time, forward_only):
        del tokens_sum
        terms = _glm5_next_flops_terms(self.config)
        return self._estimate_sparse_attention_flops(terms, batch_seqlens, delta_time, forward_only)

    def estimate_tflos(self, batch_seqlens, delta_time=1, forward_only=False):
        """TeraFLOPs for one row of sequence lengths. ``delta_time`` is ignored."""
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_dense_decoder_transformer_tflos)
        tflos = func(tokens_sum, batch_seqlens, delta_time=delta_time, forward_only=forward_only)
        adapter_multiplier = 2 if forward_only else self.trainable_flops_multiplier
        return tflos + adapter_multiplier * self.lora_params * tokens_sum / 1e12


def _deepseek_v4_flops_terms(config) -> tuple[float, float, float]:
    """Activated parameters, fixed-span attention, and causal attention per key."""
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

    span = num_layers * heads * head_dim * window + n_csa * heads * head_dim * config.index_topk
    causal = n_hca * heads * head_dim / hca_rate + n_csa * index_heads * index_dim / csa_rate
    return activated, span, causal


def _glm5_next_flops_terms(config) -> tuple[float, float, float]:
    """GLM-5.3-Flash activated parameters, top-k attention, and causal indexer per key.

    Linear-attention layers contribute their projections only.
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

    span = n_sparse * heads * value * config.index_topk
    return activated, span, n_indexer * index_heads * index_dim


# Each layout is (module name, in_features, out_features, modules per token)
# for the Linear layers a LoRA ``target_modules`` entry can match.
def _decoder_linears(config) -> list[tuple[str, int, int, int]]:
    hidden = config.hidden_size
    heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", hidden // heads)
    kv = config.num_key_value_heads * head_dim
    layers = config.num_hidden_layers
    inter = config.intermediate_size
    return [
        ("q_proj", hidden, heads * head_dim, layers),
        ("k_proj", hidden, kv, layers),
        ("v_proj", hidden, kv, layers),
        ("o_proj", heads * head_dim, hidden, layers),
        ("gate_proj", hidden, inter, layers),
        ("up_proj", hidden, inter, layers),
        ("down_proj", inter, hidden, layers),
    ]


def _qwen3_5_linears(config) -> list[tuple[str, int, int, int]]:
    hidden = config.hidden_size
    heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", hidden // heads)
    kv = config.num_key_value_heads * head_dim
    layers = config.num_hidden_layers
    n_full = layers // getattr(config, "full_attention_interval", 1)
    n_linear = layers - n_full
    gate = 2 if getattr(config, "attn_output_gate", False) else 1
    qk = config.linear_key_head_dim * config.linear_num_key_heads
    v = config.linear_value_head_dim * config.linear_num_value_heads
    value_heads = config.linear_num_value_heads
    inter = config.intermediate_size
    return [
        ("q_proj", hidden, gate * heads * head_dim, n_full),
        ("k_proj", hidden, kv, n_full),
        ("v_proj", hidden, kv, n_full),
        ("o_proj", heads * head_dim, hidden, n_full),
        ("in_proj_qkv", hidden, 2 * qk + v, n_linear),
        ("in_proj_z", hidden, v, n_linear),
        ("in_proj_a", hidden, value_heads, n_linear),
        ("in_proj_b", hidden, value_heads, n_linear),
        ("out_proj", v, hidden, n_linear),
        ("gate_proj", hidden, inter, layers),
        ("up_proj", hidden, inter, layers),
        ("down_proj", inter, hidden, layers),
    ]


def _glm5_next_linears(config) -> list[tuple[str, int, int, int]]:
    """Routed experts are 3D parameters, which ``target_modules`` does not adapt."""
    hidden = config.hidden_size
    heads = config.num_attention_heads
    layers = list(config.layer_types)
    sparse = "deepseek_sparse_attention"
    n_sparse = sum(layer == sparse for layer in layers)
    n_linear = sum(layer == "linear_attention" for layer in layers)
    indexer_types = list(getattr(config, "indexer_types", None) or [])
    n_indexer = sum(
        layer == sparse and (indexer_types[i] if i < len(indexer_types) else "full") != "shared"
        for i, layer in enumerate(layers)
    )
    mlp_types = list(getattr(config, "mlp_layer_types", None) or [])
    if mlp_types:
        n_dense = sum(kind == "dense" for kind in mlp_types)
    else:
        n_dense = min(config.first_k_dense_replace, len(layers))
    n_moe = len(layers) - n_dense
    q_lora = config.q_lora_rank
    rope = config.qk_rope_head_dim
    nope = config.qk_nope_head_dim
    value = config.v_head_dim
    kv_rank = config.kv_lora_rank
    index_heads = config.index_n_heads
    index_dim = config.index_head_dim
    linear_dim = config.linear_head_dim
    qkv = linear_dim * config.linear_num_heads
    inter = config.intermediate_size
    shared = config.n_shared_experts * config.moe_intermediate_size
    return [
        ("q_a_proj", hidden, q_lora, n_sparse),
        ("q_b_proj", q_lora, heads * (nope + rope), n_sparse),
        ("kv_a_proj_with_mqa", hidden, kv_rank + rope, n_sparse),
        ("kv_b_proj", kv_rank, heads * (nope + value), n_sparse),
        ("o_proj", heads * value, hidden, n_sparse),
        ("wq_b", q_lora, index_heads * index_dim, n_indexer),
        ("wk", hidden, index_dim, n_indexer),
        ("weights_proj", hidden, index_heads, n_indexer),
        ("q_proj", hidden, qkv, n_linear),
        ("k_proj", hidden, qkv, n_linear),
        ("v_proj", hidden, qkv, n_linear),
        ("o_proj", qkv, hidden, n_linear),
        ("b_proj", hidden, config.linear_num_heads, n_linear),
        ("f_a_proj", hidden, linear_dim, n_linear),
        ("f_b_proj", linear_dim, qkv, n_linear),
        ("g_a_proj", hidden, linear_dim, n_linear),
        ("g_b_proj", linear_dim, qkv, n_linear),
        ("gate_proj", hidden, inter, n_dense),
        ("up_proj", hidden, inter, n_dense),
        ("down_proj", inter, hidden, n_dense),
        ("gate_proj", hidden, shared, n_moe),
        ("up_proj", hidden, shared, n_moe),
        ("down_proj", shared, hidden, n_moe),
    ]


_LORA_LINEARS = {
    "llama": _decoder_linears,
    "qwen2": _decoder_linears,
    "qwen3": _decoder_linears,
    "qwen3_5": _qwen3_5_linears,
    "qwen3_5_text": _qwen3_5_linears,
    "glm5_next": _glm5_next_linears,
    "glm5_next_text": _glm5_next_linears,
}


def _lora_adapter_params(config, peft_config: dict[str, Any]) -> int:
    """Adapter parameters used per token: ``r * (in + out)`` for each targeted Linear."""
    layout = _LORA_LINEARS.get(config.model_type)
    if layout is None:
        raise ValueError(f"no LoRA module layout for model_type {config.model_type!r}")
    targets = peft_config.get("target_modules") or []
    if isinstance(targets, str):
        raise ValueError("regex target_modules are not supported; list the module names")
    if peft_config.get("target_parameters"):
        raise ValueError("target_parameters adapters are not counted")
    linears = layout(config)
    known = {name for name, *_ in linears}
    unknown = sorted(set(targets) - known)
    if unknown:
        raise ValueError(f"target_modules {unknown} match no {config.model_type} Linear; known: {sorted(known)}")
    rank = peft_config.get("r", PEFT_LORA_DEFAULT_R)
    return sum(rank * (fan_in + fan_out) * count for name, fan_in, fan_out, count in linears if name in targets)
