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

"""Qwen3.8-Flash-Next's HF reference at the precision the sampler serves it with.

Transformers' Qwen4Exp rounds five things to BF16 that vLLM's Qwen4Exp (vllm/models/qwen4_exp/nvidia) evaluates in FP32
and rounds once, and that the target trains the same way (arctic_platform/model/implementations/qwen38, fp32_parity):

- the hyper-connection mix ``mean_s(sigmoid(up(silu(down(n) / hc)))_s * n_s)``: vLLM ``hc_silu`` and ``hc_gate_mix``
  (qwen4_exp/nvidia/ops/hc.py:99-104 and :149-159, called from nvidia/hyperconnection.py:161-203); the target's
  ``Qwen4ExpTextGatedResidualPrimeRL`` and ``gated_residual(round_once=True)`` (qwen38/modeling_qwen4_exp.py:283-293,
  qwen38/hyper_connection.py:241-247). Transformers divides and SiLUs in BF16, takes the sigmoid in BF16 and rounds the
  per-stream product before the mean (transformers qwen4_exp/modeling_qwen4_exp.py:1013-1018);
- the hyper-connection combine ``X + out * 2 * sigmoid(l / hc)``: vLLM ``_hc_combine_kernel`` and
  ``_hc_combine_norm_kernel`` (ops/hc.py:217-231 and :321-326); the target's ``combine`` (hyper_connection.py:165-180).
  Transformers rounds the injection weight, the product and the sum (modeling_qwen4_exp.py:1021, :1293-1294 and
  :1299-1300);
- the GatedDeltaNet beta gate ``sigmoid(in_proj_b(x).float())``: vLLM ``fused_post_conv_prep``
  (vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py:134 and :145, from
  model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1731); the target's FP32 ``in_proj_b`` output hook
  (modeling_qwen4_exp.py:398-401). Transformers takes the sigmoid of the BF16 logits (modeling_qwen4_exp.py:570);
- the GatedDeltaNet output norm ``rmsnorm(x) * w * sigmoid(z)`` in FP32, rounded once: vLLM ``RMSNormGated``
  (model_executor/layers/layernorm.py:349-400, ``fla_rms_norm_gated`` on CUDA); the target's ``round_gated_norm_once``
  (modeling_qwen4_exp.py:397). Transformers rounds the normalized value before the weight (modeling_qwen4_exp.py:184);
- the QSA rotary embedding and sigmoid output gate: vLLM rotates the normalized BF16 q/k in FP32 against its BF16
  cos/sin cache and rounds once (qwen4_exp/nvidia/ops/qsa_prepare.py:196-218 and :94-102), and gates the BF16 attention
  output with an FP32 sigmoid, rounding once (nvidia/ops/qsa.py:233-244 and :321-325); the target's
  ``rope_rotary_slice_fp32`` and ``gate_fp32`` (qwen38/qsa_flex.py:584-587 and :606-607). Transformers rounds both
  rotary products and their sum, the sigmoid and the gated product (modeling_qwen4_exp.py:651 and :889).

Everything else on the training path already follows vLLM: the (1 + w) RMSNorms round once (hc_norm, q/k norms, PLE
norms), the PLE gate keeps Transformers' BF16 materialization points because vLLM's ``_ple_gate_kernel`` copies them
(common/ops/ple.py:217-234), the PLE short convolution accumulates in FP32 and rounds before its SiLU (ple.py:376-405),
the GatedDeltaNet decay, convolution and l2 norm run in FP32, and the router logits, shared-expert gate and LM head are
BF16 GEMMs on both sides. The QSA indexer only selects tokens and is frozen; it keeps Transformers' arithmetic.
"""

from __future__ import annotations

import types


def _wide(tensor):
    """FP32 for the 16-bit dtypes; wider tensors keep their dtype, so an FP64 model keeps FP64 math."""
    import torch

    return tensor.float() if tensor.dtype in (torch.bfloat16, torch.float16) else tensor


def fp32_hyper_connection_combine(hyper_input, block_output, injection_weights):
    """``hyper_input + block_output * injection_weights`` per stream in FP32, rounded once (vLLM ``hc_combine``)."""
    hc = injection_weights.shape[-1]
    streams = _wide(hyper_input).unflatten(-1, (hc, block_output.shape[-1]))
    combined = streams + _wide(block_output).unsqueeze(-2) * _wide(injection_weights).unsqueeze(-1)
    return combined.flatten(-2).to(hyper_input.dtype)


def fp32_rotary(x, cos, sin):
    """Partial RoPE of ``x`` ``[batch, heads, seq, head_dim]`` in FP32 against ``cos``/``sin`` ``[batch, seq, rot]``,
    rounded once to ``x``'s dtype; the channels past ``rot`` pass through (vLLM ``_qsa_prepare_kernel``)."""
    import torch

    rotary = cos.shape[-1]
    half = rotary // 2
    x_rot, x_pass = _wide(x[..., :rotary]), x[..., rotary:]
    cos, sin = _wide(cos.unsqueeze(1)), _wide(sin.unsqueeze(1))
    rotated = torch.cat((-x_rot[..., half:], x_rot[..., :half]), dim=-1)
    return torch.cat(((x_rot * cos + rotated * sin).to(x.dtype), x_pass), dim=-1)


def _gated_residual_forward(self, hyper_input):
    """``Qwen4ExpTextGatedResidual.forward`` with the SiLU, the gated mix and the injection weight in FP32.

    The injection weight stays FP32 for ``fp32_hyper_connection_combine``; vLLM never materializes it.
    """
    import torch
    import torch.nn.functional as F

    if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
        raise ValueError(
            f"Expected {self.hc_count * self.hidden_size} hyper-connection features, got {hyper_input.shape[-1]}."
        )
    normed = self.hc_norm(hyper_input)
    lowrank = F.silu(_wide(self.input_mix_weight_down(normed)) / self.hc_count).to(normed.dtype)
    gate = self.input_mix_weight_up(lowrank)
    streams = (self.hc_count, self.hidden_size)
    mixed = (torch.sigmoid(_wide(gate)).unflatten(-1, streams) * _wide(normed).unflatten(-1, streams)).mean(dim=-2)
    mixed = mixed.to(normed.dtype)
    if self.block_inject_weight is None:
        return mixed
    injection_weights = 2 * torch.sigmoid(_wide(self.block_inject_weight(normed)) / self.hc_count)
    return mixed, hyper_input, injection_weights


def _decoder_layer_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    conv_mask=None,
    past_key_values=None,
    ple_input_ids=None,
    **kwargs,
):
    """``Qwen4ExpTextDecoderLayer.forward`` with both combines in FP32 (the only change)."""
    if self.ple is not None:
        hidden_states = hidden_states + self.ple(hidden_states, ple_input_ids, past_key_values, conv_mask=conv_mask)

    hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)
    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states, cache_params=past_key_values, attention_mask=conv_mask, **kwargs
        )
    else:
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
    hidden_states = fp32_hyper_connection_combine(hyper_input, hidden_states, injection_weights)

    hidden_states, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
    hidden_states = self.mlp(hidden_states)
    return fp32_hyper_connection_combine(hyper_input, hidden_states, injection_weights)


def _attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    """``Qwen4ExpTextAttention.forward`` with the rotary embedding and the output gate in FP32 (the only changes)."""
    import torch
    from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen4_exp

    selected_token_mask = self.indexer(hidden_states, position_embeddings, attention_mask, past_key_values)
    if attention_mask.is_floating_point():
        attention_mask = attention_mask + selected_token_mask
    else:
        attention_mask = attention_mask & selected_token_mask

    cos, sin = (x[:, -hidden_states.shape[1] :, :] for x in position_embeddings)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states = fp32_rotary(query_states, cos, sin), fp32_rotary(key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attention_interface = qwen4_exp.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, qwen4_exp.eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = (_wide(attn_output) * torch.sigmoid(_wide(gate))).to(attn_output.dtype)
    return self.o_proj(attn_output), attn_weights


def _rms_norm_gated_forward(self, hidden_states, gate):
    """``rmsnorm(x) * weight * activation(gate)`` in FP32, rounded once."""
    import torch
    from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen4_exp

    hidden = _wide(hidden_states)
    hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    activated = qwen4_exp.ACT2FN[self.activation](_wide(gate))
    return (hidden * _wide(self.weight) * activated).to(hidden_states.dtype)


def _fp32_logits_forward(self, hidden_states):
    """The beta projection's BF16 logits widened, so ``b.sigmoid()`` runs in FP32."""
    return _wide(type(self).forward(self, hidden_states))


def apply_sampler_precision(model) -> dict[str, int]:
    """Patch a loaded Qwen4Exp model's hyper connections, decoder layers, GatedDeltaNets and QSA attention in place."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen4_exp

    counts = {"hyper_connections": 0, "decoder_layers": 0, "gated_norms": 0, "beta": 0, "attention": 0}
    for module in model.modules():
        if isinstance(module, qwen4_exp.Qwen4ExpTextGatedResidual):
            module.forward = types.MethodType(_gated_residual_forward, module)
            counts["hyper_connections"] += 1
        elif isinstance(module, qwen4_exp.Qwen4ExpTextDecoderLayer):
            module.forward = types.MethodType(_decoder_layer_forward, module)
            counts["decoder_layers"] += 1
        elif isinstance(module, qwen4_exp.Qwen4ExpTextRMSNormGated):
            module.forward = types.MethodType(_rms_norm_gated_forward, module)
            counts["gated_norms"] += 1
        elif isinstance(module, qwen4_exp.Qwen4ExpTextGatedDeltaNet):
            module.in_proj_b.forward = types.MethodType(_fp32_logits_forward, module.in_proj_b)
            counts["beta"] += 1
        elif isinstance(module, qwen4_exp.Qwen4ExpTextAttention):
            module.forward = types.MethodType(_attention_forward, module)
            counts["attention"] += 1
    if not counts["decoder_layers"] or counts["hyper_connections"] != 2 * counts["decoder_layers"] + 1:
        raise ValueError(f"Qwen3.8 reference precision found an unexpected hyper-connection layout: {counts}")
    if counts["gated_norms"] != counts["beta"] or counts["beta"] + counts["attention"] != counts["decoder_layers"]:
        raise ValueError(f"Qwen3.8 reference precision found an unexpected mixer layout: {counts}")
    return counts
