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

"""Qwen3.5-MoE's (Qwen3.6-35B-A3B's) HF reference at the precision the sampler serves it with.

Transformers' Qwen3_5Moe rounds five things that vLLM 0.33's Qwen3.5-MoE (model_executor/models/qwen3_5.py over
qwen3_next.py) computes in FP32 and rounds once, and that the target (arctic_platform/model/implementations/qwen35,
``fp32_parity``) computes the same way:

- every residual add that feeds a norm. vLLM's decoder passes the residual into ``GemmaRMSNorm`` (qwen3_next.py:571,
  :606 and the final norm at :739), whose ``fused_add_rms_norm`` adds in FP32, keeps the sum rounded as the next
  residual and normalizes the unrounded sum (ir/ops/layernorm.py:44-62). Transformers adds in BF16 and normalizes the
  rounded sum (modeling_qwen3_5_moe.py:978, :987, :1398). The target's ``FusedAddRMSNorm`` (gpu/vllm_rounding.py:81-99)
  matches vLLM;
- the GatedDeltaNet output norm ``rmsnorm(x) * w * silu(z)``: vLLM's ``RMSNormGated`` (layernorm.py:349-396, fla
  ``rmsnorm_fn`` on CUDA) in FP32 with one rounding; Transformers rounds the normalized value, its product with the
  weight and the gated result (modeling_qwen3_5_moe.py:213-222). The target runs fla's ``FusedRMSNormGated``;
- the GatedDeltaNet beta gate: vLLM's prefill prep stores ``sigmoid(b.float())`` in an FP32 buffer
  (third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py:134, :145, :203); Transformers takes
  ``b.sigmoid()`` in BF16 (modeling_qwen3_5_moe.py:605). The target uses ``b.float().sigmoid()``;
- the partial RoPE of q and k: vLLM's ``fused_qk_rmsnorm_rope_gate`` (fused_qk_norm_rope.py:73, :98-128) rounds the
  normed q/k once, then rotates in FP32 against the BF16 cos/sin cache (rotary_embedding/base.py:58-63) and rounds once;
  Transformers rotates in BF16, rounding both products and their sum (modeling_qwen3_5_moe.py:662-697, called at :787).
  The target's ``rope_rotary_slice_fp32`` (gpu/vllm_rounding.py:163-180) matches vLLM;
- the routing weights: vLLM keeps the renormalized top-k softmax in FP32 and multiplies it into the expert output in
  FP32 (fused_moe.py:591-597), summing the top-k in FP32 (csrc/moe/moe_align_sum_kernels.cu:420-458); Transformers
  rounds the weights to BF16 (modeling_qwen3_5_moe.py:885) and so rounds each weighted expert row. With FP32 weights,
  Transformers' grouped_mm/batched_mm experts weight and sum in FP32 and round once (integrations/moe.py:482-489), as
  the target's MoE router (moe/layers/moe.py:902, :931) and SonicMoE reduction do.

What needs nothing: the norm without a residual (one rounding everywhere), the q/k norms, the router logits (BF16
``GateLinear`` with no FP32 out_dtype for this model, as Transformers' and the target's), conv1d + SiLU (Dao's
``causal_conv1d`` in Transformers and the target, FP32 accumulation like vLLM's Triton kernel), g, the q/k L2 norm and
the chunk rule (fla in Transformers and the target; vLLM runs FlashInfer's kernel on Hopper), the attention output
gate, the SwiGLU MLPs and the shared expert gate (BF16 in vLLM and Transformers).
"""

from __future__ import annotations

import types

QWEN3_5_MOE_MODEL_TYPES = frozenset({"qwen3_5_moe", "qwen3_5_moe_text"})
# The experts implementations whose top-k reduction promotes FP32 routing weights and sums them in FP32.
FP32_REDUCING_EXPERTS = frozenset({"grouped_mm", "batched_mm"})


def _rms_norm_of_unrounded_sum(self, hidden_states):
    """The (1 + w) RMSNorm of the residual sum as given (FP32 after the first layer), rounded once to the
    activation dtype: vLLM ``fused_add_rms_norm`` normalizes the FP32 sum, not its rounding."""
    import torch

    normed = hidden_states.float()
    normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + self.eps)
    return (normed * (1.0 + self.weight.float())).to(self._sampler_activation_dtype)


def _decoder_layer_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    **kwargs,
):
    """``Qwen3_5MoeDecoderLayer.forward`` with vLLM's residual handling, the only change: each residual sum is formed
    in FP32, rounded for the residual stream and normalized unrounded. The layer returns its output sum in FP32; the
    next layer's input norm (or the final norm) normalizes it and rounds it for the residual."""
    dtype = self._sampler_activation_dtype
    residual = hidden_states.to(dtype)
    hidden_states = self.input_layernorm(hidden_states)

    if self.block_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
    elif self.block_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual.float() + hidden_states.float()
    residual = hidden_states.to(dtype)
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    if isinstance(hidden_states, tuple):
        hidden_states, _ = hidden_states
    return residual.float() + hidden_states.float()


def _rms_norm_gated_forward(self, hidden_states, gate):
    """``rmsnorm(x) * w * silu(z)`` in FP32, rounded once (vLLM ``RMSNormGated.forward_static``, norm before gate)."""
    import torch
    import torch.nn.functional as F

    normed = hidden_states.float()
    normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    activated = torch.sigmoid(gate.float()) if self.activation == "sigmoid" else F.silu(gate.float())
    return (normed * self.weight.float() * activated).to(hidden_states.dtype)


def _fp32_logits_forward(self, hidden_states):
    """The beta projection's BF16 logits widened, so the GatedDeltaNet's ``b.sigmoid()`` runs and stays in FP32."""
    return type(self).forward(self, hidden_states).float()


def fp32_rope(x, cos, sin):
    """NeoX partial RoPE of ``[batch, heads, seq, head_dim]`` in FP32 against the given cos/sin, rounded once.

    The pass-through channels are unchanged; the rotary ones are ``x * cos + rotate_half(x) * sin`` evaluated in FP32,
    as vLLM's fused q/k-norm + RoPE kernel computes ``x1 * cos - x2 * sin`` and ``x2 * cos + x1 * sin``.
    """
    import torch

    cos = cos.unsqueeze(1).float()
    sin = sin.unsqueeze(1).float()
    rotary_dim = cos.shape[-1]
    rotary, passed = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    half = rotary_dim // 2
    rotated = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
    return torch.cat(((rotary * cos + rotated * sin).to(x.dtype), passed), dim=-1)


def _attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    """``Qwen3_5MoeAttention.forward`` with the q/k RoPE in FP32 and one rounding (the only change)."""
    import torch
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen3_5_moe

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = fp32_rope(query_states, cos, sin), fp32_rope(key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attention_interface = qwen3_5_moe.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, qwen3_5_moe.eager_attention_forward
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
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _router_forward(self, hidden_states):
    """``Qwen3_5MoeTopKRouter.forward`` with the renormalized top-k weights kept in FP32 (the only change)."""
    import torch
    import torch.nn.functional as F

    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
    router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    return router_logits, router_top_value, router_indices


def apply_sampler_precision(model, activation_dtype) -> dict[str, int]:
    """Patch a loaded Qwen3_5Moe model's decoder layers, residual norms, gated norms, beta projections, attention
    layers and routers in place. ``activation_dtype`` is the dtype the residual stream is rounded to (the load dtype).
    """
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen3_5_moe

    counts = {
        "decoder_layers": 0,
        "residual_norms": 0,
        "gated_norms": 0,
        "beta_projections": 0,
        "attention_layers": 0,
        "routers": 0,
    }
    residual_norms = []
    for module in model.modules():
        if isinstance(module, qwen3_5_moe.Qwen3_5MoeDecoderLayer):
            module.forward = types.MethodType(_decoder_layer_forward, module)
            module._sampler_activation_dtype = activation_dtype
            residual_norms += [module.input_layernorm, module.post_attention_layernorm]
            counts["decoder_layers"] += 1
        elif isinstance(module, qwen3_5_moe.Qwen3_5MoeTextModel):
            residual_norms.append(module.norm)
        elif type(module) is qwen3_5_moe.Qwen3_5MoeRMSNormGated:
            module.forward = types.MethodType(_rms_norm_gated_forward, module)
            counts["gated_norms"] += 1
        elif isinstance(module, qwen3_5_moe.Qwen3_5MoeGatedDeltaNet):
            module.in_proj_b.forward = types.MethodType(_fp32_logits_forward, module.in_proj_b)
            counts["beta_projections"] += 1
        elif isinstance(module, qwen3_5_moe.Qwen3_5MoeAttention):
            module.forward = types.MethodType(_attention_forward, module)
            counts["attention_layers"] += 1
        elif type(module) is qwen3_5_moe.Qwen3_5MoeTopKRouter:
            module.forward = types.MethodType(_router_forward, module)
            counts["routers"] += 1
        elif isinstance(module, qwen3_5_moe.Qwen3_5MoeExperts):
            implementation = module.config._experts_implementation
            if implementation not in FP32_REDUCING_EXPERTS:
                raise ValueError(
                    "Qwen3.5-MoE reference precision needs experts that weight and sum the top-k in FP32 "
                    f"({sorted(FP32_REDUCING_EXPERTS)}); this model's experts implementation is {implementation!r}"
                )
    for norm in residual_norms:
        if type(norm) is not qwen3_5_moe.Qwen3_5MoeRMSNorm:
            raise ValueError(f"Qwen3.5-MoE residual norm is {type(norm).__qualname__}, not Qwen3_5MoeRMSNorm")
        norm.forward = types.MethodType(_rms_norm_of_unrounded_sum, norm)
        norm._sampler_activation_dtype = activation_dtype
    counts["residual_norms"] = len(residual_norms)
    if not counts["decoder_layers"] or counts["residual_norms"] != 2 * counts["decoder_layers"] + 1:
        raise ValueError(f"Qwen3.5-MoE reference precision found an incomplete decoder to patch: {counts}")
    return counts


def load_qwen3_5_moe_reference(model_class, model_path, *, dtype, attn_implementation, device):
    """Load the checkpoint, then apply the sampler's FP32 residual norms, gated norm, beta, RoPE and routing weights."""
    model = model_class.from_pretrained(model_path, dtype=dtype, attn_implementation=attn_implementation)
    apply_sampler_precision(model, model.get_input_embeddings().weight.dtype)
    return model.to(device)
