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

"""GLM-5.3-Flash's HF reference at the precision the sampler serves it with.

Transformers' Glm5Next runs four things in BF16 that vLLM's GLM-5.3-Flash (vllm/models/glm5next) serves in FP32 and
rounds once, and that the target trains the same way (arctic_platform/model/implementations/glm53):

- the hyper-connection remix ``post * y + comb^T @ streams``: vLLM ``mhc_post_torch``
  (model_executor/kernels/mhc/torch.py:135, the tilelang ``mhc_post`` on CUDA); the target's ``_remix`` fp32 branch
  and ``mix_streams`` (glm53/modeling_glm5_next.py:85);
- the hyper-connection ``base`` and ``scale`` as FP32 parameters: vLLM glm5next/common/model.py:471-479; the target's
  ``_FP32_PARAMETERS`` (glm53/modeling_glm5_next.py:106), so their gradients are FP32 too;
- the KDA beta gate ``sigmoid(b_proj(x).float())``: vLLM glm5next/common/kda.py:132 and :735; the target's
  ``_kda_beta`` (glm53/context_parallel.py:712);
- every ``Glm5NextTextRMSNorm`` with the FP32 weight and one rounding: vLLM's fused mHC-pre norm
  (model_executor/kernels/mhc/tilelang_kernels.py:509-521) and its CUDA ``rms_norm`` for the latent and final norms;
  the target's ``gemma``-mode Liger norm (glm53/fused_norms.py:39-46). Transformers rounds the normalized value to BF16
  and then multiplies by the BF16 weight.

The router logits need nothing: Transformers already computes ``F.linear(x.float(), weight.float())``, as vLLM's
``GateLinear`` (``moe_router_dtype: float32``) and the target's ``fp32_logits`` router do.
"""

from __future__ import annotations

import types
from contextlib import contextmanager

GLM5_NEXT_MODEL_TYPES = frozenset({"glm5_next", "glm5_next_text"})
FP32_HYPER_CONNECTION_PARAMETERS = ("attn_hc.base", "attn_hc.scale", "ffn_hc.base", "ffn_hc.scale")


def fp32_remix(post, comb, sublayer_out, streams):
    """``post * sublayer_out + comb^T @ streams`` in FP32, cast to the streams' dtype once (vLLM ``mhc_post_torch``)."""
    import torch

    mixed = torch.einsum("...ij,...ih->...jh", comb.float(), streams.float())
    return (mixed + post.float().unsqueeze(-1) * sublayer_out.float().unsqueeze(-2)).to(streams.dtype)


def _decoder_layer_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache=False,
    position_embeddings=None,
    prev_topk_indices=None,
    **kwargs,
):
    """``Glm5NextTextDecoderLayer.forward`` with both remixes in FP32 (the only change)."""
    residual = hidden_states
    post, comb, hidden_states = self.attn_hc(hidden_states)
    hidden_states = self.input_layernorm(hidden_states)
    topk_indices = None
    if self.block_type == "linear_attention":
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
    else:
        hidden_states, _, topk_indices = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            prev_topk_indices=prev_topk_indices,
            **kwargs,
        )
    hidden_states = fp32_remix(post, comb, hidden_states, residual)

    residual = hidden_states
    post, comb, hidden_states = self.ffn_hc(hidden_states)
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = fp32_remix(post, comb, hidden_states, residual)
    return hidden_states, topk_indices


def _rms_norm_forward(self, hidden_states):
    """FP32 statistics and weight, one rounding."""
    import torch

    normed = hidden_states.float()
    normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    return (normed * self.weight.float()).to(hidden_states.dtype)


def _fp32_logits_forward(self, hidden_states):
    """The beta projection's BF16 logits widened, so ``torch.sigmoid`` runs in FP32."""
    return type(self).forward(self, hidden_states).float()


@contextmanager
def fp32_hyper_connection_parameters():
    """Load the hyper-connection base/scale as FP32 parameters, whatever dtype the rest of the model loads in."""
    from transformers.models.glm5_next import modeling_glm5_next as glm5_next

    base = glm5_next.Glm5NextPreTrainedModel
    saved = base.__dict__["_keep_in_fp32_modules_strict"]
    kept = list(saved or ())
    base._keep_in_fp32_modules_strict = kept + [name for name in FP32_HYPER_CONNECTION_PARAMETERS if name not in kept]
    try:
        yield
    finally:
        base._keep_in_fp32_modules_strict = saved


def apply_sampler_precision(model) -> dict[str, int]:
    """Patch a loaded Glm5Next model's decoder layers, RMSNorms and KDA beta projections in place."""
    import torch
    from transformers.models.glm5_next import modeling_glm5_next as glm5_next

    counts = {"decoder_layers": 0, "rms_norms": 0, "kda_beta": 0}
    for module in model.modules():
        if isinstance(module, glm5_next.Glm5NextTextDecoderLayer):
            module.forward = types.MethodType(_decoder_layer_forward, module)
            counts["decoder_layers"] += 1
        elif type(module) is glm5_next.Glm5NextTextRMSNorm:
            module.forward = types.MethodType(_rms_norm_forward, module)
            counts["rms_norms"] += 1
        elif isinstance(module, glm5_next.Glm5NextTextLinearAttention):
            module.b_proj.forward = types.MethodType(_fp32_logits_forward, module.b_proj)
            counts["kda_beta"] += 1
    if not counts["decoder_layers"] or not counts["rms_norms"]:
        raise ValueError(f"GLM-5.3-Flash reference precision found no decoder layers or norms to patch: {counts}")
    narrow = [
        name
        for name, parameter in model.named_parameters()
        if name.endswith(FP32_HYPER_CONNECTION_PARAMETERS) and parameter.dtype != torch.float32
    ]
    if narrow:
        raise ValueError(f"GLM-5.3-Flash hyper-connection parameters did not load in FP32: {narrow[:4]}")
    return counts


def load_glm5_next_reference(model_class, model_path, *, dtype, attn_implementation, device):
    """Load the checkpoint with FP32 hyper-connection base/scale, then apply the sampler's FP32 remix, beta and norms."""
    with fp32_hyper_connection_parameters():
        model = model_class.from_pretrained(model_path, dtype=dtype, attn_implementation=attn_implementation)
    apply_sampler_precision(model)
    return model.to(device)
