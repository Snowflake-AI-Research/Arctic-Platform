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
"""One input-projection GEMM and grouped value heads for the stock transformers Qwen3.5 GatedDeltaNet.

The stock forward projects the layer input four times, ``in_proj_qkv``, ``in_proj_z``, ``in_proj_b`` and
``in_proj_a``: four GEMMs that each read the whole ``[tokens, hidden]`` input, two of them only 16-48 columns wide,
and four input-gradient GEMMs plus three adds in backward. It then expands q and k to the value-head count with
``repeat_interleave`` before the recurrence. Here the four projections are one GEMM over the concatenated weights
(``ConcatLinear``: the concatenation is rebuilt in backward instead of being kept), and q and k go to the rule with
their own head count when the rule groups value heads itself (FLA's: value head ``hv`` reads key head
``hv // (HV // H)``, the assignment ``repeat_interleave`` builds). The math is the stock forward's; only the GEMMs'
shapes, and so their accumulation tiling, differ.

``apply_fused_gated_delta_net`` replaces the forward of every stock dense ``Qwen3_5GatedDeltaNet`` of a model without
sequence parallelism, which projects in two GEMMs (``project_dense_gated_delta_net_inputs``: ``in_proj_qkv`` alone,
whose output the causal convolution saves, and ``z | b | a``); the head-parallel forward of ``sp/gated_delta_net.py``
packs its projections for the all-to-all and uses the one-GEMM ``project_gated_delta_net_inputs``. A projection that is not a plain bias-free ``nn.Linear``, such as one a PEFT adapter wraps, is called on its
own, as in the stock forward. A call with a cache (generation) runs the stock forward.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from torch import nn

_FUSED_ATTRIBUTE = "_arctic_fused_gated_delta_net"


class ConcatLinear(torch.autograd.Function):
    """``x @ cat(weights).T`` that saves the weights, not their concatenation, for backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, *weights: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, *weights)
        return F.linear(x, torch.cat(weights, dim=0))

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, *weights = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.matmul(grad_out, torch.cat(weights, dim=0))
        grad_weights: list[torch.Tensor | None] = [None] * len(weights)
        if any(ctx.needs_input_grad[1:]):
            grad_all = torch.matmul(grad_out.reshape(-1, grad_out.shape[-1]).t(), x.reshape(-1, x.shape[-1]))
            for index, part in enumerate(torch.split(grad_all, [int(w.shape[0]) for w in weights], dim=0)):
                if ctx.needs_input_grad[index + 1]:
                    grad_weights[index] = part
        return grad_x, *grad_weights


def _plain_linear(module: nn.Module) -> bool:
    return type(module) is nn.Linear and module.bias is None


def project_gated_delta_net_inputs(
    module: nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(in_proj_qkv(x), in_proj_z(x), in_proj_b(x), in_proj_a(x))``, as one GEMM when all four are plain linears."""
    projections = (module.in_proj_qkv, module.in_proj_z, module.in_proj_b, module.in_proj_a)
    if not all(_plain_linear(projection) for projection in projections):
        return tuple(projection(hidden_states) for projection in projections)  # type: ignore[return-value]
    weights = [projection.weight for projection in projections]
    if len({w.dtype for w in weights}) != 1:
        return tuple(projection(hidden_states) for projection in projections)  # type: ignore[return-value]
    mixed = ConcatLinear.apply(hidden_states, *weights)
    return tuple(torch.split(mixed, [int(w.shape[0]) for w in weights], dim=-1))  # type: ignore[return-value]


def project_dense_gated_delta_net_inputs(
    module: nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The four projections as two GEMMs: ``in_proj_qkv`` on its own and ``z | b | a`` in one ``ConcatLinear``.

    Without sequence parallelism the causal convolution saves its input for backward. A view of one fused
    ``qkv | z | b | a`` output would keep the z, b and a columns alive with it, z a second time next to the copy the
    gated norm saves (+0.25-0.5 GiB per layer at 64K tokens); its own GEMM output is the tensor the stock forward
    saves there. The small b and a GEMMs stay merged into z's.
    """
    projections = (module.in_proj_z, module.in_proj_b, module.in_proj_a)
    if (
        not all(_plain_linear(projection) for projection in projections)
        or len({p.weight.dtype for p in projections}) != 1
    ):
        return project_gated_delta_net_inputs(module, hidden_states)
    weights = [projection.weight for projection in projections]
    z, b, a = torch.split(ConcatLinear.apply(hidden_states, *weights), [int(w.shape[0]) for w in weights], dim=-1)
    return module.in_proj_qkv(hidden_states), z, b, a


def _dense_forward(self, hidden_states: torch.Tensor, cache_params=None, attention_mask=None, **kwargs):
    """The stock forward's training path (no cache) with one input projection and grouped value heads."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    from arctic_platform.model.implementations.gpu.sp.gated_delta_net import repeat_query_key_for_value_heads

    if cache_params is not None:
        return type(self).forward(
            self, hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs
        )
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv, z, b, a = project_dense_gated_delta_net_inputs(self, hidden_states)
    mixed_qkv = self._arctic_causal_conv1d_fn(
        mixed_qkv.transpose(1, 2),
        self.conv1d.weight.squeeze(1),
        self.conv1d.bias,
        activation=self.activation,
    ).transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
    # The stock forward's beta is ``b.sigmoid()``; the fp32 parity patch's hook on ``in_proj_b`` makes that b fp32.
    # This forward does not call ``in_proj_b`` when the projection is fused, so it takes the setting itself.
    beta = b.float().sigmoid() if self._arctic_fp32_beta else b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    rule = self._arctic_chunk_gated_delta_rule
    query, key = repeat_query_key_for_value_heads(query, key, self.num_k_heads, self.num_v_heads, rule)
    core_attn_out, _ = rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=kwargs.get("cu_seq_lens_q"),
    )
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


def apply_fused_gated_delta_net(model: nn.Module, *, fp32_beta: bool) -> int:
    """Give every stock dense Qwen3.5 GatedDeltaNet of ``model`` the fused forward; return how many.

    ``fp32_beta`` evaluates the beta gate as a sigmoid of fp32 ``b`` (the fp32 parity setting), ``False`` in ``b``'s own
    dtype, as the stock forward does. The kernels are the ones the stock forward resolves (causal-conv1d and FLA, or
    the transformers fallbacks where they are not installed). A module given the forward before is skipped.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    from arctic_platform.model.implementations.gpu.sp.gated_delta_net import _resolve_gated_delta_net_kernel

    changed = 0
    for module in model.modules():
        if type(module) is not Qwen3_5GatedDeltaNet or getattr(module, _FUSED_ATTRIBUTE, False):
            continue
        convolution = _resolve_gated_delta_net_kernel(module, "causal_conv1d_fn")
        rule = _resolve_gated_delta_net_kernel(module, "chunk_gated_delta_rule")
        module._arctic_causal_conv1d_fn = convolution
        module._arctic_chunk_gated_delta_rule = rule
        module._arctic_fp32_beta = fp32_beta
        module.forward = types.MethodType(_dense_forward, module)
        setattr(module, _FUSED_ATTRIBUTE, True)
        changed += 1
    return changed
