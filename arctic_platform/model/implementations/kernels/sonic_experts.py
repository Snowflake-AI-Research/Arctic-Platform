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
"""Routed experts on the EP receive buffer through SonicMoE's general-routing kernels.

DeepEP delivers each received token once, with the local experts it chose (``-1`` elsewhere) and their routing
weights. The default path sorts those rows by expert (a full ``[rows, hidden]`` gather), runs gate, up and down as
grouped GEMMs with SwiGLU and the routing weights as separate passes, then scatter-sums the rows back per token,
and mirrors every step in backward. SonicMoE (``kernels/sonicmoe``) gathers the token rows inside its gate|up GEMM,
applies SwiGLU in that GEMM's epilogue, and folds the routing weights and the per-token sum into the down
projection; its backward does the same in reverse and returns the routing-weight gradient. On Qwen3.8's EP8 rank
(194K received tokens, 328K routed rows, 64 experts of 2560 x 640) forward + input-gradient backward takes 12.0 ms
against 26.6 ms, with 5.1 GiB of transient memory against 7.7 GiB.

SonicMoE wants gate and up as one ``[2I, H, E]`` stack (gate rows, then up rows). Frozen experts are restacked once,
in place: ``w1`` and ``w3`` become views of one ``[E, 2I, H]`` buffer, so the stack costs no memory. Trained experts
(full fine-tuning) and PEFT's LoRA-merged expert weights are concatenated per call instead, and SonicMoE's backward
returns their weight gradients.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)
_sonicmoe_import_warned = False


def _warn_sonicmoe_import_failed(exc: BaseException) -> None:
    """One warning when SonicMoE is enabled but cannot be imported and grouped GEMM is used instead."""
    global _sonicmoe_import_warned
    if _sonicmoe_import_warned:
        return
    _sonicmoe_import_warned = True
    logger.warning(
        "moe_use_sonicmoe is enabled, but SonicMoE could not be imported (%s). BF16 routed experts are using "
        "grouped GEMM. Install the sonicmoe extra for quack, cutlass, and cuda.bindings.",
        exc,
    )


def sonic_routed_experts_applicable(experts: nn.Module) -> bool:
    """BF16 SwiGLU experts without clamps, on a Hopper/Blackwell GPU: frozen, trained (full fine-tuning), or with
    PEFT's merged expert LoRA. Not with the unfused LoRA factors (``patches.peft``), whose rows these fused GEMMs
    have nowhere to add."""
    w1 = getattr(experts, "w1", None)
    if w1 is None or not w1.is_cuda or w1.dtype != torch.bfloat16:
        return False
    if getattr(experts, "fp8_block_size", None) is not None or getattr(experts, "swiglu_limit", None) is not None:
        return False
    if not getattr(experts, "use_grouped_mm", False):
        return False
    if getattr(experts, "_ap_lora_ab", None) or getattr(experts, "_dss_lora_ab", None):
        return False
    if torch.cuda.get_device_capability(w1.device)[0] < 9:
        return False
    try:
        from arctic_platform.model.implementations.kernels import sonicmoe  # noqa: F401
    except ImportError as exc:
        _warn_sonicmoe_import_failed(exc)
        return False
    return True


@torch.no_grad()
def _stacked_gate_up(experts: nn.Module) -> torch.Tensor:
    """The ``[E, 2I, H]`` gate|up stack that ``w1`` / ``w3`` are views of, rebuilt when either storage changes."""
    w1, w3 = experts.w1, experts.w3
    stacked = getattr(experts, "_arl_gate_up", None)
    inter = w1.shape[1]
    if (
        stacked is not None
        and w1.data.data_ptr() == stacked.data_ptr()
        and w3.data.data_ptr() == stacked[:, inter:].data_ptr()
    ):
        return stacked
    stacked = torch.empty((w1.shape[0], 2 * inter, w1.shape[2]), dtype=w1.dtype, device=w1.device)
    stacked[:, :inter].copy_(w1.data)
    stacked[:, inter:].copy_(w3.data)
    w1.data = stacked[:, :inter]
    w3.data = stacked[:, inter:]
    experts._arl_gate_up = stacked
    return stacked


def sonic_routed_experts(
    experts: nn.Module, hidden_states: torch.Tensor, expert_indices: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    """``sum_k scores[t, k] * expert_{indices[t, k]}(hidden_states[t])`` over the local experts of each received
    token (``indices`` ``-1`` elsewhere), as ``[tokens, hidden]``. Differentiable in ``hidden_states`` and
    ``scores``, and in the expert weights when they train."""
    from arctic_platform.model.implementations.kernels.sonicmoe import ActivationType
    from arctic_platform.model.implementations.kernels.sonicmoe import moe_general_routing_inputs
    from arctic_platform.model.implementations.kernels.sonicmoe.functional import note_feature_dims

    tokens, topk = expert_indices.shape
    mask = expert_indices >= 0
    token_ids = torch.arange(tokens, device=hidden_states.device).unsqueeze(1).expand(tokens, topk)[mask].int()
    local_ids = expert_indices[mask].int()
    w1, w2, w3 = experts.w1, experts.w2, experts.w3  # LoRA-merged inside the PEFT wrappers
    num_experts, inter, hidden = w1.shape
    note_feature_dims(num_experts, num_experts + 1, inter, 2 * inter, hidden)
    if w1.requires_grad or w3.requires_grad:
        # Trained weights or LoRA deltas take their gradients through the stack, so it is built per call.
        gate_up = torch.cat([w1, w3], dim=1)
    else:
        gate_up = _stacked_gate_up(experts)
    out, _ = moe_general_routing_inputs(
        hidden_states,
        scores[mask],
        token_ids,
        local_ids,
        gate_up.permute(1, 2, 0),
        None,
        w2.permute(1, 2, 0),
        None,
        w1.shape[0],
        torch.cuda.current_stream().cuda_stream,
        ActivationType.SWIGLU,
        not torch.is_grad_enabled(),
        True,  # concat layout: gate rows, then up rows
    )
    return out
