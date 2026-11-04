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
"""Online redundant experts through UltraEP (https://github.com/Dots-Infra/UltraEP): ``S`` spare slots per GPU, from
the job's ``loader_options.ultraep_spares`` or, when that is unset, ``ARL_ULTRAEP_SPARES=S``.

One option serves every kind of routed experts; ``ultraep_plan`` picks the path from the experts themselves:
- frozen quantized experts, block FP8 (``GroupedExperts(fp8_block_size=...)``) or packed 4-bit
  (``GroupedExperts(packed4=...)``: INT4 / W4A8, NVFP4, MXFP4), always frozen and trained through expert LoRA or not
  at all, take ``moe/ultraep_frozen.py``, which copies their codes, scales, per-expert scalars and LoRA factors;
- BF16 experts take this module, which copies the expert parameters as they are stored: experts trained in full
  (full fine-tuning), or frozen ones without an adapter. Expert LoRA on BF16 experts is refused, as the copies would
  carry the base weights without it.

Same slot layout as ``expert_replicas.py`` (EP rank r owns physical slots ``r * (L + S) ..``: its L native experts,
then S spares), but UltraEP plans, routes and moves the copies on the GPU:
- every graph-building or no-grad forward plans its spares from that call's exact routing
  (``update_placement_sparse``: one small all-gather of loads over the EP group, solved on the device); a
  recompute reuses its forward's plan and physical routes, so the forward, the recompute and the backward route
  identically;
- ``reroute_sparse`` maps logical top-k ids to physical slots in place;
- ``weight_sync`` writes the native experts' weights into their copies' spare slots over NVLink (symmetric memory,
  copies stay inside the owner's NVLink domain), and ``grad_reduce`` sums the copies' FP32 gradients back.
DeepSpeed ZeRO-2 hooks each expert parameter's gradient as autograd accumulates it, so the copies' gradients must
be part of that gradient: the fetch below is an autograd node whose backward runs ``grad_reduce`` into an FP32
staging buffer and returns the native rows' gradient plus the copies'. UltraEP's master pointer pools hold two
contiguous blocks per expert and read the parameters in place (ZeRO-2 keeps them at fixed addresses; ZeRO-3, which
frees them between uses, is refused); Arctic keeps w1, w3 and w2 apart, so one manager carries (w1, w3) and a second
one, driven by the first one's placement, carries w2 split in halves.
"""

from __future__ import annotations

import os
from collections import deque

import torch
import torch.distributed as dist
from torch import Tensor

ULTRAEP_SPARES_ENV = "ARL_ULTRAEP_SPARES"
# Placement maps and pointer pools are sized per layer; GLM-5.3 has 75 MoE layers, Qwen3.8 48.
_MAX_LAYERS = 128
_NAMES = ("w1", "w2", "w3")

_configured: int | None = None  # loader option moe.ultraep_spares (moe/runtime_options.py); None: the env


def configure_from_options(value: int | None) -> None:
    global _configured
    _configured = value


def ultraep_spares() -> int:
    """Spare slots per GPU for both UltraEP paths: the job's ``ultraep_spares``, else ``ARL_ULTRAEP_SPARES``."""
    if _configured is not None:
        return _configured
    return int(os.environ.get(ULTRAEP_SPARES_ENV, "0") or 0)


def frozen_quantized(experts) -> bool:
    """Experts whose base weights are frozen quantized codes (block FP8 or packed 4-bit), the ``ultraep_frozen`` path."""
    return getattr(experts, "fp8_block_size", None) is not None or getattr(experts, "packed4", None) is not None


def _local(t: Tensor) -> Tensor:
    return t.to_local() if hasattr(t, "to_local") else t


def _check_copyable(experts) -> None:
    """BF16 experts whose parameters UltraEP can copy as stored: no expert LoRA, and parameters at fixed addresses."""
    parametrized = getattr(experts, "parametrizations", None)
    adapted = [n for n in _NAMES if parametrized is not None and n in parametrized]
    if adapted or getattr(experts, "_ap_lora_ab", None) or getattr(experts, "_dss_lora_ab", None):
        raise NotImplementedError(
            "UltraEP's spare slots copy BF16 routed experts' parameters as stored, without their expert LoRA: train "
            "BF16 experts in full, or use the block-FP8 checkpoint, whose frozen experts take expert LoRA with spares"
        )
    if any(hasattr(getattr(experts, n), "ds_status") for n in _NAMES):
        raise NotImplementedError(
            "UltraEP's spare slots read the expert parameters in place, which ZeRO-3 partitions and frees between "
            "uses; train with ZeRO stage 1 or 2"
        )


class UltraEPState:
    """The two UltraEP managers of one EP group, the FP32 staging for the native experts' gradients, and the
    layer index of every MoE module (in first-forward order)."""

    def __init__(self, group, experts_per_rank: int, spares: int, w1: Tensor, w2: Tensor) -> None:
        import ultra_ep

        self.group = group
        self.experts_per_rank = experts_per_rank
        self.spares = spares
        self.ranks = dist.get_world_size(group)
        n13 = w1[0].numel()
        n2 = w2[0].numel()
        self.n2_half = n2 // 2
        common = dict(
            group=group,
            num_layers=_MAX_LAYERS,
            num_local_master_experts=experts_per_rank,
            num_local_redundant_experts=spares,
            is_train=True,
            weight_data_dtype=w1.dtype,
            grad_dtype=torch.float32,
        )
        self.gate_up = ultra_ep.Manager(expert_fc1_numel=n13, expert_fc2_numel=n13, **common)
        self.down = ultra_ep.Manager(expert_fc1_numel=self.n2_half, expert_fc2_numel=n2 - self.n2_half, **common)
        # One staging buffer serves every layer: a layer's grad_reduce finishes before the next layer's starts.
        self.gate_up_grads = torch.zeros(experts_per_rank, 2, n13, dtype=torch.float32, device=w1.device)
        self.down_grads = torch.zeros(experts_per_rank, n2, dtype=torch.float32, device=w1.device)
        self.layer_of: dict[int, int] = {}
        # Each layer's physical top-k ids from its graph-building forwards, oldest first: reroute_sparse keeps
        # per-expert counters across calls, so rerouting the same tokens again in the recompute could send them to
        # different copies and change the shapes the recompute rebuilds.
        self.kept_routes: dict[int, deque] = {}

    def layer(self, module, w1: Tensor, w2: Tensor, w3: Tensor) -> int:
        key = id(module)
        if key not in self.layer_of:
            lid = len(self.layer_of)
            if lid >= _MAX_LAYERS:
                raise RuntimeError(f"UltraEP: more than {_MAX_LAYERS} MoE layers")
            L = self.experts_per_rank
            self.gate_up.construct_local_master_ptr_pool(
                lid,
                [w1[e] for e in range(L)],
                [w3[e] for e in range(L)],
                [self.gate_up_grads[e, 0] for e in range(L)],
                [self.gate_up_grads[e, 1] for e in range(L)],
            )
            flat_w2 = w2.reshape(L, -1)
            self.down.construct_local_master_ptr_pool(
                lid,
                [flat_w2[e, : self.n2_half] for e in range(L)],
                [flat_w2[e, self.n2_half :] for e in range(L)],
                [self.down_grads[e, : self.n2_half] for e in range(L)],
                [self.down_grads[e, self.n2_half :] for e in range(L)],
            )
            self.layer_of[key] = lid
        return self.layer_of[key]

    def maps(self, lid: int) -> dict:
        return dict(
            physical_to_logical_map=self.gate_up.physical_to_logical_map[lid],
            logical_to_physical_map=self.gate_up.logical_to_physical_map[lid],
            logical_replica_counts=self.gate_up.logical_replica_counts[lid],
        )


class _UltraEPFetch(torch.autograd.Function):
    """``[L, ...]`` native expert stacks -> ``[L + S, ...]`` with this rank's spare slots filled by ``weight_sync``;
    the backward reduces the spares' gradients into the native experts' (``grad_reduce``)."""

    @staticmethod
    def forward(ctx, state: UltraEPState, lid: int, w1: Tensor, w2: Tensor, w3: Tensor):
        S = state.spares
        state.gate_up.weight_sync(lid)
        state.down.weight_sync(lid, **state.maps(lid))
        spare_w1 = state.gate_up.local_replica_fc1_weight_buffer.view(S, *w1.shape[1:])
        spare_w3 = state.gate_up.local_replica_fc2_weight_buffer.view(S, *w3.shape[1:])
        down = state.down.local_replica_weight_buffer
        spare_w2 = down.reshape(S, *w2.shape[1:])
        ctx.state, ctx.lid = state, lid
        return torch.cat([w1, spare_w1]), torch.cat([w2, spare_w2]), torch.cat([w3, spare_w3])

    @staticmethod
    def backward(ctx, g1: Tensor, g2: Tensor, g3: Tensor):
        state, lid, L = ctx.state, ctx.lid, ctx.state.experts_per_rank
        S, half = state.spares, state.n2_half
        state.gate_up.local_replica_fc1_grad_buffer.copy_(g1[L:].reshape(S, -1))
        state.gate_up.local_replica_fc2_grad_buffer.copy_(g3[L:].reshape(S, -1))
        spare_g2 = g2[L:].reshape(S, -1)
        state.down.local_replica_fc1_grad_buffer.copy_(spare_g2[:, :half])
        state.down.local_replica_fc2_grad_buffer.copy_(spare_g2[:, half:])
        state.gate_up_grads.zero_()
        state.down_grads.zero_()
        state.gate_up.grad_reduce(lid)
        state.down.grad_reduce(lid, **state.maps(lid))
        # Summed in FP32, returned in the parameters' dtype.
        grad_w1 = (state.gate_up_grads[:, 0].view_as(g1[:L]) + g1[:L]).to(g1.dtype)
        grad_w3 = (state.gate_up_grads[:, 1].view_as(g3[:L]) + g3[:L]).to(g3.dtype)
        grad_w2 = (state.down_grads.view_as(g2[:L]) + g2[:L]).to(g2.dtype)
        return None, None, grad_w1, grad_w2, grad_w3


class UltraEPPlan:
    """One MoE call's UltraEP placement, with the interface of ``expert_replicas.ReplicaPlan``."""

    def __init__(self, state: UltraEPState, lid: int, building: bool, recompute: bool) -> None:
        self.state = state
        self.lid = lid
        self.building = building
        self.recompute = recompute

    @property
    def num_slots(self) -> int:
        return self.state.ranks * (self.state.experts_per_rank + self.state.spares)

    def remap(self, selected: Tensor) -> Tensor:
        kept = self.state.kept_routes.setdefault(self.lid, deque())
        if self.recompute:
            if not kept:
                raise RuntimeError("UltraEP: a recompute found no routing from its forward")
            physical = kept.popleft()
            if physical.shape != selected.shape:
                raise RuntimeError(
                    f"UltraEP: recompute routes {tuple(selected.shape)}, its forward {tuple(physical.shape)}"
                )
            return physical
        physical = selected.to(torch.int64).contiguous().clone()
        self.state.gate_up.reroute_sparse(self.lid, physical)
        if self.building:
            kept.append(physical)
        return physical

    def fetch(self, weights: tuple[Tensor, ...], group) -> tuple[Tensor, ...]:
        w1, w2, w3 = weights
        return _UltraEPFetch.apply(self.state, self.lid, w1, w2, w3)


_STATES: dict[int, UltraEPState] = {}


def ultraep_plan(module, experts, group, selected: Tensor, building: bool):
    """This MoE call's plan: planned from ``selected`` unless the call is a recompute (``building`` False), which
    reuses its forward's placement and routing. Frozen quantized experts (block FP8, packed 4-bit) get
    ``ultraep_frozen.FrozenUltraEPPlan``; BF16 experts an ``UltraEPPlan``."""
    if frozen_quantized(experts):
        from .ultraep_frozen import frozen_ultraep_plan

        ab = getattr(experts, "_ap_lora_ab", getattr(experts, "_dss_lora_ab", {}))
        return frozen_ultraep_plan(module, experts, group, selected, ab, building)
    w1, w2, w3 = (_local(getattr(experts, name)) for name in _NAMES)
    state = _STATES.get(id(group))
    if state is None or id(module) not in state.layer_of:
        _check_copyable(experts)  # before UltraEP points its pools at a new layer's parameters
    if state is None:
        state = _STATES[id(group)] = UltraEPState(group, w1.shape[0], ultraep_spares(), w1, w2)
    lid = state.layer(module, w1, w2, w3)
    if building:
        if torch.is_grad_enabled():  # a graph-building forward: whatever its last one kept has been recomputed
            state.kept_routes.setdefault(lid, deque()).clear()
        state.gate_up.update_placement_sparse(lid, selected.to(torch.int64).contiguous())
    return UltraEPPlan(state, lid, building=building and torch.is_grad_enabled(), recompute=not building)
