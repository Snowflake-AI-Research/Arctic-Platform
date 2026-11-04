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
"""UltraEP spare expert slots for frozen block-FP8 routed experts trained with expert LoRA (``ARL_ULTRAEP_SPARES``),
the block-FP8 path of ``moe/ultraep_replicas.py``'s ``ultraep_plan`` (which also owns the spare count).

UltraEP (https://github.com/Dots-Infra/UltraEP) plans, every MoE call, which of the call's hottest experts get a copy
in each EP rank's ``S`` spare slots, reroutes tokens to the copies, and moves the copies' weights and gradients over
NVLink. EP rank ``r`` owns physical slots ``r * (L + S) ..``: its ``L`` native experts, then ``S`` spares.

Here the experts' base weights are frozen E4M3 codes with their 128 x 128 FP32 block scales
(``GroupedExperts(fp8_block_size=128)``, GLM-5.3's FP8 release) and the trained parameters are the experts' LoRA
factors, so a copy carries both:
- the codes and block scales of w1 | w3 (one UltraEP manager, weights and weight scales as its two blocks) and of w2
  (a second manager, in halves) are copied with ``weight_sync`` only;
- the LoRA factors (``A`` ``[r, in]``, ``B`` ``[out, r]`` per expert and projection) are staged expert-major (PEFT
  keeps ``B`` as ``[out, r * E]`` with the expert index fastest) and copied by a trainable manager, whose
  ``grad_reduce`` returns the copies' FP32 gradients to their owners: the fetch is an autograd node, so DeepSpeed's
  ZeRO-2 hooks see the native factors' complete gradient.
A recompute reuses its forward's plan and physical routes, so forward, recompute and backward route identically.

UltraEP's pointer pools read each layer's native stacks in place. With ``moe/frozen_offload.py`` the codes exist only
during a call of the experts, as that call's copy in the layer's persistent buffer (by layer parity), at the same
addresses in every call; so a layer's pools are built at its first fetch, inside that call.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from .ultraep_replicas import ultraep_spares

# Placement maps and pointer pools are sized per layer; GLM-5.3 has 75 MoE layers.
_MAX_LAYERS = 128
_NAMES = ("w1", "w2", "w3")


def _lora_layout(ab: dict, experts: int) -> dict:
    """Per projection: (rank, in, out) of its expert LoRA factors in PEFT's ParamWrapper layout."""
    shapes = {}
    for name, (weight_a, weight_b, rank, _scaling) in ab.items():
        shapes[name] = (rank, weight_a.shape[1], weight_b.shape[0])
        if weight_a.shape[0] != experts * rank or weight_b.shape[1] != experts * rank:
            raise ValueError(
                f"{name}: LoRA factors {tuple(weight_a.shape)} / {tuple(weight_b.shape)} for {experts} experts"
            )
    return shapes


def _stage(weight_a: Tensor, weight_b: Tensor, rank: int, experts: int) -> Tensor:
    """``[E, r * in + out * r]``: each expert's ``A_e`` then its ``B_e``, contiguous."""
    a = weight_a.reshape(experts, -1)
    b = weight_b.reshape(weight_b.shape[0], rank, experts).permute(2, 0, 1).reshape(experts, -1)
    return torch.cat([a, b], dim=1)


def _unstage(staged: Tensor, rank: int, width_in: int, width_out: int) -> tuple[Tensor, Tensor]:
    """Inverse of ``_stage`` for ``staged`` ``[E, ...]``: ``(A [E * r, in], B [out, r * E])``."""
    experts = staged.shape[0]
    a = staged[:, : rank * width_in].reshape(experts * rank, width_in)
    b = (
        staged[:, rank * width_in :]
        .reshape(experts, width_out, rank)
        .permute(1, 2, 0)
        .reshape(width_out, rank * experts)
    )
    return a, b


def _local(t: Tensor) -> Tensor:
    return t.to_local() if hasattr(t, "to_local") else t


def _stacks(experts) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """The experts' code stacks ``(w1, w2, w3)`` and block scales as they are now."""
    return (
        tuple(_local(getattr(experts, name)) for name in _NAMES),
        tuple(_local(getattr(experts, f"{name}_scale_inv")) for name in _NAMES),
    )


class FrozenUltraEPState:
    """The UltraEP managers of one EP group's frozen block-FP8 experts, their LoRA staging, and each layer's index."""

    def __init__(self, group, experts, ab: dict, spares: int) -> None:
        import ultra_ep

        self.group = group
        self.spares = spares
        self.ranks = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        (w1, w2, w3), (s1, s2, s3) = _stacks(experts)
        if w1.numel() == 0:
            raise RuntimeError(
                "UltraEP: the experts' codes are not on the GPU (frozen expert offload); its state is built at the "
                "first MoE call, before the offload moves them"
            )
        L = w1.shape[0]
        self.experts_per_rank = L
        self.n2_half = w2[0].numel() // 2
        self.s2_half = s2[0].numel() // 2
        frozen = dict(
            group=group,
            num_layers=_MAX_LAYERS,
            num_local_master_experts=L,
            num_local_redundant_experts=spares,
            is_train=False,
            weight_data_dtype=w1.dtype,
            weight_scale_dtype=s1.dtype,
        )
        self.gate_up = ultra_ep.Manager(
            expert_fc1_numel=w1[0].numel(),
            expert_fc2_numel=w3[0].numel(),
            expert_fc1_weight_scale_numel=s1[0].numel(),
            expert_fc2_weight_scale_numel=s3[0].numel(),
            **frozen,
        )
        self.down = ultra_ep.Manager(
            expert_fc1_numel=self.n2_half,
            expert_fc2_numel=w2[0].numel() - self.n2_half,
            expert_fc1_weight_scale_numel=self.s2_half,
            expert_fc2_weight_scale_numel=s2[0].numel() - self.s2_half,
            **frozen,
        )
        self.lora_shapes = _lora_layout(ab, L)
        self.lora_names = [name for name in ("w1", "w3", "w2") if name in ab]
        self.lora: Any = None
        if self.lora_names:
            widths = {
                name: rank * (width_in + width_out) for name, (rank, width_in, width_out) in self.lora_shapes.items()
            }
            self.lora_width = widths
            # fc1 carries w1's factors and w2's; fc2 w3's (zero-width projections are padded to keep two blocks).
            self.fc1_names = [n for n in ("w1", "w2") if n in ab]
            self.fc2_names = [n for n in ("w3",) if n in ab]
            self.fc1_numel = sum(widths[n] for n in self.fc1_names)
            self.fc2_numel = max(sum(widths[n] for n in self.fc2_names), 8)
            dtype = next(iter(ab.values()))[0].dtype
            self.lora = ultra_ep.Manager(
                group=group,
                num_layers=_MAX_LAYERS,
                num_local_master_experts=L,
                num_local_redundant_experts=spares,
                expert_fc1_numel=self.fc1_numel,
                expert_fc2_numel=self.fc2_numel,
                is_train=True,
                weight_data_dtype=dtype,
                grad_dtype=torch.float32,
            )
            # One staging pair serves every layer: a layer's sync and reduce finish before the next layer's start.
            width = max(self.fc1_numel, self.fc2_numel)
            self.lora_stage = torch.zeros(L, 2, width, dtype=dtype, device=w1.device)
            self.lora_grads = torch.zeros(L, 2, width, dtype=torch.float32, device=w1.device)
        self.layer_of: dict[int, int] = {}
        self.pooled: set[int] = set()
        self.kept_routes: dict[int, deque] = {}

    def layer(self, module) -> int:
        key = id(module)
        if key not in self.layer_of:
            lid = len(self.layer_of)
            if lid >= _MAX_LAYERS:
                raise RuntimeError(f"UltraEP: more than {_MAX_LAYERS} MoE layers")
            self.layer_of[key] = lid
        return self.layer_of[key]

    def ensure_pools(self, lid: int, experts) -> None:
        """Point UltraEP at the layer's expert stacks at their first fetch, inside a call of the experts: with the
        frozen expert offload the codes are that call's copy, which every later call of the layer finds at the same
        addresses."""
        if lid in self.pooled:
            return
        self.pooled.add(lid)
        L = self.experts_per_rank
        (w1, w2, w3), (s1, s2, s3) = _stacks(experts)
        self.gate_up.construct_local_master_ptr_pool(
            lid,
            [w1[e] for e in range(L)],
            [w3[e] for e in range(L)],
            fc1_weight_scales=[s1[e] for e in range(L)],
            fc2_weight_scales=[s3[e] for e in range(L)],
        )
        flat_w2, flat_s2 = w2.reshape(L, -1), s2.reshape(L, -1)
        self.down.construct_local_master_ptr_pool(
            lid,
            [flat_w2[e, : self.n2_half] for e in range(L)],
            [flat_w2[e, self.n2_half :] for e in range(L)],
            fc1_weight_scales=[flat_s2[e, : self.s2_half] for e in range(L)],
            fc2_weight_scales=[flat_s2[e, self.s2_half :] for e in range(L)],
        )
        if self.lora is not None:
            self.lora.construct_local_master_ptr_pool(
                lid,
                [self.lora_stage[e, 0, : self.fc1_numel] for e in range(L)],
                [self.lora_stage[e, 1, : self.fc2_numel] for e in range(L)],
                [self.lora_grads[e, 0, : self.fc1_numel] for e in range(L)],
                [self.lora_grads[e, 1, : self.fc2_numel] for e in range(L)],
            )

    def maps(self, lid: int) -> dict:
        return dict(
            physical_to_logical_map=self.gate_up.physical_to_logical_map[lid],
            logical_to_physical_map=self.gate_up.logical_to_physical_map[lid],
            logical_replica_counts=self.gate_up.logical_replica_counts[lid],
        )


class _LoraFetch(torch.autograd.Function):
    """Native expert LoRA factors -> the ``L + S`` slots' factors (PEFT layout); the backward returns the spares'
    gradients to their owners through UltraEP's ``grad_reduce``."""

    @staticmethod
    def forward(ctx, state: FrozenUltraEPState, lid: int, names: tuple, *factors: Tensor):
        L, S = state.experts_per_rank, state.spares
        staged = {}
        for i, name in enumerate(names):
            rank, _, _ = state.lora_shapes[name]
            staged[name] = _stage(factors[2 * i], factors[2 * i + 1], rank, L)
        state.lora_stage[:, 0, : state.fc1_numel] = torch.cat([staged[n] for n in state.fc1_names], dim=1)
        if state.fc2_names:
            state.lora_stage[:, 1, : sum(state.lora_width[n] for n in state.fc2_names)] = torch.cat(
                [staged[n] for n in state.fc2_names], dim=1
            )
        state.lora.weight_sync(lid, **state.maps(lid))
        spare_fc1 = state.lora.local_replica_fc1_weight_buffer.view(S, -1)
        spare_fc2 = state.lora.local_replica_fc2_weight_buffer.view(S, -1)
        out: list[Tensor] = []
        for name in names:
            rank, width_in, width_out = state.lora_shapes[name]
            if name in state.fc1_names:
                block, names_in_block = spare_fc1, state.fc1_names
            else:
                block, names_in_block = spare_fc2, state.fc2_names
            start = sum(state.lora_width[n] for n in names_in_block[: names_in_block.index(name)])
            spare = block[:, start : start + state.lora_width[name]]
            full = torch.cat([staged[name], spare], dim=0)
            out.extend(_unstage(full, rank, width_in, width_out))
        ctx.state, ctx.lid, ctx.names = state, lid, names
        ctx.dtype = factors[0].dtype
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads: Tensor):
        state, lid, names = ctx.state, ctx.lid, ctx.names
        L, S = state.experts_per_rank, state.spares
        device = state.lora_grads.device
        staged_grads = {}
        for i, name in enumerate(names):
            rank, width_in, width_out = state.lora_shapes[name]
            ga = grads[2 * i]
            if ga is None:
                ga = torch.zeros(((L + S) * rank, width_in), device=device)
            gb = grads[2 * i + 1]
            if gb is None:
                gb = torch.zeros((width_out, rank * (L + S)), device=device)
            staged_grads[name] = _stage(ga.float(), gb.float(), rank, L + S)
        fc1 = torch.cat([staged_grads[n][L:] for n in state.fc1_names], dim=1)
        state.lora.local_replica_fc1_grad_buffer.copy_(fc1.reshape(S, -1))
        if state.fc2_names:
            fc2 = torch.cat([staged_grads[n][L:] for n in state.fc2_names], dim=1)
            state.lora.local_replica_fc2_grad_buffer.view(S, -1)[:, : fc2.shape[1]].copy_(fc2)
        state.lora_grads.zero_()
        state.lora.grad_reduce(lid, **state.maps(lid))
        out: list[Tensor | None] = [None, None, None]
        for name in names:
            rank, width_in, width_out = state.lora_shapes[name]
            names_in_block, row = (state.fc1_names, 0) if name in state.fc1_names else (state.fc2_names, 1)
            start = sum(state.lora_width[n] for n in names_in_block[: names_in_block.index(name)])
            native = staged_grads[name][:L] + state.lora_grads[:, row, start : start + state.lora_width[name]]
            ga, gb = _unstage(native, rank, width_in, width_out)
            out.extend([ga.to(ctx.dtype), gb.to(ctx.dtype)])
        return tuple(out)


class FrozenUltraEPPlan:
    """One MoE call's UltraEP placement for frozen block-FP8 experts: physical routes and the spares' weights."""

    def __init__(self, state: FrozenUltraEPState, lid: int, building: bool, recompute: bool) -> None:
        self.state = state
        self.lid = lid
        self.building = building
        self.recompute = recompute
        self.cache: tuple | None = None

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

    def extend(self, experts, weights: tuple, scales: tuple, ab: dict) -> tuple:
        """The call's ``L + S`` slot stacks: codes, block scales and LoRA factors, native experts first. One plan
        serves every dispatch chunk of its MoE call, so the spares are fetched once."""
        if self.cache is not None:
            return self.cache
        state, lid, S = self.state, self.lid, self.state.spares
        state.ensure_pools(lid, experts)
        maps = state.maps(lid)
        state.gate_up.weight_sync(lid, **maps)
        state.down.weight_sync(lid, **maps)
        w1, w2, w3 = weights
        gu, dn = state.gate_up, state.down
        spare_w1 = gu.local_replica_fc1_weight_buffer.view(S, *w1.shape[1:])
        spare_w3 = gu.local_replica_fc2_weight_buffer.view(S, *w3.shape[1:])
        spare_w2 = torch.cat(
            [dn.local_replica_fc1_weight_buffer.view(S, -1), dn.local_replica_fc2_weight_buffer.view(S, -1)], 1
        ).view(S, *w2.shape[1:])
        ext_w = (torch.cat([w1, spare_w1]), torch.cat([w2, spare_w2]), torch.cat([w3, spare_w3]))
        s1, s2, s3 = scales
        spare_s1 = gu.local_replica_fc1_weight_scale_buffer.view(S, *s1.shape[1:])
        spare_s3 = gu.local_replica_fc2_weight_scale_buffer.view(S, *s3.shape[1:])
        spare_s2 = torch.cat(
            [
                dn.local_replica_fc1_weight_scale_buffer.view(S, -1),
                dn.local_replica_fc2_weight_scale_buffer.view(S, -1),
            ],
            1,
        ).view(S, *s2.shape[1:])
        ext_s = (torch.cat([s1, spare_s1]), torch.cat([s2, spare_s2]), torch.cat([s3, spare_s3]))
        ext_ab = {}
        if ab:
            names = tuple(state.lora_names)
            factors = [t for n in names for t in ab[n][:2]]
            fetched = _LoraFetch.apply(state, lid, names, *factors)
            for i, n in enumerate(names):
                _, _, rank, scaling = ab[n]
                ext_ab[n] = (fetched[2 * i], fetched[2 * i + 1], rank, scaling)
        self.cache = (ext_w, ext_s, ext_ab)
        return self.cache


_STATES: dict[int, FrozenUltraEPState] = {}


def frozen_ultraep_plan(
    module, experts, group, selected: Tensor, ab: dict, building: bool
) -> FrozenUltraEPPlan:
    """This MoE call's plan: planned from ``selected`` unless the call is a recompute (``building`` False), which
    reuses its forward's placement and routing."""
    key = id(group)
    if key not in _STATES:
        _STATES[key] = FrozenUltraEPState(group, experts, ab, ultraep_spares())
    state = _STATES[key]
    lid = state.layer(module)
    if building:
        if torch.is_grad_enabled():  # a graph-building forward: whatever its last one kept has been recomputed
            state.kept_routes.setdefault(lid, deque()).clear()
        state.gate_up.update_placement_sparse(lid, selected.to(torch.int64).contiguous())
    return FrozenUltraEPPlan(state, lid, building=building and torch.is_grad_enabled(), recompute=not building)
