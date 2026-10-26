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
"""Online redundant experts: copies of the hottest experts in spare slots on the less-loaded EP ranks.

Every MoE call waits for the EP rank that received the most rows. A permutation of experts over ranks cannot split
one hot expert, so each rank here also holds ``spares`` extra expert slots (``ARL_EXPERT_REPLICAS``). Before each
micro-batch's forward, the EP group sums the routed-token counts every layer has seen since the last plan, keeps an
exponential moving average of them, and gives each spare slot a copy of an expert (greedily: the heaviest per-copy
share on the most-loaded rank goes to the least-loaded rank that has room and does not hold it yet, while that lowers
the maximum). A replicated expert's tokens are split evenly over its copies by their position, so the forward, its
checkpoint recompute and the backward route identically under one plan.

The spare slots read the expert's current effective weights (LoRA-merged, or trained in full fine-tuning) from its
owner in every call, through a differentiable all-to-all over the EP group, so their gradients go back to the owner's
weights and nothing is ever stale. Slot ``r * (L + S) + i`` is native expert ``r * L + i`` for ``i < L``, and spare
``i - L`` of rank ``r`` otherwise; DeepEP dispatches to the ``P * (L + S)`` slots as if they were experts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


def plan_replicas(weights: list[float], experts_per_rank: int, ranks: int, spares: int) -> list[list[int]]:
    """Spare-slot assignment ``[rank][spare] -> expert`` (``-1`` empty) for per-expert ``weights`` on the contiguous
    placement (expert ``e`` on rank ``e // experts_per_rank``)."""
    E = len(weights)
    copies = {e: [e // experts_per_rank] for e in range(E)}
    load = [0.0] * ranks
    for e in range(E):
        load[e // experts_per_rank] += weights[e]
    table = [[-1] * spares for _ in range(ranks)]
    free = [spares] * ranks
    for _ in range(ranks * spares):
        r_max = max(range(ranks), key=lambda r: load[r])
        best = None
        held = sorted((e for e in range(E) if r_max in copies[e]), key=lambda e: -weights[e] / len(copies[e]))
        for e in held:
            c = len(copies[e])
            share, new_share = weights[e] / c, weights[e] / (c + 1)
            targets = [r for r in range(ranks) if free[r] and r not in copies[e]]
            if not targets:
                continue
            r_t = min(targets, key=lambda r: load[r])
            # The copies' holders only get lighter, so the maximum drops as long as the target stays below it.
            if load[r_t] + new_share < load[r_max]:
                best = (e, r_t, share, new_share)
                break
        if best is None:
            break
        e, r_t, share, new_share = best
        for r in copies[e]:
            load[r] -= share - new_share
        load[r_t] += new_share
        copies[e].append(r_t)
        table[r_t][spares - free[r_t]] = e
        free[r_t] -= 1
    return table


@dataclass
class ReplicaPlan:
    """One layer's spare table and the device-side remap from logical experts to physical slots."""

    table: list[list[int]]  # [rank][spare] -> expert, -1 empty
    experts_per_rank: int
    spares: int
    copies_count: Tensor  # [E] int64
    copy_slots: Tensor  # [E, max_copies] int64

    @property
    def active(self) -> bool:
        return any(e >= 0 for row in self.table for e in row)

    @property
    def num_slots(self) -> int:
        return len(self.table) * (self.experts_per_rank + self.spares)

    @classmethod
    def build(cls, table: list[list[int]], num_experts: int, experts_per_rank: int, device) -> "ReplicaPlan":
        ranks = len(table)
        spares = len(table[0]) if table else 0
        width = experts_per_rank + spares
        slots: list[list[int]] = [[(e // experts_per_rank) * width + e % experts_per_rank] for e in range(num_experts)]
        for r, row in enumerate(table):
            for j, e in enumerate(row):
                if e >= 0:
                    slots[e].append(r * width + experts_per_rank + j)
        max_copies = max(len(s) for s in slots)
        copy_slots = torch.tensor([s + [s[0]] * (max_copies - len(s)) for s in slots], dtype=torch.long)
        copies_count = torch.tensor([len(s) for s in slots], dtype=torch.long)
        return cls(table, experts_per_rank, spares, copies_count.to(device), copy_slots.to(device))

    def remap(self, selected: Tensor) -> Tensor:
        """Logical expert ids ``[tokens, k]`` (``-1`` kept) -> physical slots, splitting each replicated expert's
        tokens evenly over its copies by their flat position."""
        valid = selected >= 0
        e = selected.clamp(min=0)
        position = torch.arange(selected.numel(), device=selected.device).view_as(selected)
        copy = position % self.copies_count[e]
        slots = self.copy_slots[e, copy]
        return torch.where(valid, slots, selected)

    def fetch(self, weights: tuple[Tensor, ...], group) -> tuple[Tensor, ...]:
        """Each ``weights[k]`` is this rank's ``[L, ...]`` native expert stack; returns ``[L + S, ...]`` stacks with the
        spare slots filled from their owners (zeros where a spare is empty), differentiably."""
        from torch.distributed.nn.functional import all_to_all_single

        ranks = len(self.table)
        me = dist.get_rank(group)
        L, S = self.experts_per_rank, self.spares
        send_rows: list[int] = []
        send_counts = [0] * ranks
        for r in range(ranks):
            for e in self.table[r]:
                if e >= 0 and e // L == me:
                    send_rows.append(e % L)
                    send_counts[r] += 1
        recv_from = [(j, e // L) for j, e in enumerate(self.table[me]) if e >= 0]
        recv_counts = [0] * ranks
        for _, owner in recv_from:
            recv_counts[owner] += 1
        index = torch.tensor(send_rows, dtype=torch.long, device=weights[0].device)
        differentiable = torch.is_grad_enabled() and any(w.requires_grad for w in weights)
        if not differentiable:
            # Frozen weights (e.g. packed MXFP4 codes and their FP32 scales): move raw bytes, whatever the dtypes.
            weights = tuple(w.detach() for w in weights)
            raw = [w.contiguous().view(torch.uint8).reshape(w.shape[0], -1) for w in weights]
            sizes = [r.shape[1] for r in raw]
            packed = torch.cat([r.index_select(0, index) for r in raw], dim=1)
            received = packed.new_empty((len(recv_from), sum(sizes)))
            dist.all_to_all_single(received, packed, recv_counts, send_counts, group=group)
        else:
            sizes = [w[0].numel() for w in weights]
            packed = torch.cat(
                [w.index_select(0, index).reshape(len(send_rows), n) for w, n in zip(weights, sizes)], dim=1
            )
            received = packed.new_empty((len(recv_from), sum(sizes)))
            received = all_to_all_single(received, packed.contiguous(), recv_counts, send_counts, group=group)
        # Rows arrive grouped by source rank in rank order; put each into its spare slot, and an empty spare reads
        # an appended zero row. Every rank's spares go through `received`, so every rank runs the all-to-all's
        # backward collective too, including one whose spares are all empty.
        order = sorted(range(len(recv_from)), key=lambda k: (recv_from[k][1], k))
        spare_rows = [len(recv_from)] * S
        for arrival, k in enumerate(order):
            spare_rows[recv_from[k][0]] = arrival
        padded = torch.cat([received, received.new_zeros(1, received.shape[1])])
        spare = padded.index_select(0, torch.tensor(spare_rows, dtype=torch.long, device=padded.device))
        out = []
        offset = 0
        for w, size in zip(weights, sizes):
            rows = spare[:, offset:offset + size]
            offset += size
            rows = rows.reshape(S, *w.shape[1:]) if differentiable else rows.contiguous().view(w.dtype).reshape(S, *w.shape[1:])
            out.append(torch.cat([w, rows], dim=0))
        return tuple(out)


class ExpertReplicaManager:
    """Plans the spare slots of every registered MoE layer of one EP group, once per micro-batch."""

    def __init__(self, group, spares: int, ema: float):
        self.group = group
        self.spares = spares
        self.ema = ema
        self.layers: list = []
        self.estimates: list[list[float] | None] = []
        self.plans: list[ReplicaPlan | None] = []

    def register(self, layer) -> int:
        self.layers.append(layer)
        self.estimates.append(None)
        self.plans.append(None)
        return len(self.layers) - 1

    def replan(self) -> None:
        counts = torch.stack([layer._replica_counts for layer in self.layers]).double()
        dist.all_reduce(counts, group=self.group)
        counts = counts.cpu().tolist()
        ranks = dist.get_world_size(self.group)
        for i, layer in enumerate(self.layers):
            layer._replica_counts.zero_()
            est = self.estimates[i]
            est = counts[i] if est is None else [self.ema * c + (1 - self.ema) * p for c, p in zip(counts[i], est)]
            self.estimates[i] = est
            if sum(est) == 0:
                continue
            E = len(est)
            table = plan_replicas(est, E // ranks, ranks, self.spares)
            self.plans[i] = ReplicaPlan.build(table, E, E // ranks, layer._replica_counts.device)


_MANAGERS: dict[int, ExpertReplicaManager] = {}


def replica_spares() -> int:
    return int(os.environ.get("ARL_EXPERT_REPLICAS", "0") or 0)


def manager_for(group) -> ExpertReplicaManager:
    key = id(group)
    if key not in _MANAGERS:
        _MANAGERS[key] = ExpertReplicaManager(group, replica_spares(), float(os.environ.get("ARL_EXPERT_REPLICAS_EMA", "0.5")))
    return _MANAGERS[key]
