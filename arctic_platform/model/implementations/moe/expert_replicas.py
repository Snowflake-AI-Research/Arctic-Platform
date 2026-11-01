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
one hot expert, so each rank here also holds ``spares`` extra expert slots (``loader_options.expert_replicas``, or
``ARL_EXPERT_REPLICAS`` when that is unset). Before each
micro-batch's forward, the EP group sums the routed-token counts every layer has seen since the last plan, keeps an
exponential moving average of them, and spreads each expert's expected load over its owner and spare-slot copies
(``plan_replicas``: a copy can take part or all of an expert's tokens, so this both moves and splits experts). Each
copy gets its share of the expert's tokens by their position, so the forward, its checkpoint recompute and the
backward route identically under one plan.

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


def plan_replicas(
    weights: list[float],
    experts_per_rank: int,
    ranks: int,
    spares: int,
    tol: float = 1e-3,
    node_of: list[int] | None = None,
) -> tuple[list[list[int]], dict[int, dict[int, float]]]:
    """Spread each expert's load over its owner (expert ``e`` lives on rank ``e // experts_per_rank``) and up to
    ``spares`` spare-slot copies per rank, in any proportions. Repeatedly levels the most-loaded rank against the
    best target for its heaviest piece: a rank already holding that expert (free), else the least-loaded rank with a
    free spare slot, moving half the load difference (at most the piece). This both moves experts (a copy can take an
    expert's whole load) and splits hot ones. Returns the spare table ``[rank][spare] -> expert`` (``-1`` empty) and
    ``{expert: {rank: load}}`` for every expert that is not entirely on its owner. With ``node_of`` (each rank's node),
    copies stay on their owner's node, so every fetch is intra-node."""
    E = len(weights)
    pieces: dict[int, dict[int, float]] = {e: {e // experts_per_rank: weights[e]} for e in range(E)}
    load = [0.0] * ranks
    for e in range(E):
        load[e // experts_per_rank] += weights[e]
    table = [[-1] * spares for _ in range(ranks)]
    free = [spares] * ranks
    mean = sum(load) / ranks
    for _ in range(8 * ranks * (spares + 1)):
        r_max = max(range(ranks), key=lambda r: load[r])
        best = None
        held = sorted((e for e in range(E) if pieces[e].get(r_max, 0.0) > 0), key=lambda e: -pieces[e][r_max])
        for e in held:
            holders = [(load[r], r, False) for r in pieces[e] if r != r_max]
            owner_node = None if node_of is None else node_of[e // experts_per_rank]
            spare = [
                (load[r], r, True)
                for r in range(ranks)
                if free[r] and r not in pieces[e] and (owner_node is None or node_of[r] == owner_node)
            ]
            if not holders and not spare:
                continue
            _, r_t, uses_spare = min(holders + spare)
            x = min(pieces[e][r_max], (load[r_max] - load[r_t]) / 2)
            if x > tol * mean:
                best = (e, r_t, x, uses_spare)
                break
        if best is None:
            break
        e, r_t, x, uses_spare = best
        pieces[e][r_max] -= x
        pieces[e][r_t] = pieces[e].get(r_t, 0.0) + x
        load[r_max] -= x
        load[r_t] += x
        if uses_spare:
            table[r_t][spares - free[r_t]] = e
            free[r_t] -= 1
    return table, {e: p for e, p in pieces.items() if len(p) > 1 or e // experts_per_rank not in p}


_PHI = 0.6180339887498949  # golden-ratio sequence: an even, deterministic split of token positions


class _PairwiseFetch(torch.autograd.Function):
    """Rows ``packed[start:start + count]`` go to each send peer and ``out[start:start + count]`` comes from each
    receive peer, as point-to-point NCCL operations between only the ranks that exchange rows (no collective over
    the whole group, so ranks with nothing to exchange never wait). The backward returns the received rows'
    gradients to their senders the same way."""

    @staticmethod
    def forward(ctx, packed: Tensor, sends: list, recvs: list, group, num_recv: int) -> Tensor:
        out = packed.new_empty((num_recv, packed.shape[1]))
        _exchange(packed, out, sends, recvs, group)
        ctx.sends, ctx.recvs, ctx.group, ctx.rows = sends, recvs, group, packed.shape[0]
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        grad_in = grad_out.new_zeros((ctx.rows, grad_out.shape[1]))
        _exchange(grad_out.contiguous(), grad_in, ctx.recvs, ctx.sends, ctx.group)
        return grad_in, None, None, None, None


def _exchange(source: Tensor, target: Tensor, sends: list, recvs: list, group) -> None:
    ops = [dist.P2POp(dist.isend, source[a:a + n], dist.get_global_rank(group, peer), group) for peer, a, n in sends]
    ops += [dist.P2POp(dist.irecv, target[a:a + n], dist.get_global_rank(group, peer), group) for peer, a, n in recvs]
    if ops:
        for request in dist.batch_isend_irecv(ops):
            request.wait()


@dataclass
class ReplicaPlan:
    """One layer's spare table and the device-side remap from logical experts to physical slots."""

    table: list[list[int]]  # [rank][spare] -> expert, -1 empty
    experts_per_rank: int
    spares: int
    copy_slots: Tensor  # [E, max_copies] int64 physical slots, padded with the last one
    copy_bounds: Tensor  # [E, max_copies] float64 cumulative token shares, 1.0 at and after the last copy

    @property
    def active(self) -> bool:
        return any(e >= 0 for row in self.table for e in row)

    @property
    def num_slots(self) -> int:
        return len(self.table) * (self.experts_per_rank + self.spares)

    @classmethod
    def build(
        cls,
        table: list[list[int]],
        num_experts: int,
        experts_per_rank: int,
        device,
        pieces: dict[int, dict[int, float]] | None = None,
    ) -> "ReplicaPlan":
        """``pieces`` gives each listed expert's load per rank (owner and spare copies); experts with spare copies
        but no pieces entry split their tokens evenly over the copies."""
        spares = len(table[0]) if table else 0
        width = experts_per_rank + spares
        slot_of: dict[tuple[int, int], int] = {}
        for r, row in enumerate(table):
            for j, e in enumerate(row):
                if e >= 0:
                    slot_of[(e, r)] = r * width + experts_per_rank + j
        slots: list[list[int]] = []
        shares: list[list[float]] = []
        for e in range(num_experts):
            owner = e // experts_per_rank
            native = owner * width + e % experts_per_rank
            copies = [r for (x, r) in slot_of if x == e]
            if pieces is not None and e in pieces:
                ranks = sorted(pieces[e])
                total = sum(pieces[e].values())
                es = [native if r == owner else slot_of[(e, r)] for r in ranks]
                ws = [pieces[e][r] / total if total > 0 else float(r == owner) for r in ranks]
            else:
                es = [native] + [slot_of[(e, r)] for r in copies]
                ws = [1.0 / len(es)] * len(es)
            slots.append(es)
            shares.append(ws)
        max_copies = max(len(s) for s in slots)
        bounds = []
        for ws in shares:
            cum, acc = [], 0.0
            for w in ws:
                acc += w
                cum.append(acc)
            cum[-1] = 1.0
            bounds.append(cum + [1.0] * (max_copies - len(cum)))
        copy_slots = torch.tensor([s + [s[-1]] * (max_copies - len(s)) for s in slots], dtype=torch.long)
        copy_bounds = torch.tensor(bounds, dtype=torch.float64)
        return cls(table, experts_per_rank, spares, copy_slots.to(device), copy_bounds.to(device))

    def remap(self, selected: Tensor) -> Tensor:
        """Logical expert ids ``[tokens, k]`` (``-1`` kept) -> physical slots, giving each copy of an expert its share
        of the expert's tokens by their flat position."""
        valid = selected >= 0
        e = selected.clamp(min=0)
        position = torch.arange(selected.numel(), device=selected.device, dtype=torch.float64).view_as(selected)
        u = torch.frac(position * _PHI)
        copy = (u.unsqueeze(-1) >= self.copy_bounds[e]).sum(-1).clamp(max=self.copy_slots.shape[1] - 1)
        slots = self.copy_slots[e, copy]
        return torch.where(valid, slots, selected)

    @staticmethod
    def _pairs(send_counts: list[int], recv_counts: list[int]) -> tuple[list, list]:
        """(peer, first row, rows) for every non-empty send and receive, rows grouped by peer in rank order."""
        def runs(counts):
            out, start = [], 0
            for peer, n in enumerate(counts):
                if n:
                    out.append((peer, start, n))
                start += n
            return out

        return runs(send_counts), runs(recv_counts)

    def _fetch_layout(self, me: int, device) -> tuple[Tensor, list[int], list[int], Tensor]:
        """This rank's all-to-all layout under the plan, built once: the native rows it sends (grouped by destination
        rank), the split sizes, and for each spare slot its row in the received buffer (the zero row if empty).
        Cached on the device, so a fetch issues no host-to-device copy (which would stall the host every call)."""
        cached = getattr(self, "_layout", None)
        if cached is not None and cached[0] == me:
            return cached[1]
        ranks, L, S = len(self.table), self.experts_per_rank, self.spares
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
        # Rows arrive grouped by source rank in rank order.
        order = sorted(range(len(recv_from)), key=lambda k: (recv_from[k][1], k))
        spare_rows = [len(recv_from)] * S
        for arrival, k in enumerate(order):
            spare_rows[recv_from[k][0]] = arrival
        layout = (
            torch.tensor(send_rows, dtype=torch.long).to(device),
            send_counts,
            recv_counts,
            torch.tensor(spare_rows, dtype=torch.long).to(device),
        )
        self._layout = (me, layout)
        return layout

    def fetch(self, weights: tuple[Tensor, ...], group) -> tuple[Tensor, ...]:
        """Each ``weights[k]`` is this rank's ``[L, ...]`` native expert stack; returns ``[L + S, ...]`` stacks with the
        spare slots filled from their owners (zeros where a spare is empty), differentiably."""
        S = self.spares
        send_index, send_counts, recv_counts, spare_index = self._fetch_layout(dist.get_rank(group), weights[0].device)
        num_recv = sum(recv_counts)
        index = send_index
        differentiable = torch.is_grad_enabled() and any(w.requires_grad for w in weights)
        if not differentiable:
            # Frozen weights (e.g. packed MXFP4 codes and their FP32 scales): move raw bytes, whatever the dtypes.
            weights = tuple(w.detach() for w in weights)
            raw = [w.contiguous().view(torch.uint8).reshape(w.shape[0], -1) for w in weights]
            sizes = [r.shape[1] for r in raw]
            packed = torch.cat([r.index_select(0, index) for r in raw], dim=1)
            received = _PairwiseFetch.apply(packed, *self._pairs(send_counts, recv_counts), group, num_recv)
        else:
            sizes = [w[0].numel() for w in weights]
            packed = torch.cat(
                [w.index_select(0, index).reshape(index.numel(), n) for w, n in zip(weights, sizes)], dim=1
            )
            received = _PairwiseFetch.apply(packed.contiguous(), *self._pairs(send_counts, recv_counts), group, num_recv)
        # Every rank's spares go through `received`, so every rank runs the all-to-all's backward collective too,
        # including one whose spares are all empty (they read an appended zero row).
        padded = torch.cat([received, received.new_zeros(1, received.shape[1])])
        spare = padded.index_select(0, spare_index)
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
        self.node_of: list[int] | None = None  # each EP rank's node; copies stay on their owner's node

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
        if self.node_of is None:
            import socket

            hosts = [None] * ranks
            dist.all_gather_object(hosts, socket.gethostname(), group=self.group)
            ids = {h: i for i, h in enumerate(dict.fromkeys(hosts))}
            self.node_of = [ids[h] for h in hosts]
        for i, layer in enumerate(self.layers):
            layer._replica_counts.zero_()
            est = self.estimates[i]
            est = counts[i] if est is None else [self.ema * c + (1 - self.ema) * p for c, p in zip(counts[i], est)]
            self.estimates[i] = est
            if sum(est) == 0:
                continue
            E = len(est)
            table, pieces = plan_replicas(est, E // ranks, ranks, self.spares, node_of=self.node_of)
            plan = ReplicaPlan.build(table, E, E // ranks, layer._replica_counts.device, pieces)
            plan._fetch_layout(dist.get_rank(self.group), layer._replica_counts.device)  # while the host is synced
            self.plans[i] = plan


_MANAGERS: dict[int, ExpertReplicaManager] = {}
REPLICAS_ENV = "ARL_EXPERT_REPLICAS"
REPLICAS_EMA_ENV = "ARL_EXPERT_REPLICAS_EMA"

# The job's loader options (``runtime_options.apply_moe_runtime_options``); None leaves the environment in charge.
_configured_spares: int | None = None
_configured_ema: float | None = None


def configure(spares: int | None, ema: float | None) -> None:
    """Set this process's spare count and EMA weight from the job's loader options; None for either keeps its
    environment variable (``ARL_EXPERT_REPLICAS``, ``ARL_EXPERT_REPLICAS_EMA``)."""
    global _configured_spares, _configured_ema
    _configured_spares = spares
    _configured_ema = ema


def replica_spares() -> int:
    if _configured_spares is not None:
        return _configured_spares
    return int(os.environ.get(REPLICAS_ENV, "0") or 0)


def replica_ema() -> float:
    if _configured_ema is not None:
        return _configured_ema
    return float(os.environ.get(REPLICAS_EMA_ENV, "0.5"))


def manager_for(group) -> ExpertReplicaManager:
    key = id(group)
    if key not in _MANAGERS:
        _MANAGERS[key] = ExpertReplicaManager(group, replica_spares(), replica_ema())
    return _MANAGERS[key]
