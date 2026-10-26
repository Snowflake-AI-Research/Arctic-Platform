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
"""Online redundant experts: the spare-slot planner, the token remap and the differentiable weight fetch."""

import os
import random

import pytest
import torch
import torch.multiprocessing as mp

from arctic_platform.model.implementations.moe.expert_replicas import ReplicaPlan
from arctic_platform.model.implementations.moe.expert_replicas import plan_replicas


def _rank_loads(weights, table, experts_per_rank):
    ranks = len(table)
    copies = {e: [e // experts_per_rank] for e in range(len(weights))}
    for r, row in enumerate(table):
        for e in row:
            if e >= 0:
                copies[e].append(r)
    load = [0.0] * ranks
    for e, rs in copies.items():
        for r in rs:
            load[r] += weights[e] / len(rs)
    return load, copies


def test_plan_lowers_the_hottest_rank_and_keeps_copies_on_distinct_ranks():
    random.seed(0)
    E, P, S = 64, 8, 2
    weights = [random.paretovariate(1.2) for _ in range(E)]
    weights[5] *= 40  # one expert far hotter than any rank's average
    table = plan_replicas(weights, E // P, P, S)
    before = max(sum(weights[r * 8:(r + 1) * 8]) for r in range(P))
    after, copies = _rank_loads(weights, table, E // P)
    assert max(after) < before
    assert all(len(row) == S for row in table)
    assert all(len(set(rs)) == len(rs) for rs in copies.values())
    assert len(copies[5]) > 1


def test_remap_splits_a_replicated_experts_tokens_evenly_and_keeps_padding():
    P, L, S = 4, 2, 1
    table = [[-1], [0], [0], [-1]]  # expert 0 (owner rank 0) copied to ranks 1 and 2
    plan = ReplicaPlan.build(table, P * L, L, "cpu")
    assert plan.num_slots == P * (L + S)
    selected = torch.tensor([[0, 3]] * 300 + [[-1, 5]])
    slots = plan.remap(selected)
    width = L + S
    expert0 = slots[:-1, 0]
    counts = torch.bincount(expert0, minlength=plan.num_slots)
    assert set(torch.nonzero(counts).flatten().tolist()) == {0, 1 * width + L, 2 * width + L}
    assert counts[counts > 0].max() - counts[counts > 0].min() <= 1
    assert (slots[:-1, 1] == (3 // L) * width + 3 % L).all()  # unreplicated expert keeps its native slot
    assert slots[-1, 0] == -1


def _fetch_worker(rank, world, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
    L, S = 2, 2
    # Rank r holds experts 2r, 2r+1; each expert's weights are filled with its id.
    table = [[-1, -1] for _ in range(world)]
    table[1] = [0, 2 * (world - 1)]  # rank 1: expert 0 (owner rank 0) and the last rank's first expert
    table[0] = [-1, 2 * (world - 1) + 1]
    plan = ReplicaPlan.build(table, world * L, L, torch.device("cuda", rank))
    base = torch.arange(world * L, dtype=torch.float32, device="cuda")[rank * L:(rank + 1) * L]
    w1 = (base.view(L, 1, 1) * torch.ones(L, 3, 4, device="cuda")).requires_grad_(True)
    w2 = (base.view(L, 1, 1) * torch.ones(L, 4, 3, device="cuda")).requires_grad_(True)
    e1, e2 = plan.fetch((w1, w2), torch.distributed.group.WORLD)
    assert e1.shape == (L + S, 3, 4) and e2.shape == (L + S, 4, 3)
    expected = [rank * L, rank * L + 1] + [float(e) if e >= 0 else 0.0 for e in table[rank]]
    got = [e1[i].mean().item() for i in range(L + S)]
    (e1.sum() + e2.sum()).backward()
    # Each native expert's gradient counts its own use plus one per copy fetched elsewhere.
    uses = [1 + sum(row.count(e) for row in table) for e in range(rank * L, (rank + 1) * L)]
    grads = [w1.grad[i].mean().item() for i in range(L)]
    # Frozen mixed-dtype weights (packed codes and scales) travel as bytes.
    codes = (torch.arange(L, device="cuda", dtype=torch.int8) + rank * L).view(L, 1).expand(L, 6).contiguous()
    scales = (base.view(L, 1) * torch.ones(L, 2, device="cuda")).contiguous()
    with torch.no_grad():
        c2, s2 = plan.fetch((codes, scales), torch.distributed.group.WORLD)
    frozen = ([int(c2[i, 0]) for i in range(L + S)], [s2[i, 0].item() for i in range(L + S)])
    results[rank] = (expected, got, uses, grads, frozen)
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs for an NCCL EP group")
def test_fetch_fills_spares_from_owners_and_returns_their_gradients():
    world = 4
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_fetch_worker, args=(world, 29511, results), nprocs=world, join=True)
    for rank in range(world):
        expected, got, uses, grads, (codes, scales) = results[rank]
        assert got == pytest.approx(expected), (rank, got, expected)
        assert grads == pytest.approx([float(u) for u in uses]), (rank, grads, uses)
        assert codes == [int(e) for e in expected] and scales == pytest.approx(expected), (rank, codes, scales)
