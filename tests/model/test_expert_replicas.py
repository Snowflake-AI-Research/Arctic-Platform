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


def test_plan_moves_and_splits_load_to_near_balance_within_the_spare_slots():
    random.seed(0)
    E, P, S = 64, 8, 2
    L = E // P
    weights = [random.paretovariate(1.2) for _ in range(E)]
    weights[5] *= 40  # one expert far hotter than any rank's average
    table, pieces = plan_replicas(weights, L, P, S)
    load = [0.0] * P
    for e in range(E):
        for r, amount in pieces.get(e, {e // L: weights[e]}).items():
            load[r] += amount
    before = max(sum(weights[r * L:(r + 1) * L]) for r in range(P))
    assert max(load) < before and max(load) < 1.1 * sum(weights) / P
    assert all(len(row) == S for row in table)
    for e, by_rank in pieces.items():
        assert sum(by_rank.values()) == pytest.approx(weights[e])
        for r in by_rank:
            assert r == e // L or e in table[r]  # every non-owner piece sits in one of that rank's spare slots
    assert len(pieces[5]) > 1


def test_remap_gives_each_copy_its_share_of_the_tokens():
    P, L, S = 4, 2, 1
    table = [[-1], [0], [-1], [-1]]  # expert 0 (owner rank 0) has a copy on rank 1
    plan = ReplicaPlan.build(table, P * L, L, "cpu", {0: {0: 1.0, 1: 3.0}})
    counts = torch.bincount(plan.remap(torch.zeros(4000, 1, dtype=torch.long)).flatten(), minlength=plan.num_slots)
    width = L + S
    assert counts[0] == pytest.approx(1000, abs=5) and counts[1 * width + L] == pytest.approx(3000, abs=5)
    moved = ReplicaPlan.build(table, P * L, L, "cpu", {0: {1: 2.0}})  # the copy takes all of expert 0
    assert (moved.remap(torch.zeros(10, 1, dtype=torch.long)) == 1 * width + L).all()


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
    assert counts[counts > 0].max() - counts[counts > 0].min() <= 3  # golden-ratio split of 300 positions
    assert (slots[:-1, 1] == (3 // L) * width + 3 % L).all()  # unreplicated expert keeps its native slot
    assert slots[-1, 0] == -1


def _fetch_worker(rank, world, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
    # NCCL wants a group's first call to involve every rank; ranks without spares skip the fetch's P2P exchange.
    torch.distributed.all_reduce(torch.zeros(1, device="cuda"))
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
