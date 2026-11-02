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
"""The CP-balanced DSA indexer returns the same top-k as each rank scoring its own queries."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _run(rank, world, port, failures):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    from arctic_platform.model.implementations.glm52.models.glm_moe_dsa.sparse_mla_attention import (
        _cp_balanced_indexer,
    )
    from arctic_platform.model.implementations.glm52.models.kernels.fp8_indexer import fp8_indexer

    torch.manual_seed(0)  # every rank draws the whole sequence, then takes its shard
    tokens, heads, dim, topk = 512, 32, 128, 128
    total = tokens * world
    q_all = torch.randn(total, heads, dim, device="cuda").bfloat16()
    k_all = torch.randn(total, dim, device="cuda").bfloat16()
    w_all = torch.randn(total, heads, device="cuda").bfloat16()
    # Two packed sequences: [0, 700) and [700, total).
    starts = torch.where(torch.arange(total, device="cuda") < 700, 0, 700).int()
    ks, ke = starts, torch.arange(1, total + 1, device="cuda", dtype=torch.int32)
    local = slice(rank * tokens, (rank + 1) * tokens)
    group = dist.new_group(list(range(world)))
    got = _cp_balanced_indexer(q_all[local], k_all, w_all[local], ks, ke, topk, 1.0, group, rank, world)
    want = fp8_indexer(q_all[local], k_all, w_all[local], ks[local], ke[local], topk, 1.0)
    if not torch.equal(got.sort(-1).values, want.sort(-1).values):
        failures[rank] = int((got.sort(-1).values != want.sort(-1).values).sum())
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs four GPUs")
def test_cp_balanced_indexer_matches_local_scoring():
    failures = mp.Manager().dict()
    mp.spawn(_run, args=(4, 29523, failures), nprocs=4, join=True)
    assert dict(failures) == {}
