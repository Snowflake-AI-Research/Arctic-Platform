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

import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import EPShardedEmbedding


def _worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    full_weight = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    start, end = ((0, 4), (4, 7))[rank]
    embedding = EPShardedEmbedding(7, 2)
    embedding.weight = torch.nn.Parameter(full_weight[start:end].clone())
    embedding._ep_rank = rank
    embedding._ep_world_size = 2
    embedding._ep_group = dist.group.WORLD

    same_ids = torch.tensor([2, 5])
    torch.testing.assert_close(embedding(same_ids), full_weight[same_ids])

    ids = torch.tensor([0]) if rank == 0 else torch.tensor([4, 6])
    output = embedding(ids)
    torch.testing.assert_close(output, full_weight[ids])
    output.sum().backward()

    expected_grad = torch.zeros_like(embedding.weight)
    if rank == 0:
        expected_grad[0] = 1
    else:
        expected_grad[0] = 1
        expected_grad[2] = 1
    torch.testing.assert_close(embedding.weight.grad, expected_grad)
    dist.destroy_process_group()


def test_qwen38_ep_embedding_routes_unequal_rank_local_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _worker,
            args=(f"file://{tmp}/rendezvous",),
            nprocs=2,
            join=True,
        )
