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
from torch import nn

from arctic_platform.model.implementations.deepseek_v41.engram import EngramEmbedding
from arctic_platform.model.implementations.deepseek_v41.engram import dequantize_block_fp8
from arctic_platform.model.implementations.deepseek_v41.engram import ep_row_shard


def _install(embedding: EngramEmbedding, rank: int, weight: torch.Tensor, scale: torch.Tensor | None) -> None:
    start, end = ep_row_shard(embedding.num_embeddings, 2, rank)
    embedding.weight = nn.Parameter(weight[start:end].clone(), requires_grad=False)
    embedding.weight._dss_skip_weight_sync = True
    if scale is not None:
        embedding.scale = nn.Parameter(scale[start:end].clone(), requires_grad=False)
        embedding.scale._dss_skip_weight_sync = True
    embedding._ep_rank = rank
    embedding._ep_world_size = 2
    embedding._ep_group = dist.group.WORLD
    embedding._dss_ep_sharded = True


def _worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    full_weight = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    float_table = EngramEmbedding(7, 2)
    assert float_table.weight.shape[0] == 7
    _install(float_table, rank, full_weight, None)
    assert float_table.weight.shape[0] == (4 if rank == 0 else 3)
    assert float_table.weight.requires_grad is False

    same_ids = torch.tensor([2, 5])
    torch.testing.assert_close(float_table(same_ids), full_weight[same_ids])
    ids = torch.tensor([6, 0, 3]) if rank == 0 else torch.tensor([4, 1, 4])
    torch.testing.assert_close(float_table(ids), full_weight[ids])

    values = torch.tensor([2.0, 1.0, 1.5, 1.0, 1.0, 1.0, 4.0]).view(7, 1).expand(7, 32).contiguous()
    codes = values.to(dtype=torch.float8_e4m3fn)
    scales = torch.tensor([127, 128, 126, 127, 129, 125, 0], dtype=torch.uint8).view(torch.float8_e8m0fnu).view(7, 1)
    expected = dequantize_block_fp8(codes, scales)
    quantized = EngramEmbedding(7, 32, quantized=True)
    _install(quantized, rank, codes, scales)
    quant_ids = torch.tensor([3, 6, 0]) if rank == 0 else torch.tensor([5])
    torch.testing.assert_close(quantized(quant_ids), expected[quant_ids])
    assert quantized.weight.dtype == torch.float8_e4m3fn
    assert quantized.scale.dtype == torch.float8_e8m0fnu
    dist.destroy_process_group()


def test_engram_ep_embedding_routes_unequal_shards() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(f"file://{tmp}/rendezvous",), nprocs=2, join=True)
