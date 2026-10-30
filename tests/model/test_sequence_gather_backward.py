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
"""Sequence gather preserves rank order and accumulates backward gradients in fp32."""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.distributed as dist

from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal


def _nccl_like_group(monkeypatch, world_size, rank):
    """Simulate identical rank gradients with a reduction that rounds each addition."""

    def all_gather(parts, tensor, group=None):
        for part in parts:
            assert part.is_contiguous()
            part.copy_(tensor)

    def reduce_scatter(output, inputs, op=None, group=None):
        inputs = inputs.reshape(world_size, *output.shape)
        total = inputs[rank].clone()
        for _ in range(world_size - 1):
            total += inputs[rank]
        output.copy_(total)

    for name, value in (("get_world_size", lambda group=None: world_size), ("all_gather", all_gather),
                        ("reduce_scatter_tensor", reduce_scatter)):  # fmt: skip
        monkeypatch.setattr(dist, name, value)


@pytest.mark.parametrize(
    "module",
    [
        "arctic_platform.model.implementations.glm53.context_parallel",
        "arctic_platform.model.implementations.qwen38.qsa_flex",
    ],
)
def test_sequence_gather_backward_sums_ranks_in_fp32(monkeypatch, module):
    gather_sequence = importlib.import_module(module).gather_sequence
    world_size, rank, local_length = 8, 3, 4
    _nccl_like_group(monkeypatch, world_size, rank)
    set_seed(0)
    local = torch.randn(2, 2, local_length, 16).to(torch.bfloat16).transpose(1, 2).requires_grad_(True)
    gathered = gather_sequence(local, "sp-group")
    grad = torch.randn(gathered.shape).to(torch.bfloat16)
    gathered.backward(grad)

    torch_assert_equal(gathered, local.detach().repeat(1, world_size, 1, 1))
    # Every rank holds the same gradient, so the exact sum is world_size x this rank's chunk (exact in bf16).
    expected = world_size * grad[:, rank * local_length : (rank + 1) * local_length]
    rounded = grad.chunk(world_size, dim=1)[rank].clone()
    for _ in range(world_size - 1):
        rounded += grad.chunk(world_size, dim=1)[rank]
    assert not torch.equal(rounded, expected)
    assert local.grad.dtype == torch.bfloat16
    torch_assert_equal(local.grad, expected)


def _gloo_worker(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed.nn as dist_nn

    from arctic_platform.model.implementations.gpu.sp.collectives import gather_sequence

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        local = (torch.arange(48).reshape(2, 2, 3, 4) + rank * 48).to(torch.bfloat16)
        local = local.transpose(1, 2).requires_grad_()
        gathered = gather_sequence(local, dist.group.WORLD)
        torch_assert_equal(gathered, torch.cat(dist_nn.all_gather(local.contiguous()), dim=1))
        grad = (torch.arange(gathered.numel()).reshape(gathered.shape) / 16 + rank).to(torch.bfloat16)
        expected = grad.float()
        dist.all_reduce(expected)
        gathered.backward(grad)
        torch_assert_equal(local.grad, expected.chunk(2, dim=1)[rank].to(local.dtype))
    finally:
        dist.destroy_process_group()


def test_sequence_gather_gloo(tmp_path):
    torch.multiprocessing.spawn(_gloo_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
