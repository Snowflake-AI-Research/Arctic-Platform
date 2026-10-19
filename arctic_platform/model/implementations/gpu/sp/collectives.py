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
"""Differentiable collectives shared by sequence-parallel model adapters."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

_barrier_groups: dict[object, object] = {}


def _exchange_barrier_group(group):
    if os.environ.get("ARCTIC_SP_ALLTOALL_BARRIER") != "1":
        return None
    if group not in _barrier_groups:
        _barrier_groups[group] = dist.new_group(
            ranks=dist.get_process_group_ranks(group),
            backend="gloo",
            use_local_synchronization=True,
        )
    return _barrier_groups[group]


def _finish_exchange(barrier_group):
    if barrier_group is not None:
        # Finish GPU communication before another rank can enqueue the next exchange.
        torch.cuda.current_stream().synchronize()
        dist.barrier(group=barrier_group)


class _SeqAllToAll(torch.autograd.Function):
    """Swap a sharded tensor dimension for another dimension across an SP group."""

    @staticmethod
    def forward(ctx, group, tensor, scatter_dim, gather_dim):
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(group)
        ctx.world_size = world_size
        if world_size == 1:
            return tensor
        if tensor.size(scatter_dim) % world_size != 0:
            raise ValueError(
                f"SP all-to-all scatter dim {scatter_dim} with size "
                f"{tensor.size(scatter_dim)} is not divisible by world size {world_size}"
            )
        ctx.barrier_group = _exchange_barrier_group(group)
        input_chunks = [chunk.contiguous() for chunk in tensor.chunk(world_size, dim=scatter_dim)]
        output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]
        dist.all_to_all(output_chunks, input_chunks, group=group)
        _finish_exchange(ctx.barrier_group)
        return torch.cat(output_chunks, dim=gather_dim)

    @staticmethod
    def backward(ctx, grad):
        if ctx.world_size == 1:
            return None, grad, None, None
        if grad.size(ctx.gather_dim) % ctx.world_size != 0:
            raise RuntimeError(
                f"SP all-to-all backward gather dim {ctx.gather_dim} with size "
                f"{grad.size(ctx.gather_dim)} is not divisible by world size "
                f"{ctx.world_size}"
            )
        input_chunks = [chunk.contiguous() for chunk in grad.chunk(ctx.world_size, dim=ctx.gather_dim)]
        output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(ctx.world_size)]
        dist.all_to_all(output_chunks, input_chunks, group=ctx.group)
        _finish_exchange(ctx.barrier_group)
        return None, torch.cat(output_chunks, dim=ctx.scatter_dim), None, None


def sequence_head_all_to_all(group, tensor, *, scatter_dim: int, gather_dim: int):
    """Apply an autograd-aware all-to-all between sequence and head dimensions."""
    return _SeqAllToAll.apply(group, tensor, scatter_dim, gather_dim)
