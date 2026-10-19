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
import weakref

import torch
import torch.distributed as dist

_barrier_groups: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _group_is_registered(group):
    if group is None or not dist.is_initialized():
        return False
    try:
        dist.get_process_group_ranks(group)
    except (KeyError, ValueError):
        return False
    return True


def destroy_exchange_barrier_groups(group=None):
    """Release owned Gloo groups before SP-group teardown; all members must call."""
    for source in list(_barrier_groups):
        if group is not None and source is not group:
            continue
        barrier = _barrier_groups.pop(source)()
        if barrier is not None and _group_is_registered(barrier):
            dist.destroy_process_group(barrier)


def _discard_stale_barrier_groups():
    for source, barrier_ref in list(_barrier_groups.items()):
        barrier = barrier_ref()
        if not _group_is_registered(source) or barrier is None or not _group_is_registered(barrier):
            destroy_exchange_barrier_groups(source)


def _exchange_barrier_group(group):
    # Exactly "1" enables the guard; set the same value on every SP rank.
    if os.environ.get("ARCTIC_SP_ALLTOALL_BARRIER") != "1":
        return None
    _discard_stale_barrier_groups()
    group = dist.group.WORLD if group is None else group
    if not _group_is_registered(group):
        raise ValueError("The SP all-to-all group is not registered")
    if group not in _barrier_groups:
        barrier = dist.new_group(
            ranks=dist.get_process_group_ranks(group),
            backend="gloo",
            use_local_synchronization=True,
        )
        # PyTorch owns live groups; the cache must not retain destroyed groups.
        _barrier_groups[group] = weakref.ref(barrier)
        return barrier
    return _barrier_groups[group]()


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
