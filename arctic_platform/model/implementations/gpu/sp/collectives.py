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

import torch
import torch.distributed as dist

# Bounds the fp32 packed gradient each gather backward reduce-scatters at once.
_GATHER_BACKWARD_CHUNK_BYTES = 256 * 1024 * 1024


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
        input_chunks = [chunk.contiguous() for chunk in tensor.chunk(world_size, dim=scatter_dim)]
        output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]
        dist.all_to_all(output_chunks, input_chunks, group=group)
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
        return None, torch.cat(output_chunks, dim=ctx.scatter_dim), None, None


def sequence_head_all_to_all(group, tensor, *, scatter_dim: int, gather_dim: int):
    """Apply an autograd-aware all-to-all between sequence and head dimensions."""
    return _SeqAllToAll.apply(group, tensor, scatter_dim, gather_dim)


class _GatherSequence(torch.autograd.Function):
    """All-gather along dim 1; the backward sums the ranks' gradients in fp32 and rounds once."""

    @staticmethod
    def forward(ctx, group, tensor):
        ctx.group = group
        tensor = tensor.contiguous()
        parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, tensor, group=group)
        return torch.cat(parts, dim=1)

    @staticmethod
    def backward(ctx, grad):
        # torch's own all_gather backward reduce-scatters in the gradient's dtype, and a bf16 ring rounds each partial sum.
        accumulate_dtype = torch.promote_types(grad.dtype, torch.float32)
        parts = grad.unflatten(1, (dist.get_world_size(ctx.group), -1)).movedim(1, 0)
        grad_input = grad.new_empty(parts.shape[1:])
        row_bytes = parts[:, :, :1].numel() * accumulate_dtype.itemsize
        rows = max(1, _GATHER_BACKWARD_CHUNK_BYTES // max(row_bytes, 1))
        for start in range(0, parts.shape[2], rows):
            packed = parts[:, :, start : start + rows].to(accumulate_dtype, memory_format=torch.contiguous_format)
            summed = packed.new_empty(packed.shape[1:])
            dist.reduce_scatter_tensor(summed, packed.flatten(0, 1), group=ctx.group)
            grad_input[:, start : start + rows] = summed
            del packed, summed
        return None, grad_input


def gather_sequence(tensor: torch.Tensor, group) -> torch.Tensor:
    """All-gather ``tensor`` along the sequence dimension (dim 1) across an SP group, autograd-aware."""
    if group is None or dist.get_world_size(group) == 1:
        return tensor
    return _GatherSequence.apply(group, tensor)


@torch.no_grad()
def gather_packed_positions(position_ids: torch.Tensor, group) -> torch.Tensor:
    """All-gather each rank's ``position_ids`` window into the packed call's ``[batch, sequence]`` positions.

    Takes text ``[batch, W]`` ids or packed M-RoPE ``[channels, batch, W]`` ids (channel 0 holds the row
    positions, each row restarting at 0). Shards are contiguous in rank order, so concatenating the windows on
    the token axis rebuilds the unsharded call, from which ``cu_seqlens_from_position_ids`` reads the global
    row boundaries. Concatenating M-RoPE windows on dim 1 instead would stack the ranks on the batch axis.
    """
    if position_ids.ndim == 3:
        position_ids = position_ids[0]
    if position_ids.ndim != 2:
        raise ValueError(
            "packed position_ids must be [batch, sequence] or M-RoPE [channels, batch, sequence], "
            f"got {tuple(position_ids.shape)}"
        )
    gathered = [torch.empty_like(position_ids) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, position_ids.contiguous(), group=group)
    return torch.cat(gathered, dim=-1)
