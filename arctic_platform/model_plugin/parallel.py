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

"""Collectives the platform gives plugins for sequence and expert parallelism.

Plugin code never calls ``torch.distributed`` itself. A sequence-mixing core sees full sequences and a
:class:`HeadShard` telling it which heads it holds; an expert layer sees all tokens of its expert-parallel group
through :func:`gather_tokens` / :func:`combine_tokens`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_to_all as _differentiable_all_to_all


@dataclass(frozen=True)
class HeadShard:
    """The heads a sequence-mixing core computes: shard ``rank`` of ``size`` equal head groups."""

    rank: int = 0
    size: int = 1

    def local(self, num_heads: int) -> slice:
        if num_heads % self.size != 0:
            raise ValueError(f"{num_heads} heads cannot be split into {self.size} equal shards")
        per_rank = num_heads // self.size
        return slice(self.rank * per_rank, (self.rank + 1) * per_rank)


FULL_SHARD = HeadShard()


@dataclass(frozen=True)
class SequenceWindow:
    """The tokens a sequence-parallel core sees: global positions ``[start - halo, start + local length)``.

    ``start`` is the global position of this rank's first token; the ``halo`` tokens before it were borrowed
    from the previous rank. Unsharded, both are 0.
    """

    start: int = 0
    halo: int = 0


FULL_WINDOW = SequenceWindow()


def group_size(group) -> int:
    return 1 if group is None else dist.get_world_size(group)


def group_rank(group) -> int:
    return 0 if group is None else dist.get_rank(group)


def _swap_dims(group, tensor: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
    size = group_size(group)
    if size == 1:
        return tensor
    if tensor.size(scatter_dim) % size != 0:
        raise ValueError(f"dim {scatter_dim} of size {tensor.size(scatter_dim)} is not divisible by {size} ranks")
    inputs = [chunk.contiguous() for chunk in tensor.chunk(size, dim=scatter_dim)]
    outputs = [torch.empty_like(inputs[0]) for _ in range(size)]
    outputs = _differentiable_all_to_all(outputs, inputs, group=group)
    return torch.cat(outputs, dim=gather_dim)


def seq_to_heads(group, tensor: torch.Tensor) -> torch.Tensor:
    """``[1, T/sp, H, D]`` on each rank to ``[1, T, H/sp, D]``: full sequence, local heads."""
    return _swap_dims(group, tensor, scatter_dim=2, gather_dim=1)


def heads_to_seq(group, tensor: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`seq_to_heads`."""
    return _swap_dims(group, tensor, scatter_dim=1, gather_dim=2)


def head_parallel(module: nn.Module, group) -> None:
    """Run ``module.mix`` head-parallel across ``group``.

    ``mix(*tensors, cu_seqlens, shard)`` takes head-major ``[1, T, H_i, D_i]`` tensors and returns one. After this
    call each rank passes its sequence shard; the wrapper exchanges it for the full sequence of ``1/sp`` of the
    heads, calls the original ``mix`` with that rank's :class:`HeadShard`, and exchanges the result back.
    ``cu_seqlens`` must already describe the full, unsharded sequence.
    """
    core: Callable[..., torch.Tensor] = getattr(module, "mix")
    shard = HeadShard(group_rank(group), group_size(group))

    def mix(*tensors: torch.Tensor, cu_seqlens: torch.Tensor, shard: HeadShard = shard) -> torch.Tensor:
        local = [seq_to_heads(group, tensor) for tensor in tensors]
        return heads_to_seq(group, core(*local, cu_seqlens=cu_seqlens, shard=shard))

    setattr(module, "mix", mix)


class _GatherStack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, tensor):
        ctx.group = group
        ctx.rank = group_rank(group)
        parts = [torch.empty_like(tensor) for _ in range(group_size(group))]
        dist.all_gather(parts, tensor.contiguous(), group=group)
        return torch.stack(parts)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad, group=ctx.group)
        return None, grad[ctx.rank]


def sequence_start(group, local_length: int) -> int:
    """Global position of this rank's first token when a sequence is split into equal contiguous shards."""
    return group_rank(group) * local_length


def previous_tokens(group, tensor: torch.Tensor, count: int, dim: int = 1) -> torch.Tensor:
    """The last ``count`` tokens of the previous rank's shard along ``dim``; empty on the first rank.

    Gradients flow back to the rank that owns those tokens. Every rank of ``group`` must call it.
    """
    size = group_size(group)
    if size == 1 or count == 0:
        return tensor.narrow(dim, 0, 0)
    if tensor.size(dim) < count:
        raise ValueError(f"a {tensor.size(dim)}-token shard cannot supply a {count}-token halo")
    tails = _GatherStack.apply(group, tensor.narrow(dim, tensor.size(dim) - count, count))
    rank = group_rank(group)
    # The first rank keeps an empty slice of the gathered tails, not a fresh tensor, so its backward still joins
    # the gradient all-reduce the other ranks wait on.
    return tails[rank - 1] if rank > 0 else tails[0].narrow(dim, 0, 0)


class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, tensor, dim):
        ctx.group, ctx.dim = group, dim
        parts = [torch.empty_like(tensor) for _ in range(group_size(group))]
        dist.all_gather(parts, tensor.contiguous(), group=group)
        return torch.cat(parts, dim=dim)

    @staticmethod
    def backward(ctx, grad):
        # Reduce-scatter in fp32: a bf16 activation's gradient is rounded once, after the sum, as it would be
        # unsharded (a bf16 ring would round every partial sum).
        parts = torch.stack(grad.float().chunk(group_size(ctx.group), dim=ctx.dim))
        mine = parts.new_empty(parts.shape[1:])
        dist.reduce_scatter_tensor(mine, parts.flatten(0, 1), group=ctx.group)
        return None, mine.to(grad.dtype), None


def gather_sequence(group, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Every rank's sequence shard of ``tensor``, concatenated in rank order along ``dim``: the whole sequence.

    For a policy whose core needs the whole sequence on every rank. In backward, each rank's gradient for the whole
    sequence is summed in fp32 and this rank's slice returned. Shards must have equal length; every rank of
    ``group`` must call it.
    """
    if group_size(group) == 1:
        return tensor
    return _GatherSequence.apply(group, tensor, dim)


def shard_sequence(tensor: torch.Tensor, group, dim: int = 1) -> torch.Tensor:
    """This rank's contiguous sequence shard."""
    size = group_size(group)
    if tensor.size(dim) % size != 0:
        raise ValueError(f"sequence length {tensor.size(dim)} is not divisible by sp_size {size}")
    return tensor.chunk(size, dim=dim)[group_rank(group)]


class _GatherTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, tokens):
        ctx.group = group
        ctx.rank = group_rank(group)
        ctx.rows = tokens.shape[0]
        parts = [torch.empty_like(tokens) for _ in range(group_size(group))]
        dist.all_gather(parts, tokens.contiguous(), group=group)
        return torch.cat(parts, dim=0)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad, group=ctx.group)
        return None, grad[ctx.rank * ctx.rows : (ctx.rank + 1) * ctx.rows]


class _CombineTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, partial):
        ctx.group = group
        size = group_size(group)
        ctx.size = size
        total = partial.contiguous().clone()
        dist.all_reduce(total, group=group)
        rows = total.shape[0] // size
        rank = group_rank(group)
        return total[rank * rows : (rank + 1) * rows]

    @staticmethod
    def backward(ctx, grad):
        parts = [torch.empty_like(grad) for _ in range(ctx.size)]
        dist.all_gather(parts, grad.contiguous(), group=ctx.group)
        return None, torch.cat(parts, dim=0)


def gather_tokens(group, tokens: torch.Tensor) -> torch.Tensor:
    """Every rank's ``[N, ...]`` rows of the expert-parallel group, in rank order. Shards must have equal ``N``."""
    if group_size(group) == 1:
        return tokens
    if not tokens.is_floating_point():
        parts = [torch.empty_like(tokens) for _ in range(group_size(group))]
        dist.all_gather(parts, tokens.contiguous(), group=group)
        return torch.cat(parts, dim=0)
    return _GatherTokens.apply(group, tokens)


def combine_tokens(group, partial: torch.Tensor) -> torch.Tensor:
    """Sum each rank's partial outputs for all gathered rows and return this rank's own rows."""
    if group_size(group) == 1:
        return partial
    return _CombineTokens.apply(group, partial)
