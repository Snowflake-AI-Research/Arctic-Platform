"""Context-parallel helpers for GLM sparse MLA (cp=1 is a no-op on the DSS path)."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn


def shard_for_cp(t: torch.Tensor, cp_rank: int, cp_world_size: int) -> torch.Tensor:
    assert t.shape[0] == 1, "For CP, tensor must have batch dimension of 1"
    if cp_world_size == 1:
        return t
    chunked_t = torch.chunk(t, cp_world_size, dim=1)
    return chunked_t[cp_rank]


class _GatherSequence(torch.autograd.Function):
    """All-gather [S, ...] shards into one [world * S, ...] buffer; the adjoint reduce-scatters it."""

    @staticmethod
    def forward(ctx, shard: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
        ctx.cp_group = cp_group
        world = dist.get_world_size(group=cp_group)
        out = shard.new_empty((world * shard.shape[0], *shard.shape[1:]))
        dist.all_gather_into_tensor(out, shard.contiguous(), group=cp_group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        world = dist.get_world_size(group=ctx.cp_group)
        shard = grad.new_empty((grad.shape[0] // world, *grad.shape[1:]))
        dist.reduce_scatter_tensor(shard, grad.contiguous(), group=ctx.cp_group)
        return shard, None


def gather_for_cp(t: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    if dist.get_world_size(group=cp_group) == 1:
        return t
    if t.shape[0] == 1:
        # One flat collective each way, instead of a list all-gather plus a concatenation.
        return _GatherSequence.apply(t[0], cp_group).unsqueeze(0)
    gathered_t = dist_nn.all_gather(t, group=cp_group)
    return torch.cat(gathered_t, dim=1)
