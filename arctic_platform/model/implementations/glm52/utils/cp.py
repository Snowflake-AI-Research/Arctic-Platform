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


def gather_for_cp(t: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    if dist.get_world_size(group=cp_group) == 1:
        return t
    gathered_t = dist_nn.all_gather(t, group=cp_group)
    return torch.cat(gathered_t, dim=1)
