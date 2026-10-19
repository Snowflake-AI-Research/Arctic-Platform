"""Context-parallel helpers used by DeepSeek V4 attention.

The default context is a single rank. The DeepSpeed loader installs the DSS
SP group on the model and every attention module when sequence parallel is on.
``gather_for_cp`` is the all-gather the compressor uses once that context exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn


@dataclass(frozen=True)
class CPContext:
    cp_group: dist.ProcessGroup | None = None
    cp_rank: int = 0
    cp_world_size: int = 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp_world_size > 1


def gather_for_cp(t: torch.Tensor, cp_group: dist.ProcessGroup | None) -> torch.Tensor:
    if cp_group is None or dist.get_world_size(group=cp_group) == 1:
        return t
    return torch.cat(dist_nn.all_gather(t, group=cp_group), dim=1)
