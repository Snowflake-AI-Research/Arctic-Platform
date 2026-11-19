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

"""Expert layers: how a plugin's mixture-of-experts block is sharded across an expert-parallel group.

A MoE module exposes ``experts_core(tokens, selected, weights, *, shard)``: given tokens, their chosen expert ids
and gate weights, it returns the summed contribution of the experts ``shard`` holds. The platform wraps that core
so it sees every token of the expert-parallel group, and sums the ranks' contributions back. Model code never
moves tokens itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from typing import runtime_checkable

import torch
from torch import nn

from arctic_platform.model_plugin.marks import MARKS_ATTR
from arctic_platform.model_plugin.parallel import combine_tokens
from arctic_platform.model_plugin.parallel import gather_tokens
from arctic_platform.model_plugin.parallel import group_rank
from arctic_platform.model_plugin.parallel import group_size

# Must equal ROUTER_REPLAY_FRESH in arctic_platform.model.implementations.moe.layers.moe, which the platform's own
# MoE layer reads: a replayed token with this id is routed by the trainer's own gate.
ROUTER_REPLAY_FRESH = -1

EXPERT_PARALLEL_SIZE_ATTR = "_ap_expert_parallel_size"


@dataclass(frozen=True)
class ExpertShard:
    """The routed experts one rank holds: a contiguous block of ``num_experts / size``, the ``rank``-th."""

    rank: int
    size: int
    num_experts: int

    def __post_init__(self):
        if self.num_experts % self.size != 0:
            raise ValueError(f"{self.num_experts} experts cannot be split across {self.size} ranks")

    @classmethod
    def full(cls, num_experts: int) -> ExpertShard:
        return cls(0, 1, num_experts)

    @property
    def per_rank(self) -> int:
        return self.num_experts // self.size

    @property
    def start(self) -> int:
        """Global id of the first expert held; local expert ``i`` is global expert ``start + i``."""
        return self.rank * self.per_rank


@runtime_checkable
class ExpertLayer(Protocol):
    """A plugin's mixture-of-experts integration: which tensors are experts, and how they are sharded.

    Expert weights are stacked with the expert index as dim 0. :class:`StackedExpertLayer` implements this for
    modules that follow the ``experts_core`` contract; write your own only for a different dispatch policy.
    """

    def is_instance(self, module: nn.Module) -> bool: ...

    def num_experts(self, module: nn.Module) -> int: ...

    def expert_parameters(self, module: nn.Module) -> dict[str, nn.Parameter]: ...

    def apply_expert_parallel(self, module: nn.Module, group) -> None: ...


def shard_experts(module: nn.Module, names: list[str], group) -> None:
    """Replace each stacked expert parameter ``module.<name>`` (dotted names allowed) with this rank's block."""
    size, rank = group_size(group), group_rank(group)
    for name in names:
        owner_path, _, attribute = name.rpartition(".")
        owner = module.get_submodule(owner_path) if owner_path else module
        full = getattr(owner, attribute)
        if full.shape[0] % size != 0:
            raise ValueError(f"{name}: {full.shape[0]} experts cannot be split across {size} ranks")
        per_rank = full.shape[0] // size
        local = nn.Parameter(full.detach()[rank * per_rank : (rank + 1) * per_rank].clone())
        local.requires_grad_(full.requires_grad)
        if hasattr(full, MARKS_ATTR):
            setattr(local, MARKS_ATTR, getattr(full, MARKS_ATTR))
        setattr(owner, attribute, local)


def expert_parallel(module: nn.Module, group, names: list[str], num_experts: int) -> None:
    """Shard ``module``'s stacked experts and run its ``experts_core`` expert-parallel across ``group``.

    Each rank keeps its block of experts. The wrapped core receives every rank's tokens, expert ids and gate
    weights, computes only this rank's experts (told by an :class:`ExpertShard`), and the ranks' partial outputs
    are summed back to the rank that owns each token.
    """
    shard = ExpertShard(group_rank(group), group_size(group), num_experts)
    shard_experts(module, names, group)
    core: Callable[..., torch.Tensor] = getattr(module, "experts_core")

    def experts_core(
        tokens: torch.Tensor, selected: torch.Tensor, weights: torch.Tensor, *, shard: ExpertShard = shard
    ) -> torch.Tensor:
        partial = core(
            gather_tokens(group, tokens), gather_tokens(group, selected), gather_tokens(group, weights), shard=shard
        )
        return combine_tokens(group, partial)

    setattr(module, "experts_core", experts_core)


class StackedExpertLayer:
    """An expert layer for MoE modules with a ``num_experts`` attribute, stacked experts and an ``experts_core``.

    ``expert_names`` are the stacked expert tensors relative to the module, for example ``("experts.w1", ...)``.
    """

    module_class: type[nn.Module] = nn.Module
    expert_names: tuple[str, ...] = ()

    def __init__(self, module_class: type[nn.Module] | None = None, expert_names: tuple[str, ...] | None = None):
        if module_class is not None:
            self.module_class = module_class
        if expert_names is not None:
            self.expert_names = expert_names
        if len(self.expert_names) == 0:
            raise ValueError("a stacked expert layer needs at least one expert tensor name")

    def is_instance(self, module: nn.Module) -> bool:
        return isinstance(module, self.module_class)

    def num_experts(self, module: nn.Module) -> int:
        return int(getattr(module, "num_experts"))

    def expert_parameters(self, module: nn.Module) -> dict[str, nn.Parameter]:
        return {name: module.get_parameter(name) for name in self.expert_names}

    def apply_expert_parallel(self, module: nn.Module, group) -> None:
        expert_parallel(module, group, list(self.expert_names), self.num_experts(module))
