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

"""Expert layers: how a plugin's mixture-of-experts block is sharded across an expert-parallel group."""

from __future__ import annotations

from typing import Protocol
from typing import runtime_checkable

from torch import nn

from arctic_platform.model_plugin.parallel import group_rank
from arctic_platform.model_plugin.parallel import group_size

# Must equal ROUTER_REPLAY_FRESH in arctic_platform.model.implementations.moe.layers.moe, which the platform's own
# MoE layer reads: a replayed token with this id is routed by the trainer's own gate.
ROUTER_REPLAY_FRESH = -1

EXPERT_PARALLEL_SIZE_ATTR = "_ap_expert_parallel_size"


@runtime_checkable
class ExpertLayer(Protocol):
    """A plugin's mixture-of-experts integration.

    Expert weights are stacked with the expert index as dim 0. ``apply_expert_parallel`` keeps this rank's
    ``num_experts / ep`` experts and routes the group's tokens to them, typically with
    :func:`arctic_platform.model_plugin.parallel.gather_tokens` and ``combine_tokens``.
    """

    def is_instance(self, module: nn.Module) -> bool: ...

    def num_experts(self, module: nn.Module) -> int: ...

    def expert_parameters(self, module: nn.Module) -> dict[str, nn.Parameter]: ...

    def apply_expert_parallel(self, module: nn.Module, group) -> None: ...


def shard_experts(module: nn.Module, names: list[str], group) -> None:
    """Replace each stacked expert parameter ``module.<name>`` with this rank's contiguous block of experts."""
    size, rank = group_size(group), group_rank(group)
    for name in names:
        full = getattr(module, name)
        if full.shape[0] % size != 0:
            raise ValueError(f"{name}: {full.shape[0]} experts cannot be split across {size} ranks")
        per_rank = full.shape[0] // size
        local = nn.Parameter(full.detach()[rank * per_rank : (rank + 1) * per_rank].clone())
        local.requires_grad_(full.requires_grad)
        setattr(module, name, local)
