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

"""How ToyHybridMoE tells the platform about its layers: the recurrence mixer, the expert layer, and its shape."""

from __future__ import annotations

from torch import nn

from arctic_platform.model_plugin import HeadParallelMixer
from arctic_platform.model_plugin import ParallelismInfo
from arctic_platform.model_plugin import shard_experts
from arctic_platform.model_plugin.parallel import group_rank
from arctic_platform.model_plugin.parallel import group_size

from .config import FULL_ATTENTION
from .config import GATED_RECURRENCE
from .config import MOE
from .config import ToyHybridMoEConfig
from .modeling import ToyMoE
from .modeling import ToyRecurrence


class GatedRecurrenceMixer(HeadParallelMixer):
    """Sequence parallelism for :class:`ToyRecurrence`.

    The platform's head-parallel exchange gives ``ToyRecurrence.mix`` whole rows of ``recurrence_heads / sp``
    heads; ``mix`` slices its convolution and decay parameters to those heads and restarts at row boundaries.
    Head counts are checked by :class:`HeadParallelMixer` from ``ParallelismInfo.heads``.
    """

    module_class = ToyRecurrence

    def __init__(self):
        super().__init__(GATED_RECURRENCE)


class ToyExpertLayer:
    """Expert parallelism for :class:`ToyMoE`: each rank keeps a contiguous block of experts."""

    names = ("w1", "w2", "w3")

    def is_instance(self, module: nn.Module) -> bool:
        return isinstance(module, ToyMoE)

    def num_experts(self, module: nn.Module) -> int:
        return module.num_experts

    def expert_parameters(self, module: nn.Module) -> dict[str, nn.Parameter]:
        return {f"experts.{name}": getattr(module.experts, name) for name in self.names}

    def apply_expert_parallel(self, module: nn.Module, group) -> None:
        shard_experts(module.experts, list(self.names), group)
        module.ep_group = group
        module.expert_offset = group_rank(group) * (module.num_experts // group_size(group))


def parallelism(config: ToyHybridMoEConfig) -> ParallelismInfo:
    return ParallelismInfo(
        layer_types=tuple(config.layer_types),
        heads={
            FULL_ATTENTION: (config.num_attention_heads, config.num_key_value_heads),
            GATED_RECURRENCE: (config.recurrence_heads,),
        },
        moe_layers=tuple(i for i, mlp in enumerate(config.mlp_types) if mlp == MOE),
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
    )
