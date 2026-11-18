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

"""ToyHybridMoE variants with one realistic parallelism bug each, for proving conformance rejects them.

Importable by name so conformance's spawned gloo workers can load them like any installed plugin.
"""

import dataclasses

from toy_hybrid_moe import PLUGIN
from toy_hybrid_moe.modeling import ToyHybridMoEForCausalLM
from toy_hybrid_moe.modeling import ToyRecurrence
from toy_hybrid_moe.parallel import ToyExpertLayer

from arctic_platform.model_plugin import FULL_SHARD
from arctic_platform.model_plugin import HeadShard


class FirstHeadsRecurrence(ToyRecurrence):
    """Slices its per-head parameters as if it always held the first heads, whatever its shard."""

    def mix(self, u, g, *, cu_seqlens, shard: HeadShard = FULL_SHARD):
        return super().mix(u, g, cu_seqlens=cu_seqlens, shard=HeadShard(0, shard.size))


def _build_first_heads(config):
    model = ToyHybridMoEForCausalLM(config)
    for module in model.modules():
        if type(module) is ToyRecurrence:
            module.__class__ = FirstHeadsRecurrence
    return model


class NoOffsetExpertLayer(ToyExpertLayer):
    """Shards the experts but leaves every rank thinking it holds experts 0..E/ep-1."""

    def apply_expert_parallel(self, module, group):
        super().apply_expert_parallel(module, group)
        module.expert_offset = 0


WRONG_HEAD_SLICE = dataclasses.replace(PLUGIN, build_model=_build_first_heads)
NO_EXPERT_OFFSET = dataclasses.replace(PLUGIN, expert_layer=NoOffsetExpertLayer())
