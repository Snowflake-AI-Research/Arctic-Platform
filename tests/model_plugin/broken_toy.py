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
from toy_hybrid_moe.config import SHORT_CONV
from toy_hybrid_moe.modeling import ToyHybridMoEForCausalLM
from toy_hybrid_moe.modeling import ToyMoE
from toy_hybrid_moe.modeling import ToyRecurrence
from toy_hybrid_moe.parallel import ShortConvMixer

from arctic_platform.model_plugin import FULL_SHARD
from arctic_platform.model_plugin import ExpertShard
from arctic_platform.model_plugin import HeadShard
from arctic_platform.model_plugin import SequenceWindow
from arctic_platform.model_plugin import sequence_start


class FirstHeadsRecurrence(ToyRecurrence):
    """Slices its per-head parameters as if it always held the first heads, whatever its shard."""

    def mix(self, u, g, *, cu_seqlens, shard: HeadShard = FULL_SHARD):
        return super().mix(u, g, cu_seqlens=cu_seqlens, shard=HeadShard(0, shard.size))


class FirstExpertsMoE(ToyMoE):
    """Computes its local experts as if they were always experts 0.., whatever its shard."""

    def experts_core(self, tokens, selected, weights, *, shard=None):
        if shard is not None:
            shard = ExpertShard(0, shard.size, shard.num_experts)
        return super().experts_core(tokens, selected, weights, shard=shard)


def _builder(original: type, broken: type):
    def build(config):
        model = ToyHybridMoEForCausalLM(config)
        for module in model.modules():
            if type(module) is original:
                module.__class__ = broken
        return model

    return build


class NoHaloMixer(ShortConvMixer):
    """Shards the convolution by sequence but never borrows the previous rank's tokens."""

    def apply_sequence_parallel(self, module, group):
        core = module.mix

        def mix(u, *, cu_seqlens, shard=None):
            return core(u, cu_seqlens=cu_seqlens, shard=SequenceWindow(start=sequence_start(group, u.shape[1])))

        setattr(module, "mix", mix)


WRONG_HEAD_SLICE = dataclasses.replace(PLUGIN, build_model=_builder(ToyRecurrence, FirstHeadsRecurrence))
WRONG_EXPERT_SLICE = dataclasses.replace(PLUGIN, build_model=_builder(ToyMoE, FirstExpertsMoE))
NO_HALO = dataclasses.replace(PLUGIN, mixers=dict(PLUGIN.mixers, **{SHORT_CONV: NoHaloMixer()}))
