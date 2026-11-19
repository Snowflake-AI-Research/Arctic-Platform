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

"""How ToyHybridMoE tells the platform about its layers: two mixers, the expert layer, and its shape.

The two plugin mixers show both kinds of sequence-parallel policy. The recurrence uses the platform's default,
the head-parallel exchange. The short convolution brings its own: a halo exchange with the previous rank.
"""

from __future__ import annotations

import torch
from torch import nn

from arctic_platform.model_plugin import HeadParallelMixer
from arctic_platform.model_plugin import ParallelismInfo
from arctic_platform.model_plugin import SequenceWindow
from arctic_platform.model_plugin import StackedExpertLayer
from arctic_platform.model_plugin import previous_tokens
from arctic_platform.model_plugin import sequence_start

from .config import FULL_ATTENTION
from .config import GATED_RECURRENCE
from .config import MOE
from .config import SHORT_CONV
from .config import ToyHybridMoEConfig
from .modeling import ToyMoE
from .modeling import ToyRecurrence
from .modeling import ToyShortConv


class GatedRecurrenceMixer(HeadParallelMixer):
    """Sequence parallelism for :class:`ToyRecurrence`.

    The platform's head-parallel exchange gives ``ToyRecurrence.mix`` whole rows of ``recurrence_heads / sp``
    heads; ``mix`` slices its convolution and decay parameters to those heads and restarts at row boundaries.
    Head counts are checked by :class:`HeadParallelMixer` from ``ParallelismInfo.heads``.
    """

    module_class = ToyRecurrence

    def __init__(self):
        super().__init__(GATED_RECURRENCE)


class ShortConvMixer:
    """Sequence parallelism for :class:`ToyShortConv` by halo exchange: a policy of its own.

    A causal convolution of width K reads only the K - 1 tokens before each output. So rather than the
    head-parallel exchange, which moves every activation to another rank and back, each rank borrows the last
    K - 1 tokens of the previous rank's shard, runs ``mix`` on that halo plus its own tokens, and keeps its own
    outputs. Channels are never split, so any sequence-parallel degree works as long as a shard holds at least
    K - 1 tokens.
    """

    layer_type = SHORT_CONV

    def is_instance(self, module: nn.Module) -> bool:
        return isinstance(module, ToyShortConv)

    def validate_sp(self, info: ParallelismInfo, sp_size: int) -> None:
        """No head constraint: the shard-length constraint is checked when the halo is exchanged."""

    def apply_sequence_parallel(self, module: nn.Module, group) -> None:
        core = module.mix
        halo = module.kernel - 1

        def mix(u: torch.Tensor, *, cu_seqlens: torch.Tensor, shard: SequenceWindow | None = None) -> torch.Tensor:
            before = previous_tokens(group, u, halo)
            window = SequenceWindow(start=sequence_start(group, u.shape[1]), halo=before.shape[1])
            out = core(torch.cat([before, u], dim=1), cu_seqlens=cu_seqlens, shard=window)
            return out[:, window.halo :]

        setattr(module, "mix", mix)


class ToyExpertLayer(StackedExpertLayer):
    """Expert parallelism for :class:`ToyMoE`: the platform shards the stacked experts and wraps ``experts_core``."""

    module_class = ToyMoE
    expert_names = ("experts.w1", "experts.w2", "experts.w3")


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
