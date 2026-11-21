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

"""The methods and attributes a plugin's own modules implement: one protocol per role a module plays.

The platform calls these hooks on the modules a plugin builds; the plugin never calls the platform from them. Each
protocol is the single statement of its hook's signature and obligations (``docs/model-plugins/spec.md`` §7 refers
here).

Declare the roles a module plays by listing the protocols as bases after ``nn.Module``::

    class AcmeMoE(nn.Module, StackedMoEModule): ...

Inheriting is optional, since the platform checks structure at runtime, but it documents the module's contract to
readers, and lets mypy and IDEs check the signatures against the protocol.

========================  ===============================================  ==========================
Role                      Protocol                                         Required
========================  ===============================================  ==========================
Trainer model             :class:`CausalLM`                                always
Output transform          :class:`ComputesLogits`                          optional
Non-checkpointed buffers  :class:`InitializesBuffers`                      if the module has them
Sequence mixing           :class:`SequenceMixingModule`                    every layer and sub-layer
                                                                           with a mixer
MoE routing               :class:`RoutedModule`                            every expert layer
Stacked experts           :class:`StackedMoEModule`                        with ``StackedExpertLayer``
========================  ===============================================  ==========================

The objects a plugin hands the platform alongside its model, rather than methods on its modules, have protocols of
their own: :class:`~arctic_platform.model_plugin.mixers.Mixer` and
:class:`~arctic_platform.model_plugin.experts.ExpertLayer`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import runtime_checkable

import torch
from torch import nn

if TYPE_CHECKING:
    from arctic_platform.model_plugin.experts import ExpertShard


@runtime_checkable
class CausalLM(Protocol):
    """The trainer model ``build_model(config, options)`` returns (spec §7.2).

    Decoder layer ``i``'s parameters and buffers live under ``model.layers.<i>.``; embeddings, the final norm and
    ``lm_head`` live outside. Layer count and kinds come from ``config``, so one plugin serves every size of a family.
    """

    config: Any
    """The model's config, kept as given to ``build_model``."""

    lm_head: nn.Module
    """Hidden states ``[..., D]`` to logits ``[..., V]``, used unless the model is also :class:`ComputesLogits`."""

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, *, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """The final hidden states ``[1, T_local, D]`` (after the final norm, before ``lm_head``).

        ``input_ids`` and ``position_ids`` are this rank's ``[1, T_local]`` sequence shard of a packed batch;
        ``cu_seqlens`` describes the full, unsharded batch. Never calls ``torch.distributed``, carries no routing, and
        gives the same result when a decoder layer is rerun in backward.
        """
        ...


@runtime_checkable
class ComputesLogits(Protocol):
    """A trainer model with an output transform, such as muP output scaling or logit softcapping."""

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Logits ``[..., V]`` of hidden states ``[..., D]``, used instead of ``lm_head(hidden)``.

        MUST be token-wise: the platform calls it on chunks of tokens and recomputes each chunk in backward.
        """
        ...


@runtime_checkable
class InitializesBuffers(Protocol):
    """A module with buffers that are computed rather than checkpointed (a rotary ``inv_freq``, counters)."""

    def ap_init_buffers(self) -> None:
        """Recompute this module's non-checkpointed buffers in place, on their current device.

        The platform builds models on the meta device and allocates them with ``to_empty``, which leaves such buffers
        uninitialized; it calls this on every module after ``to_empty``. Set it to ``None`` to opt a subclass out.
        """
        ...


@runtime_checkable
class SequenceMixingModule(Protocol):
    """A module whose core mixes information across tokens: a layer or sub-layer bound to a mixer (spec §7.3).

    Projections before and after ``mix`` work token by token; ``mix`` is the only cross-token code, and the layer
    type's :class:`~arctic_platform.model_plugin.mixers.Mixer` makes it sequence-parallel by wrapping it. Subclass
    :class:`~arctic_platform.model_plugin.attention.AttentionModule` for attention, which supplies ``mix``.
    """

    ap_layer_type: str
    """This module's entry in the config's ``layer_types``, or its manifest ``[sublayer_types]`` entry."""

    mix: Callable[..., Any]
    """``mix(*tensors, cu_seqlens, shard, **state) -> output | (output, *extras)``: the sequence-mixing core.

    The module's forward calls it as ``self.mix(*tensors, cu_seqlens=cu_seqlens)``. ``shard`` describes what this
    rank computes, in its policy's type: :class:`~arctic_platform.model_plugin.parallel.HeadShard` for head-parallel
    layers (default ``FULL_SHARD``), :class:`~arctic_platform.model_plugin.parallel.SequenceWindow` for halo layers
    (default ``FULL_WINDOW``). ``mix`` MUST restart all state at every ``cu_seqlens`` boundary (which describes the
    full batch), compute exactly what ``shard`` describes, and read no tokens outside its inputs. Declared as a
    callable rather than a method because its tensor arguments differ per layer type.
    """


@runtime_checkable
class RoutedModule(Protocol):
    """A mixture-of-experts module whose routing the platform owns (spec §7.6, §7.7).

    The module's forward calls ``self.route(tokens)`` exactly once, on all of the layer's local tokens. The platform
    wraps ``route`` to replay the sampler's routing, replay the forward's own routing on activation recompute, and
    read out the experts chosen.
    """

    route: Callable[..., tuple[torch.Tensor, ...]]
    """``route(tokens [N, D], routed: Tensor | None = None, ...) -> (selected [N, top_k], ...)``.

    Picks experts with :func:`~arctic_platform.model_plugin.routing.choose_experts` ``(scores, top_k, routed)`` and
    returns the selection first. ``routed`` MUST be a parameter: only the platform supplies it. Further arguments
    (for example ``input_ids`` for a hash-routed layer) are passed through. Declared as a callable rather than a
    method so those extra arguments are allowed.
    """


@runtime_checkable
class StackedMoEModule(RoutedModule, Protocol):
    """A :class:`RoutedModule` with routed experts stacked on dim 0, as
    :class:`~arctic_platform.model_plugin.experts.StackedExpertLayer` shards them (spec §7.6)."""

    num_experts: int
    """Routed experts in the full, unsharded layer."""

    def experts_core(
        self, tokens: torch.Tensor, selected: torch.Tensor, weights: torch.Tensor, *, shard: ExpertShard
    ) -> torch.Tensor:
        """The summed contribution ``[N, D]`` of the experts ``shard`` holds, for ``tokens [N, D]`` with chosen
        expert ids ``selected [N, top_k]`` and gate weights ``weights [N, top_k]``.

        Local expert ``i`` (index ``i`` of the stacked tensors) is global expert ``shard.start + i``. Under expert
        parallelism the platform wraps this method so ``tokens`` are the whole group's; it never moves tokens itself.
        Give ``shard`` a default (``None`` or ``ExpertShard.full(num_experts)``) so the module's forward can call it
        unsharded.
        """
        ...
