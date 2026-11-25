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

"""MoE routing the platform owns: sampler replay, recompute replay, and the routing readout.

A plugin's MoE module routes through one method, ``route(tokens, routed=None) -> (selected, ...)``, which picks
experts with :func:`choose_experts`. :func:`install_routing` wraps that method so that:

- on an activation-checkpoint recompute it replays the experts its own forward chose, recorded in that forward's
  checkpoint frame. Nondeterministic kernels can perturb router scores in the last bits, so a recomputed ``topk``
  could otherwise pick other experts: a shape mismatch, or silently wrong gradients;
- otherwise it replays the sampler's routing for that layer, if the caller supplied it (RL router replay);
- after a forward call it records the experts chosen, for the recompute and for the caller.

Model code never carries routing.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from arctic_platform.model_plugin.experts import ROUTER_REPLAY_FRESH
from arctic_platform.model_plugin.interfaces import RoutedModule
from arctic_platform.model_plugin.recompute import active_frames

ROUTING_TAPE_ATTR = "_ap_routing_tape"
_ROUTING_ATTR = "_ap_routing"


def choose_experts(
    scores: torch.Tensor, top_k: int, routed: torch.Tensor | None = None, *, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Expert ids ``[N, top_k]``: ``topk`` of the detached ``scores`` (plus ``bias``, which only steers the choice),
    or ``routed`` where given, whose ``ROUTER_REPLAY_FRESH`` entries fall back to that own choice."""
    choice = scores.detach() if bias is None else scores.detach() + bias
    own = torch.topk(choice, top_k, dim=-1).indices
    if routed is None:
        return own
    routed = routed.to(device=own.device, dtype=own.dtype)
    return torch.where(routed == ROUTER_REPLAY_FRESH, own, routed)


class RoutingTape:
    """One model's routing: the sampler's replay going in, and the experts each layer chose coming out."""

    def __init__(self):
        self.replay: dict[int, torch.Tensor] = {}
        # Collected only inside host.forward, which resets it per forward.
        self.readout: dict[int, torch.Tensor] | None = None

    def route(self, layer: int, call: Callable[[torch.Tensor | None], tuple]) -> tuple:
        """Run ``call(routed)``, a layer's own route, with the routing it must replay."""
        frames = active_frames()
        recomputing = [frame for frame, is_recompute in frames if is_recompute]
        if len(recomputing) > 0:
            frame = recomputing[-1]
            position = frame.positions.get(layer, 0)
            captures = frame.captures.get(layer, [])
            if position >= len(captures):
                raise RuntimeError(f"recompute of layer {layer} found no routing recorded by its checkpoint forward")
            frame.positions[layer] = position + 1
            out = call(captures[position])
        else:
            out = call(self.replay.get(layer))
        selected = out[0].detach()
        # Every frame still on its forward records the call, outer frames included; a new forward inside a
        # recompute records for its own later backward.
        for frame, is_recompute in frames:
            if not is_recompute:
                frame.captures.setdefault(layer, []).append(selected)
        if len(recomputing) == 0 and self.readout is not None:
            if layer in self.readout:
                raise RuntimeError(
                    f"decoder layer {layer} routed twice in one forward; route must be called once per forward, on all"
                    " of the layer's local tokens"
                )
            self.readout[layer] = selected
        return out


_ROUTED_CLASSES: dict[type, type] = {}


def _routed_class(cls: type) -> type:
    """``cls`` with ``route`` routed through the module's tape. A subclass rather than an instance attribute, so a
    deep copy of the model routes through its own weights."""
    if cls not in _ROUTED_CLASSES:

        def route(self, *args, **kwargs):
            kwargs.pop("routed", None)  # the platform decides what is replayed
            layer, tape = getattr(self, _ROUTING_ATTR)
            original = cls.route  # looked up per call, so the class's current method runs
            return tape.route(layer, lambda ids: original(self, *args, routed=ids, **kwargs))

        attributes = {"route": route, "__qualname__": cls.__qualname__, "_ap_routed": True}
        _ROUTED_CLASSES[cls] = type(cls.__name__, (cls,), attributes)
    return _ROUTED_CLASSES[cls]


def install_routing(module: RoutedModule, layer_index: int, tape: RoutingTape) -> None:
    """Route ``module.route`` (:class:`~arctic_platform.model_plugin.interfaces.RoutedModule`) through ``tape`` as
    decoder layer ``layer_index``. Installing again rebinds.

    The module's class is swapped for a routed subclass (deep copies route through their own weights); a prepared
    model therefore cannot be pickled whole. Save it as a checkpoint."""
    setattr(module, _ROUTING_ATTR, (layer_index, tape))
    if not getattr(type(module), "_ap_routed", False):  # inherited by anything that subclasses it later
        module.__class__ = _routed_class(type(module))
