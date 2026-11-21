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

"""Activation checkpointing of a plugin model's decoder layers, and how plugin and platform code tell a recompute
apart.

Every checkpointed call gets a :class:`CheckpointFrame`, entered on its forward and again on its recompute (through
``torch.utils.checkpoint``'s ``context_fn``, so no private checkpoint hooks are patched). The platform records MoE
routing per frame on the forward and replays it on that frame's recompute (``routing.py``), so a pending forward or
a gradient-accumulation step never consumes another forward's routing.
"""

from __future__ import annotations

import threading

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

_state = threading.local()


class CheckpointFrame:
    """One checkpointed call: what its forward recorded, for its recompute to replay in the same order."""

    def __init__(self):
        self.captures: dict[int, list[torch.Tensor]] = {}
        self.positions: dict[int, int] = {}

    def entered(self, recomputing: bool) -> _Entered:
        return _Entered(self, recomputing)


class _Entered:
    """Entering a frame, for its forward or a recompute; reusable, as ``backward(retain_graph=True)`` recomputes
    the same frame again."""

    def __init__(self, frame: CheckpointFrame, recomputing: bool):
        self.frame, self.recomputing = frame, recomputing

    def __enter__(self) -> None:
        if self.recomputing:
            self.frame.positions = {}
        active_frames().append((self.frame, self.recomputing))

    def __exit__(self, *exc) -> None:
        active_frames().pop()


def active_frames() -> list[tuple[CheckpointFrame, bool]]:
    """The checkpoint frames entered on this thread, outermost first, each with whether it is recomputing."""
    if not hasattr(_state, "frames"):
        _state.frames = []
    return _state.frames


def in_recompute() -> bool:
    """Whether the current forward is the backward pass's recompute of a checkpointed layer.

    A decoder layer runs twice under activation checkpointing. Side effects, such as updating statistics, must
    happen only on the first run: guard them with ``if not in_recompute()``.
    """
    return any(recomputing for _, recomputing in active_frames())


def _frame_contexts():
    frame = CheckpointFrame()
    return frame.entered(False), frame.entered(True)


def apply_activation_checkpointing(model: nn.Module) -> None:
    """Checkpoint every decoder layer in ``model.model.layers``: keep its inputs, recompute it in backward."""
    layers = model.get_submodule("model.layers")
    for layer in layers.children():
        forward = layer.forward

        def checkpointed(*args, _forward=forward, **kwargs):
            return checkpoint(_forward, *args, use_reentrant=False, context_fn=_frame_contexts, **kwargs)

        setattr(layer, "forward", checkpointed)
