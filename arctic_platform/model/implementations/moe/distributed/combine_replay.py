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
"""Replay a checkpointed MoE block's combine on its recompute instead of running it again.

A whole-block recompute that runs to the end of the block (no early stop) repeats the block's EP combine, an
all-to-all over the EP group, only to rebuild the tensors saved after it. The backward never reaches that
second combine: non-reentrant checkpointing runs the backward through the original forward's graph, so the
recompute's combine node is discarded once its output has been consumed. Its output only has to equal the
original one, which the original forward already produced. A ``CombineReplay`` keeps each combined output of a
graph-building forward under that forward's checkpoint frame and hands it back, chunk by chunk, to that
frame's recompute. A later forward does not discard an earlier one that has not been recomputed.
"""

from __future__ import annotations

import weakref
from collections import deque
from typing import Callable

import torch

from arctic_platform.model.implementations.gpu.router_replay_recompute import _in_recompute
from arctic_platform.model.implementations.gpu.router_replay_recompute import _install_recompute_detector

_OUTSIDE = object()


def _checkpoint_frame() -> object | None:
    """The non-reentrant checkpoint frame whose forward or recompute is running, if there is one.

    The frame is the closure of the saved-tensor hook ``checkpoint`` installs for that call. Forward and
    recompute of one call share it, and two calls get two frames, so interleaved forwards do not share a queue.
    """
    top = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None)
    if top is None:
        return None
    try:
        hooks = top(False)
    except (AttributeError, TypeError):
        return None
    if not hooks:
        return None
    return _frame_in_closure(hooks[0], set())


def _frame_in_closure(fn: object, seen: set[int]) -> object | None:
    closure = getattr(fn, "__closure__", None)
    if not closure or id(fn) in seen:
        return None
    seen.add(id(fn))
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if type(value).__name__ == "_CheckpointFrame":
            return value
        if isinstance(value, weakref.ref):
            target = value()
            if target is not None and type(target).__name__ == "_CheckpointFrame":
                return target
        if getattr(value, "__closure__", None):
            found = _frame_in_closure(value, seen)
            if found is not None:
                return found
    return None


class CombineReplay:
    """Combined outputs of one MoE layer, one queue per checkpointed forward, oldest chunk first."""

    def __init__(self) -> None:
        _install_recompute_detector()
        self._by_forward: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._outside: deque[tuple[torch.Tensor, int]] | None = None
        self._writing: object | None = None

    def start(self) -> None:
        """Begin a graph-building forward. Its recompute reads this queue; other forwards keep theirs."""
        if not torch.is_grad_enabled() or _in_recompute():
            return
        frame = _checkpoint_frame()
        bucket: deque[tuple[torch.Tensor, int]] = deque()
        if frame is None:
            self._outside = bucket
            self._writing = _OUTSIDE
            return
        self._by_forward[frame] = bucket
        self._writing = None

    def _queue(self, *, reading: bool) -> deque[tuple[torch.Tensor, int]] | None:
        frame = _checkpoint_frame()
        if frame is not None:
            return self._by_forward.get(frame)
        if reading:
            return self._outside
        if self._writing is _OUTSIDE:
            return self._outside
        return None

    def run(
        self,
        combine: Callable[[], torch.Tensor],
        replayed: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """``combine()`` on a forward, keeping its output when the forward builds a graph; on the recompute,
        ``replayed(output)``: a node that stands in for the combine without communicating."""
        if _in_recompute():
            queue = self._queue(reading=True)
            if not queue:
                raise RuntimeError("combine replay: a recompute found no combined output from its forward")
            combined, version = queue.popleft()
            if combined._version != version:
                raise RuntimeError("combine replay: the combined output was modified in place after its forward")
            return replayed(combined)
        combined = combine()
        if torch.is_grad_enabled():
            queue = self._queue(reading=False)
            if queue is None:
                raise RuntimeError("combine replay: run() before start() on a graph-building forward")
            queue.append((combined, combined._version))
        return combined
