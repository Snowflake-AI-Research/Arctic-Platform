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
graph-building forward and hands it back, in order, to that forward's recompute.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import torch

from arctic_platform.model.implementations.gpu.router_replay_recompute import _in_recompute
from arctic_platform.model.implementations.gpu.router_replay_recompute import _install_recompute_detector


class CombineReplay:
    """The combined outputs of one MoE layer's checkpointed forward, oldest chunk first."""

    def __init__(self) -> None:
        _install_recompute_detector()
        self._combined: deque[tuple[torch.Tensor, int]] = deque()

    def start(self) -> None:
        """Begin a forward: a graph-building one is recomputed before this layer's next, so what it held is spent."""
        if torch.is_grad_enabled() and not _in_recompute():
            self._combined.clear()

    def run(
        self,
        combine: Callable[[], torch.Tensor],
        replayed: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """``combine()`` on a forward, keeping its output when the forward builds a graph; on the recompute,
        ``replayed(output)``: a node that stands in for the combine without communicating."""
        if _in_recompute():
            if not self._combined:
                raise RuntimeError("combine replay: a recompute found no combined output from its forward")
            combined, version = self._combined.popleft()
            if combined._version != version:
                raise RuntimeError("combine replay: the combined output was modified in place after its forward")
            return replayed(combined)
        combined = combine()
        if torch.is_grad_enabled():
            self._combined.append((combined, combined._version))
        return combined
