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

"""Activation checkpointing of a plugin model's decoder layers, and how plugin code tells a recompute apart."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextlib import nullcontext

from torch import nn
from torch.utils.checkpoint import checkpoint

_state = threading.local()


def in_recompute() -> bool:
    """Whether the current forward is the backward pass's recompute of a checkpointed layer.

    A decoder layer runs twice under activation checkpointing. Side effects, such as updating statistics, must
    happen only on the first run: guard them with ``if not in_recompute()``.
    """
    return getattr(_state, "active", False)


@contextmanager
def _recomputing() -> Iterator[None]:
    previous = in_recompute()
    _state.active = True
    try:
        yield
    finally:
        _state.active = previous


def apply_activation_checkpointing(model: nn.Module) -> None:
    """Checkpoint every decoder layer in ``model.model.layers``: keep its inputs, recompute it in backward."""
    layers = model.get_submodule("model.layers")
    for layer in layers.children():
        forward = layer.forward

        def checkpointed(*args, _forward=forward, **kwargs):
            return checkpoint(
                _forward, *args, use_reentrant=False, context_fn=lambda: (nullcontext(), _recomputing()), **kwargs
            )

        setattr(layer, "forward", checkpointed)
