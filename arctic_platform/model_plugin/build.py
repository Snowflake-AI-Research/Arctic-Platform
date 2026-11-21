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

"""Building a plugin's trainer model: its options, and the meta-device build the platform uses at scale.

At scale the platform builds the model on the meta device, which costs no memory, then allocates it with
``to_empty`` and fills it from the checkpoint (ZeRO then partitions it). A buffer computed in ``__init__`` and not checkpointed
(a rotary ``inv_freq``, a counter) would be left as uninitialized memory, so a module with such buffers implements
``ap_init_buffers()`` (:class:`~arctic_platform.model_plugin.interfaces.InitializesBuffers`), which :func:`materialize` calls on every module after ``to_empty``.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from arctic_platform.model_plugin.interfaces import InitializesBuffers
from arctic_platform.model_plugin.marks import apply_marks
from arctic_platform.model_plugin.marks import cast_model
from arctic_platform.model_plugin.plugin import ModelPlugin


def build(plugin: ModelPlugin, config: Any, options: Any = None, *, meta: bool = False) -> nn.Module:
    """The plugin's trainer model for ``config`` and the job's ``options`` mapping; with ``meta``, on the meta
    device, to be allocated by :func:`materialize`."""
    parsed = plugin.parse_options(options)
    with torch.device("meta") if meta else nullcontext():
        model = plugin.build_model(config, parsed)
    apply_marks(model)
    return model


def init_buffers(model: nn.Module) -> None:
    """Call ``ap_init_buffers()`` on every :class:`InitializesBuffers` module."""
    for module in model.modules():
        # callable(): a subclass opts out with ``ap_init_buffers = None`` but still inherits the protocol.
        if isinstance(module, InitializesBuffers) and callable(module.ap_init_buffers):
            module.ap_init_buffers()


def materialize(model: nn.Module, device: str | torch.device, dtype: torch.dtype | None = None) -> nn.Module:
    """Allocate a meta-device model on ``device`` (contents uninitialized until loaded), recompute its
    non-checkpointed buffers, and cast it to ``dtype`` keeping fp32-marked tensors; returns the model.

    ``to_empty`` replaces parameter objects, giving each name of a tied parameter its own; ties are restored, and
    attributes set on a tensor itself (marks, and the expert-parallel tags :func:`prepare_model` sets on a model
    prepared on the meta device) are carried over by name."""
    aliases: dict[int, list[str]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(param), []).append(name)
    own_attributes = {
        name: dict(vars(tensor))
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]
        if vars(tensor)
    }
    model.to_empty(device=device)
    for first, *others in aliases.values():
        for name in others:
            owner, _, attribute = name.rpartition(".")
            setattr(model.get_submodule(owner), attribute, model.get_parameter(first))
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        for attribute, value in own_attributes.get(name, {}).items():
            setattr(tensor, attribute, value)
    init_buffers(model)
    if dtype is not None:
        cast_model(model, dtype)
    apply_marks(model)
    return model
