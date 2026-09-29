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
"""Per-transformer-layer compilation patch."""

from __future__ import annotations

import torch.nn as nn

from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch


def _transformer_layers(model: nn.Module):
    target = model
    for part in ("model", "layers"):
        target = getattr(target, part, None)
        if target is None:
            raise ValueError("compile patch requires model.model.layers")
    return target


@register_patch("compile")
def apply_compile(model: nn.Module, ctx: LoaderContext) -> None:
    settings = ctx.spec.patches.compile
    if settings is None:
        raise ValueError("compile patch requires patches.compile configuration")
    for layer in _transformer_layers(model):
        layer.compile(fullgraph=settings.fullgraph)
