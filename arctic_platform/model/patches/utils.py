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
"""Utilities shared by model patches."""

from __future__ import annotations

import torch.nn as nn


def text_decoder(model: nn.Module, *, patch_name: str) -> nn.Module:
    """The model's text decoder: the module holding the decoder layers (``layers``) and the input embedding.

    ``get_decoder()`` when it returns one (a multimodal model such as Gemma 4 returns its language model), else the one
    module with both a ``layers`` ModuleList and ``embed_tokens``.
    """
    get_decoder = getattr(model, "get_decoder", None)
    decoder = get_decoder() if callable(get_decoder) else None
    if decoder is not None and isinstance(getattr(decoder, "layers", None), nn.ModuleList):
        return decoder
    found = [
        module
        for module in model.modules()
        if isinstance(getattr(module, "layers", None), nn.ModuleList) and hasattr(module, "embed_tokens")
    ]
    if len(found) != 1:
        raise ValueError(
            f"{patch_name}: found {len(found)} modules with decoder layers and an input embedding; the patch needs "
            "the text decoder of a causal LM"
        )
    return found[0]


def transformer_layers(model: nn.Module, *, patch_name: str):
    target = model
    for part in ("model", "layers"):
        target = getattr(target, part, None)
        if target is None:
            raise ValueError(f"{patch_name} patch requires model.model.layers")
    return target
