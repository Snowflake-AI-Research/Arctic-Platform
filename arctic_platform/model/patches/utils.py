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


def transformer_layers(model: nn.Module, *, patch_name: str):
    target = model
    for part in ("model", "layers"):
        target = getattr(target, part, None)
        if target is None:
            raise ValueError(f"{patch_name} patch requires model.model.layers")
    return target
