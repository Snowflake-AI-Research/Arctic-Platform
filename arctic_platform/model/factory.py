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
"""Build a torch model from a ModelSpec."""

from __future__ import annotations

from typing import Any

import torch

from arctic_platform.model.config import ModelSpec
from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import select_loader
from arctic_platform.model.patch import apply_patches
from arctic_platform.peft import apply_peft


def build_model(spec: ModelSpec, parallel_groups: Any | None = None) -> LoadedModel:
    """Run the spec's resolved loader, apply its patches, and return the result."""
    ctx = LoaderContext(spec=spec, parallel_groups=parallel_groups)
    loaded = select_loader(ctx)(ctx)
    apply_patches(loaded, ctx)
    if spec.peft_config:
        dtype = torch.bfloat16 if spec.dtype == "auto" else getattr(torch, spec.dtype)
        loaded.model = apply_peft(loaded.model, spec.peft_config, optimization_dtype=dtype)
    return loaded
