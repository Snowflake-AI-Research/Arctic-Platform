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
"""PEFT wrapping after model patches and before optimizer construction."""

from __future__ import annotations

import torch
from torch import nn

from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch
from arctic_platform.peft import apply_peft


@register_patch("peft")
def apply_peft_patch(model: nn.Module, ctx: LoaderContext) -> nn.Module:
    dtype = torch.bfloat16 if ctx.spec.dtype == "auto" else getattr(torch, ctx.spec.dtype)
    return apply_peft(model, ctx.spec.patches.peft, optimization_dtype=dtype)
