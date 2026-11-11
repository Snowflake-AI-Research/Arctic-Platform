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
"""Add each LoRA linear's low-rank update to the base output inside the ``lora_B`` GEMM.

PEFT's ``lora.Linear`` computes ``result + lora_B(lora_A(dropout(x))) * scaling``: the ``lora_B`` GEMM writes a
``[tokens, out]`` tensor, the scaling reads and writes it again, and the add reads it and the base output to write a
third one (and backward repeats the scaling pass). With every operand in one dtype, a single ``addmm`` computes
``result + scaling * (lora_A(x) @ lora_B.weight.T)`` with the base output as the GEMM's accumulator input, so the
update costs one read of the base output and one write. Modules with dropout, a LoRA variant (DoRA, ...), a bias on
the LoRA factors, merged or disabled adapters, mixed dtypes or extra forward arguments keep PEFT's forward.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch


def _fusable_adapters(module, x) -> list[str] | None:
    """The active vanilla adapters whose update ``addmm`` computes as PEFT's composition does, or None."""
    base = module.base_layer
    if type(base) is not nn.Linear or x.dtype != base.weight.dtype:
        return None
    dtype = base.weight.dtype
    names = []
    for name in module.active_adapters:
        if name not in module.lora_A.keys():
            continue
        lora_a, lora_b = module.lora_A[name], module.lora_B[name]
        if (
            name in module.lora_variant
            or not isinstance(module.lora_dropout[name], nn.Identity)
            or type(lora_a) is not nn.Linear
            or type(lora_b) is not nn.Linear
            or lora_a.bias is not None
            or lora_b.bias is not None
            or lora_a.weight.dtype != dtype
            or lora_b.weight.dtype != dtype
        ):
            return None
        names.append(name)
    return names


def fused_lora_linear_forward(self, x, *args, **kwargs):
    """PEFT ``lora.Linear.forward`` with the update added in the ``lora_B`` GEMM when every operand shares a dtype."""
    names = None if (args or kwargs or self.disable_adapters or self.merged) else _fusable_adapters(self, x)
    if names is None:
        return type(self).forward(self, x, *args, **kwargs)
    result = self.base_layer(x)
    shape = result.shape
    flat = result.reshape(-1, shape[-1])
    for name in names:
        lora_a, lora_b = self.lora_A[name], self.lora_B[name]
        update = lora_a(x).reshape(-1, lora_a.out_features)
        flat = torch.addmm(flat, update, lora_b.weight.t(), alpha=self.scaling[name])
    return flat.view(shape)


@register_patch("fused_lora_linear")
def apply_fused_lora_linear(model: nn.Module, ctx: LoaderContext) -> None:
    from peft.tuners.lora.layer import Linear as LoraLinear

    patched = 0
    for module in model.modules():
        if type(module) is LoraLinear:
            module.forward = types.MethodType(fused_lora_linear_forward, module)
            patched += 1
    if patched == 0:
        raise ValueError("fused_lora_linear found no PEFT LoRA linear layer; it needs peft with a LoRA config")
