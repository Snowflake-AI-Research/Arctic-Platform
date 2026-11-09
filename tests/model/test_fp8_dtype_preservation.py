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

"""FP8 weights must survive DeepSpeed's mixed-precision cast.

Older DeepSpeed called ``module.bfloat16()``. Current DeepSpeed walks
``parameters()`` in ``_cast_module_mixed_precision`` and does
``p.data.to(bfloat16)``. Torch reports e4m3 as floating point, so an unguarded
cast doubles frozen FP8 storage -- 42 GiB per GPU of expert weights on GLM-5.3
at EP=16 -- and forces a 96 MiB ``o_proj`` recast during AC recompute.
"""

import pytest
import torch
from torch import nn

from arctic_platform.model import preserve_fp8_params_through_ds_cast


def _preserve(model) -> int:
    preserved, _marked = preserve_fp8_params_through_ds_cast(model)
    return preserved


def _deepspeed_engine_or_skip():
    """DeepSpeed old enough to call ``module.bfloat16()`` has no cast to patch."""
    from deepspeed.runtime.engine import DeepSpeedEngine

    if not hasattr(DeepSpeedEngine, "_cast_module_mixed_precision"):
        pytest.skip("DeepSpeed predates the per-parameter mixed-precision cast")
    return DeepSpeedEngine


def _fp8_moe_like_model() -> nn.Module:
    from arctic_platform.model import mark_keep_fp32

    model = nn.Module()
    model.expert_weight = nn.Parameter(torch.empty(8, 4, dtype=torch.float8_e4m3fn), requires_grad=False)
    model.weight_scale_inv = mark_keep_fp32(nn.Parameter(torch.ones(1, 1, dtype=torch.float32), requires_grad=False))
    model.lora_a = nn.Parameter(torch.ones(4, 4, dtype=torch.float32))
    return model


def test_fp8_params_survive_bfloat16_cast():
    model = _fp8_moe_like_model()
    preserved = _preserve(model)

    assert preserved == 1
    model.bfloat16()

    assert model.expert_weight.dtype == torch.float8_e4m3fn
    assert model.expert_weight.element_size() == 1
    assert model.weight_scale_inv.dtype == torch.float32
    # Trainable non-FP8 params must still be cast, or DeepSpeed's bf16 path breaks.
    assert model.lora_a.dtype == torch.bfloat16


def test_deepspeed_per_param_cast_keeps_scales_fp32():
    model = _fp8_moe_like_model()
    model.weight_scale_inv.data = torch.tensor([[1.0 + 2**-10]], dtype=torch.float32)
    exact = model.weight_scale_inv.item()
    DeepSpeedEngine = _deepspeed_engine_or_skip()
    _preserve(model)

    class _Engine:
        def __init__(self, module):
            self.module = module

    DeepSpeedEngine._cast_module_mixed_precision(_Engine(model), torch.bfloat16, None, False)

    assert model.weight_scale_inv.dtype == torch.float32
    assert model.weight_scale_inv.item() == exact


def _ds_cast(model) -> None:
    DeepSpeedEngine = _deepspeed_engine_or_skip()

    class _Engine:
        def __init__(self, module):
            self.module = module

    DeepSpeedEngine._cast_module_mixed_precision(_Engine(model), torch.bfloat16, None, False)


def _rebuilt_like_meta_materialization(model) -> None:
    """Replace every parameter object, dropping attributes set at construction."""
    for module in model.modules():
        for pname, p in list(module.named_parameters(recurse=False)):
            module.register_parameter(pname, nn.Parameter(p.data.clone(), requires_grad=p.requires_grad))


def _fp8_linear_model():
    from arctic_platform.model.implementations.fp8 import BlockFp8Linear

    model = nn.Module()
    model.attn = BlockFp8Linear(in_features=256, out_features=256, block_size=128, device="cpu")
    return model


def test_scales_survive_a_rebuild_before_deepspeed_init():
    model = _fp8_linear_model()
    model.attn.weight_scale_inv.data = torch.full((2, 2), 1.0 + 2**-10, dtype=torch.float32)
    exact = model.attn.weight_scale_inv[0, 0].item()
    _rebuilt_like_meta_materialization(model)

    _preserve(model)
    _ds_cast(model)

    assert model.attn.weight.dtype == torch.float8_e4m3fn
    assert model.attn.weight_scale_inv.dtype == torch.float32
    assert model.attn.weight_scale_inv[0, 0].item() == exact


def test_models_without_fp8_are_left_alone():
    model = nn.Linear(4, 4)

    assert _preserve(model) == 0
    assert "bfloat16" not in vars(model)

    model.bfloat16()
    assert model.weight.dtype == torch.bfloat16
