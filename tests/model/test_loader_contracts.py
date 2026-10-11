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

"""Tests for public model-loading contracts."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from arctic_platform.model import ModelParallelismMetadata
from arctic_platform.model import canonical_parameter_name
from arctic_platform.model import finalize_model_for_training
from arctic_platform.model import model_parallelism_metadata_from_config


def test_model_parallelism_metadata_from_flat_config():
    metadata = model_parallelism_metadata_from_config(
        {
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }
    )

    assert metadata == ModelParallelismMetadata(
        num_attention_heads=32,
        num_key_value_heads=8,
        has_linear_attention=False,
        linear_num_key_heads=None,
        linear_num_value_heads=None,
    )
    assert ModelParallelismMetadata.from_dict(metadata.to_dict()) == metadata


def test_model_parallelism_metadata_uses_nested_text_config():
    metadata = model_parallelism_metadata_from_config(
        {
            "num_attention_heads": 64,
            "text_config": {
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "layer_types": ["full_attention", "linear_attention"],
                "linear_num_key_heads": 8,
                "linear_num_value_heads": 16,
            },
        }
    )

    assert metadata == ModelParallelismMetadata(
        num_attention_heads=16,
        num_key_value_heads=4,
        has_linear_attention=True,
        linear_num_key_heads=8,
        linear_num_value_heads=16,
    )


def test_model_parallelism_metadata_nested_config_falls_back_to_root():
    metadata = model_parallelism_metadata_from_config(
        {
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "layer_types": ["linear_attention"],
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
            "text_config": {"model_type": "text"},
        }
    )

    assert metadata == ModelParallelismMetadata(
        num_attention_heads=32,
        num_key_value_heads=8,
        has_linear_attention=True,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
    )


def test_model_parallelism_metadata_uses_glm_kda_head_fallback():
    metadata = model_parallelism_metadata_from_config(
        {
            "text_config": {
                "num_attention_heads": 64,
                "layer_types": ["linear_attention"],
                "linear_attn_config": {"num_heads": 16},
            }
        }
    )

    assert metadata.has_linear_attention is True
    assert metadata.linear_num_key_heads == 16
    assert metadata.linear_num_value_heads == 16


def test_model_parallelism_metadata_detects_linear_head_fields():
    metadata = model_parallelism_metadata_from_config(
        {
            "num_attention_heads": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
        }
    )

    assert metadata.has_linear_attention is True


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "num_attention_heads must be a positive integer"),
        ({"num_attention_heads": True}, "num_attention_heads must be a positive integer"),
        ({"num_attention_heads": 0}, "num_attention_heads must be a positive integer"),
        (
            {"num_attention_heads": 8, "num_key_value_heads": 0},
            "num_key_value_heads must be a positive integer when set",
        ),
        (
            {"num_attention_heads": 8, "layer_types": "linear_attention"},
            "layer_types must be a list",
        ),
        (
            {"num_attention_heads": 8, "layer_types": ["linear_attention"]},
            "linear_num_key_heads must be a positive integer",
        ),
    ],
)
def test_model_parallelism_metadata_validates_config(config, message):
    with pytest.raises(ValueError, match=message + " in checkpoint config"):
        model_parallelism_metadata_from_config(
            config,
            source="checkpoint config",
        )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.0.weight", "model.layers.0.weight"),
        (
            "model._checkpoint_wrapped_module.layers.0.weight",
            "model.layers.0.weight",
        ),
        (
            "model._checkpoint_wrapped_module._checkpoint_wrapped_module.weight",
            "model.weight",
        ),
        (
            "_checkpoint_wrapped_module.model._checkpoint_wrapped_module.weight",
            "model.weight",
        ),
    ],
)
def test_canonical_parameter_name(name, expected):
    assert canonical_parameter_name(name) == expected


class _ParameterHolder(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(1, dtype=dtype),
            requires_grad=True,
        )


class _TransformedModel(nn.Module):
    def __init__(self, float8_dtypes: list[torch.dtype]):
        super().__init__()
        self.fp8 = nn.ModuleList([_ParameterHolder(dtype) for dtype in float8_dtypes])
        self.weight_scale_inv = nn.Parameter(torch.ones(1), requires_grad=True)
        self.lora_A = nn.Parameter(torch.ones(1), requires_grad=True)
        self.adapter_bias = nn.Parameter(torch.ones(1), requires_grad=True)


def test_finalize_model_for_training_refreezes_storage_and_scales():
    float8_dtypes = [
        dtype
        for name in dir(torch)
        if name.startswith("float8_") and isinstance((dtype := getattr(torch, name)), torch.dtype)
    ]
    model = _TransformedModel(float8_dtypes)

    frozen = finalize_model_for_training(model)

    assert frozen == len(float8_dtypes) + 1
    assert all(not holder.weight.requires_grad for holder in model.fp8)
    assert model.weight_scale_inv.requires_grad is False
    assert model.lora_A.requires_grad is True
    assert model.adapter_bias.requires_grad is True
    assert finalize_model_for_training(model) == 0
