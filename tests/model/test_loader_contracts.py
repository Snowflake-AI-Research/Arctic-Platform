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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from arctic_platform.model import ModelParallelismMetadata
from arctic_platform.model import canonical_parameter_name
from arctic_platform.model import freeze_fp8_weights_and_scales
from arctic_platform.model import model_parallelism_metadata_from_config
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close


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


def test_freeze_fp8_weights_and_scales():
    float8_dtypes = [
        dtype
        for name in dir(torch)
        if name.startswith("float8_") and isinstance((dtype := getattr(torch, name)), torch.dtype)
    ]
    model = _TransformedModel(float8_dtypes)

    frozen = freeze_fp8_weights_and_scales(model)

    assert frozen == len(float8_dtypes) + 1
    assert all(not holder.weight.requires_grad for holder in model.fp8)
    assert model.weight_scale_inv.requires_grad is False
    assert model.lora_A.requires_grad is True
    assert model.adapter_bias.requires_grad is True
    assert freeze_fp8_weights_and_scales(model) == 0


class TestQwen3BoundedLossLoader(TestCasePlus):
    def _tiny_model(self):
        from transformers import Qwen3Config
        from transformers import Qwen3ForCausalLM

        set_seed(42)
        config = Qwen3Config(
            vocab_size=37,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
        )
        config._attn_implementation = "eager"
        return Qwen3ForCausalLM(config)

    def _load(self, path, dtype="float32"):
        from arctic_platform.model.config import ModelSpec
        from arctic_platform.model.loader import LoaderContext
        from arctic_platform.model.loaders.huggingface import load_huggingface

        return load_huggingface(
            LoaderContext(
                spec=ModelSpec(
                    model_path_or_name=str(path),
                    loader="huggingface",
                    dtype=dtype,
                    attn_implementation="eager",
                )
            )
        ).model

    def _assert_loaded_forward_and_backward(self, dtype, device):
        from transformers import Qwen3ForCausalLM

        from arctic_platform.sft.chunked_cross_entropy import chunked_causal_lm_loss

        baseline = self._tiny_model().to(dtype=dtype)
        path = self.get_auto_remove_tmp_dir()
        baseline.save_pretrained(path)
        loaded = self._load(path, dtype=str(dtype).removeprefix("torch.")).to(device)
        baseline = Qwen3ForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="eager").to(device)
        self.assertIs(loaded.loss_function, chunked_causal_lm_loss)
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], device=device)
        labels = inputs.clone()
        labels[0, :2] = -100
        expected_parameters = dict(baseline.named_parameters())
        for name, parameter in loaded.named_parameters():
            torch_assert_close(parameter, expected_parameters[name], rtol=0, atol=0, msg=name)
        for training in (False, True):
            with self.subTest(training=training):
                baseline.train(training)
                loaded.train(training)
                baseline.zero_grad(set_to_none=True)
                loaded.zero_grad(set_to_none=True)
                expected = baseline(input_ids=inputs, labels=labels, num_items_in_batch=9)
                actual = loaded(input_ids=inputs, labels=labels, num_items_in_batch=9)
                torch_assert_close(actual.loss, expected.loss, rtol=0, atol=1e-6)
                torch_assert_close(actual.logits, expected.logits, rtol=0, atol=0)
                expected.loss.backward()
                actual.loss.backward()
                for name, parameter in loaded.named_parameters():
                    torch_assert_close(parameter.grad, expected_parameters[name].grad, rtol=0, atol=1e-6, msg=name)

    def test_real_qwen3_loading_forward_and_backward(self):
        self._assert_loaded_forward_and_backward(torch.float32, "cpu")

    @require_torch_gpu
    def test_real_qwen3_gpu_loading_forward_and_backward(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._assert_loaded_forward_and_backward(dtype, "cuda")

    def test_custom_qwen3_loss_is_preserved(self):
        from unittest.mock import patch

        from arctic_platform.model.loaders import huggingface

        model = self._tiny_model()

        def custom_loss(**kwargs):
            return kwargs["logits"].sum()

        model.loss_function = custom_loss
        with patch.object(huggingface.AutoModelForCausalLM, "from_pretrained", return_value=model):
            loaded = self._load("custom-qwen3")
        self.assertIs(loaded.loss_function, custom_loss)

    def test_other_model_families_keep_default_loss(self):
        from unittest.mock import patch

        from transformers.loss.loss_utils import ForCausalLMLoss

        from arctic_platform.model.loaders import huggingface

        for family in ("llama", "qwen3_next", "qwen3_5", "qwen3_5_moe"):
            with self.subTest(family=family):
                model = nn.Module()
                model.config = SimpleNamespace(model_type=family)
                model.loss_function = ForCausalLMLoss
                with (
                    patch.object(huggingface.AutoModelForCausalLM, "from_pretrained", return_value=model),
                    patch("arctic_platform.model.implementations.gpu.vllm_rounding.round_gated_norms_once"),
                    patch("arctic_platform.model.implementations.gpu.qwen3_5_dense_parity.apply_fp32_parity"),
                ):
                    loaded = self._load(f"other-{family}")
                self.assertIs(loaded.loss_function, ForCausalLMLoss)
