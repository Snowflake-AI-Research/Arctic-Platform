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

"""Generic PrimeRL MoE family loader selection and sequence-parallelism guard."""

import json

import pytest
from torch import nn

from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model import Patches
from arctic_platform.model import build_model
from arctic_platform.model.loaders.generic_moe import GENERIC_MOE_MODEL_TYPES


def _checkpoint(tmp_path, model_type: str, *, composite: bool = False) -> str:
    config = {"model_type": model_type}
    if composite:
        config = {"model_type": "wrapper", "text_config": config}
    path = tmp_path / model_type
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


@pytest.mark.parametrize("model_type", sorted(GENERIC_MOE_MODEL_TYPES))
@pytest.mark.parametrize("composite", [False, True])
def test_selection_uses_model_type(tmp_path, model_type, composite):
    spec = ModelSpec(
        model_path_or_name=_checkpoint(tmp_path, model_type, composite=composite),
        parallelism=ParallelismConfig(expert_parallel=2),
    )
    assert spec.loader == "generic_moe"


@pytest.mark.parametrize("model_type", sorted(GENERIC_MOE_MODEL_TYPES))
def test_expert_parallel_one_does_not_select_generic_loader(tmp_path, model_type):
    spec = ModelSpec(
        model_path_or_name=_checkpoint(tmp_path, model_type),
        parallelism=ParallelismConfig(expert_parallel=1),
    )
    assert spec.loader == "huggingface"


def test_legacy_minimax_config_uses_custom_impl(tmp_path):
    from transformers import AutoConfig

    from arctic_platform.model.implementations.qwen35.models import supports_custom_impl

    config = AutoConfig.from_pretrained(_checkpoint(tmp_path, "minimax"))
    assert supports_custom_impl(config)
    assert config.attention_bias is False
    assert config.pad_token_id is None
    assert config.rope_scaling is None
    assert config.use_grouped_mm is True
    assert config.partial_rotary_factor == pytest.approx(config.rotary_dim / config.head_dim)
    assert config.rope_parameters["partial_rotary_factor"] == pytest.approx(0.5)


def test_sequence_parallel_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="generic MoE families do not support sequence parallelism"):
        ModelSpec(
            model_path_or_name=_checkpoint(tmp_path, "qwen3_moe"),
            parallelism=ParallelismConfig(expert_parallel=2, sequence_parallel=2),
        )


def test_loader_preserves_options_and_process_group(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as integration

    seen = {}
    model = nn.Identity()

    def load(**kwargs):
        seen.update(kwargs)
        return model

    monkeypatch.setattr(integration, "load_generic_moe_model", load)
    ep_group = object()
    spec = ModelSpec(
        model_path_or_name=_checkpoint(tmp_path, "nemotron_h"),
        parallelism=ParallelismConfig(expert_parallel=4),
        loader_options={"ep_comm_backend": "uccl", "fused_cross_entropy": False},
    )
    result = build_model(spec, parallel_groups={"ep_group": ep_group})
    assert result.model is model
    assert seen["ep_group"] is ep_group
    assert seen["ep_size"] == 4
    assert seen["sp_size"] == 1
    assert seen["options"].ep_comm_backend == "uccl"
    assert seen["options"].fused_cross_entropy is False


def test_peft_patch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="expert adapter integration"):
        ModelSpec(
            model_path_or_name=_checkpoint(tmp_path, "afmoe"),
            parallelism=ParallelismConfig(expert_parallel=2),
            patches=Patches(peft={"peft_type": "LORA"}),
        )
