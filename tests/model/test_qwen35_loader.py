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

"""Qwen loader configuration and runtime ownership boundaries."""

import sys
from types import SimpleNamespace

import pytest
from torch import nn

from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model import Patches
from arctic_platform.model import build_model
from arctic_platform.model import loader
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import execute_subprocess_async


@pytest.mark.parametrize("composite", [False, True])
def test_selection_uses_text_config_not_checkpoint_name(monkeypatch, composite):
    text = SimpleNamespace(model_type="qwen3_5_moe_text")
    config = SimpleNamespace(model_type="qwen3_5_moe", text_config=text) if composite else text
    monkeypatch.setattr(loader, "_load_hf_config", lambda _: config)
    spec = ModelSpec(model_path_or_name="renamed-checkpoint", parallelism=ParallelismConfig(expert_parallel=2))
    assert spec.loader == "qwen3_5_moe"


@pytest.mark.parametrize("backend", ["deepep", "uccl"])
def test_loader_preserves_options_and_process_groups(monkeypatch, backend):
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen

    seen = {}
    model = nn.Identity()

    def load(**kwargs):
        seen.update(kwargs)
        return model

    monkeypatch.setattr(qwen, "load_qwen3_5_moe_model", load)
    ep_group, sp_group = object(), object()
    spec = ModelSpec(
        model_path_or_name="local-checkpoint",
        loader="qwen3_5_moe",
        parallelism=ParallelismConfig(expert_parallel=2, sequence_parallel=2),
        loader_options={
            "ep_comm_backend": backend,
            "deepep_num_sms": 24,
            "tiled_mlp_token_chunk_size": 32,
            "weight_conversion_cache_dir": "",
            "trust_remote_code": True,
            "ac_config": {"offload_config": {"pin_memory_enabled": False, "pin_memory_max_size_gib": 0}},
        },
    )
    result = build_model(spec, parallel_groups={"ep_group": ep_group, "sp_group": sp_group})
    assert result.model is model
    assert seen["ep_group"] is ep_group and seen["sp_group"] is sp_group
    assert seen["ep_size"] == seen["sp_size"] == 2
    assert seen["optimization_dtype"] == "bfloat16"
    assert seen["attn_implementation"] == "flash_attention_3"
    assert seen["options"] == Qwen3_5MoeOptions.model_validate(spec.loader_options)
    assert seen["options"].tiled_mlp_token_chunk_size == 32
    assert seen["options"].ac_config.offload_config.pin_memory_enabled is False
    assert ModelSpec.model_validate_json(spec.model_dump_json()) == spec


@pytest.mark.parametrize(
    "options",
    [
        {"ep_comm_backend": "unknown"},
        {"unsupported_option": True},
        {"debug": {"unknown": True}},
        {"tiled_mlp_token_chunk_size": 0},
        {"ac_config": {"freq": 0}},
        {"ac_config": {"offload_config": {"pin_memory_max_size_gib": -1}}},
        {"ac_config": {"mode": "selective", "offload_config": {}}},
        {"fused_cross_entropy": "liger", "fused_lm_head_token_chunk_size": 128},
    ],
)
def test_unsupported_options_are_rejected(options):
    with pytest.raises(ValueError):
        Qwen3_5MoeOptions(**options)


def test_liger_fused_cross_entropy_allows_fp32_lm_head():
    options = Qwen3_5MoeOptions(fused_cross_entropy="liger", fp32_lm_head=True)
    assert options.fused_cross_entropy == "liger"
    assert options.fp32_lm_head is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype": "float16"},
        {"patches": Patches(peft={"peft_type": "Lora"})},
    ],
)
def test_spec_cannot_silently_ignore_settings(kwargs):
    with pytest.raises(ValueError):
        ModelSpec(model_path_or_name="local", loader="qwen3_5_moe", **kwargs)


def test_runtime_groups_are_required():
    spec = ModelSpec(model_path_or_name="local", loader="qwen3_5_moe")
    with pytest.raises(ValueError, match="ep_group"):
        build_model(spec)


def test_sequence_parallel_group_is_required():
    spec = ModelSpec(
        model_path_or_name="local",
        loader="qwen3_5_moe",
        parallelism=ParallelismConfig(expert_parallel=2, sequence_parallel=2),
    )
    with pytest.raises(ValueError, match="sp_group"):
        build_model(spec, parallel_groups={"ep_group": object()})


@pytest.mark.parametrize("patches", [Patches(gradient_checkpointing=True), Patches(zorro_train={})])
def test_generic_forward_patches_are_rejected(patches):
    with pytest.raises(ValueError, match="generic forward patches"):
        ModelSpec(model_path_or_name="local", loader="qwen3_5_moe", patches=patches)


def test_legacy_disabled_offload_normalizes_to_none():
    options = Qwen3_5MoeOptions(ac_config={"offload_config": {"enabled": False}})
    assert options.ac_config is not None
    assert options.ac_config.offload_config is None


def test_runtime_config_is_derived_from_validated_options():
    from arctic_platform.model.implementations.qwen35.deepspeed_integration import _build_model_config

    options = Qwen3_5MoeOptions(
        seq_len=1024,
        ep_comm_backend="uccl",
        ac_config={
            "freq": 2,
            "offload_config": {
                "keep_last_n": 3,
                "pin_memory_max_size_gib": 0,
            },
        },
    )
    config = _build_model_config(
        "local",
        4,
        2,
        "float32",
        "sdpa",
        options,
    )

    assert config.seq_len == 1024
    assert config.ep == 4
    assert config.dp_replicate == 2
    assert config.optimization_dtype == "float32"
    assert config.attn == "sdpa"
    assert config.ep_comm_backend == "uccl"
    assert config.ac is options.ac_config
    assert config.ac.offload_config.keep_last_n == 3


def test_legacy_qwen_types_are_shared():
    from arctic_platform.model.implementations.moe.layers.moe import MoE
    from arctic_platform.model.implementations.qwen35.models.layers.moe import MoE as LegacyMoE

    assert LegacyMoE is MoE


class TestStandaloneImports(TestCasePlus):
    def test_model_implementation_has_no_service_imports(self):
        code = """
import importlib.abc
import sys
class BlockService(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('dss', 'dss_client'):
            raise AssertionError(f'service import: {fullname}')
sys.meta_path.insert(0, BlockService())
from arctic_platform.model.implementations.qwen35 import deepspeed_integration
from arctic_platform.model.implementations import fp8
print('Standalone MoE and FP8 imports passed')
"""
        execute_subprocess_async([sys.executable, "-c", code], env=self.get_env(), timeout=60)
