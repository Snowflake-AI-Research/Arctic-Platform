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

"""GLM MoE DSA loader configuration and ownership boundaries."""

import json
import sys

import pytest
from torch import nn

from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model import Patches
from arctic_platform.model import build_model
from arctic_platform.model.loaders.glm_moe_dsa import GlmMoeDsaOptions
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import execute_subprocess_async


@pytest.mark.parametrize("composite", [False, True])
def test_selection_uses_model_type(tmp_path, composite):
    config = {"model_type": "glm_moe_dsa"}
    if composite:
        config = {
            "model_type": "glm_vlm",
            "text_config": config,
        }
    (tmp_path / "config.json").write_text(json.dumps(config))
    spec = ModelSpec(model_path_or_name=str(tmp_path), parallelism=ParallelismConfig(expert_parallel=2))
    assert spec.loader == "glm_moe_dsa"


@pytest.mark.parametrize("backend", ["deepep", "deepep_v2", "uccl"])
def test_loader_preserves_options_and_process_group(monkeypatch, backend):
    from arctic_platform.model.implementations.glm52 import deepspeed_integration as glm

    seen = {}
    model = nn.Identity()

    def load(**kwargs):
        seen.update(kwargs)
        return model

    monkeypatch.setattr(glm, "load_glm_moe_dsa_model", load)
    ep_group = object()
    spec = ModelSpec(
        model_path_or_name="local-checkpoint",
        loader="glm_moe_dsa",
        parallelism=ParallelismConfig(expert_parallel=2),
        loader_options={
            "ep_comm_backend": backend,
            "sparse_mla_backend": "dense",
            "deepep_num_sms": 24,
            "tiled_mlp_token_chunk_size": 32,
            "weight_conversion_cache_dir": "",
            "trust_remote_code": False,
            "ac_config": {"offload_config": {"pin_memory_enabled": False, "pin_memory_max_size_gib": 0}},
        },
    )
    result = build_model(spec, parallel_groups={"ep_group": ep_group})
    assert result.model is model
    assert seen["ep_group"] is ep_group
    assert seen["ep_size"] == 2
    assert seen["sp_size"] == 1
    assert seen["optimization_dtype"] == "bfloat16"
    assert seen["attn_implementation"] == "flash_attention_2"
    assert seen["options"] == GlmMoeDsaOptions.model_validate(spec.loader_options)
    assert seen["options"].sparse_mla_backend == "dense"
    assert seen["options"].ac_config.offload_config.pin_memory_enabled is False
    assert ModelSpec.model_validate_json(spec.model_dump_json()) == spec


@pytest.mark.parametrize(
    "options",
    [
        {"ep_comm_backend": "unknown"},
        {"sparse_mla_backend": "unknown"},
        {"unsupported_option": True},
        {"debug": {"unknown": True}},
        {"tiled_mlp_token_chunk_size": 0},
        {"deepep_num_sms": 21},
        {"ac_config": {"freq": 0}},
    ],
)
def test_unsupported_options_are_rejected(options):
    with pytest.raises(ValueError):
        GlmMoeDsaOptions(**options)


def test_omitted_fused_cross_entropy_defaults_to_liger(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa"}))
    spec = ModelSpec(model_path_or_name=str(tmp_path), parallelism=ParallelismConfig(expert_parallel=2))
    assert spec.loader_options["fused_cross_entropy"] == "liger"

    explicit = ModelSpec(
        model_path_or_name=str(tmp_path),
        parallelism=ParallelismConfig(expert_parallel=2),
        loader_options={"fused_cross_entropy": False},
    )
    assert explicit.loader_options["fused_cross_entropy"] is False


def test_chunked_lm_head_rejects_fused_cross_entropy():
    with pytest.raises(ValueError, match="cannot combine fused_cross_entropy"):
        GlmMoeDsaOptions(
            fused_cross_entropy="liger",
            fused_lm_head_token_chunk_size=128,
        )
    options = GlmMoeDsaOptions(
        fused_cross_entropy=False,
        fused_lm_head_token_chunk_size=128,
    )
    assert options.fused_cross_entropy is False
    assert options.fused_lm_head_token_chunk_size == 128


def test_quack_fused_cross_entropy_is_supported():
    assert GlmMoeDsaOptions(fused_cross_entropy="quack").fused_cross_entropy == "quack"
    with pytest.raises(ValueError, match="fp32_lm_head"):
        GlmMoeDsaOptions(fused_cross_entropy="quack", fp32_lm_head=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype": "float16"},
        {"parallelism": ParallelismConfig(expert_parallel=2, sequence_parallel=2)},
        {"patches": Patches(peft={"peft_type": "Lora"})},
    ],
)
def test_spec_cannot_silently_ignore_settings(kwargs):
    with pytest.raises(ValueError):
        ModelSpec(model_path_or_name="local", loader="glm_moe_dsa", **kwargs)


def test_runtime_group_is_required():
    spec = ModelSpec(model_path_or_name="local", loader="glm_moe_dsa")
    with pytest.raises(ValueError, match="ep_group"):
        build_model(spec)


def test_runtime_config_is_derived_from_validated_options():
    from arctic_platform.model.implementations.glm52.deepspeed_integration import _build_model_config

    options = GlmMoeDsaOptions(
        seq_len=1024,
        trust_remote_code=False,
        ep_comm_backend="uccl",
        sparse_mla_backend="flashmla",
        ac_config={"freq": 2},
        debug={"num_layers": 6, "random_init": True},
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
    assert config.sparse_mla_backend == "flashmla"
    assert config.ac is options.ac_config
    assert config.debug.num_layers == 6
    assert config.debug.random_init is True


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
from arctic_platform.model.implementations.glm52 import deepspeed_integration
print('Standalone GLM MoE import passed')
"""
        execute_subprocess_async([sys.executable, "-c", code], env=self.get_env(), timeout=60)
