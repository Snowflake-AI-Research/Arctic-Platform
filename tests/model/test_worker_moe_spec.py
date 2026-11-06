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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU configuration checks for the expert-parallel spec both servers build from typed model settings."""

import json
from copy import deepcopy

import pytest

from arctic_platform.common.model_settings import ModelSettings
from arctic_platform.common.utils import sp_size_from_job_config
from arctic_platform.model import settings_moe_loader_options
from arctic_platform.model import settings_moe_model_spec


@pytest.fixture
def training_config():
    return {
        "model_provider": "prime_rl",
        "attn_implementation": "sparse_mla",
        "ep_size": 8,
        "activation_checkpointing": True,
        "ac_config": {
            "mode": "full",
            "freq": 1,
            "router_replay_recompute": True,
            "offload_config": {"enabled": True, "keep_last_n": 1, "pin_memory_enabled": True},
        },
        "tiled_mlp_token_chunk_size": 1024,
        "prime_rl": {
            "ep_comm_backend": "deepep",
            "deepep_token_chunk_size": 8192,
            "fused_cross_entropy": False,
            "fused_lm_head_token_chunk_size": 4096,
        },
        "model_options": {"lm_head": {"cross_entropy": False, "token_chunk_size": 4096}},
        "peft_config": {"peft_type": "Lora", "r": 32, "lora_alpha": 32, "target_modules": ["q_a_proj"]},
    }


def _model_path(tmp_path, model_type="glm5_next"):
    path = tmp_path / model_type
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(path)


def _spec(model_path, training_config, *, sequence_parallel=8, **kwargs):
    return settings_moe_model_spec(
        ModelSettings.model_validate(training_config),
        model_path=model_path,
        sequence_parallel=sequence_parallel,
        **kwargs,
    )


def test_moe_spec_preserves_glm53_recipe_and_input(tmp_path, training_config):
    original = deepcopy(training_config)
    spec = _spec(_model_path(tmp_path), training_config)

    assert spec.loader == "glm5_next"
    assert spec.parallelism.expert_parallel == spec.parallelism.sequence_parallel == 8
    assert spec.dtype == "bfloat16"
    assert spec.attn_implementation == "sparse_mla"
    assert spec.patches.gradient_checkpointing is False
    assert spec.patches.zorro_train is None
    assert spec.patches.peft["r"] == 32
    for key, value in training_config["prime_rl"].items():
        assert spec.loader_options[key] == value
    assert spec.loader_options["ac_config"]["offload_config"]["enabled"] is True
    assert spec.loader_options["ac_config"]["router_replay_recompute"] is True
    assert spec.loader_options["tiled_mlp_token_chunk_size"] == 1024
    assert training_config == original


def test_moe_spec_glm52_still_rejects_sp(tmp_path, training_config):
    with pytest.raises(ValueError, match="glm_moe_dsa does not support sequence parallelism"):
        _spec(_model_path(tmp_path, "glm_moe_dsa"), training_config)


@pytest.mark.parametrize(
    "patch",
    [
        {"use_liger": True},
        {"model_options": {"lm_head": {"token_chunk_size": 128}}},
        {"model_options": {"unsupported": True}},
        {"prime_rl": {"attn": "sdpa"}},
        {"prime_rl": {"unsupported_option": 1}},
        {"model_options": {"moe": {"comm_token_chunk": 1024}}},
        {"model_options": {"moe": {"unsupported": 1}}},
    ],
)
def test_moe_spec_conflicting_or_unsupported_options_fail(tmp_path, training_config, patch):
    training_config.update(patch)
    with pytest.raises(ValueError):
        _spec(_model_path(tmp_path), training_config)


def test_moe_spec_preserves_selected_dtype_and_margin(tmp_path, training_config):
    training_config["prime_rl"]["optimization_dtype"] = "float32"
    spec = _spec(_model_path(tmp_path), training_config, debug={"gpu_mem_preallocate_gib": 2})
    assert spec.dtype == "float32"
    assert spec.debug == {"gpu_mem_preallocate_gib": 2}


@pytest.mark.parametrize("activation_checkpointing", [True, False])
def test_ac_config_alone_switches_checkpointing_on_the_moe_route(training_config, activation_checkpointing):
    training_config["activation_checkpointing"] = activation_checkpointing
    with_block = settings_moe_loader_options(ModelSettings.model_validate(training_config))
    training_config.pop("ac_config")
    without_block = settings_moe_loader_options(ModelSettings.model_validate(training_config))

    assert with_block["ac_config"]["mode"] == "full"
    assert without_block["ac_config"] is None


def test_the_block_overrides_ac_config_and_seq_len_and_job_keys_override_the_block(training_config):
    training_config["prime_rl"].update(
        ac_config={"mode": "selective"}, seq_len=2048, weight_conversion_cache_dir="/block"
    )
    options = settings_moe_loader_options(
        ModelSettings.model_validate(training_config), seq_len=8192, weight_conversion_cache_dir="/job"
    )

    assert options["ac_config"] == {"mode": "selective"}
    assert options["seq_len"] == 2048
    assert options["weight_conversion_cache_dir"] == "/job"
    assert options["attn"] == "sparse_mla"


def test_seq_len_comes_from_the_job_when_the_block_names_none(training_config):
    options = settings_moe_loader_options(ModelSettings.model_validate(training_config), seq_len=8192)
    assert options["seq_len"] == 8192
    assert "seq_len" not in settings_moe_loader_options(ModelSettings.model_validate(training_config))


@pytest.mark.parametrize("key", ["sequence_parallel", "sequence_parallel_size", "parallelism"])
def test_worker_resolves_canonical_sp_aliases(key):
    worker_config = {key: {"sequence_parallel": 8} if key == "parallelism" else 8}
    assert sp_size_from_job_config({"ds_worker_config": worker_config}) == 8


@pytest.mark.parametrize("key", ["sequence_parallel", "sequence_parallel_size", "parallelism"])
def test_worker_rejects_conflicting_sp_aliases(key):
    worker_config = {"sp_size": 8, key: {"sequence_parallel": 2} if key == "parallelism" else 2}
    with pytest.raises(ValueError, match="conflicting sequence-parallel degrees"):
        sp_size_from_job_config({"ds_worker_config": worker_config})


def test_moe_spec_alias_only_moe_options(tmp_path, training_config):
    training_config["prime_rl"].pop("deepep_token_chunk_size")
    training_config["model_options"]["moe"] = {"comm_token_chunk": 2048, "grouped_mm": False}
    spec = _spec(_model_path(tmp_path), training_config)
    assert spec.loader_options["deepep_token_chunk_size"] == 2048
    assert spec.loader_options["moe_use_grouped_mm"] is False


def test_moe_spec_requires_expert_parallelism(training_config):
    training_config["ep_size"] = 1
    training_config.pop("model_options")
    with pytest.raises(ValueError, match="requires ep_size > 1"):
        _spec("unused", training_config)


def test_moe_spec_requires_the_arctic_provider(training_config):
    training_config["model_provider"] = "huggingface"
    with pytest.raises(ValueError, match="only supported for the arctic model provider"):
        _spec("unused", training_config)
