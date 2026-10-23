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
"""CPU configuration checks for the on-prem expert-parallel worker bridge."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from arctic_platform.model import ModelSpec


@pytest.fixture
def worker_config():
    return {
        "model_provider": "prime_rl",
        "attn_implementation": "sparse_mla",
        "ep_size": 8,
        "sp_size": 8,
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


def test_worker_moe_spec_preserves_glm53_recipe_and_input(tmp_path, worker_config):
    original = deepcopy(worker_config)
    spec = ModelSpec.from_ds_worker_config(_model_path(tmp_path), worker_config)

    assert spec.loader == "glm5_next"
    assert spec.parallelism.expert_parallel == spec.parallelism.sequence_parallel == 8
    assert spec.dtype == "bfloat16"
    assert spec.attn_implementation == "sparse_mla"
    assert spec.patches.gradient_checkpointing is False
    assert spec.patches.zorro_train is None
    assert spec.patches.peft["r"] == 32
    for key, value in worker_config["prime_rl"].items():
        assert spec.loader_options[key] == value
    assert spec.loader_options["ac_config"]["offload_config"]["enabled"] is True
    assert spec.loader_options["ac_config"]["router_replay_recompute"] is True
    assert spec.loader_options["tiled_mlp_token_chunk_size"] == 1024
    assert worker_config == original


@pytest.mark.parametrize(
    "relative",
    [
        "glm5.3-flash-bf16/h200/train-sft-16gpus-sp16-256k.config",
        "glm5.3-flash-bf16/h200/train-rl-32gpus-sp16-256k.config",
        "glm5.3-flash/h200/train-sft-lora-8gpus-sp8-256k.config",
        "glm5.3-flash/h200/train-rl-lora-16gpus-sp8-256k.config",
    ],
)
def test_worker_moe_spec_current_glm_recipes(tmp_path, relative):
    root = Path(__file__).resolve().parents[2] / "arctic_platform/correctness/configs"
    config = json.loads((root / relative).read_text())
    worker = deepcopy(config["training"]["ds_worker_config"])
    peft = config["training"].get("peft")
    if peft is not None:
        worker["peft_config"] = deepcopy(peft)
    spec = ModelSpec.from_ds_worker_config(_model_path(tmp_path), worker)

    assert spec.loader == "glm5_next"
    assert spec.parallelism.expert_parallel == worker["ep_size"]
    assert spec.parallelism.sequence_parallel == worker["sp_size"]
    assert spec.patches.peft == peft
    assert spec.patches.gradient_checkpointing is False
    assert spec.loader_options["fused_cross_entropy"] == worker["prime_rl"]["fused_cross_entropy"]
    assert spec.loader_options["ac_config"]["offload_config"]["enabled"] is True
    for key, value in worker["prime_rl"].items():
        assert spec.loader_options[key] == value


def test_worker_moe_spec_glm52_still_rejects_sp(tmp_path, worker_config):
    with pytest.raises(ValueError, match="glm_moe_dsa does not support sequence parallelism"):
        ModelSpec.from_ds_worker_config(_model_path(tmp_path, "glm_moe_dsa"), worker_config)


@pytest.mark.parametrize(
    "patch",
    [
        {"zorro_train_enable": True},
        {"use_liger": True},
        {"enable_gradient_checkpointing": True},
        {"activation_checkpointing": False},
        {"model_options": {"lm_head": {"token_chunk_size": 128}}},
        {"model_options": {"lm_head": {"vocab_chunk_size": 1024}}},
        {"model_options": {"unsupported": True}},
        {"prime_rl": {"attn": "sdpa"}},
        {"prime_rl": {"ac_config": {"mode": "selective"}}},
        {"prime_rl": {"unsupported_option": 1}},
        {"model_options": {"moe": {"comm_token_chunk": 1024}}},
        {"model_options": {"moe": {"unsupported": 1}}},
    ],
)
def test_worker_moe_spec_conflicting_or_unsupported_options_fail(tmp_path, worker_config, patch):
    worker_config.update(patch)
    with pytest.raises(ValueError):
        ModelSpec.from_ds_worker_config(_model_path(tmp_path), worker_config)


def test_worker_moe_spec_preserves_selected_dtype_and_margin(tmp_path, worker_config):
    worker_config["prime_rl"]["optimization_dtype"] = "float32"
    worker_config["debug"] = {"gpu_mem_preallocate_gib": 2}
    spec = ModelSpec.from_ds_worker_config(_model_path(tmp_path), worker_config)
    assert spec.dtype == "float32"
    assert spec.debug == {"gpu_mem_preallocate_gib": 2}


def test_worker_moe_spec_explicit_checkpointing_without_overrides(tmp_path, worker_config):
    worker_config.pop("ac_config")
    spec = ModelSpec.from_ds_worker_config(_model_path(tmp_path), worker_config)
    assert spec.loader_options["ac_config"]["mode"] == "full"
    assert spec.loader_options["ac_config"]["freq"] == 1
    assert spec.loader_options["ac_config"]["offload_config"]["enabled"] is False


def test_worker_moe_spec_dense_bridge_keeps_generic_patches():
    spec = ModelSpec.from_ds_worker_config("unused", {"attn_implementation": "sdpa"})
    assert spec.parallelism.expert_parallel == spec.parallelism.sequence_parallel == 1
    assert spec.patches.gradient_checkpointing is True
    assert spec.loader == "huggingface"


def test_worker_moe_spec_alias_only_moe_options(tmp_path, worker_config):
    worker_config["prime_rl"].pop("deepep_token_chunk_size")
    worker_config["model_options"]["moe"] = {"comm_token_chunk": 2048, "grouped_mm": False}
    spec = ModelSpec.from_ds_worker_config(_model_path(tmp_path), worker_config)
    assert spec.loader_options["deepep_token_chunk_size"] == 2048
    assert spec.loader_options["moe_use_grouped_mm"] is False
