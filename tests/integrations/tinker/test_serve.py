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

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from arctic_platform.integrations.tinker.router import LoraConfig
from arctic_platform.integrations.tinker.serve import TinkerServeConfig
from arctic_platform.integrations.tinker.serve import _client_config
from arctic_platform.integrations.tinker.serve import _isolation
from arctic_platform.integrations.tinker.serve import _served_lora
from arctic_platform.integrations.tinker.serve import _teacher_config


def test_client_config_uses_packaged_types(monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://cortex.test")

    config = _client_config(TinkerServeConfig())

    assert config.backend.base_url == "http://cortex.test"
    assert config.training.peft is None
    assert config.training.ds_config["zero_optimization"] == {"stage": 2}
    assert config.sampling.vllm == {"gpu_memory_utilization": 0.8}


def test_client_config_file_and_existing_job(tmp_path):
    path = tmp_path / "connection.json"
    path.write_text(json.dumps({"connection": {"base_url": "http://cortex.test"}}), encoding="utf-8")

    config = _client_config(TinkerServeConfig(config=str(path), job_id="job-1"))

    assert config.training_job_id == "job-1:training:0"
    assert config.sampling_job_id == "job-1:sampling:0"


def test_teacher_is_a_sampling_only_job_of_its_own(monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://cortex.test")
    cfg = TinkerServeConfig(
        model="Qwen/Qwen3-0.6B",
        teacher_model="Qwen/Qwen3-8B",
        teacher_sampling_gpus=2,
        max_prompt_length=512,
        max_response_length=1024,
        job_id="student-job",
    )

    config = _teacher_config(cfg)

    assert config.model_name == "Qwen/Qwen3-8B"
    assert (config.training_gpus, config.sampling_gpus) == (0, 2)
    # It scores a full student sequence, then samples one token past it.
    assert config.max_seq_len == cfg.max_seq_len + 1
    assert config.training_job_id is None and config.sampling_job_id is None


def test_adam_is_provisioned_as_the_cookbook_sends_it(monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://cortex.test")
    cfg = TinkerServeConfig(training_gpus=1, sampling_gpus=0)

    training = _client_config(cfg).to_cortex()[0]["training_config"]

    assert training["optimizer"] == {
        "name": "AdamW",
        "lr": cfg.learning_rate,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    assert training["gradient_clipping"] == 0.0
    assert cfg.fixed_adam == {"beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.0, "grad_clip_norm": 0.0}


def _model_config(layer_types, nested=True):
    text = SimpleNamespace(layer_types=layer_types)
    return SimpleNamespace(text_config=text) if nested else text


@pytest.mark.parametrize(
    ("setting", "model_config", "isolated"),
    [
        ("auto", _model_config(["linear_attention", "full_attention"]), True),
        ("auto", _model_config(["full_attention"], nested=False), False),
        ("auto", SimpleNamespace(), False),
        ("on", SimpleNamespace(), True),
        ("off", _model_config(["linear_attention"]), False),
    ],
    ids=["qwen3.5", "dense", "no-layer-types", "on", "off"],
)
def test_sequences_are_isolated_for_linear_attention(setting, model_config, isolated):
    cfg = TinkerServeConfig(isolate_sequences=setting, max_prompt_length=1024, max_response_length=512)

    job_cfg, capacity = _isolation(cfg, model_config)

    if isolated:
        # One full-length sequence per micro-batch.
        assert capacity == job_cfg.max_tokens_per_mb == 1536
    else:
        assert (capacity, job_cfg) == (None, cfg)


def test_unknown_isolation_setting_refused():
    with pytest.raises(ValueError, match="--isolate-sequences"):
        _isolation(TinkerServeConfig(isolate_sequences="yes"), SimpleNamespace())


def test_lora_matches_tinkers_module_groups(monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://cortex.test")
    cfg = TinkerServeConfig(lora_rank=32, lora_modules="mlp,unembed", teacher_model="Qwen/Qwen3-8B")

    peft = _client_config(cfg).training.peft

    assert peft == {
        "peft_type": "Lora",
        "r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["gate_proj", "up_proj", "down_proj", "lm_head"],
    }
    assert _served_lora(cfg) == LoraConfig(rank=32, train_mlp=True, train_attn=False, train_unembed=True)
    # The teacher serves its base weights.
    assert _teacher_config(cfg).training.peft is None


def test_full_fine_tuning_serves_no_lora(monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://cortex.test")
    assert _served_lora(TinkerServeConfig()) is None


def test_unknown_lora_module_group_refused(monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://cortex.test")
    with pytest.raises(ValueError, match="--lora-modules"):
        _client_config(TinkerServeConfig(lora_rank=8, lora_modules="mlp,embed"))
