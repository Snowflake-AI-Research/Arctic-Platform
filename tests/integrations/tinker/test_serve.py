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

from arctic_platform.integrations.tinker.serve import TinkerServeConfig
from arctic_platform.integrations.tinker.serve import _client_config
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
