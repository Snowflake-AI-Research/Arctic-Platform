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
"""Sequence length and the job-create comment, without a live Cortex job."""

from __future__ import annotations

from types import SimpleNamespace

from arctic_platform.client.transports.cortex import job_create_payload
from arctic_platform.integrations._cortex_dispatch import _to_unified_config
from arctic_platform.integrations._cortex_dispatch import legacy_max_seq_len
from arctic_platform.rl.config import ArcticRLClientConfig


def test_missing_length_keeps_the_default():
    assert legacy_max_seq_len(SimpleNamespace(vllm_config={}, training_config={})) is None


def test_vllm_length_wins_over_the_training_fallback():
    legacy = SimpleNamespace(vllm_config=dict(max_model_len=40960), training_config=dict(max_length=8192))
    assert legacy_max_seq_len(legacy) == 40960


def test_gsm8k_sized_length_keeps_the_8192_default():
    # SkyRL writes max_model_len = prompt + response for every recipe. GSM8K is 1536.
    legacy = SimpleNamespace(vllm_config=dict(max_model_len=1536), training_config=dict(max_length=1536))
    assert legacy_max_seq_len(legacy) is None


def test_unified_config_carries_the_sequence_length(monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://example.invalid")
    legacy = ArcticRLClientConfig(model_name="Qwen/Qwen3.5-4B", training_gpus=1, vllm_config=dict(max_model_len=40960))
    unified = _to_unified_config(legacy)
    assert unified.max_seq_len == 40960


def test_unified_config_without_a_length_stays_at_8192(monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://example.invalid")
    legacy = ArcticRLClientConfig(model_name="Qwen/Qwen3-0.6B", training_gpus=1)
    unified = _to_unified_config(legacy)
    assert unified.max_seq_len == 8192


def test_unified_config_keeps_8192_for_a_shorter_vllm_length(monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://example.invalid")
    legacy = ArcticRLClientConfig(model_name="Qwen/Qwen3-0.6B", training_gpus=1, vllm_config=dict(max_model_len=1536))
    unified = _to_unified_config(legacy)
    assert unified.max_seq_len == 8192


def test_job_comment_is_added_only_when_set():
    body = job_create_payload([dict(role="training")], "karthik openhands")
    assert body["comment"] == "karthik openhands"
    assert "comment" not in job_create_payload([dict(role="training")], None)
    assert "comment" not in job_create_payload([dict(role="training")], "")
