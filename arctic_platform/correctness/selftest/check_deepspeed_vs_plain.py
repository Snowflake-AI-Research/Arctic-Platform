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

"""Selftests for the one-GPU DeepSpeed-vs-plain diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from arctic_platform.correctness.diagnostics.deepspeed_vs_plain import loss_settings
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.selftest.config_factory import native_config


def _write_config(tmp_path: Path, training: dict) -> Path:
    path = tmp_path / "configs" / "qwen3.6-35b-a3b" / "h200" / "workload.config"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(native_config(training, model_name="Qwen/Qwen3.6-35B-A3B")))
    return path


def test_loss_settings_use_reference_defaults_when_lm_head_chunks_are_omitted(tmp_path: Path) -> None:
    cfg = load_config(
        _write_config(
            tmp_path,
            {
                "n_gpus": 8,
                "sp_size": 8,
                "model_provider": "prime_rl",
            },
        )
    )

    assert loss_settings(cfg) == (False, "liger", None, 8192)


def test_loss_settings_preserve_configured_lm_head_chunks(tmp_path: Path) -> None:
    cfg = load_config(
        _write_config(
            tmp_path,
            {
                "n_gpus": 8,
                "sp_size": 8,
                "model_provider": "prime_rl",
                "fused_cross_entropy": False,
                "fused_lm_head_token_chunk_size": 4096,
                "fused_lm_head_vocab_chunk_size": 16384,
            },
        )
    )

    assert loss_settings(cfg) == (False, False, 4096, 16384)
