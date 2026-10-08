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

"""GPU-specific correctness attention selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arctic_platform.client.config import ArcticClientConfig
from arctic_platform.correctness.harness.config import config_checksum
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.config import validate_against_host
from arctic_platform.correctness.selftest.config_factory import native_config


def _write_config(tmp_path: Path, gpu_type: str, attention: str | None, n_gpus: int = 8) -> Path:
    path = tmp_path / "configs" / "model" / gpu_type / "workload.config"
    path.parent.mkdir(parents=True)
    training = {"n_gpus": n_gpus, "sp_size": n_gpus}
    if attention is not None:
        training["attn_implementation"] = attention
    path.write_text(json.dumps(native_config(training)))
    return path


@pytest.mark.parametrize(
    ("gpu_type", "expected"),
    [
        ("h200", "flash_attention_3"),
        ("b200", "flash_attention_4"),
        ("b300", "flash_attention_4"),
    ],
)
def test_attention_is_derived_from_gpu_type(tmp_path: Path, gpu_type: str, expected: str) -> None:
    config = load_config(_write_config(tmp_path, gpu_type, None))

    assert config.attention_implementation == expected


def test_config_rejects_attention_for_another_gpu_family(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, "b200", "flash_attention_3"))

    with pytest.raises(ValueError, match="b200 requires attn_implementation='flash_attention_4'"):
        validate_against_host(config, available_gpus=8, check_gpu_type=False)


def test_config_accepts_gpu_family_attention(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, "b300", "flash_attention_4"))

    validate_against_host(config, available_gpus=8, check_gpu_type=False)


def test_config_uses_fewer_gpus_than_the_host_when_requested(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, "h200", "flash_attention_3", n_gpus=4))

    validate_against_host(config, available_gpus=8, check_gpu_type=False)

    assert config.n_gpus == 4
    assert config.sp_size == 4


def test_effective_training_applies_nested_prime_rl_overrides(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "h200", "flash_attention_3")
    raw = json.loads(path.read_text())
    training = raw["training"]["ds_worker_config"]
    training["fused_lm_head_token_chunk_size"] = 4096
    training["prime_rl"] = {
        "fused_lm_head_token_chunk_size": 8192,
        "fused_cross_entropy": False,
    }
    path.write_text(json.dumps(raw))

    effective = load_config(path).effective_training

    assert effective["fused_lm_head_token_chunk_size"] == 8192
    assert effective["fused_cross_entropy"] is False


def test_disabled_lm_head_chunk_resolves_to_none(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "h200", "flash_attention_3")
    raw = json.loads(path.read_text())
    raw["training"]["ds_worker_config"]["prime_rl"] = {
        "fused_lm_head_token_chunk_size": "disabled",
    }
    path.write_text(json.dumps(raw))

    assert load_config(path).lm_head_token_chunk_size is None


def test_config_checksum_is_canonical_and_tracks_training_behavior(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "h200", "flash_attention_3", n_gpus=4)
    first = config_checksum(load_config(path))
    raw = json.loads(path.read_text())
    path.write_text(json.dumps(raw, indent=4))

    assert config_checksum(load_config(path)) == first

    raw["training_gpus"] = 8
    path.write_text(json.dumps(raw))

    assert config_checksum(load_config(path)) != first


def test_optimizer_dtype_follows_explicit_deepspeed_state_precision(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "h200", "flash_attention_3")
    raw = json.loads(path.read_text())
    raw["training"]["ds_config"] = {
        "bf16": {
            "enabled": True,
            "bf16_master_weights_and_grads": True,
            "bf16_optimizer_states": True,
        }
    }
    path.write_text(json.dumps(raw))

    assert load_config(path).optimizer_dtype == "bfloat16"


def test_optimizer_dtype_rejects_mixed_master_and_state_precision(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "h200", "flash_attention_3")
    raw = json.loads(path.read_text())
    raw["training"]["ds_config"] = {"bf16": {"enabled": True, "bf16_master_weights_and_grads": True}}
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="matching Adam master and state precision"):
        _ = load_config(path).optimizer_dtype


def test_optimizer_dtype_matches_deepspeed_bf16_default_without_override(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "h200", "flash_attention_3")

    assert load_config(path).optimizer_dtype == "bfloat16"


def _write(tmp_path, training_config: dict):
    """Write a minimal single-training-sub-job config and return its path."""
    import json

    path = tmp_path / "h200" / "probe.config"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(native_config(training_config)))
    return path


def test_declared_row_width_is_used_as_written(tmp_path) -> None:
    """A config that states max_seq_len is measured at that width."""
    cfg = _write(tmp_path, {"max_seq_len": 4096, "n_gpus": 8, "attn_implementation": "flash_attention_3"})

    assert load_config(cfg).max_seq_len == 4096


def test_absent_row_width_follows_the_engine_default(tmp_path) -> None:
    """A config with no max_seq_len uses the Arctic client schema default."""
    cfg = _write(tmp_path, {"n_gpus": 8, "attn_implementation": "flash_attention_3"})

    assert load_config(cfg).max_seq_len == ArcticClientConfig.model_fields["max_seq_len"].default
