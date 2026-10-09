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

"""Selftests for the Qwen3.6 grouping diagnostic."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from arctic_platform.correctness.diagnostics.qwen36_grouping_control import _ap_dp_non_sp_training
from arctic_platform.correctness.diagnostics.qwen36_grouping_control import _build_dss_payload
from arctic_platform.correctness.diagnostics.qwen36_grouping_control import _compare_optimizer_artifacts


def test_optimizer_diagnostic_payload_skips_gradient_telemetry_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_build_payload(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"training_config": {"debug": {"gradient_norms_per_param": kwargs["gradient_norms_per_param"]}}}

    monkeypatch.setattr(
        "arctic_platform.correctness.diagnostics.qwen36_grouping_control.build_payload",
        fake_build_payload,
    )

    payload = _build_dss_payload(
        {"n_gpus": 8, "mb_spec": {"max_tokens_per_mb": 8192}},
        "model",
        "flash_attention_3",
        tmp_path / "optimizer",
        include_gradient_telemetry=False,
    )

    assert captured["args"] == ({"n_gpus": 8, "mb_spec": {"max_tokens_per_mb": 8192}}, "model", 1234)
    assert captured["kwargs"]["attn_implementation"] == "flash_attention_3"
    assert captured["kwargs"]["optimizer_state_output_dir"] == tmp_path / "optimizer"
    assert payload["training_config"]["debug"]["gradient_norms_per_param"] is False


def test_optimizer_diagnostic_payload_can_request_gradient_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_build_payload(*_args, **kwargs):
        return {"training_config": {"debug": {"gradient_norms_per_param": kwargs["gradient_norms_per_param"]}}}

    monkeypatch.setattr(
        "arctic_platform.correctness.diagnostics.qwen36_grouping_control.build_payload",
        fake_build_payload,
    )

    payload = _build_dss_payload(
        {"n_gpus": 8, "mb_spec": {"max_tokens_per_mb": 8192}},
        "model",
        "flash_attention_3",
        tmp_path / "optimizer",
        include_gradient_telemetry=True,
    )

    assert payload["training_config"]["debug"]["gradient_norms_per_param"] is True


def test_ap_dp_non_sp_training_recomputes_deepspeed_train_batch_size() -> None:
    cfg = SimpleNamespace(
        n_gpus=8,
        training={
            "n_gpus": 8,
            "sp_size": 8,
            "train_batch_size": 1,
            "ds_config": {
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "train_batch_size": 1,
            },
        },
    )

    training = _ap_dp_non_sp_training(cfg)

    assert training["sp_size"] == 1
    assert training["n_gpus"] == 8
    assert training["train_batch_size"] == 8
    assert training["ds_config"]["train_batch_size"] == 8
    assert cfg.training["train_batch_size"] == 1
    assert cfg.training["ds_config"]["train_batch_size"] == 1


def test_optimizer_comparator_dynamic_import_is_registered(tmp_path: Path) -> None:
    module_name = "arctic_platform.correctness.checks.optimizer_state_grouping_control"
    sys.modules.pop(module_name, None)

    dss_manifest = tmp_path / "dss.json"
    reference_manifest = tmp_path / "reference.json"
    manifest = {"step": 1, "parameters": []}
    dss_manifest.write_text(json.dumps(manifest))
    reference_manifest.write_text(json.dumps(manifest))

    comparison = _compare_optimizer_artifacts(
        dss_manifest,
        reference_manifest,
        optimizer_config={"betas": [0.9, 0.999], "eps": 1e-8},
        learning_rate=1e-2,
    )

    assert module_name in sys.modules
    assert comparison.deltas == []
    assert comparison.only_dss == []
    assert comparison.only_reference == []
