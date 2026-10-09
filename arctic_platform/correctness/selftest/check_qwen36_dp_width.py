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

"""Selftests for the Qwen3.6 DP-width diagnostic helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _stub_module(name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_dp_width(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    def _unused(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("selftest should not call diagnostic runtime dependencies")

    def _train_batch_size(training: dict, gpus: int) -> int:
        ds_config = training["ds_config"]
        micro = int(ds_config["train_micro_batch_size_per_gpu"])
        gas = int(ds_config["gradient_accumulation_steps"])
        sp_size = int(training.get("sp_size", 1))
        return micro * gas * (gpus // sp_size)

    stubs = {
        "arctic_platform.correctness.harness.arms": _stub_module(
            "arctic_platform.correctness.harness.arms",
            correctness_microbatch_tokens=lambda value: min(int(value), 65536),
        ),
        "arctic_platform.correctness.harness.batches": _stub_module(
            "arctic_platform.correctness.harness.batches",
            build_batch=_unused,
            load=_unused,
            save=_unused,
        ),
        "arctic_platform.correctness.harness.config": _stub_module(
            "arctic_platform.correctness.harness.config",
            LoadedConfig=object,
            deepspeed_train_batch_size=_train_batch_size,
            load_config=_unused,
        ),
        "arctic_platform.correctness.harness.dss_driver": _stub_module(
            "arctic_platform.correctness.harness.dss_driver",
            build_payload=_unused,
            fwd_bwd_step=_unused,
            gateway=_unused,
            pack=_unused,
            running_job=_unused,
        ),
        "arctic_platform.correctness.harness.names": _stub_module(
            "arctic_platform.correctness.harness.names",
            align=lambda left, right: (
                [(name, left[name], right[name]) for name in sorted(left.keys() & right.keys())],
                [],
                [],
            ),
        ),
        "arctic_platform.correctness.harness.runner": _stub_module(
            "arctic_platform.correctness.harness.runner",
            run_reference=_unused,
        ),
        "arctic_platform.correctness.harness.seeds": _stub_module(
            "arctic_platform.correctness.harness.seeds",
            SEED=1234,
        ),
        "arctic_platform.correctness.harness.spec": _stub_module(
            "arctic_platform.correctness.harness.spec",
            ArmSpec=object,
            TestSpec=object,
        ),
        "arctic_platform.correctness.harness.workdir": _stub_module(
            "arctic_platform.correctness.harness.workdir",
            correctness_workdir=_unused,
        ),
        "arctic_platform.correctness.reference.model_features": _stub_module(
            "arctic_platform.correctness.reference.model_features",
            uses_mixer_packing=lambda _model_path: True,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).resolve().parents[1] / "diagnostics" / "qwen36_dp_width.py"
    spec = importlib.util.spec_from_file_location("_qwen36_dp_width_selftest", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _Config:
    def __init__(self) -> None:
        self.training = {
            "n_gpus": 8,
            "sp_size": 8,
            "ep_size": 4,
            "ds_config": {
                "train_micro_batch_size_per_gpu": 2,
                "gradient_accumulation_steps": 3,
                "train_batch_size": 8,
            },
            "ds_worker_config": {
                "ep_size": 4,
                "kept": True,
            },
        }
        self.effective_training = {
            **self.training,
            "fused_lm_head_token_chunk_size": 8192,
            "fused_cross_entropy": False,
        }


def test_parse_widths_accepts_comma_list(monkeypatch: pytest.MonkeyPatch) -> None:
    dp_width = _load_dp_width(monkeypatch)

    assert dp_width.parse_widths("1, 2,4,8") == (1, 2, 4, 8)


def test_parse_widths_rejects_empty_and_non_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    dp_width = _load_dp_width(monkeypatch)

    with pytest.raises(ValueError, match="at least one"):
        dp_width.parse_widths("")
    with pytest.raises(ValueError, match="positive"):
        dp_width.parse_widths("1,0")


def test_training_for_width_forces_sp1_ep1_and_resizes_deepspeed_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dp_width = _load_dp_width(monkeypatch)
    cfg = _Config()

    training = dp_width.training_for_width(cfg, 4)

    assert training["n_gpus"] == 4
    assert training["sp_size"] == 1
    assert training["ep_size"] == 1
    assert training["ds_worker_config"] == {"ep_size": 1, "kept": True}
    assert training["fused_lm_head_token_chunk_size"] == 8192
    assert training["fused_cross_entropy"] is False
    assert training["train_batch_size"] == 24
    assert training["ds_config"]["train_batch_size"] == 24
    assert cfg.training["sp_size"] == 8
    assert cfg.training["ds_worker_config"]["ep_size"] == 4
    assert "fused_lm_head_token_chunk_size" not in cfg.training


def test_summarize_width_counts_ratios_and_rel_abs_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    dp_width = _load_dp_width(monkeypatch)
    dss = type(
        "Dss",
        (),
        {
            "grad_norms": {"a": 1.0, "b": 2.002, "c": 0.0002},
            "avg_loss": 3.25,
            "model_calls": 2,
            "packed_rows": 8,
        },
    )()
    reference = {"grad_norms": {"a": 1.0, "b": 2.0, "c": 0.0001}, "loss": 3.0}

    summary = dp_width.summarize_width(2, dss, reference)

    assert summary.width == 2
    assert summary.compared == 3
    assert summary.above_one == 2
    assert summary.over_gate == 2
    assert summary.worst_name == "b"
    assert summary.loss_delta == pytest.approx(0.25)
    assert summary.model_calls == 2
    assert summary.packed_rows == 8
