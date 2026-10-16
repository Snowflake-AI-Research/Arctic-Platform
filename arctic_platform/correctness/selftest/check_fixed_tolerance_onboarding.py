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

# !/usr/bin/env python3
"""CPU-only checks for onboarding a check whose gate is a constant rather than a calibrated one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from arctic_platform.correctness.harness import spec as SPEC
from arctic_platform.correctness.harness.config import config_checksum
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import registered_tests
from arctic_platform.correctness.onboarding import onboard as MODULE
from arctic_platform.correctness.selftest.config_factory import native_config

LOCAL_TEST = "inference-checkpoint-loss"
RESUME_TEST = "checkpoint-resume-loss"


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            native_config(
                {
                    "n_gpus": 4,
                    "sp_size": 4,
                    "max_seq_len": 32768,
                    "attn_implementation": "flash_attention_3",
                    "train_batch_size": 4,
                    "optimizer": {"lr": 2e-05},
                }
            )
        )
    )
    return path


def _write_spec(config_path: Path, spec_path: Path) -> None:
    """A reviewed spec in the state the local check needs: onboarded, with a tolerance and settings."""
    cfg = load_config(config_path)
    SPEC.TestSpec(
        config_id=cfg.config_id,
        config_path=str(config_path),
        model=SPEC.ModelSpec(
            source_checkpoint="Qwen/Qwen3-8B",
            num_hidden_layers=4,
            vision_depth=0,
            seed=1234,
            param_count=1_000_000,
            cache_path="/data-fast/base-models/synthetic/Qwen3-8B-4L",
            content_hash="0123456789abcdef",
            sized_against_gib=140.0,
        ),
        arms=[
            SPEC.ArmSpec(
                name="gas1",
                global_batch_size=1,
                max_seq_len=10240,
                total_tokens=10240,
                active_tokens=10240,
                dss_microbatches=1,
                pad_fraction=0.0,
            )
        ],
        attn_implementations=["flash_attention_3"],
        applicable_tests=["single-step-grads", "single-step-optimizer"],
        config_checksum=config_checksum(cfg),
        test_tolerances={
            "single-step-grads": SPEC.TestTolerance(
                absolute=0.004,
                calibration_runs=16,
                raw_max_same_tensor_range=0.0017,
                multiplier=2.0,
                computed_gate=0.0034,
                worst_tensor="embed_tokens.weight",
                status="calibrated",
            )
        },
        test_settings={"single-step-optimizer": {"reference_token_budget": 10240}},
    ).write(spec_path)


def _onboarding_args(config_path: Path, spec_path: Path, tmp_path: Path, test_id: str):
    output_dir = tmp_path / "onboarding-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return argparse.Namespace(
        config=config_path, test_spec=spec_path, test_id=test_id, output_dir=output_dir, checkout_root=tmp_path
    )


def _prepared(tmp_path: Path, test_id: str = LOCAL_TEST):
    config_path = _write_config(tmp_path)
    spec_path = tmp_path / "spec.json"
    _write_spec(config_path, spec_path)
    return (_onboarding_args(config_path, spec_path, tmp_path, test_id), load_config(config_path), spec_path)


def _regression(status: str):
    def run(args, selected, **kwargs):
        return {
            "layers": selected["layers"],
            "status": status,
            "returncode": 0 if status == "pass" else 1,
            "log": "/tmp/run.log",
            "report": "/tmp/report.json" if status == "pass" else None,
        }

    return run


def _must_not_run(args, selected, **kwargs):
    raise AssertionError("the check was run for a config whose spec was already refused")


def test_unknown_test_id_is_still_refused() -> None:
    with pytest.raises(ValueError, match="onboarding does not support test id"):
        MODULE._onboarding_kind("no-such-check")


def test_a_check_with_no_reference_takes_the_fixed_tolerance_path() -> None:
    assert MODULE._onboarding_kind(LOCAL_TEST) == MODULE.FIXED_TOLERANCE


def test_checkpoint_resume_measures_its_own_gate() -> None:
    assert MODULE._onboarding_kind(RESUME_TEST) == RESUME_TEST


def test_reference_checks_keep_their_own_paths() -> None:
    assert MODULE._onboarding_kind("single-step-grads") == "single-step-grads"
    assert MODULE._onboarding_kind("single-step-optimizer") == "single-step-optimizer"


def test_no_check_declares_it_needs_the_hosted_control_plane() -> None:
    # TODO: assert ``requires_hosted_control_plane`` for RESUME_TEST again when it returns to the hosted transport.
    tests = registered_tests()

    assert not tests[RESUME_TEST].requires_hosted_control_plane
    assert not tests[LOCAL_TEST].requires_hosted_control_plane


def test_passing_check_adds_one_id_and_changes_nothing_else(monkeypatch, tmp_path: Path) -> None:
    args, cfg, spec_path = _prepared(tmp_path)
    before = json.loads(spec_path.read_text())
    monkeypatch.setattr(MODULE, "_run_final_regression", _regression("pass"))

    assert MODULE._onboard_fixed_tolerance_test(args, cfg) == 0

    after = json.loads(spec_path.read_text())
    assert set(after["applicable_tests"]) - set(before["applicable_tests"]) == {LOCAL_TEST}
    assert after["applicable_tests"] == sorted(after["applicable_tests"])
    assert after["test_tolerances"] == before["test_tolerances"]
    assert after["test_settings"] == before["test_settings"]
    assert LOCAL_TEST not in after["test_tolerances"]
    assert LOCAL_TEST not in after["test_settings"]
    assert {key: value for key, value in after.items() if key != "applicable_tests"} == {
        key: value for key, value in before.items() if key != "applicable_tests"
    }


def test_failing_check_restores_the_previous_spec_bytes(monkeypatch, tmp_path: Path) -> None:
    args, cfg, spec_path = _prepared(tmp_path)
    before = spec_path.read_bytes()
    monkeypatch.setattr(MODULE, "_run_final_regression", _regression("failed"))

    with pytest.raises(RuntimeError, match=f"{LOCAL_TEST} failed"):
        MODULE._onboard_fixed_tolerance_test(args, cfg)

    assert spec_path.read_bytes() == before


def test_an_exception_during_the_run_restores_the_previous_spec_bytes(monkeypatch, tmp_path: Path) -> None:
    args, cfg, spec_path = _prepared(tmp_path)
    before = spec_path.read_bytes()

    def explode(args, selected, **kwargs):
        raise RuntimeError("the gateway never came up")

    monkeypatch.setattr(MODULE, "_run_final_regression", explode)

    with pytest.raises(RuntimeError, match="the gateway never came up"):
        MODULE._onboard_fixed_tolerance_test(args, cfg)

    assert spec_path.read_bytes() == before


def test_a_config_with_no_spec_is_refused(monkeypatch, tmp_path: Path) -> None:
    args, cfg, spec_path = _prepared(tmp_path)
    spec_path.unlink()
    monkeypatch.setattr(MODULE, "_run_final_regression", _must_not_run)

    with pytest.raises(ValueError, match="requires an existing reviewed spec"):
        MODULE._onboard_fixed_tolerance_test(args, cfg)

    assert not spec_path.exists()


def test_a_stale_checksum_is_refused_rather_than_repaired(monkeypatch, tmp_path: Path) -> None:
    args, cfg, spec_path = _prepared(tmp_path)
    before = spec_path.read_bytes()
    edited = json.loads(args.config.read_text())
    edited["max_seq_len"] = 16384
    args.config.write_text(json.dumps(edited))
    monkeypatch.setattr(MODULE, "_run_final_regression", _must_not_run)

    with pytest.raises(ValueError, match="hashes to"):
        MODULE._onboard_fixed_tolerance_test(args, load_config(args.config))

    assert spec_path.read_bytes() == before


def test_a_hosted_only_check_is_refused_and_names_the_hosted_command(monkeypatch, tmp_path: Path) -> None:
    args, cfg, spec_path = _prepared(tmp_path, RESUME_TEST)
    before = spec_path.read_bytes()
    monkeypatch.setattr(registered_tests()[RESUME_TEST], "requires_hosted_control_plane", True)
    monkeypatch.setattr(MODULE, "_run_final_regression", _must_not_run)

    with pytest.raises(ValueError) as refusal:
        MODULE._onboard_fixed_tolerance_test(args, cfg)

    assert "arctic_platform.correctness hosted" in str(refusal.value)
    assert RESUME_TEST in str(refusal.value)
    assert spec_path.read_bytes() == before
