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

"""Checkpoint onboarding adds one contract without replacing reviewed assets or gates."""

import json
import sys
from types import SimpleNamespace

import pytest

from arctic_platform.correctness.checks import checkpoint_resume
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.onboarding import onboard
from arctic_platform.correctness.selftest.check_fixed_tolerance_onboarding import RESUME_TEST
from arctic_platform.correctness.selftest.check_fixed_tolerance_onboarding import _prepared
from arctic_platform.correctness.selftest.check_fixed_tolerance_onboarding import _regression


@pytest.fixture
def calibration(monkeypatch, tmp_path):
    args, config, spec_path = _prepared(tmp_path, RESUME_TEST)
    original = spec_path.read_bytes()
    calls = []

    def model_config(path, **kwargs):
        calls.append(("model", path))
        return SimpleNamespace(text_config=SimpleNamespace(vocab_size=256))

    def measure(training, model, **kwargs):
        assert spec_path.read_bytes() == original
        calls.append(("measure", model, kwargs))
        return 0.1001, "gas4 loss at iteration 10"

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=model_config)),
    )
    monkeypatch.setattr(checkpoint_resume, "measure_undeterministic_loss_range", measure)
    monkeypatch.setattr(onboard, "_run_final_regression", _regression("pass"))
    return args, config, spec_path, original, calls


def test_missing_checkpoint_contract_preserves_existing_reviewed_fields(calibration):
    args, config, path, original, calls = calibration
    before = json.loads(original)

    assert onboard._onboard_checkpoint_resume(args, config) == 0

    after = json.loads(path.read_text())
    assert set(after["applicable_tests"]) - set(before["applicable_tests"]) == {RESUME_TEST}
    assert after["applicable_tests"] == sorted(after["applicable_tests"])
    assert {key: value for key, value in after.items() if key not in {"applicable_tests", "test_tolerances"}} == {
        key: value for key, value in before.items() if key not in {"applicable_tests", "test_tolerances"}
    }
    assert {key: value for key, value in after["test_tolerances"].items() if key != RESUME_TEST} == before[
        "test_tolerances"
    ]
    assert after["test_tolerances"][RESUME_TEST]["absolute"] == 0.201
    assert [(call[0], call[1]) for call in calls] == [
        ("model", before["model"]["cache_path"]),
        ("measure", before["model"]["cache_path"]),
    ]
    assert calls[1][2]["slots"] == config.n_gpus
    assert calls[1][2]["vocab_size"] == 256
    assert calls[1][2]["workdir"] == args.output_dir / "checkpoint-resume-calibration"


@pytest.mark.parametrize("status", ["failed", "error"])
def test_unsuccessful_validation_restores_exact_reviewed_bytes(calibration, monkeypatch, status):
    args, config, path, original, _ = calibration
    monkeypatch.setattr(onboard, "_run_final_regression", _regression(status))

    with pytest.raises(RuntimeError, match=f"{RESUME_TEST} {status}"):
        onboard._onboard_checkpoint_resume(args, config)

    assert path.read_bytes() == original


@pytest.mark.parametrize("exception", [RuntimeError("gateway unavailable"), KeyboardInterrupt()])
def test_interrupted_validation_restores_exact_reviewed_bytes(calibration, monkeypatch, exception):
    args, config, path, original, _ = calibration

    def fail(*args, **kwargs):
        assert RESUME_TEST in json.loads(path.read_text())["applicable_tests"]
        raise exception

    monkeypatch.setattr(onboard, "_run_final_regression", fail)
    with pytest.raises(type(exception)):
        onboard._onboard_checkpoint_resume(args, config)

    assert path.read_bytes() == original


def test_stale_config_is_rejected_before_measurement_or_spec_write(calibration):
    args, _, path, original, calls = calibration
    config = json.loads(args.config.read_text())
    config["max_seq_len"] = 16384
    args.config.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="hashes to"):
        onboard._onboard_checkpoint_resume(args, load_config(args.config))

    assert calls == []
    assert path.read_bytes() == original


def test_calibration_failure_does_not_modify_reviewed_spec(calibration, monkeypatch):
    args, config, path, original, _ = calibration

    def fail(*args, **kwargs):
        raise RuntimeError("calibration failed")

    monkeypatch.setattr(checkpoint_resume, "measure_undeterministic_loss_range", fail)
    with pytest.raises(RuntimeError, match="calibration failed"):
        onboard._onboard_checkpoint_resume(args, config)

    assert path.read_bytes() == original
