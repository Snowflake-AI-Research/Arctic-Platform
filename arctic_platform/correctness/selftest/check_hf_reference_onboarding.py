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

# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit HF preparation uses frozen workloads without resizing or target execution."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import transformers

from arctic_platform.correctness.harness import reference_records
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.config import config_checksum
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.spec import TestSpec as ConfigSpec
from arctic_platform.correctness.onboarding import onboard
from arctic_platform.correctness.selftest.check_applicable_tests import _spec

CONFIG = Path("arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config")


@pytest.fixture
def prepared_args(tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    onboard.add_arguments(parser)
    args = parser.parse_args(["--config", str(CONFIG), "--prepare-references"])
    cfg = load_config(CONFIG)
    spec = _spec(list(reference_records.CRITERIA))
    spec.config_id = cfg.config_id
    spec.config_checksum = config_checksum(cfg)
    spec.model.cache_path = str(tmp_path / "external-model")
    monkeypatch.setattr(ConfigSpec, "read", lambda path: spec)
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda path, **kwargs: SimpleNamespace(vocab_size=128),
    )
    return args, spec


@pytest.mark.parametrize("test_id", list(reference_records.CRITERIA))
def test_preparation_dispatches_frozen_workload_without_target_or_sizing(prepared_args, monkeypatch, test_id):
    args, spec = prepared_args
    args.test_id = test_id
    calls = []
    monkeypatch.setattr(runner, "execute", lambda *a, **k: calls.append((a, k)) or [])
    monkeypatch.setattr(onboard, "_run_probe", lambda *a: pytest.fail("preparation must not resize"))
    monkeypatch.setattr(onboard, "_write_initial_spec", lambda *a: pytest.fail("preparation must not rewrite spec"))
    assert onboard.run(args) == 0
    assert len(calls) == 1
    positional, keywords = calls[0]
    assert positional[1] is spec
    assert keywords["prepare_references"] is True
    assert keywords["select"] == [test_id]
    assert keywords["vocab_size"] == 128
    assert keywords["reference_calibration_runs"] == 16
    assert keywords["reference_calibration_workers"] == 1
    assert keywords["client_backend"].base_url == "http://127.0.0.1:8003"
    assert keywords["reference_records_dir"] == reference_records.DEFAULT_RECORDS_DIR


@pytest.mark.parametrize("field", ["config_checksum", "config_id"])
def test_preparation_rejects_mismatched_frozen_spec(prepared_args, monkeypatch, field):
    args, spec = prepared_args
    setattr(spec, field, "wrong")
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("mismatched workload executed"))
    with pytest.raises(ValueError, match="matching reviewed"):
        onboard.run(args)


@pytest.mark.parametrize("test_id", ["checkpoint-resume-loss", "inference-checkpoint-loss", "rl-weight-sync", "bad"])
def test_preparation_rejects_live_reference_checks(prepared_args, monkeypatch, test_id):
    args, _ = prepared_args
    args.test_id = test_id
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("normal reference prepared"))
    with pytest.raises(ValueError, match="three HF"):
        onboard.run(args)


def test_preparation_rejects_inapplicable_check(prepared_args, monkeypatch):
    args, spec = prepared_args
    spec.applicable_tests = []
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("inapplicable workload executed"))
    with pytest.raises(ValueError, match="not applicable"):
        onboard.run(args)


def test_preparation_does_not_accept_model_sizing_override(prepared_args, monkeypatch):
    args, _ = prepared_args
    args.probe_layers = 4
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("model override executed"))
    with pytest.raises(ValueError, match="model sizing"):
        onboard.run(args)


def test_preparation_requires_git_record_destination(prepared_args, tmp_path, monkeypatch):
    args, _ = prepared_args
    args.reference_records_dir = tmp_path / "untracked-outside-checkout"
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("outside Git preparation executed"))
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="Git checkout"):
        onboard.run(args)


def test_full_model_preparation_is_only_for_e2e(prepared_args, monkeypatch):
    args, _ = prepared_args
    args.local_gateway_full_model = True
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("wrong full-model workload executed"))
    with pytest.raises(ValueError, match="only for E2E"):
        onboard.run(args)


def test_preparation_passes_explicit_calibration_and_full_e2e_options(prepared_args, monkeypatch):
    args, _ = prepared_args
    args.test_id = "e2e-sft-train-validate"
    args.local_gateway_full_model = True
    args.reference_calibration_workers = 8
    calls = []
    monkeypatch.setattr(runner, "execute", lambda *a, **k: calls.append(k) or [])
    assert onboard.run(args) == 0
    assert calls[0]["local_gateway_full_model"] is True
    assert calls[0]["reference_calibration_workers"] == 8


def test_preparation_preserves_combined_check_workload(prepared_args, monkeypatch):
    args, _ = prepared_args
    args.reference_tests = ["single-step-grads", "single-step-optimizer"]
    calls = []
    monkeypatch.setattr(runner, "execute", lambda *a, **k: calls.append(k) or [])
    assert onboard.run(args) == 0
    assert calls[0]["select"] == args.reference_tests


def test_combined_selection_cannot_prepare_live_reference(prepared_args, monkeypatch):
    args, _ = prepared_args
    args.reference_tests = ["single-step-grads", "checkpoint-resume-loss"]
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("live reference prepared"))
    with pytest.raises(ValueError, match="three HF"):
        onboard.run(args)


def test_reference_selection_requires_preparation_mode(prepared_args, monkeypatch):
    args, _ = prepared_args
    args.prepare_references = False
    args.reference_tests = ["single-step-grads", "single-step-optimizer"]
    monkeypatch.setattr(onboard, "_parent", lambda *a: pytest.fail("normal onboarding executed"))
    with pytest.raises(ValueError, match="requires --prepare-references"):
        onboard.run(args)


def test_failed_reference_retains_complete_traceback_on_fast_storage(tmp_path, monkeypatch):
    output = tmp_path / "reference.json"
    stderr = "Traceback (most recent call last):\n" + "detail\n" * 1000 + "RuntimeError: actual failure\n"
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="actual stdout\n", stderr=stderr),
    )
    with pytest.raises(RuntimeError, match="full stderr:") as error:
        runner.run_reference(
            "external-model",
            tmp_path / "batch.pt",
            output,
            token_budget=64,
            ce_chunk=1,
            attn="sdpa",
            fp32_lm_head=False,
        )
    assert str(tmp_path / "reference.stderr.log") in str(error.value)
    assert (tmp_path / "reference.stderr.log").read_text() == stderr
    assert (tmp_path / "reference.stdout.log").read_text() == "actual stdout\n"


@pytest.mark.parametrize(
    "selection", [None, [], ["checkpoint-resume-loss"], ["single-step-grads", "rl-router-replay"]]
)
def test_runner_preparation_cannot_execute_normal_references(selection, tmp_path, monkeypatch):
    cfg = load_config(CONFIG)
    monkeypatch.setattr(runner, "registered_tests", lambda: pytest.fail("invalid selection reached execution"))
    with pytest.raises(ValueError, match="explicit HF"):
        runner.execute(cfg, _spec([]), [], vocab_size=128, prepare_references=True, select=selection, workdir=tmp_path)
