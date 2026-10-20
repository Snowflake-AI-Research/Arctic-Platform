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

"""Operator selection and batch behavior without models, GPUs or backend requests."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from arctic_platform.correctness import api
from arctic_platform.correctness.harness import execution
from arctic_platform.correctness.harness.dss_driver import UnsafeCleanupError
from arctic_platform.correctness.harness.reference_records import ReferencePrerequisiteError
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import TestResult as Result

CONFIG = "qwen3-8b-h200-train-sft-full-4gpus-64k"
TEST = "inference-checkpoint-loss"


def test_pair_selection_accepts_catalog_id_or_path():
    expected = api.select(api.RunRequest(config=CONFIG, tests=(TEST,)))
    assert expected == api.select(api.RunRequest(config=expected[0].config_path, tests=(TEST,)))
    assert [(case.config_id, case.test_id) for case in expected] == [(CONFIG, TEST)]


def test_config_only_selects_each_maintained_test_once():
    selected = api.select(api.RunRequest(config=CONFIG))
    assert [case.test_id for case in selected] == sorted(api.registered_tests())
    assert len(set(selected)) == len(selected)


def test_test_only_selects_all_reviewed_configs_in_order():
    selected = api.select(api.RunRequest(tests=(TEST, TEST)))
    assert [case.config_id for case in selected] == sorted(api.catalog())
    assert all(case.test_id == TEST for case in selected)
    assert len(set(selected)) == len(selected)


@pytest.mark.parametrize(
    "selection,match",
    [
        (api.RunRequest(), "give --config"),
        (api.RunRequest(config="missing"), "unknown configuration"),
        (api.RunRequest(config=CONFIG, tests=("missing",)), "unknown check"),
        (api.RunRequest(config=CONFIG, all_configs=True), "not both"),
    ],
)
def test_bad_selectors_fail_before_backend(selection, match, monkeypatch):
    monkeypatch.setattr(execution, "_client_backend", lambda *a: pytest.fail("backend accessed"))
    with pytest.raises(ValueError, match=match):
        api.run(selection)


def test_empty_catalog_is_not_a_success(tmp_path):
    with pytest.raises(ValueError, match="no active"):
        api.select(api.RunRequest(tests=(TEST,), config_dir=tmp_path))


def test_archive_configs_are_not_discovered(tmp_path, monkeypatch):
    root = tmp_path / "configs"
    archived = root / "archive" / "model" / "h200" / "train.config"
    archived.parent.mkdir(parents=True)
    archived.write_text("{}")
    monkeypatch.setattr(api, "PACKAGE_DIR", tmp_path)
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "archive-model-h200-train.json").write_text("{}")
    assert api.catalog(root) == {}


def _fake_batches(monkeypatch, verdicts):
    monkeypatch.setattr(execution, "write_report", lambda *a, **k: None)

    def fake(args, *, executor, snapshot):
        outcome = verdicts.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        results = [
            Result(check, api.config_id(Path(args.config)), outcome, summary="synthetic comparison")
            for check in args.test
        ]
        snapshot(results, {}, {})
        return api.aggregate_exit_code(results)

    monkeypatch.setattr(execution, "_run_configs", fake)


def test_mixed_batch_continues_and_retains_every_pair(tmp_path, monkeypatch, capsys):
    tests = ("checkpoint-resume-loss", "e2e-sft-train-validate", TEST)
    _fake_batches(monkeypatch, [Outcome.FAIL, ReferencePrerequisiteError("prepared reference missing"), Outcome.PASS])
    result = api.run(api.RunRequest(config=CONFIG, tests=tests), api.RunOptions(out=tmp_path))
    assert [case["status"] for case in result.cases] == ["fail", "blocked", "pass"]
    assert result.exit_code == 1 and not result.stopped_unsafe
    assert len(result.results) == len(tests)
    assert "ReferencePrerequisiteError" in (tmp_path / "full.log").read_text()
    assert "Traceback" not in capsys.readouterr().out
    assert (tmp_path / "batch.json").is_file()


def test_actual_execution_preserves_selection_order_and_combined_step_identity(tmp_path, monkeypatch):
    selected = api.select(api.RunRequest(config=CONFIG))
    observed = []
    monkeypatch.setattr(execution, "write_report", lambda *a, **k: None)

    def record(args, *, executor, snapshot):
        observed.append(tuple(args.test))
        snapshot([Result(check, CONFIG, Outcome.PASS) for check in args.test], {}, {})

    monkeypatch.setattr(execution, "_run_configs", record)
    result = api.run(api.RunRequest(config=CONFIG), api.RunOptions(out=tmp_path))
    assert [check for group in observed for check in group] == [case.test_id for case in selected]
    assert ("single-step-grads", "single-step-optimizer") in observed
    assert [case["test_id"] for case in result.cases] == [case.test_id for case in selected]
    assert result.exit_code == 0


@pytest.mark.parametrize("module", ["rl_weight_sync", "rl_router_replay"])
def test_rl_wrapper_traceback_is_saved_in_batch_log(module, tmp_path, monkeypatch, capsys):
    from importlib import import_module

    check = import_module(f"arctic_platform.correctness.checks.{module}")

    def fail(ctx):
        raise RuntimeError("synthetic RL failure")

    def execute(args, *, executor, snapshot):
        snapshot(check.run(SimpleNamespace(config_id=CONFIG)), {}, {})

    monkeypatch.setattr(check, "_run", fail)
    monkeypatch.setattr(execution, "_run_configs", execute)
    monkeypatch.setattr(execution, "write_report", lambda *a, **k: None)
    result = api.run(api.RunRequest(config=CONFIG, tests=(check.TEST_ID,)), api.RunOptions(out=tmp_path))
    assert result.cases[0]["status"] == "error" and result.exit_code == 1
    assert "Traceback" in (tmp_path / "full.log").read_text()
    assert "RuntimeError: synthetic RL failure" in (tmp_path / "full.log").read_text()
    assert "Traceback" not in capsys.readouterr().err


def test_inapplicable_is_visible_and_not_a_pass(tmp_path, monkeypatch):
    _fake_batches(monkeypatch, [Outcome.INAPPLICABLE])
    result = api.run(api.RunRequest(config=CONFIG, tests=(TEST,)), api.RunOptions(out=tmp_path))
    assert result.cases[0]["status"] == "inapplicable"
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("unexpected failure"), UnsafeCleanupError("cleanup did not complete"), KeyboardInterrupt()],
)
def test_unsafe_exit_or_interruption_stops_remaining_cases(tmp_path, monkeypatch, failure):
    _fake_batches(monkeypatch, [failure])
    result = api.run(
        api.RunRequest(config=CONFIG, tests=("e2e-sft-train-validate", TEST)),
        api.RunOptions(out=tmp_path),
    )
    assert result.stopped_unsafe and result.exit_code == 1
    assert [case["status"] for case in result.cases] == ["error", "blocked"]
    assert all(result.outcome is Outcome.FAIL for result in result.results)


def test_empty_execution_is_explicitly_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "write_report", lambda *a, **k: None)
    monkeypatch.setattr(execution, "_run_configs", lambda *a, **k: 0)
    result = api.run(api.RunRequest(config=CONFIG, tests=(TEST,)), api.RunOptions(out=tmp_path))
    assert result.cases[0]["status"] == "blocked" and result.exit_code == 1


@pytest.mark.parametrize(
    "error_type,status",
    [(None, "fail"), ("RuntimeError", "error"), ("ReferencePrerequisiteError", "blocked")],
)
def test_comparison_failure_and_execution_errors_are_distinct(error_type, status):
    case = api.SelectedCase(CONFIG, api.catalog()[CONFIG], TEST)
    result = Result(TEST, CONFIG, Outcome.FAIL, summary="comparison did not pass", error_type=error_type)
    assert execution._case_summary(case, [result])["status"] == status


def test_parse_flags_is_shared_with_operator_command():
    request, options = api.parse_run_args(["--config", CONFIG, "--test", TEST, "--backend", "onprem"])
    assert request.config == CONFIG and request.tests == (TEST,)
    assert options.backend == "onprem"


def test_static_guide_is_unchanged_by_run_and_onboarding(tmp_path, monkeypatch):
    guide = api.PACKAGE_DIR / "docs" / "running-tests.doc"
    before = guide.read_bytes()
    monkeypatch.setattr(execution, "_run_vocab_size", lambda *a, **k: 128)
    from transformers import AutoConfig

    from arctic_platform.correctness.harness import report
    from arctic_platform.correctness.harness import runner

    monkeypatch.setattr(report, "collect", lambda: {"host": "synthetic"})
    result = api.run(
        api.RunRequest(config=CONFIG, tests=(TEST,)),
        api.RunOptions(out=tmp_path, base_url="http://existing.test"),
        executor=lambda cfg, *a, **k: [Result(TEST, cfg.config_id, Outcome.PASS)],
    )
    assert result.exit_code == 0 and (tmp_path / "report.md").exists()
    prepared = []
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(vocab_size=128))
    monkeypatch.setattr(runner, "execute", lambda *a, **k: prepared.append(k))
    assert (
        execution.main(
            [
                "onboard",
                "--config",
                str(api.catalog()[CONFIG]),
                "--prepare-references",
                "--output-dir",
                str(tmp_path / "preparation"),
                "--test-id",
                "single-step-grads",
            ]
        )
        == 0
    )
    assert len(prepared) == 1 and prepared[0]["prepare_references"] is True
    assert guide.read_bytes() == before


def test_logger_traceback_is_saved_and_handlers_are_restored(tmp_path, monkeypatch, capsys):
    root = logging.getLogger()
    previous = root.handlers[:]
    monkeypatch.setattr(execution, "write_report", lambda *a, **k: None)

    def run_with_logged_failure(args, *, executor, snapshot):
        try:
            raise ReferencePrerequisiteError("missing prepared identity")
        except ReferencePrerequisiteError:
            logging.getLogger(__name__).exception("reference unavailable")
        snapshot([Result(TEST, CONFIG, Outcome.FAIL, summary="ReferencePrerequisiteError: missing identity")], {}, {})

    monkeypatch.setattr(execution, "_run_configs", run_with_logged_failure)
    result = api.run(api.RunRequest(config=CONFIG, tests=(TEST,)), api.RunOptions(out=tmp_path))
    assert result.cases[0]["status"] == "blocked"
    assert "Traceback" in (tmp_path / "full.log").read_text()
    assert "Traceback" not in capsys.readouterr().err
    assert root.handlers == previous


@pytest.mark.parametrize(
    "module",
    ["e2e_sft_train_validate", "inference_checkpoint_loss", "rl_weight_sync", "rl_router_replay", "checkpoint_resume"],
)
def test_check_wrappers_propagate_unsafe_cleanup(module, tmp_path, monkeypatch):
    import contextlib
    from importlib import import_module

    check = import_module(f"arctic_platform.correctness.checks.{module}")
    failure = UnsafeCleanupError("owned cleanup failed")

    def fail(*args):
        raise failure

    context = SimpleNamespace(
        workdir=tmp_path,
        arms=[SimpleNamespace(name="gas1")],
        transport_for=lambda *args: contextlib.nullcontext(),
    )
    monkeypatch.setattr(check, "_run_arm" if module == "checkpoint_resume" else "_run", fail)
    with pytest.raises(UnsafeCleanupError) as caught:
        check.run(context)
    assert caught.value is failure
