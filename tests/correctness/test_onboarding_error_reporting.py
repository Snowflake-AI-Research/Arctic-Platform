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

"""CPU controls for owned startup diagnostics; no workers or model downloads are started."""

import io
import json
import logging
import os
import subprocess
import traceback
from contextlib import contextmanager
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from arctic_platform.client import CortexConfig
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.correctness.harness import config as config_module
from arctic_platform.correctness.harness import cortex
from arctic_platform.correctness.harness import execution
from arctic_platform.correctness.harness import registry
from arctic_platform.correctness.harness.cortex import CorrectnessCortexTransport
from arctic_platform.correctness.harness.cortex import UnsafeCleanupError
from arctic_platform.correctness.onboarding import onboard

BATCH_ERROR = (
    "AssertionError: Check batch related parameters. train_batch_size is not equal to "
    "micro_batch_per_gpu * gradient_acc_step * world_size 8 != 1 * 1 * 2.0"
)
TEST_IDS = tuple(registry.registered_tests())
PRODUCTION_RUN = onboard.run


@pytest.mark.parametrize("encoding", ["mapping", "json", "http-repr", "human-log", "inline-msg"])
def test_local_cortex_creation_failure_reports_owned_deepspeed_cause(tmp_path, encoding):
    child = "owned-native-child"
    path = tmp_path / "training-zone-control.log"
    remote_traceback = (
        "Traceback (most recent call last):\n"
        '  File "deepspeed/runtime/config.py", line 1000, in _batch_assertion\n'
        "    assert train_batch == micro_batch * grad_acc * self.world_size\n"
        + BATCH_ERROR
    )
    if encoding == "human-log":
        text = f"12:00:00 ERROR [zone] Failed to create job {child}\n{remote_traceback}\n12:00:01 INFO done\n"
    elif encoding == "inline-msg":
        text = json.dumps({"msg": f"Failed to create job {child}\n{remote_traceback}"}) + "\n"
    else:
        text = json.dumps({"msg": f"Failed to create job {child}", "exc": remote_traceback}) + "\n"
    path.write_text(text)
    detail = {
        "message": f"job {child} failed",
        "zone_id": path.stem,
        "zone_log_path": str(path),
        "init_debug": None,
    }
    reason = (
        json.dumps({"detail": detail})
        if encoding == "json"
        else f"sub-job training failed: 500: {detail!r}" if encoding == "http-repr" else detail
    )
    result = cortex._local_creation_failure(reason, {"training": child})
    assert result.startswith(f"Remote server error: {BATCH_ERROR}\n")
    assert f"Server log: {path}" in result and remote_traceback in result
    assert "INFO done" not in result


@pytest.mark.parametrize(
    "message",
    [
        "Failed to create job owned-native-child-extra\n",
        "Failed to create job owned-native-child \n",
        "Failed to create job unrelated-child\n",
        "prefix Failed to create job owned-native-child\n",
        "Failed to create job owned-native-child: ",
    ],
)
def test_local_cortex_creation_failure_rejects_inline_message_near_prefixes(message):
    record = json.dumps({"msg": message + BATCH_ERROR})
    assert cortex._creation_failure_traceback(record, "owned-native-child") is None


@pytest.mark.parametrize(
    "broken",
    [
        "different-child",
        "different-message-child",
        "different-zone",
        "missing",
        "directory",
        "symlink",
        "unreadable",
        "malformed-reason",
        "oversized-reason",
        "no-traceback",
    ],
)
def test_local_cortex_creation_failure_never_attributes_unrelated_or_unreadable_log(tmp_path, monkeypatch, broken):
    child = "owned-native-child"
    path = tmp_path / "training-zone-control.log"
    detail = {"message": f"job {child} failed", "zone_id": path.stem, "zone_log_path": str(path)}
    if broken == "directory":
        path.mkdir()
    elif broken != "missing":
        path.write_text(
            json.dumps(
                {
                    "msg": f"Failed to create job {('unrelated-child' if broken == 'different-child' else child)}",
                    "exc": "no exception in this record" if broken == "no-traceback" else BATCH_ERROR,
                }
            )
            + "\n"
        )
    if broken == "different-message-child":
        detail["message"] = "job unrelated-child failed"
    elif broken == "different-zone":
        detail["zone_id"] = "training-zone-unrelated"
    elif broken == "symlink":
        link = tmp_path / "training-zone-link.log"
        link.symlink_to(path)
        detail.update(zone_id=link.stem, zone_log_path=str(link))
    elif broken == "unreadable":
        original_open = os.open

        def opened(value, *args, **kwargs):
            if Path(value) == path:
                raise PermissionError("unreadable control")
            return original_open(value, *args, **kwargs)

        monkeypatch.setattr(cortex.os, "open", opened)
    reason = (
        "sub-job failed: {malformed}"
        if broken == "malformed-reason"
        else "{" + "x" * 65536 if broken == "oversized-reason" else detail
    )
    assert cortex._local_creation_failure(reason, {"training": child}) is None


def test_local_cortex_creation_failure_selects_exact_json_record_and_bounds_log_read(tmp_path):
    path = tmp_path / "training-zone-control.log"
    child = "owned-native-child"
    path.write_text(
        "x" * (1 << 20)
        + "\n"
        + json.dumps({"msg": f"Failed to create job {child}", "exc": BATCH_ERROR})
        + "\n"
        + json.dumps({"msg": "Failed to create job unrelated-child", "exc": "ValueError: stale unrelated failure"})
        + "\n"
    )
    result = cortex._local_creation_failure(
        {"message": f"job {child} failed", "zone_id": path.stem, "zone_log_path": str(path)},
        {"training": child},
    )
    assert BATCH_ERROR in result
    assert "stale unrelated failure" not in result and len(result) < 2000


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("recoverable", [False, True])
def test_local_cortex_creation_failure_real_poll_preserves_cause_and_banner(
    tmp_path, monkeypatch, capsys, local, recoverable
):
    child = "owned-native-child"
    path = tmp_path / "training-zone-control.log"
    path.write_text(json.dumps({"msg": f"Failed to create job {child}", "exc": BATCH_ERROR}))
    transport = object.__new__(CorrectnessCortexTransport)
    transport.config = CortexConfig(base_url="http://127.0.0.1:8003" if local else "http://remote.test")
    transport.poll_interval = 0.1
    transport.job_id = "owned-parent"
    transport._native_children = {"training": child}
    detail = {"message": f"job {child} failed", "zone_id": path.stem, "zone_log_path": str(path)}
    if not recoverable:
        detail["zone_log_path"] = str(tmp_path / "missing.log")
    monkeypatch.setattr(CortexTransport, "_job", lambda self: {"status": "failed", "reason": detail})
    with pytest.raises(RuntimeError) as caught:
        transport._wait_running()
    reason = str(caught.value)
    assert "terminal state 'failed'" in reason
    if local and recoverable:
        assert reason.startswith(f"Remote server error: {BATCH_ERROR}\n")
        assert isinstance(caught.value.__cause__, RuntimeError)
    else:
        assert "Remote server error:" not in reason
    onboard._onboarding_failure(reason)
    assert reason.splitlines()[0] in capsys.readouterr().err


@pytest.mark.parametrize("unsafe_cleanup", [False, True])
@pytest.mark.parametrize("test_id", TEST_IDS)
def test_onboarding_target_creation_error_leads_final_banner_and_preserves_unsafe_stop(
    tmp_path, monkeypatch, capsys, unsafe_cleanup, test_id
):
    args = execution.parse_args(["onboard", "--config", str(tmp_path / "model.json"), "--test", test_id])
    args.output_dir = tmp_path
    args.checkout_root = tmp_path
    cause = f"Remote server error: {BATCH_ERROR}"
    reason = cause + "\nServer log: /tmp/raydss/training-zone-control.log"
    if unsafe_cleanup:
        reason = "UnsafeCleanupError: Original workload failed: RuntimeError: " + reason

    def child(command, **kwargs):
        out = args.output_dir / "validation" / "4-layers"
        (out / "report.json").write_text(
            json.dumps(
                {
                    "totals": {"error": 1},
                    "results": [{"test_id": args.test_id, "arm": "gas1", "outcome": "error", "reason": reason}],
                }
            )
        )
        (out / "batch.json").write_text(json.dumps({"stopped_unsafe": unsafe_cleanup}))
        return SimpleNamespace(stdout=io.StringIO(reason + "\n"), wait=lambda: 1)

    monkeypatch.setattr(onboard.subprocess, "Popen", child)
    error_type = UnsafeCleanupError if unsafe_cleanup else RuntimeError
    with pytest.raises(error_type, match="8 != 1") as caught:
        validation = onboard._run_final_regression(args, {"layers": 4})
        onboard._require_validation_pass(validation, "final 4-layer regression")
    assert str(caught.value).splitlines()[0] == f"{test_id} / gas1: {reason.splitlines()[0]}"
    assert "Server log:" in str(caught.value) and "Complete log:" in str(caught.value)
    if unsafe_cleanup:
        assert "Further onboarding stopped" in str(caught.value)
        assert not (tmp_path / "validation" / "4-layers" / "validation.json").exists()
    onboard._onboarding_failure(str(caught.value))
    printed = capsys.readouterr().err
    assert f"!!!!! ONBOARDING FAILED : {str(caught.value).splitlines()[0]} !!!!!" in printed
    assert "8 != 1 * 1 * 2.0" in printed


@pytest.fixture
def onboarding_attempt(tmp_path, monkeypatch):
    source = tmp_path / "model.json"
    source.write_text(json.dumps({"model_name": "Qwen/Qwen3-0.6B", "training_gpus": 1, "training": {}}))
    spec = tmp_path / "model-spec.json"
    spec.write_text("{}")
    args = execution.parse_args(["onboard", "--config", str(source), "--test-spec", str(spec)])
    args.output_dir = tmp_path / "output"
    monkeypatch.setenv("AP_CORRECTNESS_LOGDIR", str(tmp_path / "logs"))
    monkeypatch.setattr(onboard, "run", onboard._onboard_configuration)
    # Applicability/model execution is separate from these shared failure-path controls.
    monkeypatch.setattr(registry, "onboarding_inapplicable_reason", lambda *a: None)
    monkeypatch.setattr(
        onboard, "_onboarding_runtime_sources", lambda: {"python": "control", "dependencies": {}, "code": {}}
    )
    monkeypatch.setattr(onboard, "_onboarding_preflight", lambda *a: None)
    return args


@pytest.mark.parametrize("test_id", TEST_IDS)
@pytest.mark.parametrize("phase", ["preflight", "backend", "exit"])
def test_every_onboarding_test_retains_failure_log_and_original_cause(
    onboarding_attempt, monkeypatch, capsys, test_id, phase
):
    args = onboarding_attempt
    args.test_id = test_id
    before_handlers = logging.getLogger().handlers[:]

    def fail(*a):
        try:
            raise ValueError("AP worker original cause")
        except ValueError:
            logging.exception("owned backend operation failed")
            if phase == "exit":
                return 9
            raise

    monkeypatch.setattr(onboard, "_parent", fail)
    if phase == "preflight":
        monkeypatch.setattr(onboard, "_onboarding_preflight", fail)
        monkeypatch.setattr(onboard, "_parent", lambda *a: pytest.fail("preflight reached model execution"))
    assert onboard.logged_run(args) == 1
    assert logging.getLogger().handlers == before_handlers
    result = json.loads(args.test_spec.with_suffix(".onboarding.json").read_text())["results"][test_id]
    path = Path(result["log"])
    assert path == args._diagnostics_dir / test_id / "test.log"
    assert "ValueError: AP worker original cause" in path.read_text()
    assert "Traceback (most recent call last)" in path.read_text()
    assert "AP worker original cause" in result["reason"]
    assert str(path) in result["reason"]
    assert "AP worker original cause" in (args._diagnostics_dir / "onboarding.log").read_text()
    assert "AP worker original cause" in (args._diagnostics_dir / "report.md").read_text()
    console = capsys.readouterr()
    assert "AP worker original cause" in console.err
    assert str(path) in console.err


def test_config_onboarding_preserves_separate_failure_logs_and_continues(onboarding_attempt, monkeypatch):
    args = onboarding_attempt
    called = []

    def fail(child):
        called.append(child.test_id)
        raise ValueError(f"AP cause for {child.test_id}")

    monkeypatch.setattr(onboard, "_parent", fail)
    assert onboard.logged_run(args) == 1
    assert set(called) == set(TEST_IDS)
    results = json.loads(args.test_spec.with_suffix(".onboarding.json").read_text())["results"]
    for test_id in TEST_IDS:
        text = Path(results[test_id]["log"]).read_text()
        assert f"AP cause for {test_id}" in text
        assert all(f"AP cause for {other}" not in text for other in TEST_IDS if other != test_id)


@pytest.mark.parametrize("test_id", TEST_IDS)
@pytest.mark.parametrize("output", ["silent", "print", "logger"])
def test_validation_retains_log_when_gateway_fails_before_child_start(tmp_path, monkeypatch, test_id, output):
    args = execution.parse_args(["onboard", "--config", str(tmp_path / "model.json"), "--test", test_id])
    args.output_dir = args.checkout_root = tmp_path

    @contextmanager
    def gateway(*a):
        try:
            raise ValueError("AP gateway startup original cause")
        except ValueError:
            if output == "print":
                print("AP gateway startup original cause", flush=True)
            elif output == "logger":
                logging.exception("AP gateway startup failed")
            raise
        yield

    monkeypatch.setattr(onboard, "_validation_gateway", gateway)
    monkeypatch.setattr(onboard.subprocess, "Popen", lambda *a, **k: pytest.fail("failed gateway launched child"))
    with pytest.raises(ValueError, match="AP gateway startup original cause"):
        onboard._run_final_regression(args, {"layers": 4})
    log = (tmp_path / "validation" / "4-layers" / "run.log").read_text()
    assert "ValueError: AP gateway startup original cause" in log
    assert "Traceback (most recent call last)" in log


@pytest.mark.parametrize("test_id", TEST_IDS)
@pytest.mark.parametrize("phase", ["runtime", "backend", "exit"])
def test_existing_spec_single_test_preserves_failure_log_and_cause(
    onboarding_attempt, monkeypatch, capsys, test_id, phase
):
    args = onboarding_attempt
    args.test_id = test_id
    monkeypatch.setattr(onboard, "run", PRODUCTION_RUN)
    monkeypatch.setattr(onboard, "_validate_storage", lambda *a: None)
    monkeypatch.setattr(config_module, "local_config_path", lambda path: path)
    monkeypatch.setattr(config_module, "validate_first_two_model_dtype", lambda *a: None)
    monkeypatch.setattr(onboard, "_completed_onboarding", lambda *a: None)
    monkeypatch.setattr(
        onboard, "_onboarding_runtime_sources", lambda: {"python": "control", "dependencies": {}, "code": {}}
    )

    def fail(*a):
        try:
            raise ValueError("AP worker original cause")
        except ValueError as cause:
            if phase == "exit":
                logging.exception("owned backend operation failed")
                return 9
            raise RuntimeError("owned operation failed") from cause

    monkeypatch.setattr(onboard, "_parent", fail)
    if phase == "runtime":
        monkeypatch.setattr(onboard, "_onboarding_runtime_sources", fail)
    assert onboard.logged_run(args) == (9 if phase == "exit" else 1)
    result = json.loads(args.test_spec.with_suffix(".onboarding.json").read_text())["results"][test_id]
    log = Path(result["log"])
    assert "AP worker original cause" in log.read_text()
    assert "AP worker original cause" in result["reason"]
    assert str(log) in result["reason"]
    console = capsys.readouterr().err
    assert "AP worker original cause" in console
    if phase != "exit":
        assert console.splitlines().count("ValueError: AP worker original cause") == 1
        assert log.read_text().splitlines().count("ValueError: AP worker original cause") == 1


@pytest.mark.parametrize("test_id", TEST_IDS)
@pytest.mark.parametrize("outcome", ["returned-failure", "exception", "pass"])
def test_regression_batch_surfaces_saved_backend_failure_only_for_failed_tests(
    tmp_path, monkeypatch, capsys, test_id, outcome
):
    from arctic_platform.correctness.api import SelectedCase

    args = execution.parse_args(["run", "--config", str(tmp_path / "model.json"), "--test", test_id])
    args.out = tmp_path / "run"
    monkeypatch.setattr(execution, "_load_dense_checkpoint_allocation", lambda *a: None)
    monkeypatch.setattr(execution, "write_report", lambda *a, **k: None)
    before_handlers = logging.getLogger().handlers[:]

    def run_configs(child, *, executor, snapshot):
        try:
            raise ValueError("AP backend actual failure")
        except ValueError:
            if outcome == "exception":
                raise RuntimeError("outer harness failure")
            if outcome == "returned-failure":
                logging.exception("AP backend operation failed")
                traceback.print_exc()
            else:
                print("quiet successful backend progress")
        result = registry.TestResult(
            test_id,
            "model",
            registry.TestOutcome.PASS if outcome == "pass" else registry.TestOutcome.FAIL,
            summary="pass" if outcome == "pass" else "outer harness failure",
            error_type=None if outcome == "pass" else "RuntimeError",
        )
        snapshot([result], {}, {})
        return int(outcome != "pass")

    monkeypatch.setattr(execution, "_run_configs", run_configs)
    batch = execution.run_batch(args, selected=[SelectedCase("model", Path(args.config), test_id)])
    assert logging.getLogger().handlers == before_handlers
    console = capsys.readouterr().out
    if outcome == "pass":
        assert batch.exit_code == 0
        assert "quiet successful backend progress" not in console
        assert "Last execution output" not in console
    else:
        assert batch.exit_code != 0
        assert "ValueError: AP backend actual failure" in console
        assert str(args.out / "full.log") in console
        assert "ValueError: AP backend actual failure" in (args.out / "full.log").read_text()


@pytest.mark.parametrize("test_id", ["inference-checkpoint-loss", "e2e-sft-train-validate"])
def test_real_check_exception_handler_surfaces_original_backend_error(tmp_path, monkeypatch, capsys, test_id):
    from arctic_platform.correctness.api import SelectedCase
    from arctic_platform.correctness.checks import e2e_sft_train_validate
    from arctic_platform.correctness.checks import inference_checkpoint_loss

    check = {
        "inference-checkpoint-loss": inference_checkpoint_loss,
        "e2e-sft-train-validate": e2e_sft_train_validate,
    }[test_id]
    args = execution.parse_args(["run", "--config", str(tmp_path / "model.json"), "--test", test_id])
    args.out = tmp_path / "run"
    monkeypatch.setattr(execution, "_load_dense_checkpoint_allocation", lambda *a: None)
    monkeypatch.setattr(execution, "write_report", lambda *a, **k: None)

    def fail(ctx):
        try:
            raise ValueError(BATCH_ERROR)
        except ValueError as cause:
            raise RuntimeError("AP backend job failed") from cause

    monkeypatch.setattr(check, "_run", fail)

    def run_configs(child, *, executor, snapshot):
        results = check.run(SimpleNamespace(config_id="model"))
        snapshot(results, {}, {})
        return 1

    monkeypatch.setattr(execution, "_run_configs", run_configs)
    batch = execution.run_batch(args, selected=[SelectedCase("model", Path(args.config), test_id)])
    assert batch.exit_code == 1
    assert batch.cases[0]["status"] == "error"
    assert BATCH_ERROR in capsys.readouterr().out
    assert BATCH_ERROR in (args.out / "full.log").read_text()


def test_shared_failed_group_replays_output_once_without_previous_success_output(tmp_path, monkeypatch, capsys):
    from arctic_platform.correctness.api import SelectedCase

    args = execution.parse_args(["run", "--config", str(tmp_path / "model.json")])
    args.out = tmp_path / "run"
    monkeypatch.setattr(execution, "_load_dense_checkpoint_allocation", lambda *a: None)
    monkeypatch.setattr(execution, "write_report", lambda *a, **k: None)

    def run_configs(child, *, executor, snapshot):
        passed = child.test == ["inference-checkpoint-loss"]
        print("private successful output" if passed else "shared backend failure")
        snapshot(
            [
                registry.TestResult(
                    test_id,
                    "model",
                    registry.TestOutcome.PASS if passed else registry.TestOutcome.FAIL,
                    summary="pass" if passed else "AP backend job failed",
                    error_type=None if passed else "RuntimeError",
                )
                for test_id in child.test
            ],
            {},
            {},
        )
        return int(not passed)

    monkeypatch.setattr(execution, "_run_configs", run_configs)
    execution.run_batch(
        args,
        selected=[
            SelectedCase("model", Path(args.config), test_id)
            for test_id in ["inference-checkpoint-loss", "single-step-grads", "single-step-optimizer"]
        ],
    )
    console = capsys.readouterr().out
    assert "private successful output" not in console
    assert console.count("Last execution output:") == 1
    assert console.count("shared backend failure") == 1


@pytest.mark.parametrize("test_id", TEST_IDS)
@pytest.mark.parametrize("report", ["invalid JSON", '{"results": [null]}'])
def test_malformed_validation_report_cannot_hide_unsafe_cleanup(tmp_path, monkeypatch, test_id, report):
    args = execution.parse_args(["onboard", "--config", str(tmp_path / "model.json"), "--test", test_id])
    args.output_dir = args.checkout_root = tmp_path
    directory = tmp_path / "validation" / "4-layers"
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(report)
    (directory / "batch.json").write_text(json.dumps({"stopped_unsafe": True}))
    monkeypatch.setattr(onboard, "_validation_gateway", lambda *a: nullcontext(None))
    monkeypatch.setattr(
        onboard.subprocess,
        "Popen",
        lambda *a, **k: SimpleNamespace(stdout=io.StringIO("AP backend failed\n"), wait=lambda: 1),
    )
    with pytest.raises(UnsafeCleanupError, match="Further onboarding stopped"):
        onboard._run_final_regression(args, {"layers": 4})


@pytest.mark.parametrize("error", [KeyboardInterrupt, UnsafeCleanupError])
def test_failure_logging_restores_handlers_environment_and_stops_on_unsafe_cleanup(
    onboarding_attempt, monkeypatch, error
):
    args = onboarding_attempt
    handlers = logging.getLogger().handlers[:]
    root = os.environ["AP_CORRECTNESS_LOGDIR"]
    called = []

    def fail(child):
        called.append(child.test_id)
        raise error("owned execution could not continue")

    monkeypatch.setattr(onboard, "_parent", fail)
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            onboard.logged_run(args)
    else:
        assert onboard.logged_run(args) == 1
    assert len(called) == 1
    assert logging.getLogger().handlers == handlers
    assert os.environ["AP_CORRECTNESS_LOGDIR"] == root
    assert "owned execution could not continue" in (args._diagnostics_dir / called[0] / "test.log").read_text()


def test_subprocess_failure_keeps_original_stderr_in_reason(tmp_path):
    log = tmp_path / "test.log"
    log.write_text("Traceback (most recent call last):\nValueError: original calibration failure\n")
    error = subprocess.CalledProcessError(1, ["calibrate"])
    assert "original calibration failure" in onboard._failure_reason(error, log)
    assert str(log) in onboard._failure_reason(error, log)


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_early_onboarding_interruption_retains_traceback_and_restores_state(tmp_path, monkeypatch, capsys, error):
    args = SimpleNamespace(config="model.json")
    root = tmp_path / "logs"
    monkeypatch.setenv("AP_CORRECTNESS_LOGDIR", str(root))
    handlers = logging.getLogger().handlers[:]

    def fail(*a):
        raise error("interrupted before test preflight")

    monkeypatch.setattr(onboard, "run", fail)
    with pytest.raises(error, match="interrupted before test preflight"):
        onboard.logged_run(args)
    assert "interrupted before test preflight" in (args._diagnostics_dir / "onboarding.log").read_text()
    console = capsys.readouterr().err
    assert console.count("Traceback (most recent call last)") == (0 if error is KeyboardInterrupt else 1)
    assert logging.getLogger().handlers == handlers
    assert os.environ["AP_CORRECTNESS_LOGDIR"] == str(root)


@pytest.mark.parametrize("oom", [False, True])
def test_sizing_logs_unexpected_traceback_but_preserves_expected_oom(tmp_path, monkeypatch, capsys, oom):
    import torch

    from arctic_platform.correctness.onboarding import synth_model

    args = SimpleNamespace(
        config=tmp_path / "model.json",
        cache_root=tmp_path,
        source_checkpoint="model",
        probe_layers=4,
        probe_output=tmp_path / "sizing.json",
    )
    monkeypatch.setattr(onboard, "_validate_storage", lambda *a: None)
    monkeypatch.setattr(onboard, "_memory_snapshot", lambda: {})
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a: None)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(config_module, "load_config", lambda *a: SimpleNamespace(effective_training={}))
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "0")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    def fail(*a):
        if oom:
            raise torch.OutOfMemoryError("CUDA out of memory")
        try:
            raise ValueError("original model construction failure")
        except ValueError as error:
            raise RuntimeError("sizing model could not start") from error

    monkeypatch.setattr(synth_model, "materialize_pretrained", fail)
    assert onboard._probe_worker(args) == (onboard.OOM_EXIT if oom else 1)
    result = json.loads(args.probe_output.read_text())
    assert result["status"] == ("oom" if oom else "error")
    console = capsys.readouterr().err
    if oom:
        assert "Traceback" not in console
    else:
        assert "ValueError: original model construction failure" in console
        assert "original model construction failure" in result["error"]
