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

"""Static correctness selftests: operator interface."""

from __future__ import annotations

import ast
import copy
import inspect
import io
import json
import logging
import os
import runpy
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from arctic_platform.correctness import api
from arctic_platform.correctness.checks import checkpoint_resume
from arctic_platform.correctness.diagnostics import __main__ as dispatch
from arctic_platform.correctness.harness import cortex as cortex_fixture
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness import execution
from arctic_platform.correctness.harness import execution as cli
from arctic_platform.correctness.harness import report
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness import storage as preflight
from arctic_platform.correctness.harness import storage as workdir
from arctic_platform.correctness.harness.dss_driver import UnsafeCleanupError
from arctic_platform.correctness.harness.reference_records import ReferencePrerequisiteError
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import TestResult as Result
from arctic_platform.correctness.harness.report import activity
from arctic_platform.correctness.harness.report import arm_verdict
from arctic_platform.correctness.harness.report import write
from arctic_platform.correctness.harness.storage import CORRECTNESS_TMPDIR_ENV
from arctic_platform.correctness.reference import hf_single_gpu
from arctic_platform.correctness.reference.adamw_trajectory import AdamWTrajectory

from .check_catalog_configuration import _applicable_tests_spec
from .helpers import _config_factory_native_config

# Clean port.


@pytest.mark.parametrize("provider", ["huggingface", "prime_rl"])
@pytest.mark.parametrize("case,rows,calls", [("gas1", 4, 1), ("gas4", 16, 4)])
def test_clean_port_checkpoint_cases_use_plain_supervised_requests(provider, case, rows, calls):
    ctx = SimpleNamespace(
        config_id="qwen3-8b-h200-train-sft-full-4gpus-64k",
        cfg=SimpleNamespace(training={"model_provider": provider}),
        vocab_size=128,
    )
    arm = SimpleNamespace(name=case, global_batch_size=rows, max_seq_len=8, dss_microbatches=calls)
    bodies = checkpoint_resume._iteration_bodies(ctx, arm)
    batches = checkpoint_resume.iteration_batches(ctx, arm)
    for body, batch in zip(bodies, batches, strict=True):
        assert isinstance(body["batch"], dict)
        assert body["processing"] == {"loss_fn": "sft"}
        lowered = dss_driver.cortex_training_body(body)
        assert "processing" not in lowered
        assert lowered["kwargs"]["input_ids"].shape == (rows, 8)
        labels = batch.shifted_labels() if provider == "prime_rl" else batch.labels
        torch.testing.assert_close(lowered["kwargs"]["labels"], labels, rtol=0, atol=0)


def test_clean_port_payload_does_not_invent_deepep_chunking():
    payload = dss_driver.build_payload({"ep_size": 4, "mb_spec": {"max_tokens_per_mb": 10240}}, "model", 42)
    assert "deepep_token_chunk_size" not in payload["training_config"]
    configured = dss_driver.build_payload({"ep_size": 4, "deepep_token_chunk_size": 4096}, "model", 42)
    assert configured["training_config"]["deepep_token_chunk_size"] == 4096


def test_clean_port_normal_runner_has_no_execution_overlay_or_config_specific_loss_selector():
    assert "optimizer_capture_worker" not in vars(runner)
    assert not hasattr(dss_driver, "ap_batch_processing_options")
    assert not hasattr(dss_driver, "pack_microbatches")
    assert "gradient_accumulation_steps" not in inspect.signature(dss_driver.build_payload).parameters
    assert "processing_config" not in inspect.signature(dss_driver.pack).parameters


def test_clean_port_core_fixture_binding_has_no_upload_account_or_subprocess_dependency():
    assert not hasattr(cortex_fixture, "require_upload_credentials")
    assert not hasattr(cortex_fixture, "DEV_STAGE")
    assert not hasattr(cortex_fixture, "subprocess")


@pytest.mark.parametrize("torch_adam", [False, True])
def test_clean_port_golden_trajectory_cannot_select_product_optimizer(torch_adam):
    assert "optimizer_backend" not in inspect.signature(AdamWTrajectory).parameters
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16, device="cpu")
    config = {"name": "AdamW", "torch_adam": torch_adam, "betas": [0.8, 0.95], "eps": 1e-06, "weight_decay": 0.125}
    trajectory = AdamWTrajectory(model, config, learning_rate=0.003, gradient_clipping=None, optimizer_dtype="float32")
    trajectory.step({"weight": torch.tensor([[0.25, -0.5]], dtype=torch.float32, device="cpu")})
    optimizer = trajectory._optimizer
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.defaults["fused"] is False
    assert optimizer.defaults["foreach"] is False
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == 0.003
    assert group["betas"] == (0.8, 0.95)
    assert group["eps"] == 1e-06
    assert group["weight_decay"] == 0.125
    assert all((parameter.device.type == "cpu" and parameter.dtype == torch.float32 for parameter in group["params"]))


@pytest.mark.parametrize("fp32_lm_head,accum_dtype", [(False, None), (True, torch.float32)])
def test_clean_port_golden_fused_cross_entropy_uses_configured_accumulation(monkeypatch, fp32_lm_head, accum_dtype):
    loss = object()
    constructor = Mock(return_value=loss)
    monkeypatch.setitem(sys.modules, "liger_kernel", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "liger_kernel.transformers.fused_linear_cross_entropy",
        SimpleNamespace(LigerFusedLinearCrossEntropyLoss=constructor),
    )
    assert hf_single_gpu._build_liger_fused_cross_entropy(fp32_lm_head=fp32_lm_head) is loss
    constructor.assert_called_once_with(ignore_index=-100, reduction="mean", accum_dtype=accum_dtype)


def test_clean_port_sampling_does_not_select_gdn_backend_to_match_training():
    payload = dss_driver.sampling_payload("model", 42, dtype="bfloat16", n_gpus=4, max_seq_len=128)
    assert "gdn_prefill_backend" not in payload["inference_config"]["vllm_config"]


# Cli config selection.


def _cli_config_selection_write_config(root: Path, gpu_type: str) -> Path:
    path = root / "model" / gpu_type / "train.config"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_config_factory_native_config({"n_gpus": 8, "sp_size": 8})))
    return path


def _cli_config_selection_args(config_dir: Path, *, any_gpu: bool = False, config: str | None = None) -> Namespace:
    return Namespace(all_configs=config is None, any_gpu=any_gpu, config_dir=str(config_dir), config=config)


@pytest.mark.parametrize("auth_mode", [None, "disabled", "token"])
def test_cli_config_selection_cli_entry_preserves_ray_auth_mode(monkeypatch, auth_mode) -> None:
    if auth_mode is None:
        monkeypatch.delenv("RAY_AUTH_MODE", raising=False)
    else:
        monkeypatch.setenv("RAY_AUTH_MODE", auth_mode)
    monkeypatch.setattr(cli.sys, "argv", ["correctness", "--help"])

    def forbidden_restart(*args):
        pytest.fail("the correctness CLI must not reexec to change Ray authentication")

    monkeypatch.setattr(os, "execv", forbidden_restart)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("arctic_platform.correctness", run_name="__main__")
    assert exited.value.code == 0
    assert os.environ.get("RAY_AUTH_MODE") == auth_mode


def test_cli_config_selection_all_configs_selects_only_the_local_gpu_family(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "configs"
    h200 = _cli_config_selection_write_config(config_dir, "h200")
    _cli_config_selection_write_config(config_dir, "b200")
    monkeypatch.setattr(cli, "local_gpu_type", lambda: "h200")
    assert cli._resolve_configs(_cli_config_selection_args(config_dir)) == [h200]


def test_cli_config_selection_all_configs_any_gpu_keeps_every_gpu_family(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "configs"
    h200 = _cli_config_selection_write_config(config_dir, "h200")
    b200 = _cli_config_selection_write_config(config_dir, "b200")
    monkeypatch.setattr(cli, "local_gpu_type", lambda: "h200")
    assert cli._resolve_configs(_cli_config_selection_args(config_dir, any_gpu=True)) == sorted([h200, b200])


def test_cli_config_selection_explicit_config_is_not_filtered_before_validation(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "configs"
    b200 = _cli_config_selection_write_config(config_dir, "b200")
    monkeypatch.setattr(cli, "local_gpu_type", lambda: "h200")
    assert cli._resolve_configs(_cli_config_selection_args(config_dir, config=str(b200))) == [b200]


# Console.


class _console_TTYBuffer(io.StringIO):

    def isatty(self) -> bool:
        return True


def test_console_activity_rotates_and_erases_the_temporary_line() -> None:
    stream = _console_TTYBuffer()
    with activity("Running reference gas1", stream=stream, interval=0.001):
        deadline = time.monotonic() + 0.2
        while "\r  Running reference gas1 /" not in stream.getvalue() and time.monotonic() < deadline:
            time.sleep(0.001)
    output = stream.getvalue()
    assert "\r  Running reference gas1 |" in output
    assert "\r  Running reference gas1 /" in output
    assert output.endswith("\r")
    assert " " * len("  Running reference gas1 |") in output


def test_console_activity_prints_one_progress_line_when_output_is_not_interactive() -> None:
    stream = io.StringIO()
    with activity("Running Arctic Platform gas1", stream=stream):
        pass
    assert stream.getvalue() == "  Running Arctic Platform gas1 ...\n"


def test_console_passing_gate_reports_passed_tensors() -> None:
    result = Result(
        test_id="single-step-grads",
        config_id="example",
        outcome=Outcome.PASS,
        metrics={"tensors_compared": 47.0, "tensors_over_stated_criterion": 0.0, "stated_criterion_abs": 0.013},
    )
    rendered = arm_verdict(result, None)
    assert "PASSED 1.3e-02 tolerance" in rendered
    assert "47/47" in rendered
    assert "FAILED 1.3e-02 tolerance" not in rendered


def test_console_failing_gate_reports_failed_tensors() -> None:
    result = Result(
        test_id="single-step-grads",
        config_id="example",
        outcome=Outcome.FAIL,
        summary="two mismatches",
        metrics={"tensors_compared": 47.0, "tensors_over_stated_criterion": 2.0, "stated_criterion_abs": 0.013},
    )
    rendered = arm_verdict(result, None)
    assert "FAILED 1.3e-02 tolerance" in rendered
    assert "2/47" in rendered


def test_console_optimizer_gate_names_residual_delta_and_reports_passed_values() -> None:
    result = Result(
        test_id="single-step-optimizer",
        config_id="example",
        outcome=Outcome.PASS,
        worst_name="layer.weight::exp_avg",
        metrics={
            "optimizer_values_compared": 141.0,
            "optimizer_values_over_criterion": 0.0,
            "max_optimizer_delta_norm": 0.004,
            "stated_criterion_abs": 0.012,
        },
    )
    rendered = arm_verdict(result, None)
    assert "optimizer moment and update-residual delta L2 norms" in rendered
    assert "PASSED 1.2e-02 tolerance" in rendered
    assert "141/141" in rendered
    assert "layer.weight::exp_avg" in rendered


# Diagnostics dispatch.


def test_diagnostics_dispatch_catalog_covers_existing_commands_but_not_helper_modules():
    directory = Path(dispatch.__file__).parent
    commands = set()
    for path in directory.glob("*.py"):
        if path.name == "__main__.py":
            continue
        tree = ast.parse(path.read_text())
        if any(
            (
                isinstance(node, ast.If)
                and ast.dump(node.test) == ast.dump(ast.parse('__name__ == "__main__"', mode="eval").body)
                for node in tree.body
            )
        ):
            commands.add(path.stem)
    assert set(dispatch.DIAGNOSTICS) == commands
    assert commands == {"lm_head_chunking"}


def test_diagnostics_dispatch_list_is_sorted_and_does_not_execute_modules(monkeypatch, capsys):

    def forbidden(*args, **kwargs):
        raise AssertionError("listing must not load an experiment")

    monkeypatch.setattr(dispatch.runpy, "run_module", forbidden)
    assert dispatch.main(["list"]) == 0
    names = [line.split("\t")[0] for line in capsys.readouterr().out.splitlines()]
    assert names == sorted(dispatch.DIAGNOSTICS)


def test_diagnostics_dispatch_actual_list_entry_needs_only_standard_library():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "arctic_platform.correctness.diagnostics", "list"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert [line.split("\t")[0] for line in result.stdout.splitlines()] == sorted(dispatch.DIAGNOSTICS)
    assert result.stderr == ""


@pytest.mark.parametrize("arguments", [[], ["--help"], ["config.config", "--local_rank=0", "--flag", "two words"]])
def test_diagnostics_dispatch_run_preserves_selected_module_and_argument_order(monkeypatch, arguments):
    original_argv = sys.argv
    observed = []

    def run_module(module, **kwargs):
        observed.append((module, kwargs, list(sys.argv)))

    monkeypatch.setattr(dispatch.runpy, "run_module", run_module)
    assert dispatch.main(["run", "lm_head_chunking", *arguments]) == 0
    assert observed == [
        (
            "arctic_platform.correctness.diagnostics.lm_head_chunking",
            {"run_name": "__main__", "alter_sys": True},
            ["arctic_platform.correctness.diagnostics.lm_head_chunking", *arguments],
        )
    ]
    assert sys.argv is original_argv


@pytest.mark.parametrize(
    "arguments", [[], ["run"], ["run", "../ladder"], ["run", "execution_controls"], ["run", "ladder"], ["list", "x"]]
)
def test_diagnostics_dispatch_invalid_selection_fails_before_module_execution(monkeypatch, arguments):

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid selection must not execute")

    monkeypatch.setattr(dispatch.runpy, "run_module", forbidden)
    with pytest.raises(SystemExit) as caught:
        dispatch.main(arguments)
    assert caught.value.code == 2


@pytest.mark.parametrize("code", [None, 0, 7, "diagnostic failure"])
def test_diagnostics_dispatch_module_exit_is_preserved_and_argv_restored(monkeypatch, code):
    original_argv = sys.argv

    def run_module(*args, **kwargs):
        raise SystemExit(code)

    monkeypatch.setattr(dispatch.runpy, "run_module", run_module)
    if isinstance(code, str):
        with pytest.raises(SystemExit) as caught:
            dispatch.main(["run", "lm_head_chunking"])
        assert caught.value.code == code
    else:
        assert dispatch.main(["run", "lm_head_chunking"]) == (0 if code is None else code)
    assert sys.argv is original_argv


@pytest.mark.parametrize("error", [RuntimeError("original failure"), KeyboardInterrupt()])
def test_diagnostics_dispatch_original_error_and_interruption_propagate(monkeypatch, error):
    original_argv = sys.argv

    def run_module(*args, **kwargs):
        raise error

    monkeypatch.setattr(dispatch.runpy, "run_module", run_module)
    with pytest.raises(type(error)) as caught:
        dispatch.main(["run", "lm_head_chunking"])
    assert caught.value is error
    assert sys.argv is original_argv


def test_diagnostics_dispatch_real_module_entry_runs_main_with_native_argv(monkeypatch, tmp_path, capsys):
    from arctic_platform.correctness import diagnostics

    source = tmp_path / "dispatch_fixture.py"
    source.write_text(
        'import json\nimport sys\nif __name__ == "__main__":\n    print(json.dumps([__name__, sys.argv,'
        " sys.modules['__main__'].__file__]))\n    raise SystemExit(7)\n"
    )
    monkeypatch.setattr(diagnostics, "__path__", [*diagnostics.__path__, str(tmp_path)])
    monkeypatch.setitem(dispatch.DIAGNOSTICS, "dispatch_fixture", "CPU-only fixture")
    original_argv = sys.argv
    original_main = sys.modules["__main__"]
    assert dispatch.main(["run", "dispatch_fixture", "--help", "two words"]) == 7
    assert json.loads(capsys.readouterr().out) == ["__main__", [str(source), "--help", "two words"], str(source)]
    assert sys.argv is original_argv
    assert sys.modules["__main__"] is original_main


def _diagnostics_dispatch_chunking_config():
    from arctic_platform.correctness.harness.config import LoadedConfig

    training = {
        "n_gpus": 4,
        "fused_lm_head_token_chunk_size": 256,
        "prime_rl": {"fused_lm_head_token_chunk_size": 512, "unchanged": 1},
        "unchanged": [1, 2],
    }
    sub_job = {"job_type": "training", "model_name": "test", "training_config": training}
    second = {"training_config": {"fused_lm_head_token_chunk_size": 128, "unchanged": True}}
    return LoadedConfig("test", Path("test.config"), sub_job, [sub_job, second])


def test_diagnostics_dispatch_chunking_removes_only_knob_from_copied_training_configs():
    from arctic_platform.correctness.diagnostics.lm_head_chunking import KNOB
    from arctic_platform.correctness.diagnostics.lm_head_chunking import without_chunked_head

    cfg = _diagnostics_dispatch_chunking_config()
    before = copy.deepcopy(cfg)
    stripped = without_chunked_head(cfg)
    assert cfg == before
    assert stripped is not cfg
    for sub in [stripped.sub_job, *stripped.sub_jobs]:
        assert KNOB not in sub["training_config"]
        assert KNOB not in sub["training_config"].get("prime_rl", {})
    expected = copy.deepcopy(before)
    for sub in [expected.sub_job, *expected.sub_jobs]:
        sub["training_config"].pop(KNOB, None)
        sub["training_config"].get("prime_rl", {}).pop(KNOB, None)
    assert stripped == expected


@pytest.mark.parametrize("arguments", [None, [], ["--config", "test.config", "--test", "single-step-grads"]])
def test_diagnostics_dispatch_chunking_uses_injected_api_executor_without_global_mutation(monkeypatch, arguments):
    from arctic_platform.correctness import harness
    from arctic_platform.correctness.diagnostics import lm_head_chunking as diagnostic

    cfg = _diagnostics_dispatch_chunking_config()
    before = copy.deepcopy(cfg)
    request, options = (object(), object())
    parse_calls, execute_calls = ([], [])
    result = object()

    def parse_run_args(argv):
        parse_calls.append(list(argv))
        return (request, options)

    def execute(config, *args, **kwargs):
        execute_calls.append((config, args, kwargs))
        return result

    def run(actual_request, actual_options, *, executor):
        assert actual_request is request and actual_options is options
        assert executor(cfg, "spec", workdir="work") is result
        assert runner.execute is execute
        return SimpleNamespace(exit_code=7)

    runner = ModuleType("arctic_platform.correctness.harness.runner")
    runner.execute = execute
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    monkeypatch.setattr(harness, "runner", runner, raising=False)
    monkeypatch.setattr(diagnostic.api, "parse_run_args", parse_run_args)
    monkeypatch.setattr(diagnostic.api, "run", run)
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--test", "single-step-grads"])
    assert diagnostic.main(arguments) == 7
    assert diagnostic.main(arguments) == 7
    expected_arguments = ["--test", "single-step-grads"] if arguments is None else arguments
    assert parse_calls == [expected_arguments, expected_arguments]
    assert len(execute_calls) == 2
    assert all((config is not cfg for config, _, _ in execute_calls))
    assert all((args == ("spec",) and kwargs == {"workdir": "work"} for _, args, kwargs in execute_calls))
    assert cfg == before
    assert runner.execute is execute


def test_diagnostics_dispatch_chunking_parse_failure_does_not_load_executor(monkeypatch):
    from arctic_platform.correctness.diagnostics import lm_head_chunking as diagnostic

    error = ValueError("original argument refusal")
    empty_runner = ModuleType("arctic_platform.correctness.harness.runner")
    monkeypatch.setitem(sys.modules, empty_runner.__name__, empty_runner)

    def refuse(argv):
        raise error

    monkeypatch.setattr(diagnostic.api, "parse_run_args", refuse)
    with pytest.raises(ValueError) as caught:
        diagnostic.main(["--unknown"])
    assert caught.value is error


def test_diagnostics_dispatch_chunking_executor_failure_keeps_original_config_and_binding(monkeypatch):
    from arctic_platform.correctness import harness
    from arctic_platform.correctness.diagnostics import lm_head_chunking as diagnostic

    cfg = _diagnostics_dispatch_chunking_config()
    before = copy.deepcopy(cfg)
    error = RuntimeError("original executor failure")

    def execute(config, *args, **kwargs):
        assert config is not cfg
        raise error

    def run(request, options, *, executor):
        return executor(cfg)

    runner = ModuleType("arctic_platform.correctness.harness.runner")
    runner.execute = execute
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    monkeypatch.setattr(harness, "runner", runner, raising=False)
    monkeypatch.setattr(diagnostic.api, "parse_run_args", lambda argv: (object(), object()))
    monkeypatch.setattr(diagnostic.api, "run", run)
    with pytest.raises(RuntimeError) as caught:
        diagnostic.main([])
    assert caught.value is error
    assert cfg == before
    assert runner.execute is execute


# Gateway lifecycle.
def test_gateway_lifecycle_available_gpus_reads_a_hostfile_without_a_prestarted_gateway(tmp_path, monkeypatch) -> None:
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("node-a slots=8\nnode-b slots=8\n")
    monkeypatch.delenv("DSS_GATEWAY_URL", raising=False)
    monkeypatch.setenv("DSS_GATEWAY_HOSTFILE", str(hostfile))
    assert dss_driver.available_gpus() == 16


# Operator api.


_operator_api_CONFIG = "qwen3-8b-h200-train-sft-full-4gpus-64k"

_operator_api_TEST = "inference-checkpoint-loss"


def test_operator_api_pair_selection_accepts_catalog_id_or_path():
    expected = api.select(api.RunRequest(config=_operator_api_CONFIG, tests=(_operator_api_TEST,)))
    assert expected == api.select(api.RunRequest(config=expected[0].config_path, tests=(_operator_api_TEST,)))
    assert [(case.config_id, case.test_id) for case in expected] == [(_operator_api_CONFIG, _operator_api_TEST)]


def test_operator_api_config_only_selects_each_maintained_test_once():
    selected = api.select(api.RunRequest(config=_operator_api_CONFIG))
    assert [case.test_id for case in selected] == sorted(api.registered_tests())
    assert len(set(selected)) == len(selected)


def test_operator_api_test_only_selects_all_reviewed_configs_in_order():
    selected = api.select(api.RunRequest(tests=(_operator_api_TEST, _operator_api_TEST)))
    assert [case.config_id for case in selected] == sorted(api.catalog())
    assert all((case.test_id == _operator_api_TEST for case in selected))
    assert len(set(selected)) == len(selected)


@pytest.mark.parametrize(
    "selection,match",
    [
        (api.RunRequest(), "give --config"),
        (api.RunRequest(config="missing"), "unknown configuration"),
        (api.RunRequest(config=_operator_api_CONFIG, tests=("missing",)), "unknown check"),
        (api.RunRequest(config=_operator_api_CONFIG, all_configs=True), "not both"),
    ],
)
def test_operator_api_bad_selectors_fail_before_backend(selection, match, monkeypatch):
    monkeypatch.setattr(execution, "_client_backend", lambda *a: pytest.fail("backend accessed"))
    with pytest.raises(ValueError, match=match):
        api.run(selection)


def test_operator_api_empty_catalog_is_not_a_success(tmp_path):
    with pytest.raises(ValueError, match="no active"):
        api.select(api.RunRequest(tests=(_operator_api_TEST,), config_dir=tmp_path))


def test_operator_api_archive_configs_are_not_discovered(tmp_path, monkeypatch):
    root = tmp_path / "configs"
    archived = root / "archive" / "model" / "h200" / "train.config"
    archived.parent.mkdir(parents=True)
    archived.write_text("{}")
    monkeypatch.setattr(api, "PACKAGE_DIR", tmp_path)
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "archive-model-h200-train.json").write_text("{}")
    assert api.catalog(root) == {}


def _operator_api_fake_batches(monkeypatch, verdicts):
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


def test_operator_api_mixed_batch_continues_and_retains_every_pair(tmp_path, monkeypatch, capsys):
    tests = ("checkpoint-resume-loss", "e2e-sft-train-validate", _operator_api_TEST)
    _operator_api_fake_batches(
        monkeypatch, [Outcome.FAIL, ReferencePrerequisiteError("prepared reference missing"), Outcome.PASS]
    )
    result = api.run(api.RunRequest(config=_operator_api_CONFIG, tests=tests), api.RunOptions(out=tmp_path))
    assert [case["status"] for case in result.cases] == ["fail", "blocked", "pass"]
    assert result.exit_code == 1 and (not result.stopped_unsafe)
    assert len(result.results) == len(tests)
    assert "ReferencePrerequisiteError" in (tmp_path / "full.log").read_text()
    assert "Traceback" not in capsys.readouterr().out
    assert (tmp_path / "batch.json").is_file()


def test_operator_api_actual_execution_preserves_selection_order_and_combined_step_identity(tmp_path, monkeypatch):
    selected = api.select(api.RunRequest(config=_operator_api_CONFIG))
    observed = []
    monkeypatch.setattr(execution, "write_report", lambda *a, **k: None)

    def record(args, *, executor, snapshot):
        observed.append(tuple(args.test))
        snapshot([Result(check, _operator_api_CONFIG, Outcome.PASS) for check in args.test], {}, {})

    monkeypatch.setattr(execution, "_run_configs", record)
    result = api.run(api.RunRequest(config=_operator_api_CONFIG), api.RunOptions(out=tmp_path))
    assert [check for group in observed for check in group] == [case.test_id for case in selected]
    assert ("single-step-grads", "single-step-optimizer") in observed
    assert [case["test_id"] for case in result.cases] == [case.test_id for case in selected]
    assert result.exit_code == 0


@pytest.mark.parametrize("module", ["rl_weight_sync", "rl_router_replay"])
def test_operator_api_rl_wrapper_traceback_is_saved_in_batch_log(module, tmp_path, monkeypatch, capsys):
    from importlib import import_module

    check = import_module(f"arctic_platform.correctness.checks.{module}")

    def fail(ctx):
        raise RuntimeError("synthetic RL failure")

    def execute(args, *, executor, snapshot):
        snapshot(check.run(SimpleNamespace(config_id=_operator_api_CONFIG)), {}, {})

    monkeypatch.setattr(check, "_run", fail)
    monkeypatch.setattr(execution, "_run_configs", execute)
    monkeypatch.setattr(execution, "write_report", lambda *a, **k: None)
    result = api.run(api.RunRequest(config=_operator_api_CONFIG, tests=(check.TEST_ID,)), api.RunOptions(out=tmp_path))
    assert result.cases[0]["status"] == "error" and result.exit_code == 1
    assert "Traceback" in (tmp_path / "full.log").read_text()
    assert "RuntimeError: synthetic RL failure" in (tmp_path / "full.log").read_text()
    assert "Traceback" not in capsys.readouterr().err


def test_operator_api_inapplicable_is_visible_and_not_a_pass(tmp_path, monkeypatch):
    _operator_api_fake_batches(monkeypatch, [Outcome.INAPPLICABLE])
    result = api.run(
        api.RunRequest(config=_operator_api_CONFIG, tests=(_operator_api_TEST,)), api.RunOptions(out=tmp_path)
    )
    assert result.cases[0]["status"] == "inapplicable"
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("unexpected failure"), UnsafeCleanupError("cleanup did not complete"), KeyboardInterrupt()],
)
def test_operator_api_unsafe_exit_or_interruption_stops_remaining_cases(tmp_path, monkeypatch, failure):
    _operator_api_fake_batches(monkeypatch, [failure])
    result = api.run(
        api.RunRequest(config=_operator_api_CONFIG, tests=("e2e-sft-train-validate", _operator_api_TEST)),
        api.RunOptions(out=tmp_path),
    )
    assert result.stopped_unsafe and result.exit_code == 1
    assert [case["status"] for case in result.cases] == ["error", "blocked"]
    assert all((result.outcome is Outcome.FAIL for result in result.results))


def test_operator_api_empty_execution_is_explicitly_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "write_report", lambda *a, **k: None)
    monkeypatch.setattr(execution, "_run_configs", lambda *a, **k: 0)
    result = api.run(
        api.RunRequest(config=_operator_api_CONFIG, tests=(_operator_api_TEST,)), api.RunOptions(out=tmp_path)
    )
    assert result.cases[0]["status"] == "blocked" and result.exit_code == 1


@pytest.mark.parametrize(
    "error_type,status", [(None, "fail"), ("RuntimeError", "error"), ("ReferencePrerequisiteError", "blocked")]
)
def test_operator_api_comparison_failure_and_execution_errors_are_distinct(error_type, status):
    case = api.SelectedCase(_operator_api_CONFIG, api.catalog()[_operator_api_CONFIG], _operator_api_TEST)
    result = Result(
        _operator_api_TEST,
        _operator_api_CONFIG,
        Outcome.FAIL,
        summary="comparison did not pass",
        error_type=error_type,
    )
    assert execution._case_summary(case, [result])["status"] == status


def test_operator_api_parse_flags_is_shared_with_operator_command():
    request, options = api.parse_run_args(
        ["--config", _operator_api_CONFIG, "--test", _operator_api_TEST, "--backend", "onprem"]
    )
    assert request.config == _operator_api_CONFIG and request.tests == (_operator_api_TEST,)
    assert options.backend == "onprem"


def test_operator_api_static_guide_is_unchanged_by_run_and_onboarding(tmp_path, monkeypatch):
    guide = api.PACKAGE_DIR / "docs" / "running-tests.doc"
    before = guide.read_bytes()
    monkeypatch.setattr(execution, "_run_vocab_size", lambda *a, **k: 128)
    from transformers import AutoConfig

    from arctic_platform.correctness.harness import report
    from arctic_platform.correctness.harness import runner

    monkeypatch.setattr(report, "collect", lambda: {"host": "synthetic"})
    result = api.run(
        api.RunRequest(config=_operator_api_CONFIG, tests=(_operator_api_TEST,)),
        api.RunOptions(out=tmp_path, base_url="http://existing.test"),
        executor=lambda cfg, *a, **k: [Result(_operator_api_TEST, cfg.config_id, Outcome.PASS)],
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
                str(api.catalog()[_operator_api_CONFIG]),
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


def test_operator_api_logger_traceback_is_saved_and_handlers_are_restored(tmp_path, monkeypatch, capsys):
    root = logging.getLogger()
    previous = root.handlers[:]
    monkeypatch.setattr(execution, "write_report", lambda *a, **k: None)

    def run_with_logged_failure(args, *, executor, snapshot):
        try:
            raise ReferencePrerequisiteError("missing prepared identity")
        except ReferencePrerequisiteError:
            logging.getLogger(__name__).exception("reference unavailable")
        snapshot(
            [
                Result(
                    _operator_api_TEST,
                    _operator_api_CONFIG,
                    Outcome.FAIL,
                    summary="ReferencePrerequisiteError: missing identity",
                )
            ],
            {},
            {},
        )

    monkeypatch.setattr(execution, "_run_configs", run_with_logged_failure)
    result = api.run(
        api.RunRequest(config=_operator_api_CONFIG, tests=(_operator_api_TEST,)), api.RunOptions(out=tmp_path)
    )
    assert result.cases[0]["status"] == "blocked"
    assert "Traceback" in (tmp_path / "full.log").read_text()
    assert "Traceback" not in capsys.readouterr().err
    assert root.handlers == previous


@pytest.mark.parametrize(
    "module",
    ["e2e_sft_train_validate", "inference_checkpoint_loss", "rl_weight_sync", "rl_router_replay", "checkpoint_resume"],
)
def test_operator_api_check_wrappers_propagate_unsafe_cleanup(module, tmp_path, monkeypatch):
    import contextlib
    from importlib import import_module

    check = import_module(f"arctic_platform.correctness.checks.{module}")
    failure = UnsafeCleanupError("owned cleanup failed")

    def fail(*args):
        raise failure

    context = SimpleNamespace(
        workdir=tmp_path, arms=[SimpleNamespace(name="gas1")], transport_for=lambda *args: contextlib.nullcontext()
    )
    monkeypatch.setattr(check, "_run_arm" if module == "checkpoint_resume" else "_run", fail)
    with pytest.raises(UnsafeCleanupError) as caught:
        check.run(context)
    assert caught.value is failure


# Preflight.


def _preflight_run_node_script(root) -> str:
    return subprocess.run(
        ["bash", "-c", preflight.node_script(str(root))], capture_output=True, text=True, check=True
    ).stdout


def test_preflight_a_lock_without_a_shared_object_is_reported(tmp_path) -> None:
    build = tmp_path / "py312_cu130" / "fused_adam"
    build.mkdir(parents=True)
    (build / "lock").touch()
    (build / "fused_adam_frontend.o").touch()
    found = preflight.parse(_preflight_run_node_script(tmp_path))
    assert [directory for _, directory in found] == [str(build)]


def test_preflight_a_linked_build_and_a_clean_build_are_not_reported(tmp_path) -> None:
    linked = tmp_path / "py312_cu130" / "fused_adam"
    linked.mkdir(parents=True)
    (linked / "lock").touch()
    (linked / "fused_adam.so").touch()
    (tmp_path / "py312_cu130" / "cpu_adam").mkdir()
    assert preflight.parse(_preflight_run_node_script(tmp_path)) == []


def test_preflight_parse_reads_hosts_behind_the_pdsh_prefix() -> None:
    output = (
        f"hostfile=/data-fast/hostfile\n10.0.0.2: {preflight.MARKER} node-1"
        " /home/u/.cache/torch_extensions/py312_cu130/fused_adam\n"
    )
    assert preflight.parse(output) == [("node-1", "/home/u/.cache/torch_extensions/py312_cu130/fused_adam")]


def _preflight_hostfile(tmp_path, lines: int):
    path = tmp_path / "hostfile"
    path.write_text("".join((f"10.0.0.{i} slots=8\n" for i in range(1, lines + 1))))
    return str(path)


def test_preflight_the_run_stops_with_every_interrupted_directory_named(monkeypatch, tmp_path) -> None:
    out = (
        f"10.0.0.1: {preflight.MARKER} node-0 /c/py312_cu130/fused_adam\n10.0.0.2: {preflight.MARKER} node-1"
        " /c/py312_cu130/cpu_adam\n"
    )
    monkeypatch.setattr(
        preflight.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr="")
    )
    with pytest.raises(SystemExit) as stopped:
        preflight.ensure_no_interrupted_builds(_preflight_hostfile(tmp_path, 2))
    assert "node-0: /c/py312_cu130/fused_adam" in str(stopped.value)
    assert "node-1: /c/py312_cu130/cpu_adam" in str(stopped.value)


def test_preflight_a_single_node_allocation_is_not_inspected(monkeypatch, tmp_path) -> None:

    def refuse(*args, **kwargs):
        raise AssertionError("a one-node allocation must not be inspected over ssh")

    monkeypatch.setattr(preflight.subprocess, "run", refuse)
    preflight.ensure_no_interrupted_builds(_preflight_hostfile(tmp_path, 1))


def test_preflight_without_a_named_gateway_nothing_is_inspected(monkeypatch, tmp_path) -> None:

    def refuse(*args, **kwargs):
        raise AssertionError("a gateway the harness starts itself serves this node alone")

    monkeypatch.delenv("DSS_GATEWAY_URL", raising=False)
    monkeypatch.setenv("DSS_GATEWAY_HOSTFILE", _preflight_hostfile(tmp_path, 2))
    monkeypatch.setattr(preflight.subprocess, "run", refuse)
    preflight.ensure_no_interrupted_builds()


def test_preflight_an_inspection_that_could_not_run_stops_the_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="ssh: timeout"),
    )
    with pytest.raises(SystemExit, match="ssh: timeout"):
        preflight.ensure_no_interrupted_builds(_preflight_hostfile(tmp_path, 2))


# Report.


def test_report_reports_are_timestamped_and_latest_links_move_without_deleting_history(tmp_path: Path) -> None:
    write([], {}, tmp_path, timestamp="2026-09-24-15-50")
    write([], {}, tmp_path, timestamp="2026-09-24-15-51")
    assert (tmp_path / "report-2026-09-24-15-50.md").is_file()
    assert (tmp_path / "report-2026-09-24-15-50.json").is_file()
    assert (tmp_path / "report-2026-09-24-15-51.md").is_file()
    assert (tmp_path / "report-2026-09-24-15-51.json").is_file()
    assert (tmp_path / "report.md").is_symlink()
    assert (tmp_path / "report.json").is_symlink()
    assert (tmp_path / "report.md").readlink() == Path("report-2026-09-24-15-51.md")
    assert (tmp_path / "report.json").readlink() == Path("report-2026-09-24-15-51.json")


@pytest.mark.parametrize(
    "selected",
    [
        ["inference-checkpoint-loss"],
        ["inference-checkpoint-loss", "e2e-sft-train-validate"],
        ["e2e-sft-train-validate"],
    ],
)
def test_report_cortex_fixture_report_distinguishes_full_inference_model(tmp_path, monkeypatch, selected):
    monkeypatch.setattr(report, "collect", lambda: {"host": "client-host"})
    spec = _applicable_tests_spec(selected)
    metadata = {
        "backend": "cortex",
        "requested_model_name": "published-reduced-model",
        "fixture_sha256": "fixture-hash",
        "fixture_stage_path": "snow://fixture-stage/exact",
    }
    inference_selected = "inference-checkpoint-loss" in selected
    if inference_selected:
        metadata["inference_model_name"] = "catalog-full-model"
    execution = {"unit": metadata}
    results = [Result(test_id=test_id, config_id="unit", outcome=Outcome.PASS) for test_id in selected]
    write(results, {"unit": spec}, tmp_path, execution=execution)
    markdown = (tmp_path / "report.md").read_text()
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["execution"] == execution
    assert [result["test_id"] for result in payload["results"]] == selected
    assert "local model/tokenizer SHA-256 binding: `fixture-hash`" in markdown
    assert "publication stage: `snow://fixture-stage/exact`" in markdown
    if inference_selected:
        assert "`inference-checkpoint-loss` requested full model: `catalog-full-model`" in markdown
        assert "requested published reduced model (fixture-based checks only): `published-reduced-model`" in markdown
        assert "- requested published reduced model: `published-reduced-model`" not in markdown
        assert "reviewed reference fixture (fixture-based checks only; not used by inference)" in markdown
        assert "reviewed reference fixture (not loaded on Cortex)" not in markdown
    else:
        assert "requested published reduced model: `published-reduced-model`" in markdown
        assert "requested full model" not in markdown


def test_report_local_cortex_fixture_report_does_not_require_publication_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "collect", lambda: {"host": "client-host"})
    spec = _applicable_tests_spec(["single-step-grads"])
    execution = {
        "unit": {
            "backend": "cortex",
            "requested_model_name": "/data-fast/base-models/synthetic/unit",
            "fixture_sha256": "fixture-hash",
            "fixture_distribution": "local-gateway",
            "gateway_url": "http://127.0.0.1:8003",
        }
    }
    results = [Result(test_id="single-step-grads", config_id="unit", outcome=Outcome.PASS)]
    write(results, {"unit": spec}, tmp_path, execution=execution)
    markdown = (tmp_path / "report.md").read_text()
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["execution"] == execution
    assert "requested local reduced model: `/data-fast/base-models/synthetic/unit`" in markdown
    assert "local gateway: `http://127.0.0.1:8003`" in markdown
    assert "local model/tokenizer SHA-256 binding: `fixture-hash`" in markdown
    assert "publication stage:" not in markdown
    assert "reviewed reference fixture (not loaded on Cortex)" not in markdown


@pytest.mark.parametrize(
    ("metadata", "expected_backend"),
    [
        ({"gpus": 16}, "unknown"),
        ({"gpus": 16, "backend": None}, "unknown"),
        ({"gpus": 16, "backend": ""}, "unknown"),
        ({"gpus": 16, "backend": "onprem"}, "onprem"),
    ],
    ids=["missing-backend", "null-backend", "empty-backend", "known-backend"],
)
def test_report_execution_error_report_preserves_original_failure_with_partial_metadata(
    tmp_path, monkeypatch, metadata, expected_backend
):
    monkeypatch.setattr(report, "collect", lambda: {"host": "client-host"})
    spec = _applicable_tests_spec(["inference-checkpoint-loss"])
    reason = "Traceback (most recent call last):\nRuntimeError: exported tensor inventory is incomplete"
    result = Result(
        test_id="inference-checkpoint-loss",
        config_id="unit",
        outcome=Outcome.FAIL,
        summary="Checkpoint export failed before comparison",
        reason=reason,
    )
    execution = {"unit": metadata}
    write([result], {"unit": spec}, tmp_path, execution=execution)
    markdown = (tmp_path / "report.md").read_text()
    payload = json.loads((tmp_path / "report.json").read_text())
    assert f"execution backend: `{expected_backend}`" in markdown
    assert reason in markdown
    assert payload["execution"] == execution
    assert payload["results"][0]["reason"] == reason
    assert payload["results"][0]["outcome"] == "fail"
    assert payload["totals"]["fail"] == 1
    assert payload["totals"]["pass"] == 0


# Selftest storage.
def test_selftest_storage_selftest_default_uses_unique_fast_scratch(monkeypatch):
    commands = []
    monkeypatch.setattr(cli, "correctness_workdir", lambda prefix: Path("/data-fast/tmp/unique-selftests"))
    monkeypatch.setattr(pytest, "main", lambda command: commands.append(command) or 0)
    assert cli.cmd_selftest(SimpleNamespace(runner_args=[])) == 0
    assert commands[0][-2:] == ["--basetemp", "/data-fast/tmp/unique-selftests/pytest"]
    assert commands[0][0] == "-sv"


@pytest.mark.parametrize(
    "arguments",
    [["--basetemp=/tmp/tests"], ["--basetemp", "/code/tests"], ["--basetemp", "/data-fast"], ["--basetemp"]],
)
def test_selftest_storage_selftest_rejects_off_fast_override_before_pytest(monkeypatch, arguments):

    def unexpected(*args, **kwargs):
        pytest.fail("unsafe storage must be rejected before allocation or pytest")

    monkeypatch.setattr(cli, "correctness_workdir", unexpected)
    monkeypatch.setattr(pytest, "main", unexpected)
    with pytest.raises(ValueError, match="/data-fast"):
        cli.cmd_selftest(SimpleNamespace(runner_args=arguments))


@pytest.mark.parametrize("joined", [False, True])
def test_selftest_storage_selftest_accepts_explicit_fast_override(tmp_path, monkeypatch, joined):
    commands = []
    arguments = [f"--basetemp={tmp_path / 'tests'}"] if joined else ["--basetemp", str(tmp_path / "tests")]
    monkeypatch.setattr(pytest, "main", lambda command: commands.append(command) or 0)
    assert cli.cmd_selftest(SimpleNamespace(runner_args=arguments)) == 0
    assert commands[0][-len(arguments) :] == arguments
    assert not (tmp_path / "tests").exists()


# Workdir root.


def _workdir_root_reset_tmp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CORRECTNESS_TMPDIR_ENV, raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.delenv("TEMP", raising=False)
    monkeypatch.delenv("TMP", raising=False)


def test_workdir_root_correctness_workdir_prefers_data_fast_over_unsafe_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_fast_tmp = tmp_path / "data-fast" / "tmp"
    data_fast_tmp.parent.mkdir()
    _workdir_root_reset_tmp_env(monkeypatch)
    monkeypatch.setattr(workdir, "_DATA_FAST_TMPDIR", data_fast_tmp)
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", data_fast_tmp.parent)
    monkeypatch.setenv("TMPDIR", "/code/users/stas/ap3tmp")
    created = workdir.correctness_workdir("dss-correctness-")
    assert created.parent == data_fast_tmp
    assert created.name.startswith("dss-correctness-")


def test_workdir_root_correctness_workdir_fails_closed_without_data_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_root = tmp_path / "missing-data-fast"
    _workdir_root_reset_tmp_env(monkeypatch)
    monkeypatch.setattr(workdir, "_DATA_FAST_TMPDIR", missing_root / "tmp")
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", missing_root)
    monkeypatch.setenv("TMPDIR", "/code/users/stas/ap3tmp")
    with pytest.raises(ValueError, match="existing /data-fast"):
        workdir.correctness_workdir("dss-correctness-")
    assert not missing_root.exists()


def test_workdir_root_correctness_workdir_honors_safe_explicit_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_fast = tmp_path / "data-fast"
    data_fast.mkdir()
    override = data_fast / "override"
    _workdir_root_reset_tmp_env(monkeypatch)
    monkeypatch.setattr(workdir, "_DATA_FAST_TMPDIR", data_fast / "tmp")
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", data_fast)
    monkeypatch.setenv("TMPDIR", "/code/users/stas/ap3tmp")
    monkeypatch.setenv(CORRECTNESS_TMPDIR_ENV, str(override))
    created = workdir.correctness_workdir("dss-correctness-")
    assert created.parent == override


def test_workdir_root_correctness_workdir_rejects_unsafe_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _workdir_root_reset_tmp_env(monkeypatch)
    monkeypatch.setenv(CORRECTNESS_TMPDIR_ENV, "/code/users/stas/ap3tmp")
    with pytest.raises(ValueError, match=CORRECTNESS_TMPDIR_ENV):
        workdir.correctness_workdir("dss-correctness-")
