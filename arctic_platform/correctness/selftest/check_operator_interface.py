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
import hashlib
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

import pytest

from arctic_platform.correctness import api
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

from .check_catalog_configuration import _applicable_tests_spec
from .helpers import _config_factory_native_config
from .helpers import _synthetic_config_spec


@pytest.fixture(autouse=True)
def _external_operator_catalog(tmp_path, monkeypatch):
    root = tmp_path / "external-inputs"
    config, _ = _synthetic_config_spec(root)
    for name in ("second", "third"):
        extra = root / "configs" / "synthetic" / "h200" / f"{name}.json"
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_bytes(config.read_bytes())
    original_catalog = api.catalog
    default_root = api.PACKAGE_DIR / "configs"

    def catalog(config_dir=None):
        selected = root / "configs" if config_dir is None or Path(config_dir) == default_root else config_dir
        return original_catalog(selected)

    monkeypatch.setattr(api, "catalog", catalog)
    monkeypatch.setattr(cli, "SPEC_DIR", root / "specs")


@pytest.fixture
def _historical_gradient_cli(tmp_path):
    path = tmp_path / "receipts" / "bindings.json"
    path.parent.mkdir()
    value = {
        "schema": "reviewed-gradient-references-v1",
        "records": {
            arm: {
                "record_path": f"records/{arm}.json" if arm == "gas1" else str(tmp_path / "absolute-record.json"),
                "record_sha256": hashlib.sha256(arm.encode()).hexdigest(),
                "review": f"Reviewed historical {arm} gradient names only",
            }
            for arm in ("gas1", "gas4")
        },
    }
    path.write_text(json.dumps(value))
    config = tmp_path / "external.config"
    config.write_text(
        json.dumps(_config_factory_native_config({"n_gpus": 4, "max_seq_len": 16, "attn_implementation": "sdpa"}))
    )
    args = execution.parse_args(
        [
            "run",
            "--config",
            str(config),
            "--test",
            "single-step-grads",
            "--backend",
            "cortex",
            "--base-url",
            "http://127.0.0.1:8003",
            "--out",
            str(tmp_path / "out"),
            "--historical-gradient-references",
            str(path),
            "--historical-gradient-references-sha256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
        ]
    )
    return args, value


def test_historical_gradient_cli_api_preserve_original_pinned_pair(_historical_gradient_cli, monkeypatch):
    args, _ = _historical_gradient_cli
    request, options = api.parse_run_args(
        [
            "--config",
            str(args.config),
            "--test",
            "single-step-grads",
            "--historical-gradient-references",
            str(args.historical_gradient_references),
            "--historical-gradient-references-sha256",
            args.historical_gradient_references_sha256,
        ]
    )
    assert options.historical_gradient_references == args.historical_gradient_references
    assert isinstance(options.historical_gradient_references, Path)
    assert options.historical_gradient_references_sha256 == args.historical_gradient_references_sha256
    namespace = api._namespace(request, options)
    assert namespace.historical_gradient_references == args.historical_gradient_references
    assert namespace.historical_gradient_references_sha256 == args.historical_gradient_references_sha256
    expected_selection = object()
    result = object()
    monkeypatch.setattr(
        api, "select", lambda actual: expected_selection if actual == request else pytest.fail("request changed")
    )

    def capture(actual, *, selected, executor):
        assert actual.historical_gradient_references == args.historical_gradient_references
        assert actual.historical_gradient_references_sha256 == args.historical_gradient_references_sha256
        assert selected is expected_selection and executor is None
        return result

    monkeypatch.setattr(execution, "run_batch", capture)
    assert api.run(request, options) is result


def test_historical_gradient_absent_pair_keeps_default():
    assert execution._load_historical_gradient_references(Namespace()) is None
    assert api.RunOptions().historical_gradient_references is None
    assert api.RunOptions().historical_gradient_references_sha256 is None
    assert api.RunOptions().backend == "onprem"


@pytest.mark.parametrize("missing", ["historical_gradient_references", "historical_gradient_references_sha256"])
def test_historical_gradient_pair_rejected_before_setup(_historical_gradient_cli, monkeypatch, missing):
    args, _ = _historical_gradient_cli
    setattr(args, missing, None)
    monkeypatch.setattr(execution, "_resolve_configs", lambda *a, **kw: pytest.fail("invalid pair reached configs"))
    monkeypatch.setattr(execution, "correctness_workdir", lambda *a: pytest.fail("invalid pair allocated workdir"))
    with pytest.raises(ValueError, match="required together"):
        execution._run_configs(args)


@pytest.mark.parametrize("digest", ["0" * 64, "0" * 63, "A" * 64, "g" * 64, 123])
def test_historical_gradient_pin_checked_before_json(_historical_gradient_cli, monkeypatch, digest):
    args, _ = _historical_gradient_cli
    args.historical_gradient_references.write_bytes(b"not JSON")
    args.historical_gradient_references_sha256 = digest
    monkeypatch.setattr(execution.json, "loads", lambda *a, **kw: pytest.fail("bad pin reached JSON parsing"))
    with pytest.raises(ValueError, match="SHA256"):
        execution._load_historical_gradient_references(args)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema":"first","schema":"second","records":{}}',
        b'{"schema":"reviewed-gradient-references-v1","records":{"gas1":{},"gas1":{}}}',
        b'{"schema":"reviewed-gradient-references-v1","records":{"gas1":{"review":"first","review":"second"}}}',
    ],
)
def test_historical_gradient_duplicate_keys_rejected(_historical_gradient_cli, raw):
    args, _ = _historical_gradient_cli
    args.historical_gradient_references.write_bytes(raw)
    args.historical_gradient_references_sha256 = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="duplicate"):
        execution._load_historical_gradient_references(args)


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_historical_gradient_nonfinite_metadata_rejected(_historical_gradient_cli, number):
    args, _ = _historical_gradient_cli
    raw = ('{"schema":"reviewed-gradient-references-v1","records":{"gas1":{"review":' + number + "}}}").encode()
    args.historical_gradient_references.write_bytes(raw)
    args.historical_gradient_references_sha256 = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        execution._load_historical_gradient_references(args)


@pytest.mark.parametrize(
    "corruption",
    [
        "root-list",
        "root-null",
        "schema",
        "extra-root",
        "missing-records",
        "empty-records",
        "records-list",
        "empty-arm",
        "binding-list",
        "extra-binding",
        "missing-review",
        "path-type",
        "empty-path",
        "record-pin",
        "review-type",
        "blank-review",
    ],
)
def test_historical_gradient_descriptor_schema_is_strict(_historical_gradient_cli, corruption):
    args, value = _historical_gradient_cli
    binding = value["records"]["gas1"]
    if corruption == "root-list":
        value = []
    elif corruption == "root-null":
        value = None
    elif corruption == "schema":
        value["schema"] = "unreviewed"
    elif corruption == "extra-root":
        value["ignored"] = True
    elif corruption == "missing-records":
        del value["records"]
    elif corruption == "empty-records":
        value["records"] = {}
    elif corruption == "records-list":
        value["records"] = []
    elif corruption == "empty-arm":
        value["records"][""] = value["records"].pop("gas1")
    elif corruption == "binding-list":
        value["records"]["gas1"] = []
    elif corruption == "extra-binding":
        binding["ignored"] = True
    elif corruption == "missing-review":
        del binding["review"]
    elif corruption == "path-type":
        binding["record_path"] = 1
    elif corruption == "empty-path":
        binding["record_path"] = ""
    elif corruption == "record-pin":
        binding["record_sha256"] = "A" * 64
    elif corruption == "review-type":
        binding["review"] = True
    else:
        binding["review"] = " \n\t "
    raw = json.dumps(value).encode()
    args.historical_gradient_references.write_bytes(raw)
    args.historical_gradient_references_sha256 = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="schema|binding"):
        execution._load_historical_gradient_references(args)


def test_historical_gradient_descriptor_size_bounded(_historical_gradient_cli):
    args, _ = _historical_gradient_cli
    raw = b" " * (1024 * 1024 + 1)
    args.historical_gradient_references.write_bytes(raw)
    args.historical_gradient_references_sha256 = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="exceeds 1 MiB"):
        execution._load_historical_gradient_references(args)


def test_historical_gradient_relative_paths_use_caller_and_descriptor_directory(
    _historical_gradient_cli, monkeypatch, tmp_path
):
    args, value = _historical_gradient_cli
    original = args.historical_gradient_references
    pin = args.historical_gradient_references_sha256
    args.historical_gradient_references = original.relative_to(tmp_path)
    monkeypatch.chdir(tmp_path)
    bindings = execution._load_historical_gradient_references(args)
    assert bindings["gas1"]["record_path"] == str(original.parent / "records" / "gas1.json")
    assert bindings["gas4"]["record_path"] == str(tmp_path / "absolute-record.json")
    for arm in value["records"]:
        assert bindings[arm]["record_sha256"] == value["records"][arm]["record_sha256"]
        assert bindings[arm]["review"] == value["records"][arm]["review"]
    assert hashlib.sha256(original.read_bytes()).hexdigest() == pin
    assert args.historical_gradient_references == original.relative_to(tmp_path)
    assert args.historical_gradient_references_sha256 == pin


@pytest.mark.parametrize(
    "field,value",
    [
        ("config", None),
        ("all_configs", True),
        ("test", None),
        ("test", []),
        ("test", ["single-step-optimizer"]),
        ("test", ["single-step-grads", "single-step-optimizer"]),
        ("test", ["single-step-grads", "single-step-grads"]),
        ("prepare_references", True),
        ("probe_layers", 0),
        ("probe_layers", 4),
    ],
)
def test_historical_gradient_runbatch_checks_original_selection_before_narrowing(
    _historical_gradient_cli, monkeypatch, tmp_path, field, value
):
    args, _ = _historical_gradient_cli
    selected = [api.SelectedCase("synthetic", Path(args.config), "single-step-grads")]
    setattr(args, field, value)
    args.out = None
    monkeypatch.setattr(execution, "_run_configs", lambda *a, **kw: pytest.fail("invalid selection narrowed"))
    monkeypatch.setattr(
        execution, "correctness_workdir", lambda *a: pytest.fail("invalid selection allocated workdir")
    )
    with pytest.raises(ValueError, match="one explicit config and only single-step-grads"):
        execution.run_batch(args, selected=selected)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("selection", ["all-configs", "all-tests", "optimizer", "mixed"])
def test_historical_gradient_api_guard_precedes_catalog(_historical_gradient_cli, monkeypatch, selection):
    args, _ = _historical_gradient_cli
    request = api.RunRequest(
        config=None if selection == "all-configs" else args.config,
        tests={
            "all-configs": ("single-step-grads",),
            "all-tests": (),
            "optimizer": ("single-step-optimizer",),
            "mixed": ("single-step-grads", "single-step-optimizer"),
        }[selection],
        all_configs=selection == "all-configs",
    )
    options = api.RunOptions(
        historical_gradient_references=args.historical_gradient_references,
        historical_gradient_references_sha256=args.historical_gradient_references_sha256,
    )
    monkeypatch.setattr(api, "select", lambda *a: pytest.fail("invalid selection reached catalog"))
    with pytest.raises(ValueError, match="one explicit config and only single-step-grads"):
        api.run(request, options)


def test_historical_gradient_runbatch_forwards_original_pair(_historical_gradient_cli, monkeypatch):
    args, _ = _historical_gradient_cli
    expected = execution._load_historical_gradient_references(args)
    captured = []
    selected = [api.SelectedCase("synthetic", Path(args.config), "single-step-grads")]
    monkeypatch.setattr(execution, "write_report", lambda *a, **kw: None)

    def capture(child, *, executor, snapshot):
        assert child.historical_gradient_references == args.historical_gradient_references
        assert child.historical_gradient_references_sha256 == args.historical_gradient_references_sha256
        assert child.test == ["single-step-grads"]
        captured.append(execution._load_historical_gradient_references(child))
        snapshot([Result("single-step-grads", "synthetic", Outcome.PASS)], {}, {})

    monkeypatch.setattr(execution, "_run_configs", capture)
    result = execution.run_batch(args, selected=selected)
    assert captured == [expected]
    assert result.exit_code == 0


def test_historical_gradient_execution_forwards_exact_bindings(_historical_gradient_cli, monkeypatch):
    args, _ = _historical_gradient_cli
    cfg = execution.load_config(Path(args.config))
    arms = execution.arms_for(cfg.max_seq_len, cfg.n_gpus)
    spec = SimpleNamespace(applicable_tests=["single-step-grads"])
    expected = execution._load_historical_gradient_references(args)
    captured = []
    monkeypatch.setattr(execution, "_frozen_spec", lambda *a, **kw: (spec, arms))
    monkeypatch.setattr(execution, "_run_vocab_size", lambda *a, **kw: 128)
    monkeypatch.setattr(execution, "_finish", lambda *a, **kw: 0)

    def capture(config, reviewed, definitions, **kwargs):
        assert config == cfg and reviewed is spec and definitions == arms
        captured.append(kwargs)
        return []

    assert execution._run_configs(args, executor=capture) == 0
    assert len(captured) == 1
    assert captured[0]["historical_gradient_references"] == expected
    assert captured[0]["select"] == ["single-step-grads"]


def test_historical_gradient_execution_refuses_rl_before_native_or_spec_work(_historical_gradient_cli, monkeypatch):
    args, _ = _historical_gradient_cli
    Path(args.config).write_text(
        json.dumps(
            _config_factory_native_config(
                {"n_gpus": 4, "max_seq_len": 16, "attn_implementation": "sdpa"},
                sampling={"n_gpus": 4, "max_seq_len": 16},
            )
        )
    )
    assert execution.load_config(Path(args.config)).is_rl
    assert not hasattr(execution, "execute_rl")
    recorded = []
    monkeypatch.setattr(execution, "_frozen_spec", lambda *a, **kw: pytest.fail("RL rejection reached frozen spec"))
    monkeypatch.setattr(execution, "_run_vocab_size", lambda *a, **kw: pytest.fail("RL rejection read model metadata"))
    monkeypatch.setattr(execution, "arms_for", lambda *a, **kw: pytest.fail("RL rejection reached native arms"))
    monkeypatch.setattr(
        execution, "reference_tests", lambda: pytest.fail("historical bindings reached RL inapplicable branch")
    )

    def finish(results, *a, **kw):
        recorded.extend(results)
        return 1

    monkeypatch.setattr(execution, "_finish", finish)
    assert execution._run_configs(args, executor=lambda *a, **kw: pytest.fail("RL reached SFT executor")) == 1
    assert len(recorded) == 1
    assert recorded[0].outcome is Outcome.FAIL
    assert recorded[0].test_id == "-"
    assert recorded[0].reason == "historical gradient references require a supervised config"


def test_historical_gradient_execution_keeps_declared_width_with_larger_pool(_historical_gradient_cli, monkeypatch):
    args, _ = _historical_gradient_cli
    config_bytes = Path(args.config).read_bytes()
    cfg = execution.load_config(Path(args.config))
    args.gpu_pool = cfg.n_gpus * 4
    arms = execution.arms_for(cfg.max_seq_len, cfg.n_gpus)
    spec = SimpleNamespace(applicable_tests=["single-step-grads"])
    bindings = execution._load_historical_gradient_references(args)
    captured = []
    monkeypatch.setattr(execution, "_frozen_spec", lambda *a, **kw: (spec, arms))
    monkeypatch.setattr(execution, "_run_vocab_size", lambda *a, **kw: 128)
    monkeypatch.setattr(execution, "_finish", lambda *a, **kw: 0)

    def capture(config, reviewed, definitions, **kwargs):
        captured.append((config.n_gpus, definitions, kwargs))
        assert reviewed is spec
        return []

    assert execution._run_configs(args, executor=capture) == 0
    assert len(captured) == 1
    width, definitions, kwargs = captured[0]
    assert width == cfg.n_gpus == 4
    assert definitions == arms
    assert kwargs["historical_gradient_references"] == bindings
    assert kwargs["select"] == ["single-step-grads"]
    assert Path(args.config).read_bytes() == config_bytes


@pytest.mark.parametrize(
    "corruption", ["missing-arm", "extra-arm", "empty", "type", "optimizer", "mixed", "prep", "rl"]
)
def test_historical_gradient_runner_guard_precedes_workdir_and_target(
    _historical_gradient_cli, monkeypatch, corruption
):
    args, _ = _historical_gradient_cli
    cfg = execution.load_config(Path(args.config))
    bindings = execution._load_historical_gradient_references(args)
    select = ["single-step-grads"]
    if corruption == "missing-arm":
        bindings.pop("gas4")
    elif corruption == "extra-arm":
        bindings["foreign"] = bindings["gas1"]
    elif corruption == "empty":
        bindings = {}
    elif corruption == "type":
        bindings = []
    elif corruption == "optimizer":
        select = ["single-step-optimizer"]
    elif corruption == "mixed":
        select.append("single-step-optimizer")
    elif corruption == "rl":
        Path(args.config).write_text(
            json.dumps(
                _config_factory_native_config(
                    {"n_gpus": 4, "max_seq_len": 16, "attn_implementation": "sdpa"},
                    sampling={"n_gpus": 4, "max_seq_len": 16},
                )
            )
        )
        cfg = execution.load_config(Path(args.config))
    monkeypatch.setattr(runner, "correctness_workdir", lambda *a: pytest.fail("invalid binding reached workdir"))
    monkeypatch.setattr(runner, "gateway", lambda *a, **kw: pytest.fail("invalid binding reached target"))
    with pytest.raises(ValueError, match="exactly the selected SFT arms"):
        runner.execute(
            cfg,
            None,
            [SimpleNamespace(name=arm) for arm in ("gas1", "gas4")],
            vocab_size=128,
            select=select,
            prepare_references=corruption == "prep",
            historical_gradient_references=bindings,
        )


def test_historical_gradient_runner_records_provenance_before_target_and_keeps_bindings(
    _historical_gradient_cli, monkeypatch, tmp_path
):
    from arctic_platform.correctness.harness import reference_records
    from arctic_platform.correctness.harness.config import ArmDefinition

    args, _ = _historical_gradient_cli
    cfg = execution.load_config(Path(args.config))
    bindings = execution._load_historical_gradient_references(args)
    before = copy.deepcopy(bindings)
    definitions = [ArmDefinition("gas1", 2, 8), ArmDefinition("gas4", 4, 8)]
    spec = SimpleNamespace(
        applicable_tests=["single-step-grads"],
        test_settings={},
        model=SimpleNamespace(cache_path="/data-fast/tmp/unmaterialized-synthetic-model"),
    )
    loaded = []
    provenance = {}
    work = tmp_path / "runner"
    monkeypatch.setattr(runner, "require_data_fast_path", lambda path, **kw: Path(path))
    monkeypatch.setattr(runner, "uses_mixer_packing", lambda *a: False)
    monkeypatch.setattr(runner, "run_reference", lambda *a, **kw: pytest.fail("historical binding rebuilt HF"))
    monkeypatch.setattr(reference_records, "model_identity", lambda *a: {"synthetic": True})
    monkeypatch.setattr(reference_records, "workload_identity", lambda ctx, check, recipe: recipe)

    def load(ctx, identity, *, historical_binding):
        arm = "gas1" if identity["synthetic_inputs"]["rows"] == 2 else "gas4"
        assert historical_binding == before[arm]
        provenance[arm] = copy.deepcopy(historical_binding)
        ctx.reference_record_provenance[arm] = provenance[arm]
        loaded.append(arm)
        historical_binding["review"] = "mutated loader-local copy"
        return {"loss": 1.0, "grad_norms": {"synthetic.weight": 0.5}}

    def no_target(*a, **kw):
        assert loaded == ["gas1", "gas4"]
        assert json.loads((work / "reference-record-provenance.json").read_text()) == provenance
        assert bindings == before
        raise RuntimeError("test-only stop before target allocation")

    monkeypatch.setattr(reference_records, "load_record", load)
    monkeypatch.setattr(runner, "gateway", no_target)
    with pytest.raises(RuntimeError, match="test-only stop before target allocation"):
        runner.execute(
            cfg,
            spec,
            definitions,
            vocab_size=128,
            select=["single-step-grads"],
            workdir=work,
            historical_gradient_references=bindings,
        )
    assert bindings == before


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


def test_gateway_lifecycle_available_gpus_reads_a_hostfile_without_a_prestarted_gateway(tmp_path, monkeypatch) -> None:
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("node-a slots=8\nnode-b slots=8\n")
    monkeypatch.delenv("DSS_GATEWAY_URL", raising=False)
    monkeypatch.setenv("DSS_GATEWAY_HOSTFILE", str(hostfile))
    assert dss_driver.available_gpus() == 16


_operator_api_CONFIG = "qwen3-8b-h200-train-sft-full-4gpus-64k"


_operator_api_TEST = "single-step-grads"


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
    with pytest.raises(ValueError, match="no configurations"):
        api.select(api.RunRequest(tests=(_operator_api_TEST,), config_dir=tmp_path / "empty"))


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
    tests = (_operator_api_TEST,)
    _operator_api_fake_batches(
        monkeypatch, [Outcome.FAIL, ReferencePrerequisiteError("prepared reference missing"), Outcome.PASS]
    )
    result = api.run(api.RunRequest(all_configs=True, tests=tests), api.RunOptions(out=tmp_path))
    assert [case["status"] for case in result.cases] == ["fail", "blocked", "pass"]
    assert result.exit_code == 1 and (not result.stopped_unsafe)
    assert len(result.results) == 3
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
        api.RunRequest(all_configs=True, tests=(_operator_api_TEST,)),
        api.RunOptions(out=tmp_path),
    )
    assert result.stopped_unsafe and result.exit_code == 1
    assert [case["status"] for case in result.cases] == ["error", "blocked", "blocked"]
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
    monkeypatch.setattr(execution, "available_gpus", lambda: 4)
    monkeypatch.setattr(execution, "validate_against_host", lambda *a, **k: None)
    monkeypatch.setattr(execution, "ensure_no_interrupted_builds", lambda: None)
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
                "--test-spec",
                str(cli.SPEC_DIR / f"{_operator_api_CONFIG}.json"),
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
    spec = _applicable_tests_spec(["single-step-optimizer"])
    reason = "Traceback (most recent call last):\nRuntimeError: exported tensor inventory is incomplete"
    result = Result(
        test_id="single-step-optimizer",
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
