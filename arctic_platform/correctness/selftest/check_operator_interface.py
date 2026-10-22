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
import signal
import subprocess
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import Mock

import pytest
import torch

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


@pytest.mark.parametrize(
    "document",
    [
        "PR_DESCRIPTION.md",
        "arctic_platform/correctness/docs/running-tests.md",
        "arctic_platform/correctness/docs/getting-started.md",
    ],
)
def test_quick_demo_config_is_tiny_and_retains_both_gas_arms(tmp_path, monkeypatch, document):
    from arctic_platform.correctness.harness.config import arms_for
    from arctic_platform.correctness.harness.config import load_config

    root = Path(__file__).resolve().parents[3]
    text = (root / document).read_text()
    snippet = text.split("python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    downloads = []
    hub = ModuleType("huggingface_hub")

    def snapshot_download(**options):
        downloads.append(options)
        return str(tmp_path / "tiny-model")

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "work" / "configs").mkdir(parents=True)
    exec(compile(snippet, document, "exec"), {})
    config_path = tmp_path / "work" / "configs" / "tiny-qwen3-129m-one-h200.json"
    raw = json.loads(config_path.read_text())
    config = load_config(config_path)
    assert raw["backend"] == {
        "type": "remote",
        "protocol": "cortex",
        "base_url": "http://127.0.0.1:1",
        "database": "dev",
        "schema": "dev",
    }
    assert config.native.backend.type == "remote"
    assert config.native.backend.protocol == "cortex"
    assert downloads == [
        {
            "repo_id": "axolotl-ai-co/tiny-qwen3-129m",
            "cache_dir": "/checkpoint/huggingface/hub",
        }
    ]
    assert config.n_gpus == 1 and config.max_seq_len == 128
    assert config.attention_implementation == "flash_attention_3"
    assert raw["training"]["ds_config"]["train_batch_size"] == 1
    assert raw["training"]["ds_worker_config"]["activation_checkpointing"] is False
    assert [(arm.name, arm.global_batch_size, arm.max_seq_len) for arm in arms_for(128, 1)] == [
        ("gas1", 1, 128),
        ("gas4", 4, 128),
    ]
    assert "--config ./work/configs/tiny-qwen3-129m-one-h200.json" in text
    assert "qwen3-8b-four-h200" not in text


def test_getting_started_matches_pr_and_is_first_read():
    root = Path(__file__).resolve().parents[3]
    guide = root / "arctic_platform/correctness/docs/getting-started.md"
    assert guide.read_bytes() == (root / "PR_DESCRIPTION.md").read_bytes()
    index = (root / "docs/correctness.md").read_text()
    first_paragraph = index.split("\n\n", 2)[1]
    assert first_paragraph.startswith(
        "**Read [Getting started](../arctic_platform/correctness/docs/getting-started.md) first."
    )


@pytest.mark.parametrize(
    "document",
    [
        "PR_DESCRIPTION.md",
        "arctic_platform/correctness/docs/running-tests.md",
        "arctic_platform/correctness/docs/getting-started.md",
    ],
)
@pytest.mark.parametrize("log_override", ["demo", "other-run", "traversal"])
def test_demo_artifact_commands_stage_reusable_inputs_and_clean_only_owned_scratch(tmp_path, document, log_override):
    root = Path(__file__).resolve().parents[3]
    text = (root / document).read_text()
    staging = (
        text.split("## Add To Repo\n", 1)[1].split("## Cleanup\n", 1)[0].split("```bash\n", 1)[1].split("```", 1)[0]
    )
    cleanup = text.split("## Cleanup\n", 1)[1]
    commands = [block.split("```", 1)[0] for block in cleanup.split("```bash\n")[1:3]]
    assert all("/data-fast" not in command and "DSS_CORRECTNESS_TMPDIR" not in command for command in commands)
    assert commands == [
        "rm -f -- ./work/configs/tiny-qwen3-129m-one-h200.json\n",
        "rm -rf -- ./work/logs/correctness-demo/\nrm -rf -- ./work/reports/both/\n",
    ]
    assert 'export DSS_CORRECTNESS_LOGDIR="$PWD/work/logs/correctness-demo"' in text
    identifier = "tiny-qwen3-129m-one-h200"
    config = f"arctic_platform/correctness/configs/{identifier}.json"
    specification = f"arctic_platform/correctness/specs/{identifier}.json"
    record = f"arctic_platform/correctness/reference_data/{identifier}/single-step-grads/record.json"
    unrelated_record = "arctic_platform/correctness/reference_data/other/single-step-grads/record.json"
    original = f"work/configs/{identifier}.json"
    report_path = "work/reports/both/report.json"
    for relative in (config, specification, record, unrelated_record, original, report_path):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    (tmp_path / ".gitignore").write_text(
        "/arctic_platform/correctness/configs/\n/arctic_platform/correctness/specs/\n/arctic_platform/correctness/reference_data/\n/work/\n"
    )

    def run(*args, env=None):
        return subprocess.run(args, cwd=tmp_path, env=env, check=True, text=True, capture_output=True).stdout

    run("git", "init", "--quiet")
    run("bash", "-eu", "-c", staging)
    assert set(run("git", "ls-files", "--cached").splitlines()) == {config, specification}
    assert set(run("git", "check-ignore", record, unrelated_record).splitlines()) == {record, unrelated_record}
    assert (
        text.index("Use skills/correctness/SKILL.md to fix onboarding for\n")
        < text.index("## Add To Repo\n")
        < text.index("## Cleanup\n")
    )
    assert "mktemp -d /data-fast/tmp/correctness-demo.XXXXXXXX" in text

    logs = tmp_path / "work" / "logs" / "correctness-demo"
    logs.mkdir(parents=True)
    (logs / "onboarding.log").write_text("saved diagnostic")
    other_logs = tmp_path / "work" / "logs" / "keep-this-run"
    other_logs.mkdir()
    (other_logs / "onboarding.log").write_text("unrelated diagnostic")
    unrelated_demo_logs = tmp_path / "work" / "logs" / "correctness-demo.previous"
    unrelated_demo_logs.mkdir()
    (unrelated_demo_logs / "onboarding.log").write_text("retained earlier diagnostic")
    overrides = {
        "demo": str(logs),
        "other-run": str(other_logs),
        "traversal": str(logs) + "/../..",
    }
    with tempfile.TemporaryDirectory(prefix="correctness-demo.", dir="/data-fast/tmp") as scratch:
        scratch_path = Path(scratch)
        (scratch_path / "batch.pt").write_bytes(b"temporary")
        (scratch_path / "retained-input").symlink_to(tmp_path / config)
        env = {**os.environ, "DSS_CORRECTNESS_TMPDIR": scratch, "DSS_CORRECTNESS_LOGDIR": overrides[log_override]}
        for command in commands:
            run("bash", "-eu", "-c", command, env=env)
        assert (scratch_path / "batch.pt").read_bytes() == b"temporary"
        assert (scratch_path / "retained-input").is_symlink()
        assert not logs.exists()
        assert all((tmp_path / relative).is_file() for relative in (config, specification, record, unrelated_record))
        assert (other_logs / "onboarding.log").is_file()
        assert (unrelated_demo_logs / "onboarding.log").is_file()
        assert not (tmp_path / original).exists()
        assert not (tmp_path / report_path).exists()


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
    from arctic_platform.correctness.harness.config import local_config_path

    def local_config(path, config_dir=None):
        selected = root / "configs" if config_dir is None or Path(config_dir) == default_root else config_dir
        return local_config_path(path, selected)

    monkeypatch.setattr(api, "local_config_path", local_config)
    monkeypatch.setattr(cli, "local_config_path", local_config)
    monkeypatch.setattr("arctic_platform.correctness.harness.config.local_config_path", local_config)
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
        api,
        "select",
        lambda actual: (expected_selection if actual == request else pytest.fail("request changed")),
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
    assert api.RunOptions().backend == "cortex-training"


@pytest.mark.parametrize(
    "missing",
    ["historical_gradient_references", "historical_gradient_references_sha256"],
)
def test_historical_gradient_pair_rejected_before_setup(_historical_gradient_cli, monkeypatch, missing):
    args, _ = _historical_gradient_cli
    setattr(args, missing, None)
    monkeypatch.setattr(
        execution,
        "_resolve_configs",
        lambda *a, **kw: pytest.fail("invalid pair reached configs"),
    )
    monkeypatch.setattr(
        execution,
        "correctness_workdir",
        lambda *a: pytest.fail("invalid pair allocated workdir"),
    )
    with pytest.raises(ValueError, match="required together"):
        execution._run_configs(args)


@pytest.mark.parametrize("digest", ["0" * 64, "0" * 63, "A" * 64, "g" * 64, 123])
def test_historical_gradient_pin_checked_before_json(_historical_gradient_cli, monkeypatch, digest):
    args, _ = _historical_gradient_cli
    args.historical_gradient_references.write_bytes(b"not JSON")
    args.historical_gradient_references_sha256 = digest
    monkeypatch.setattr(
        execution.json,
        "loads",
        lambda *a, **kw: pytest.fail("bad pin reached JSON parsing"),
    )
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
    monkeypatch.setattr(
        execution,
        "_run_configs",
        lambda *a, **kw: pytest.fail("invalid selection narrowed"),
    )
    monkeypatch.setattr(
        execution,
        "correctness_workdir",
        lambda *a: pytest.fail("invalid selection allocated workdir"),
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
        assert config == execution.load_config(execution.local_config_path(cfg.path))
        assert config.path.read_bytes() == cfg.path.read_bytes()
        assert reviewed is spec and definitions == arms
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
    monkeypatch.setattr(
        execution,
        "_frozen_spec",
        lambda *a, **kw: pytest.fail("RL rejection reached frozen spec"),
    )
    monkeypatch.setattr(
        execution,
        "_run_vocab_size",
        lambda *a, **kw: pytest.fail("RL rejection read model metadata"),
    )
    monkeypatch.setattr(
        execution,
        "arms_for",
        lambda *a, **kw: pytest.fail("RL rejection reached native arms"),
    )
    monkeypatch.setattr(
        execution,
        "reference_tests",
        lambda: pytest.fail("historical bindings reached RL inapplicable branch"),
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
    "corruption",
    ["missing-arm", "extra-arm", "empty", "type", "optimizer", "mixed", "prep", "rl"],
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
    monkeypatch.setattr(
        runner,
        "correctness_workdir",
        lambda *a: pytest.fail("invalid binding reached workdir"),
    )
    monkeypatch.setattr(
        runner,
        "gateway",
        lambda *a, **kw: pytest.fail("invalid binding reached target"),
    )
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
    monkeypatch.setattr(
        runner,
        "run_reference",
        lambda *a, **kw: pytest.fail("historical binding rebuilt HF"),
    )
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
    return Namespace(
        all_configs=config is None,
        any_gpu=any_gpu,
        config_dir=str(config_dir),
        config=config,
    )


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
        metrics={
            "tensors_compared": 47.0,
            "tensors_over_stated_criterion": 0.0,
            "stated_criterion_abs": 0.013,
        },
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
        metrics={
            "tensors_compared": 47.0,
            "tensors_over_stated_criterion": 2.0,
            "stated_criterion_abs": 0.013,
        },
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


@pytest.mark.parametrize(
    "arguments",
    [[], ["--help"], ["config.config", "--local_rank=0", "--flag", "two words"]],
)
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
    "arguments",
    [
        [],
        ["run"],
        ["run", "../ladder"],
        ["run", "execution_controls"],
        ["run", "ladder"],
        ["list", "x"],
    ],
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
    assert json.loads(capsys.readouterr().out) == [
        "__main__",
        [str(source), "--help", "two words"],
        str(source),
    ]
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
    sub_job = {
        "job_type": "training",
        "model_name": "test",
        "training_config": training,
    }
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


def test_diagnostics_dispatch_chunking_parse_failure_does_not_load_executor(
    monkeypatch,
):
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


def test_diagnostics_dispatch_chunking_executor_failure_keeps_original_config_and_binding(
    monkeypatch,
):
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


def test_operator_api_config_only_selects_each_applicable_test_once():
    selected = api.select(api.RunRequest(config=_operator_api_CONFIG))
    config = api.load_config(api.catalog()[_operator_api_CONFIG])
    assert [case.test_id for case in selected] == sorted(
        test_id
        for test_id, test in api.registered_tests().items()
        if api.workload_inapplicable_reason(test, config) is None
    )
    assert len(set(selected)) == len(selected)


@pytest.mark.parametrize("is_rl", [False, True])
def test_operator_api_config_mode_selects_compatible_checks_without_a_backend(tmp_path, monkeypatch, is_rl):
    config = tmp_path / "external.json"
    config.write_text(
        json.dumps(
            _config_factory_native_config(
                {"n_gpus": 4, "max_seq_len": 16},
                sampling={"n_gpus": 4} if is_rl else None,
            )
        )
    )
    tests = {
        "gradients": SimpleNamespace(compares_to_reference=True, workload_modes=("sft",)),
        "rl-only": SimpleNamespace(test_id="rl-only", compares_to_reference=False, workload_modes=("rl",)),
        "shared": SimpleNamespace(compares_to_reference=False, workload_modes=("sft", "rl")),
    }
    monkeypatch.setattr(api, "registered_tests", lambda: tests)
    monkeypatch.setattr(api, "catalog", lambda *a: {"external": config})
    monkeypatch.setattr(
        execution,
        "_client_backend",
        lambda *a: pytest.fail("selection accessed backend"),
    )
    local_root = tmp_path / "checkout" / "configs"
    selected = api.select(api.RunRequest(config=config, config_dir=local_root))
    assert [case.test_id for case in selected] == (["rl-only", "shared"] if is_rl else ["gradients", "shared"])
    assert all(case.config_path == local_root / config.name for case in selected)
    assert selected[0].config_path.read_bytes() == config.read_bytes()
    incompatible = "gradients" if is_rl else "rl-only"
    explicit = api.select(api.RunRequest(config=config, config_dir=local_root, tests=(incompatible,)))
    assert [case.test_id for case in explicit] == [incompatible]


@pytest.mark.parametrize("suffix", [".json", ".config"])
def test_operator_external_config_copy_is_local_and_idempotent(tmp_path, suffix):
    from arctic_platform.correctness.harness.config import config_id
    from arctic_platform.correctness.harness.config import load_config
    from arctic_platform.correctness.harness.config import local_config_path

    source = tmp_path / "incoming" / f"model{suffix}"
    source.parent.mkdir()
    source.write_text(json.dumps(_config_factory_native_config({"n_gpus": 4, "max_seq_len": 16}), indent=2) + "\n")
    root = tmp_path / "checkout" / "configs"
    local = local_config_path(source, root)
    assert local == root / source.name
    assert local.read_bytes() == source.read_bytes()
    assert config_id(local) == config_id(source)
    assert load_config(local).native == load_config(source).native
    modified = local.stat().st_mtime_ns
    assert local_config_path(source, root) == local
    assert local.stat().st_mtime_ns == modified
    source.unlink()
    assert local_config_path(local, root) == local


def test_operator_external_config_copy_preserves_local_recipe_on_collision(tmp_path):
    from arctic_platform.correctness.harness.config import local_config_path

    source = tmp_path / "model.json"
    source.write_text(json.dumps(_config_factory_native_config({"n_gpus": 4, "max_seq_len": 16})))
    root = tmp_path / "checkout" / "configs"
    local = local_config_path(source, root)
    original = local.read_bytes()
    source.write_text(json.dumps(_config_factory_native_config({"n_gpus": 8, "max_seq_len": 16})))
    with pytest.raises(ValueError, match="different contents"):
        local_config_path(source, root)
    assert local.read_bytes() == original


def test_operator_external_config_copy_reuses_id_across_extensions_and_layouts(
    tmp_path,
):
    from arctic_platform.correctness.harness.config import config_id
    from arctic_platform.correctness.harness.config import local_config_path

    root = tmp_path / "checkout" / "configs"
    local = root / "model" / "h200" / "train.config"
    local.parent.mkdir(parents=True)
    local.write_text(json.dumps(_config_factory_native_config({"n_gpus": 4, "max_seq_len": 16})))
    source = tmp_path / "model-h200-train.json"
    source.write_bytes(local.read_bytes())
    assert config_id(source) == config_id(local)
    assert local_config_path(source, root) == local
    assert not (root / source.name).exists()
    source.write_text(json.dumps(_config_factory_native_config({"n_gpus": 8, "max_seq_len": 16})))
    with pytest.raises(ValueError, match="different contents"):
        local_config_path(source, root)
    assert not (root / source.name).exists()


def test_operator_external_config_copy_rejects_invalid_input_before_import(tmp_path):
    from arctic_platform.correctness.harness.config import local_config_path

    source = tmp_path / "not-a-config.json"
    source.write_text('{"unrelated": true}')
    root = tmp_path / "checkout" / "configs"
    with pytest.raises(ValueError):
        local_config_path(source, root)
    assert not root.exists()


def test_operator_onboarding_uses_imported_config_before_preparation(tmp_path, monkeypatch):
    from arctic_platform.correctness.harness.config import local_config_path
    from arctic_platform.correctness.onboarding import onboard

    source = tmp_path / "model.json"
    source.write_text(json.dumps(_config_factory_native_config({"n_gpus": 4, "max_seq_len": 16})))
    root = tmp_path / "checkout" / "configs"
    args = execution.parse_args(["onboard", "--config", str(source)])
    monkeypatch.setattr(onboard, "_validate_storage", lambda *a: None)
    monkeypatch.setattr(
        "arctic_platform.correctness.harness.config.local_config_path",
        lambda path: local_config_path(path, root),
    )
    used = []
    monkeypatch.setattr(onboard, "_onboard_configuration", lambda child: used.append(child.config) or 0)
    assert onboard.run(args) == 0
    assert used == [root / source.name]
    assert used[0].read_bytes() == source.read_bytes()


# Config-driven onboarding prerequisites.


@pytest.fixture
def _onboarding_prerequisites(tmp_path, monkeypatch):
    from arctic_platform.correctness.harness import config as config_module
    from arctic_platform.correctness.harness import registry
    from arctic_platform.correctness.onboarding import onboard

    args = execution.parse_args(["onboard", "--config", str(tmp_path / "model.json")])
    args.test_spec = tmp_path / "spec.json"
    args.output_dir = tmp_path / "scratch"
    args.cache_root = tmp_path / "cache"
    args.reference_records_dir = tmp_path / "records"
    config = SimpleNamespace(
        config_id="synthetic",
        is_rl=False,
        n_gpus=4,
        sp_size=1,
        gpu_type="h200",
        model_name="Qwen/synthetic",
        max_seq_len=128,
        training={},
        attention_implementation="flash_attention_3",
        reference_attention_implementation="flash_attention_3",
    )
    tests = {
        test_id: SimpleNamespace(
            test_id=test_id,
            compares_to_reference=True,
            requires_hosted_control_plane=False,
            workload_modes=("sft",),
        )
        for test_id in ("single-step-grads", "single-step-optimizer")
    }
    tests["hosted"] = SimpleNamespace(
        test_id="hosted",
        compares_to_reference=False,
        requires_hosted_control_plane=True,
        workload_modes=("sft",),
    )
    monkeypatch.setattr(config_module, "load_config", lambda *a: config)
    monkeypatch.setattr(registry, "registered_tests", lambda: tests)
    monkeypatch.setattr(dss_driver, "_allocation_hostfile", lambda: None)
    calls = []
    result = {
        "host": "head",
        "python": sys.executable,
        "gpu_count": 4,
        "versions": {"torch": "test"},
        "errors": [],
    }

    def inspect_node(request):
        calls.append(copy.deepcopy(request))
        return copy.deepcopy(result)

    monkeypatch.setattr(preflight, "onboarding_node_prerequisites", inspect_node)
    return SimpleNamespace(args=args, config=config, calls=calls, result=result, onboard=onboard)


@pytest.mark.parametrize("selection", ["all", "single-step-optimizer"])
def test_onboarding_prerequisites_fail_before_workdir_fixture_or_weights(
    _onboarding_prerequisites, monkeypatch, selection
):
    state = _onboarding_prerequisites
    state.args.test_id = selection
    state.result["errors"] = ["attention backend flash_attention_3: incompatible CUDA extension"]
    monkeypatch.setattr(state.onboard, "_parent", lambda *a: pytest.fail("preflight reached model work"))
    monkeypatch.setattr(
        state.onboard,
        "correctness_workdir",
        lambda *a: pytest.fail("preflight created scratch"),
    )
    with pytest.raises(ValueError, match="prerequisites failed before model work") as caught:
        state.onboard._onboard_configuration(state.args)
    assert "head" in str(caught.value) and sys.executable in str(caught.value)
    assert "flash_attention_3" in str(caught.value)
    assert len(state.calls) == 1
    assert not state.args.output_dir.exists() and not state.args.test_spec.exists()
    assert not getattr(state.args, "_prerequisites_checked", False)


def test_onboarding_prerequisites_run_once_before_all_selected_checks(_onboarding_prerequisites, monkeypatch, capsys):
    state = _onboarding_prerequisites
    seen = []

    def parent(child):
        assert child._prerequisites_checked and len(state.calls) == 1
        seen.append(child.test_id)
        child.test_spec.write_text("{}")
        return 0

    monkeypatch.setattr(state.onboard, "_parent", parent)
    monkeypatch.setattr(state.onboard, "_prepare_references", lambda child: 0)
    assert state.onboard._onboard_configuration(state.args) == 0
    assert seen == ["single-step-grads", "single-step-optimizer"]
    assert state.calls[0]["attention"] == ["flash_attention_3"]
    assert "vllm" not in state.calls[0]["modules"]
    output = capsys.readouterr().out
    assert "### Onboarding 1/2: Single Step Grads ###" in output
    assert "### Onboarding 2/2: Single Step Optimizer ###" in output
    assert "*** Onboarding" not in output


@pytest.mark.parametrize(
    "phase, marker",
    [("Onboarding 1/1", "###"), ("Phase 2/3", "***"), ("Target", "***")],
)
def test_onboarding_heading_marker_is_distinct_from_inner_stages(capsys, phase, marker):
    from arctic_platform.correctness.onboarding import onboard

    onboard._stage(phase, "single step optimizer", "Config: example")
    assert capsys.readouterr().out == f"\n{marker} {phase}: Single Step Optimizer {marker}\nConfig: example\n\n"


def test_onboarding_prerequisites_respect_reference_backend_and_config_lora(
    _onboarding_prerequisites,
):
    state = _onboarding_prerequisites
    state.config.attention_implementation = "sparse_mla"
    state.config.reference_attention_implementation = "sdpa"
    state.config.training["peft_config"] = {"rank": 8}
    state.args.backend = "cortex"
    state.args.base_url = "http://127.0.0.1:8003"
    state.onboard._onboarding_preflight(state.args, state.config)
    assert state.calls[0]["attention"] == state.calls[0]["hf_attention"] == ["sdpa"]
    assert "peft" in state.calls[0]["modules"]
    assert "deepspeed" not in state.calls[0]["modules"]
    assert "fla" not in state.calls[0]["modules"] and "causal_conv1d" not in state.calls[0]["modules"]


@pytest.mark.parametrize("fault", [None, "version", "worker", "capacity", "no-hostfile"])
def test_onboarding_prerequisites_check_actual_allocation_nodes(
    _onboarding_prerequisites, tmp_path, monkeypatch, fault
):
    state = _onboarding_prerequisites
    state.config.n_gpus = 8
    hostfile = tmp_path / "hosts"
    hostfile.write_text("head slots=4\npeer slots=4\n")
    nodes = {host: copy.deepcopy(state.result) for host in ("head", "peer")}
    if fault == "version":
        nodes["peer"]["versions"]["torch"] = "different"
    if fault == "worker":
        nodes["peer"]["errors"] = ["attention backend flash_attention_3: cannot load kernel"]
    if fault == "capacity":
        hostfile.write_text("head slots=4\npeer slots=2\n")
    monkeypatch.setattr(
        dss_driver,
        "_allocation_hostfile",
        lambda: None if fault == "no-hostfile" else hostfile,
    )
    remote = Mock(return_value=nodes)
    monkeypatch.setattr(preflight, "allocation_onboarding_prerequisites", remote)
    if fault:
        with pytest.raises(ValueError):
            state.onboard._onboarding_preflight(state.args, state.config)
        assert not getattr(state.args, "_prerequisites_checked", False)
    else:
        state.onboard._onboarding_preflight(state.args, state.config)
        assert state.args._preflight_gpu_capacity == 8
    if fault != "no-hostfile":
        request, actual_hostfile = remote.call_args.args
        assert actual_hostfile == hostfile and request["hf_attention"] == []
        assert "specification output" not in request["writable"]
        assert "liger_kernel.transformers.fused_linear_cross_entropy" not in request["modules"]
        assert request["min_gpus"] == 1 and request["source"] is None and request["model_id"] is None


def test_onboarding_prerequisites_reference_free_cortex_needs_no_author_gpus(
    _onboarding_prerequisites,
):
    state = _onboarding_prerequisites
    state.args.test_id = "hosted"
    state.args.backend = "cortex"
    state.args.test_spec.write_text("{}")
    state.result["gpu_count"] = 0
    state.onboard._onboarding_preflight(state.args, state.config, ["hosted"])
    assert state.calls[0]["min_gpus"] == 0 and state.calls[0]["attention"] == []


def test_onboarding_prerequisites_reference_workers_are_not_target_requirements(
    _onboarding_prerequisites, tmp_path, monkeypatch
):
    state = _onboarding_prerequisites
    state.config.n_gpus = 16
    state.args.reference_calibration_workers = 8
    state.result["gpu_count"] = 8
    state.result["versions"]["liger_kernel.transformers.fused_linear_cross_entropy"] = "local-only"
    hostfile = tmp_path / "hosts"
    hostfile.write_text("head slots=8\npeer slots=8\n")
    remote = Mock(
        return_value={
            host: {
                "host": host,
                "python": sys.executable,
                "gpu_count": 8,
                "versions": {"torch": "test"},
                "errors": [],
            }
            for host in ("head", "peer")
        }
    )
    monkeypatch.setattr(dss_driver, "_allocation_hostfile", lambda: hostfile)
    monkeypatch.setattr(preflight, "allocation_onboarding_prerequisites", remote)
    state.onboard._onboarding_preflight(state.args, state.config)
    assert state.calls[0]["min_gpus"] == 8
    assert remote.call_args.args[0]["min_gpus"] == 1


@pytest.fixture
def _node_prerequisites(tmp_path, monkeypatch):
    device = SimpleNamespace(name="NVIDIA H200", major=9, minor=0)
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 4,
        get_device_properties=lambda index: device,
    )
    torch_module = SimpleNamespace(cuda=cuda, __version__="synthetic-cuda")
    imported = preflight.importlib.import_module

    def import_module(name):
        if name == "torch":
            return torch_module
        if name == "flash_attn_interface":
            return SimpleNamespace(flash_attn_varlen_func=lambda: None)
        if name == "transformers.utils":
            return SimpleNamespace(is_flash_attn_3_available=lambda: True)
        return imported(name)

    monkeypatch.setattr(preflight.importlib, "import_module", import_module)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"synthetic file, never loaded")
    request = {
        "modules": ["torch"],
        "attention": ["flash_attention_3"],
        "hf_attention": ["flash_attention_3"],
        "gpu_type": "h200",
        "min_gpus": 4,
        "source": str(source),
        "writable": {"scratch": str(tmp_path / "not-yet-created")},
    }
    return SimpleNamespace(request=request, device=device, cuda=cuda, importer=import_module)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "cuda",
        "family",
        "count",
        "capability",
        "abi",
        "hf",
        "source",
        "output",
        "no-gpu-work",
    ],
)
def test_onboarding_node_prerequisites_do_not_allocate_models(_node_prerequisites, monkeypatch, fault):
    state = _node_prerequisites
    before = copy.deepcopy(state.request)
    if fault == "cuda":
        state.cuda.is_available = lambda: False
    elif fault == "family":
        state.device.name = "NVIDIA B200"
    elif fault == "count":
        state.cuda.device_count = lambda: 2
    elif fault == "capability":
        state.device.major = 8
    elif fault in {"abi", "hf"}:

        def import_module(name):
            if fault == "abi" and name == "flash_attn_interface":
                raise ImportError("undefined CUDA symbol")
            if fault == "hf" and name == "transformers.utils":
                return SimpleNamespace(is_flash_attn_3_available=lambda: False)
            return state.importer(name)

        monkeypatch.setattr(preflight.importlib, "import_module", import_module)
    elif fault == "source":
        (Path(state.request["source"]) / "config.json").unlink()
    elif fault == "output":
        state.request["writable"]["scratch"] = str(Path(state.request["source"]) / "config.json")
    elif fault == "no-gpu-work":
        state.request["min_gpus"] = 0
        state.request["attention"] = []
        state.cuda.is_available = lambda: pytest.fail("inspected CUDA for a reference-free Cortex request")
    result = preflight.onboarding_node_prerequisites(state.request)
    assert bool(result["errors"]) == (fault not in {None, "no-gpu-work"})
    assert result["python"] == sys.executable
    assert state.request["attention"] == ([] if fault == "no-gpu-work" else before["attention"])
    assert not Path(before["writable"]["scratch"]).exists()
    if fault in {"abi", "hf"}:
        assert "another FlashAttention version is not a substitute" in "\n".join(result["errors"])


@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "malformed", "slots", "transport"])
def test_onboarding_allocation_prerequisites_require_every_host_response(tmp_path, monkeypatch, fault):
    hostfile = tmp_path / "hosts"
    hostfile.write_text("10.0.0.1 slots=4\n10.0.0.2 slots=4\n")
    result = {
        "host": "node",
        "python": sys.executable,
        "gpu_count": 4,
        "versions": {},
        "errors": [],
    }
    if fault == "slots":
        result["gpu_count"] = 2
    output = f"10.0.0.1: CORRECTNESS_PREREQUISITES {json.dumps(result)}\n"
    if fault == "malformed":
        output = "10.0.0.1: CORRECTNESS_PREREQUISITES []\n"
    if fault != "missing":
        output += f"10.0.0.2: CORRECTNESS_PREREQUISITES {json.dumps(result)}\n"
    if fault == "duplicate":
        output += output.splitlines()[0] + "\n"
    run = Mock(return_value=subprocess.CompletedProcess([], int(fault == "transport"), output, "synthetic stderr"))
    monkeypatch.setattr(preflight.subprocess, "run", run)
    if fault in {"missing", "duplicate", "malformed", "transport"}:
        with pytest.raises(ValueError):
            preflight.allocation_onboarding_prerequisites({}, hostfile)
    else:
        nodes = preflight.allocation_onboarding_prerequisites({}, hostfile)
        assert set(nodes) == {"10.0.0.1", "10.0.0.2"}
        assert bool(nodes["10.0.0.2"]["errors"]) == (fault == "slots")
    command = run.call_args.args[0]
    assert command[1:3] == ["-f", str(hostfile)]
    assert sys.executable in command[-1] and "base64 -d" in command[-1]


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
        (
            api.RunRequest(config=_operator_api_CONFIG, tests=("missing",)),
            "unknown check",
        ),
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
            Result(
                check,
                api.config_id(Path(args.config)),
                outcome,
                summary="synthetic comparison",
            )
            for check in args.test
        ]
        snapshot(results, {}, {})
        return api.aggregate_exit_code(results)

    monkeypatch.setattr(execution, "_run_configs", fake)


def test_operator_api_mixed_batch_continues_and_retains_every_pair(tmp_path, monkeypatch, capsys):
    tests = (_operator_api_TEST,)
    _operator_api_fake_batches(
        monkeypatch,
        [
            Outcome.FAIL,
            ReferencePrerequisiteError("prepared reference missing"),
            Outcome.PASS,
        ],
    )
    result = api.run(api.RunRequest(all_configs=True, tests=tests), api.RunOptions(out=tmp_path))
    assert [case["status"] for case in result.cases] == ["fail", "blocked", "pass"]
    assert result.exit_code == 1 and (not result.stopped_unsafe)
    assert len(result.results) == 3
    assert "ReferencePrerequisiteError" in (tmp_path / "full.log").read_text()
    assert "Traceback" not in capsys.readouterr().out
    assert (tmp_path / "batch.json").is_file()


def test_operator_api_actual_execution_preserves_selection_order_and_combined_step_identity(
    tmp_path, monkeypatch, capsys
):
    selected = api.select(api.RunRequest(config=_operator_api_CONFIG))
    observed = []
    monkeypatch.setattr(execution, "write_report", lambda *a, **k: None)

    def record(args, *, executor, snapshot):
        observed.append(tuple(args.test))
        snapshot(
            [Result(check, _operator_api_CONFIG, Outcome.PASS) for check in args.test],
            {},
            {},
        )

    monkeypatch.setattr(execution, "_run_configs", record)
    result = api.run(api.RunRequest(config=_operator_api_CONFIG), api.RunOptions(out=tmp_path))
    assert [check for group in observed for check in group] == [case.test_id for case in selected]
    assert ("single-step-grads", "single-step-optimizer") in observed
    assert [case["test_id"] for case in result.cases] == [case.test_id for case in selected]
    assert result.exit_code == 0
    expected = ", ".join(dict.fromkeys(case.test_id for case in selected))
    assert f"Selected tests: {expected}" in capsys.readouterr().out.splitlines()


def test_operator_api_inapplicable_is_visible_and_not_a_pass(tmp_path, monkeypatch):
    _operator_api_fake_batches(monkeypatch, [Outcome.INAPPLICABLE])
    result = api.run(
        api.RunRequest(config=_operator_api_CONFIG, tests=(_operator_api_TEST,)),
        api.RunOptions(out=tmp_path),
    )
    assert result.cases[0]["status"] == "inapplicable"
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("unexpected failure"),
        UnsafeCleanupError("cleanup did not complete"),
        KeyboardInterrupt(),
    ],
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
        api.RunRequest(config=_operator_api_CONFIG, tests=(_operator_api_TEST,)),
        api.RunOptions(out=tmp_path),
    )
    assert result.cases[0]["status"] == "blocked" and result.exit_code == 1


@pytest.mark.parametrize(
    "error_type,status",
    [
        (None, "fail"),
        ("RuntimeError", "error"),
        ("ReferencePrerequisiteError", "blocked"),
    ],
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
        [
            "--config",
            _operator_api_CONFIG,
            "--test",
            _operator_api_TEST,
            "--backend",
            "cortex-training",
        ]
    )
    assert request.config == _operator_api_CONFIG and request.tests == (_operator_api_TEST,)
    assert options.backend == api.RunOptions().backend == "cortex-training"
    assert api._namespace(request, options).backend == "cortex"
    with pytest.raises(ValueError, match="no longer supported"):
        api._namespace(request, api.RunOptions(backend="onprem"))


def test_operator_api_static_guide_is_unchanged_by_run_and_onboarding(tmp_path, monkeypatch):
    monkeypatch.setenv("DSS_CORRECTNESS_LOGDIR", str(tmp_path / "logs"))
    guide = api.PACKAGE_DIR / "docs" / "running-tests.md"
    before = guide.read_bytes()
    monkeypatch.setattr(execution, "_run_vocab_size", lambda *a, **k: 128)
    from transformers import AutoConfig

    from arctic_platform.correctness.harness import report
    from arctic_platform.correctness.harness import runner
    from arctic_platform.correctness.onboarding import onboard

    monkeypatch.setattr(onboard, "_onboarding_preflight", lambda *a: None)
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
                str(execution.SPEC_DIR / f"{_operator_api_CONFIG}.json"),
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
        api.RunRequest(config=_operator_api_CONFIG, tests=(_operator_api_TEST,)),
        api.RunOptions(out=tmp_path),
    )
    assert result.cases[0]["status"] == "blocked"
    assert "Traceback" in (tmp_path / "full.log").read_text()
    assert "Traceback" not in capsys.readouterr().err
    assert root.handlers == previous


def _preflight_run_node_script(root) -> str:
    return subprocess.run(
        ["bash", "-c", preflight.node_script(str(root))],
        capture_output=True,
        text=True,
        check=True,
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
        preflight.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr=""),
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


def test_report_reports_are_timestamped_and_latest_links_move_without_deleting_history(
    tmp_path: Path,
) -> None:
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
    monkeypatch.setattr(
        cli,
        "correctness_workdir",
        lambda prefix: Path("/data-fast/tmp/unique-selftests"),
    )
    monkeypatch.setattr(pytest, "main", lambda command: commands.append(command) or 0)
    assert cli.cmd_selftest(SimpleNamespace(runner_args=[])) == 0
    assert commands[0][-2:] == ["--basetemp", "/data-fast/tmp/unique-selftests/pytest"]
    assert commands[0][0] == "-sv"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--basetemp=/tmp/tests"],
        ["--basetemp", "/code/tests"],
        ["--basetemp", "/data-fast"],
        ["--basetemp"],
    ],
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


def test_workdir_root_correctness_workdir_rejects_unsafe_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workdir_root_reset_tmp_env(monkeypatch)
    monkeypatch.setenv(CORRECTNESS_TMPDIR_ENV, "/code/users/stas/ap3tmp")
    with pytest.raises(ValueError, match=CORRECTNESS_TMPDIR_ENV):
        workdir.correctness_workdir("dss-correctness-")


# Repeating onboarding resumes completed preparation, not another reference measurement.


@pytest.mark.parametrize("failure", ["reported", "missing-report", "inconsistent-totals"])
def test_onboarding_target_failure_reason_survives_into_summary(tmp_path, monkeypatch, failure):
    from arctic_platform.correctness.onboarding import onboard

    args = execution.parse_args(
        [
            "onboard",
            "--config",
            str(tmp_path / "model.json"),
            "--test",
            "single-step-grads",
        ]
    )
    args.output_dir = tmp_path
    args.checkout_root = tmp_path
    cause = "The target worker rejected the optimizer snapshot metadata."
    target_report = {
        "totals": {"pass": 1, "fail": int(failure != "inconsistent-totals")},
        "results": [
            {"test_id": args.test_id, "arm": "gas1", "outcome": "pass"},
            {
                "test_id": args.test_id,
                "arm": "gas4",
                "outcome": "fail",
                "reason": cause,
            },
        ],
    }

    def child(command, **kwargs):
        out = args.output_dir / "validation" / "4-layers"
        if failure != "missing-report":
            (out / "report.json").write_text(json.dumps(target_report))
        return SimpleNamespace(
            stdout=io.StringIO(cause + "\n"),
            wait=lambda: 0 if failure == "inconsistent-totals" else 1,
        )

    monkeypatch.setattr(onboard.subprocess, "Popen", child)
    if failure == "inconsistent-totals":
        with pytest.raises(RuntimeError, match=cause) as caught:
            onboard._run_final_regression(args, {"layers": 4})
    else:
        validation = onboard._run_final_regression(args, {"layers": 4})
        assert validation["status"] == "failed" and cause in validation["reason"]
        with pytest.raises(RuntimeError, match=cause) as caught:
            onboard._require_validation_pass(validation, "Gradient validation")
    assert "Complete log:" in str(caught.value)
    assert str(tmp_path / "validation" / "4-layers") in str(caught.value)
    if failure != "missing-report":
        assert "gas4" in str(caught.value)


@pytest.mark.parametrize("outcome", ["blocked", "error"])
def test_onboarding_zero_exit_with_nonpassing_totals_is_not_success(outcome):
    from arctic_platform.correctness.onboarding import onboard

    with pytest.raises(RuntimeError, match="failing test results"):
        onboard._validation_outcome(0, "", {"totals": {"pass": 1, "fail": 0, outcome: 1}})


def test_onboarding_resume_retries_validation_without_repeating_preparation(tmp_path):
    from arctic_platform.correctness.onboarding import onboard

    args = Namespace(
        _resume_dir=tmp_path / "state",
        test_spec=tmp_path / "spec.json",
        reference_records_dir=tmp_path / "records",
    )
    args._resume_dir.mkdir()
    calibration = tmp_path / "calibration.json"
    calls = []

    def sizing():
        calls.append("sizing")
        return {"status": "fit", "layers": 4}

    def fixture():
        calls.append("fixture")
        args.test_spec.write_text('{"fixture": 4}')

    def calibrate():
        calls.append("calibration")
        calibration.write_text('{"tolerance": {"selected_gate": 0.005}}')
        args.test_spec.write_text('{"fixture": 4, "gate": 0.005}')
        return str(calibration)

    def references():
        calls.append("references")
        return 0

    def validate():
        calls.append("validation")
        return {"status": "error" if calls.count("validation") == 1 else "pass"}

    for attempt in range(2):
        onboard._resume_phase(args, "sizing", sizing)
        onboard._resume_phase(args, "fixture", fixture)
        onboard._resume_phase(args, "calibration", calibrate)
        onboard._resume_phase(args, "references", references)
        result = onboard._resume_phase(args, "validation", validate)
        assert result["status"] == ("error" if attempt == 0 else "pass")
        if attempt == 0:
            assert not (args._resume_dir / "validation.json").exists()
            args.test_spec.unlink()  # The failed parent rolls back its unpublished specification.
    assert calls == [
        "sizing",
        "fixture",
        "calibration",
        "references",
        "validation",
        "validation",
    ]
    assert json.loads(args.test_spec.read_text())["gate"] == 0.005


@pytest.mark.parametrize(
    "drift",
    ["deleted-calibration", "changed-calibration", "changed-reference", "changed-spec"],
)
def test_onboarding_resume_rebuilds_changed_phase_and_invalidates_successors(tmp_path, drift):
    from arctic_platform.correctness.onboarding import onboard

    args = Namespace(
        _resume_dir=tmp_path / "state",
        test_spec=tmp_path / "spec.json",
        reference_records_dir=tmp_path / "records",
    )
    args._resume_dir.mkdir()
    args.test_spec.write_text("{}")
    record = args.reference_records_dir / args.test_spec.stem / "reference.json"
    record.parent.mkdir(parents=True)
    record.write_text('{"measurement": 1}')
    calibration = tmp_path / "calibration.json"
    calls = []

    def calibrate():
        calls.append("calibration")
        calibration.write_text('{"gate": 0.005}')
        return str(calibration)

    onboard._resume_phase(args, "calibration", calibrate)
    (args._resume_dir / "validation.json").write_text("{}")
    if drift == "deleted-calibration":
        calibration.unlink()
    elif drift == "changed-calibration":
        calibration.write_text('{"gate": 0.100}')
    elif drift == "changed-reference":
        record.write_text('{"measurement": 2}')
    else:
        args.test_spec.write_text('{"changed": true}')
    onboard._resume_phase(args, "calibration", calibrate)
    assert calls == ["calibration", "calibration"]
    assert not (args._resume_dir / "validation.json").exists()


@pytest.mark.parametrize("status", ["oom", "error"])
def test_onboarding_resume_never_admits_failed_sizing(tmp_path, status):
    from arctic_platform.correctness.onboarding import onboard

    args = Namespace(_resume_dir=tmp_path, test_spec=tmp_path / "spec.json")
    attempts = []

    def probe():
        attempts.append(status)
        return {"status": status}

    assert onboard._resume_phase(args, "sizing", probe)["status"] == status
    assert onboard._resume_phase(args, "sizing", probe)["status"] == status
    assert attempts == [status, status] and not (tmp_path / "sizing.json").exists()


@pytest.mark.parametrize("drift", ["config", "options", "dependencies"])
def test_onboarding_resume_state_is_bound_to_recipe_and_environment(tmp_path, monkeypatch, drift):
    from arctic_platform.correctness.onboarding import onboard

    config = tmp_path / "config.json"
    config.write_text('{"seed": 42}')
    args = Namespace(
        config=config,
        source_checkpoint="synthetic/model",
        test_id="single-step-grads",
        test_spec=tmp_path / "spec.json",
        tokens=16,
        _resume_root=tmp_path / "logs",
    )
    onboard._resume_setup(args)
    original = args._resume_dir
    onboard._resume_setup(args)
    assert args._resume_dir == original
    if drift == "config":
        config.write_text('{"seed": 43}')
    elif drift == "options":
        args.tokens = 32
    else:
        monkeypatch.setattr(onboard.importlib.metadata, "version", lambda name: "changed")
    onboard._resume_setup(args)
    assert args._resume_dir != original


@pytest.fixture
def _onboarding_retry_sources(tmp_path, monkeypatch):
    from arctic_platform.correctness.onboarding import onboard

    root = tmp_path / "package"
    for directory in ("onboarding", "harness", "reference", "checks"):
        (root / directory).mkdir(parents=True)
    (root / "onboarding" / "onboard.py").write_text(
        "def recipe():\n    return 42\n\ndef _resume_bookkeeping():\n    return 1\n"
    )
    (root / "harness" / "runner.py").write_text("if stepping:\n    gate = 0.005\n")
    (root / "harness" / "cortex.py").write_text("save_route = 'old'\n")
    (root / "reference" / "worker.py").write_text("seed = 42\n")
    (root / "harness" / "batches.py").write_text("seed = 42\n")
    monkeypatch.setattr(onboard, "__file__", str(root / "onboarding" / "onboard.py"))
    config = tmp_path / "config.json"
    config.write_text('{"seed": 42}')
    args = Namespace(
        config=config,
        source_checkpoint="synthetic/model",
        test_id="single-step-optimizer",
        test_spec=tmp_path / "spec.json",
        reference_records_dir=tmp_path / "records",
        _resume_root=tmp_path / "state",
    )
    args.test_spec.write_text("{}")
    return onboard, args, root


def test_onboarding_retry_target_fix_reuses_preparation_in_place_but_not_validation(
    _onboarding_retry_sources,
):
    onboard, args, root = _onboarding_retry_sources
    onboard._resume_setup(args)
    original = args._resume_dir
    calibration = root / "calibration.json"
    calibration.write_text('{"gate": 0.005}')
    onboard._resume_phase(args, "calibration", lambda: str(calibration))
    onboard._resume_phase(args, "validation", lambda: {"status": "pass"})
    saved = (original / "calibration.json").read_bytes()
    (root / "harness" / "cortex.py").write_text("save_route = 'fixed'\n")
    onboard._resume_setup(args)
    assert args._resume_dir != original
    assert onboard._resume_phase(
        args,
        "calibration",
        lambda: pytest.fail("a target-only fix must not rerun calibration"),
    ) == str(calibration)
    assert not (args._resume_dir / "calibration.json").exists()
    assert (original / "calibration.json").read_bytes() == saved
    calls = []
    result = onboard._resume_phase(
        args,
        "validation",
        lambda: calls.append("current-target") or {"status": "error"},
    )
    assert result["status"] == "error" and calls == ["current-target"]
    assert not (args._resume_dir / "validation.json").exists()


@pytest.mark.parametrize(
    "relative,source",
    [
        ("reference/worker.py", "seed = 43\n"),
        ("harness/batches.py", "seed = 43\n"),
        ("harness/runner.py", "if stepping:\n    gate = 0.100\n"),
        ("onboarding/onboard.py", "def recipe():\n    return 43\n"),
    ],
)
def test_onboarding_retry_reference_or_recipe_change_cannot_reuse_preparation(
    _onboarding_retry_sources, relative, source
):
    onboard, args, root = _onboarding_retry_sources
    onboard._resume_setup(args)
    (root / relative).write_text(source)
    onboard._resume_setup(args)
    assert args._resume_candidates == []


def test_onboarding_retry_reviewed_cache_and_summary_changes_preserve_preparation(
    _onboarding_retry_sources,
):
    onboard, args, root = _onboarding_retry_sources
    onboard._resume_setup(args)
    original = args._resume_dir
    (root / "onboarding" / "onboard.py").write_text(
        "import ast\n\ndef recipe():\n    return 42\n\ndef _resume_bookkeeping():\n    return 2\n"
    )
    (root / "harness" / "runner.py").write_text(
        'if stepping and "single-step-optimizer" not in records:\n'
        '    print(f"\\n*** Optimizer Norm Summary: {arm.name} ***\\n"\n'
        '          f"Reading saved optimizer states from {len(manifests)} completed HF runs.", flush=True)\n'
        "    summary_started = time.monotonic()\n"
        "    gate = 0.005\n"
        '    print(f"Optimizer norm summary complete: arm={arm.name}, "\n'
        '          f"elapsed={time.monotonic() - summary_started:.1f}s, "\n'
        '          f"maximum_pairwise_norm_delta={spread:.9e}\\n", flush=True)\n'
    )
    onboard._resume_setup(args)
    assert args._resume_candidates == [original]


@pytest.mark.parametrize("damage", ["missing-manifest", "wrong-key", "changed-artifact"])
def test_onboarding_retry_incomplete_or_corrupt_legacy_state_is_not_admitted(_onboarding_retry_sources, damage):
    onboard, args, root = _onboarding_retry_sources
    onboard._resume_setup(args)
    original = args._resume_dir
    calibration = root / "calibration.json"
    calibration.write_text('{"gate": 0.005}')
    onboard._resume_phase(args, "calibration", lambda: str(calibration))
    if damage == "missing-manifest":
        (original / "request.json").unlink()
    elif damage == "wrong-key":
        original.rename(original.with_name("not-the-request-digest"))
    else:
        calibration.write_text('{"gate": 0.100}')
    (root / "harness" / "cortex.py").write_text("save_route = 'fixed'\n")
    onboard._resume_setup(args)
    calls = []
    onboard._resume_phase(args, "calibration", lambda: calls.append("rebuild"))
    assert calls == ["rebuild"] and args._resume_candidates == []


@pytest.mark.parametrize("selection", ["all", "single-step-grads", "single-step-optimizer"])
def test_onboarding_config_only_automatically_selects_cortex_fixture(tmp_path, monkeypatch, selection):
    from arctic_platform.correctness.onboarding import onboard

    for name in ("ARCTIC_CORTEX_HOST", "ARCTIC_CORTEX_BASE_URL", "ARCTIC_CORTEX_PAT"):
        monkeypatch.delenv(name, raising=False)
    args = execution.parse_args(["onboard", "--config", str(tmp_path / "model.json"), "--test", selection])
    config = SimpleNamespace(is_rl=False)
    onboard._configure_fixture_target(args, config, list(onboard.CALIBRATED_TEST_IDS))
    assert args._automatic_cortex_gateway
    if selection != "all":
        assert args.local_cortex_fixture
    else:
        child = Namespace(**{**vars(args), "test_id": "single-step-grads"})
        onboard._configure_fixture_target(child, config, ["single-step-grads"])
        assert child.local_cortex_fixture and child._automatic_cortex_gateway


@pytest.mark.parametrize("body_error", [False, True])
@pytest.mark.parametrize("job_hostfile", [False, True])
def test_onboarding_automatic_gateway_uses_upstream_server_and_closes_owned_process(
    tmp_path, monkeypatch, body_error, job_hostfile
):
    import requests

    from arctic_platform.correctness.onboarding import onboard

    args = Namespace(_automatic_cortex_gateway=True, output_dir=tmp_path)
    process = Mock()
    process.pid = 12345
    process.poll.return_value = None
    process.wait.return_value = 0
    start = Mock(return_value=process)
    monkeypatch.setattr(onboard.subprocess, "Popen", start)
    monkeypatch.setattr(dss_driver, "_allocation_hostfile", lambda: None)
    is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: job_hostfile if str(path) == "/job/hostfile" else is_file(path),
    )
    listener = MagicMock()
    listener.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 54321)
    monkeypatch.setattr(dss_driver.socket, "socket", lambda: listener)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(status_code=200))
    monkeypatch.setattr(
        dss_driver,
        "_gateway_process_identity",
        lambda pid: {"pid": pid, "uid": os.getuid(), "start_time": "1"},
    )
    if body_error:
        with pytest.raises(RuntimeError, match="synthetic validation failed"):
            with onboard._validation_gateway(args):
                raise RuntimeError("synthetic validation failed")
    else:
        with onboard._validation_gateway(args) as url:
            assert url.startswith("http://127.0.0.1:")
    command = start.call_args.args[0]
    assert command[:3] == [sys.executable, "-m", "dss.ray_dss.gateway_server"]
    assert "--backend" in command and "gpu" in command
    assert command[-2:] == [
        "--hostfile",
        "/job/hostfile" if job_hostfile else str(tmp_path / "hostfile"),
    ]
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=120)


def test_onboarding_automatic_gateway_does_not_start_or_stop_configured_target(tmp_path, monkeypatch):
    from arctic_platform.correctness.onboarding import onboard

    monkeypatch.setattr(
        onboard.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("external target was replaced"),
    )
    with onboard._validation_gateway(Namespace(output_dir=tmp_path)) as url:
        assert url is None


def test_onboarding_resume_config_only_command_retries_failed_optimizer(tmp_path, monkeypatch, capsys):
    from arctic_platform.correctness.harness import config as config_module
    from arctic_platform.correctness.harness import registry
    from arctic_platform.correctness.onboarding import onboard

    monkeypatch.chdir(tmp_path)
    for name in (
        "ARCTIC_CORTEX_HOST",
        "ARCTIC_CORTEX_BASE_URL",
        "ARCTIC_CORTEX_PAT",
        "DSS_CORRECTNESS_LOGDIR",
    ):
        monkeypatch.delenv(name, raising=False)
    config_path = tmp_path / "recipe.json"
    config_path.write_text(json.dumps(_config_factory_native_config({"n_gpus": 4, "attn_implementation": "sdpa"})))
    config = config_module.load_config(config_path)
    spec_path = tmp_path / "spec.json"
    model_path = tmp_path / "model"
    calls = []
    entries = registry.registered_tests()
    monkeypatch.setattr(
        registry,
        "registered_tests",
        lambda: {test_id: entries[test_id] for test_id in onboard.CALIBRATED_TEST_IDS},
    )
    monkeypatch.setattr(config_module, "local_config_path", lambda path: Path(path))
    monkeypatch.setattr(onboard, "_validate_storage", lambda args: None)
    monkeypatch.setattr(onboard, "_checkout_root", lambda path: tmp_path)
    monkeypatch.setattr(onboard, "_onboarding_preflight", lambda *args: None)
    monkeypatch.setattr(onboard, "correctness_workdir", lambda prefix: tmp_path / "scratch")
    monkeypatch.setattr(onboard, "_model_shape", lambda path: (36, 1, 4))
    monkeypatch.setattr(onboard, "_optimizer_reference_model", lambda *args: (model_path, 4))
    spec = SimpleNamespace(
        config_id=config.config_id,
        config_checksum=config_module.config_checksum(config),
        applicable_tests=list(onboard.CALIBRATED_TEST_IDS),
        model=SimpleNamespace(
            source_checkpoint="synthetic/model",
            cache_path=str(model_path),
            num_hidden_layers=4,
            vision_depth=0,
        ),
    )
    monkeypatch.setattr(config_module.TestSpec, "read", lambda path: spec)

    def probe(args, layers):
        calls.append("sizing")
        return {"status": "fit", "layers": layers, "peak_gib": 1.0}

    def fixture(args, selected, first_oom):
        calls.append("fixture")
        args.test_spec.write_text('{"tests": ["single-step-grads"]}')

    def calibrate(args, *rest):
        calls.append(f"{args.test_id}:calibration")
        destination = args.output_dir / "calibration.json"
        destination.write_text(
            json.dumps(
                {
                    "tolerance": {"selected_gate": 0.005},
                    "criterion": "synthetic",
                    "arms": {},
                }
            )
        )
        if args.test_id == "single-step-optimizer":
            args.test_spec.write_text('{"tests": ["single-step-grads", "single-step-optimizer"]}')
        return destination

    def prepare(args):
        calls.append(f"{args.test_id}:references")
        return 0

    def validate(args, selected):
        calls.append(f"{args.test_id}:validation")
        failure = args.test_id == "single-step-optimizer" and calls.count("single-step-optimizer:validation") == 1
        log = args._diagnostics_dir / "validation.log"
        report_path = args._diagnostics_dir / "validation.json"
        log.write_text("synthetic target failure" if failure else "pass")
        report_path.write_text("{}")
        return {
            "status": "error" if failure else "pass",
            "log": str(log),
            "report": str(report_path),
        }

    monkeypatch.setattr(onboard, "_run_probe", probe)
    monkeypatch.setattr(onboard, "_write_initial_spec", fixture)
    monkeypatch.setattr(onboard, "_render_and_run_calibration", calibrate)
    monkeypatch.setattr(onboard, "_run_local_optimizer_calibration", calibrate)
    monkeypatch.setattr(onboard, "_prepare_references", prepare)
    monkeypatch.setattr(onboard, "_run_final_regression", validate)
    command = [
        "onboard",
        "--config",
        str(config_path),
        "--test-spec",
        str(spec_path),
        "--reference-records-dir",
        str(tmp_path / "records"),
    ]
    assert execution.cmd_onboard(execution.parse_args(command)) == 1
    assert json.loads(spec_path.read_text())["tests"] == ["single-step-grads"]
    first_calls = calls[:]
    assert execution.cmd_onboard(execution.parse_args(command)) == 0
    assert calls == first_calls + [
        "single-step-optimizer:validation",
        "single-step-grads:references",
    ]
    assert json.loads(spec_path.read_text())["tests"] == list(onboard.CALIBRATED_TEST_IDS)
    logs = list((tmp_path / "work" / "logs").glob("onboarding-*/onboarding.log"))
    assert len(logs) == 2
    assert any("Traceback" in path.read_text() and "regression error" in path.read_text() for path in logs)
    assert "Reusing the verified completed phase" in capsys.readouterr().out


@pytest.mark.parametrize(
    "verdict",
    [
        Outcome.PASS,
        Outcome.FAIL,
        Outcome.INAPPLICABLE,
        ReferencePrerequisiteError("missing"),
        RuntimeError("error"),
    ],
)
def test_operator_batch_total_runtime_is_last_after_reports_and_failure_guidance(
    tmp_path, monkeypatch, capsys, verdict
):
    _operator_api_fake_batches(monkeypatch, [verdict])
    ticks = iter([100.0, 165.0, 167.01])
    monkeypatch.setattr(execution.time, "monotonic", lambda: next(ticks))
    result = api.run(
        api.RunRequest(config=_operator_api_CONFIG, tests=(_operator_api_TEST,)),
        api.RunOptions(out=tmp_path),
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "Total run time: 00:01:08"
    assert sum(line.startswith("Total run time:") for line in lines) == 1
    assert f"Selected tests: {_operator_api_TEST}" in lines
    assert any(line.startswith("Report:") for line in lines[:-1])
    if isinstance(verdict, Exception):
        assert lines[-2].startswith("Action required:")
    assert result.elapsed_seconds == 65.0
    assert json.loads((tmp_path / "batch.json").read_text())["elapsed_seconds"] == 65.0


@pytest.mark.parametrize("verdict", [Outcome.PASS, Outcome.FAIL, Outcome.INAPPLICABLE])
def test_compatibility_total_runtime_follows_all_report_and_verdict_output(tmp_path, monkeypatch, capsys, verdict):
    monkeypatch.setattr(execution, "write_report", lambda *args, **kwargs: print("report saved"))
    monkeypatch.setattr(execution.time, "monotonic", lambda: 165.0)
    result = Result("single-step-grads", "synthetic", verdict, summary="synthetic result")
    code = execution._finish([result], {}, tmp_path, configs=1, started=100.0)
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "total run time across 1 config(s): 00:01:05"
    assert sum("total run time" in line for line in lines) == 1
    assert code == (0 if verdict is Outcome.PASS else 1)


@pytest.mark.parametrize("failure", [False, True])
def test_onboarding_total_runtime_is_last_and_retained_in_console_log(tmp_path, monkeypatch, capsys, failure):
    from arctic_platform.correctness.onboarding import onboard

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DSS_CORRECTNESS_LOGDIR", raising=False)
    ticks = iter([100.0, 167.01])
    monkeypatch.setattr(onboard.time, "monotonic", lambda: next(ticks))

    def run(args):
        print("completed phase")
        if failure:
            raise ValueError("synthetic onboarding failure")
        print("spec published")
        return 0

    monkeypatch.setattr(onboard, "run", run)
    args = Namespace(config="synthetic.json")
    assert onboard.logged_run(args) == int(failure)
    assert capsys.readouterr().out.splitlines()[-1] == "Total run time: 00:01:08"
    assert (args._diagnostics_dir / "onboarding.log").read_text().splitlines()[-1] == "Total run time: 00:01:08"
    assert "DSS_CORRECTNESS_LOGDIR" not in os.environ


@pytest.mark.parametrize("destination", ["workspace", "environment", "explicit"])
def test_regression_failure_logs_use_durable_report_directory(tmp_path, monkeypatch, destination):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DSS_CORRECTNESS_LOGDIR", raising=False)
    root = tmp_path / "work" / "logs"
    options = api.RunOptions()
    if destination == "environment":
        root = tmp_path / "retained-logs"
        monkeypatch.setenv("DSS_CORRECTNESS_LOGDIR", str(root))
    elif destination == "explicit":
        root = tmp_path / "explicit-report"
        monkeypatch.setenv("DSS_CORRECTNESS_LOGDIR", str(tmp_path / "unused-logs"))
        options = api.RunOptions(out=root)
    _operator_api_fake_batches(monkeypatch, [RuntimeError("synthetic target failure")])
    result = api.run(
        api.RunRequest(config=_operator_api_CONFIG, tests=(_operator_api_TEST,)),
        options,
    )
    if destination == "explicit":
        assert result.report_dir == root
        assert not (tmp_path / "unused-logs").exists()
    else:
        assert result.report_dir.parent == root
        assert result.report_dir.name.startswith("ap-correctness-run-")
    assert result.exit_code == 1 and result.cases[0]["status"] == "error"
    assert "RuntimeError: synthetic target failure" in (result.report_dir / "full.log").read_text()
    assert json.loads((result.report_dir / "batch.json").read_text())["cases"] == result.cases


@pytest.mark.parametrize("drift", ["display-only", "changed-run-count"])
@pytest.mark.parametrize("formatter", ["duration", "total_duration"])
def test_onboarding_retry_runtime_footer_and_calibration_bar_preserve_only_unchanged_recipe(
    _onboarding_retry_sources, drift, formatter
):
    onboard, args, root = _onboarding_retry_sources
    (root / "onboarding" / "onboard.py").write_text(
        "def logged_run(args):\n    try:\n        return run(args)\n    except Exception:\n        return 1\n"
    )
    (root / "harness" / "runner.py").write_text(
        "if stepping:\n    for start in range(reference_calibration_runs):\n        measure_reference(start)\n"
    )
    onboard._resume_setup(args)
    original = args._resume_dir
    (root / "onboarding" / "onboard.py").write_text(
        "def logged_run(args):\n"
        f"    from arctic_platform.correctness.harness.report import {formatter}\n"
        "    started = time.monotonic()\n"
        "    try:\n"
        "        return run(args)\n"
        "    except Exception:\n"
        "        return 1\n"
        "    finally:\n"
        f'        print(f"Total run time: {{{formatter}(time.monotonic() - started)}}", flush=True)\n'
    )
    runs = "reference_calibration_runs" if drift == "display-only" else "reference_calibration_runs + 1"
    (root / "harness" / "runner.py").write_text(
        "if stepping:\n"
        "    from tqdm.auto import tqdm\n"
        "    with tqdm(total=reference_calibration_runs,\n"
        '              desc=f"Optimizer calibration {arm.name}", unit="run", file=sys.stdout,\n'
        "              dynamic_ncols=True, mininterval=0, miniters=1) as progress:\n"
        f"        for start in range({runs}):\n"
        "            measure_reference(start)\n"
        "            progress.update(1)\n"
    )
    onboard._resume_setup(args)
    assert args._resume_candidates == ([original] if drift == "display-only" else [])


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "00:00:00"),
        (0.001, "00:00:01"),
        (59, "00:00:59"),
        (59.01, "00:01:00"),
        (60, "00:01:00"),
        (3599.01, "01:00:00"),
        (3600, "01:00:00"),
        (90061.01, "25:01:02"),
    ],
)
def test_total_runtime_rounds_up_with_full_hours_minutes_seconds(seconds, expected):
    assert report.total_duration(seconds) == expected


@pytest.fixture
def _automatic_regression_gateway(tmp_path, monkeypatch):
    import requests

    for name in ("ARCTIC_CORTEX_HOST", "ARCTIC_CORTEX_BASE_URL", "ARCTIC_CORTEX_PAT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dss_driver, "_allocation_hostfile", lambda: None)
    is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: False if str(path) == "/job/hostfile" else is_file(path),
    )
    listener = MagicMock()
    listener.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 54321)
    monkeypatch.setattr(dss_driver.socket, "socket", lambda: listener)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(status_code=200))
    process = Mock()
    process.pid = 12345
    process.poll.return_value = None
    process.wait.return_value = 0
    start = Mock(return_value=process)
    monkeypatch.setattr(
        dss_driver,
        "_gateway_process_identity",
        lambda pid: {"pid": pid, "uid": os.getuid(), "start_time": "1"},
    )
    original_popen = dss_driver.subprocess.Popen
    monkeypatch.setattr(
        dss_driver.subprocess,
        "Popen",
        lambda command, *args, **kwargs: (
            start(command, *args, **kwargs)
            if command[:3] == [sys.executable, "-m", "dss.ray_dss.gateway_server"]
            else original_popen(command, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(execution, "_run_vocab_size", lambda *a, **k: 128)
    real_load_local_fixture = cortex_fixture.load_local_fixture
    monkeypatch.setattr(
        cortex_fixture,
        "load_local_fixture",
        lambda backend, spec: {
            "model_name": "synthetic-fixture",
            "fixture_sha256": "f" * 64,
        },
    )
    return SimpleNamespace(
        process=process,
        start=start,
        out=tmp_path / "reports",
        real_load_local_fixture=real_load_local_fixture,
    )


@pytest.mark.parametrize("failure", [None, "health", "identity", "interrupt"])
def test_regression_automatic_gateway_starts_before_real_fixture_health_and_cleans_up(
    _automatic_regression_gateway, monkeypatch, failure
):
    import requests

    state = _automatic_regression_gateway
    first_two = {key: value for key, value in api.registered_tests().items() if key.startswith("single-step-")}
    monkeypatch.setattr(api, "registered_tests", lambda: first_two)
    monkeypatch.setattr(cortex_fixture, "load_local_fixture", state.real_load_local_fixture)
    monkeypatch.setattr(cortex_fixture, "require_data_fast_path", lambda path, **kw: Path(path))
    session = MagicMock()
    session.__enter__.return_value = session
    monkeypatch.setattr(requests, "Session", lambda: session)
    events = []

    def health(url, **options):
        state.start.assert_called_once()
        state.process.terminate.assert_not_called()
        assert url == "http://127.0.0.1:54321/health"
        assert session.trust_env is False and options["allow_redirects"] is False
        events.append("health")
        if failure == "interrupt":
            raise KeyboardInterrupt()
        return SimpleNamespace(
            status_code=503 if failure == "health" else 200,
            json=lambda: {"status": "OK"},
        )

    def describe(spec, path):
        assert events == ["health"]
        events.append("identity")
        if failure == "identity":
            raise ValueError("synthetic fixture identity mismatch")
        return {"fixture_sha256": "f" * 64}

    def vocabulary(*args, **options):
        assert events == ["health", "identity"]
        events.append("vocabulary")
        return 128

    def execute(cfg, spec, arms, **options):
        assert events == ["health", "identity", "vocabulary"]
        events.append("execute")
        return [Result(test, cfg.config_id, Outcome.PASS, arm=arm.name) for test in options["select"] for arm in arms]

    session.get.side_effect = health
    monkeypatch.setattr(cortex_fixture, "describe_fixture", describe)
    monkeypatch.setattr(execution, "_run_vocab_size", vocabulary)
    result = api.run(
        api.RunRequest(config=_operator_api_CONFIG),
        api.RunOptions(out=state.out),
        executor=execute,
    )
    expected = {
        None: "pass",
        "health": "blocked",
        "identity": "blocked",
        "interrupt": "error",
    }
    assert [case["status"] for case in result.cases] == [expected[failure]] * 2
    assert result.exit_code == (0 if failure is None else 1)
    assert events == (
        ["health", "identity", "vocabulary", "execute"]
        if failure is None
        else ["health", "identity"] if failure == "identity" else ["health"]
    )
    state.process.terminate.assert_called_once()
    state.process.wait.assert_called_once_with(timeout=120)
    assert (state.out / "full.log").is_file() and (state.out / "report.json").is_file()


@pytest.mark.parametrize("proxy_name", ["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"])
def test_gateway_readiness_ignores_proxy_environment(_automatic_regression_gateway, monkeypatch, proxy_name):
    import requests

    state = _automatic_regression_gateway
    for name in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(proxy_name, "http://proxy.invalid:3128")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setattr(requests, "get", requests.api.get)
    observed = []

    def send(adapter, request, **options):
        observed.append(options["proxies"])
        assert request.url == "http://127.0.0.1:54321/health"
        assert not any(options["proxies"].values()), "loopback readiness inherited a proxy"
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    with dss_driver.local_cortex_gateway(state.out) as url:
        assert url == "http://127.0.0.1:54321"
    assert len(observed) == 1
    state.process.terminate.assert_called_once()
    state.process.wait.assert_called_once_with(timeout=120)


@pytest.mark.parametrize("failure", [None, "comparison", "executor", "interrupt", "reference"])
def test_regression_automatic_gateway_single_command_runs_both_checks_and_closes_owned_service(
    _automatic_regression_gateway, monkeypatch, failure
):
    state = _automatic_regression_gateway
    first_two = {key: value for key, value in api.registered_tests().items() if key.startswith("single-step-")}
    monkeypatch.setattr(api, "registered_tests", lambda: first_two)
    observed = []

    def execute(cfg, spec, arms, **options):
        assert state.process.terminate.call_count == 0
        assert options["local_cortex_fixture"] is True
        backend = options["client_backend"]
        assert backend.base_url == "http://127.0.0.1:54321"
        assert backend.database == backend.schema_ == "dev"
        assert options["select"] == ["single-step-grads", "single-step-optimizer"]
        observed.extend(options["select"])
        if failure == "executor":
            raise RuntimeError("synthetic execution failed")
        if failure == "interrupt":
            raise KeyboardInterrupt()
        if failure == "reference":
            raise ReferencePrerequisiteError("synthetic prepared record missing; onboard first")
        return [
            Result(
                test,
                cfg.config_id,
                Outcome.FAIL if failure == "comparison" else Outcome.PASS,
                arm=arm.name,
            )
            for test in options["select"]
            for arm in arms
        ]

    result = api.run(
        api.RunRequest(config=_operator_api_CONFIG),
        api.RunOptions(out=state.out),
        executor=execute,
    )
    assert observed == ["single-step-grads", "single-step-optimizer"]
    assert result.exit_code == (0 if failure is None else 1)
    expected = {
        None: "pass",
        "comparison": "fail",
        "executor": "error",
        "interrupt": "error",
        "reference": "blocked",
    }
    assert [case["status"] for case in result.cases] == [expected[failure]] * 2
    assert len(result.results) == (4 if failure in (None, "comparison") else 2)
    state.process.terminate.assert_called_once()
    state.process.wait.assert_called_once_with(timeout=120)
    assert state.start.call_count == 1
    assert (state.out / "cases" / _operator_api_CONFIG / "0" / "gateway.log").is_file()
    assert (state.out / "full.log").is_file() and (state.out / "report.json").is_file()


@pytest.mark.parametrize("source", ["argument", "environment"])
def test_regression_automatic_gateway_does_not_replace_explicit_endpoint(
    _automatic_regression_gateway, monkeypatch, source
):
    state = _automatic_regression_gateway
    monkeypatch.setattr(
        execution,
        "local_cortex_gateway",
        lambda *a: pytest.fail("explicit service replaced"),
    )
    values = {"out": state.out, "local_cortex_fixture": True}
    if source == "argument":
        values["base_url"] = "http://127.0.0.1:8003"
    else:
        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://127.0.0.1:8003")

    def execute(cfg, *a, **options):
        assert options["client_backend"].base_url == "http://127.0.0.1:8003"
        return [Result("single-step-grads", cfg.config_id, Outcome.PASS)]

    result = api.run(
        api.RunRequest(config=_operator_api_CONFIG, tests=("single-step-grads",)),
        api.RunOptions(**values),
        executor=execute,
    )
    assert result.exit_code == 0
    state.start.assert_not_called()
    state.process.terminate.assert_not_called()


@pytest.mark.parametrize("mode", ["host", "pat", "later-check", "rl", "publication", "full-model", "onprem"])
def test_regression_automatic_gateway_respects_authentication_and_workload_boundaries(
    _automatic_regression_gateway, monkeypatch, mode
):
    args = execution.parse_args(["run", "--config", _operator_api_CONFIG, "--test", "single-step-grads"])
    if mode == "host":
        monkeypatch.setenv("ARCTIC_CORTEX_HOST", "account.invalid")
    elif mode == "pat":
        monkeypatch.setenv("ARCTIC_CORTEX_PAT", "synthetic-not-a-credential")
    elif mode == "later-check":
        args.test = ["inference-checkpoint-loss"]
    elif mode == "rl":
        monkeypatch.setattr(execution, "load_config", lambda *a: SimpleNamespace(is_rl=True))
    elif mode == "publication":
        args.cortex_fixture = Path("publication.json")
    elif mode == "full-model":
        args.local_gateway_full_model = True
    else:
        args.backend = "onprem"
    if mode == "pat":
        with pytest.raises(ValueError, match="unauthenticated"):
            execution._configure_regression_target(args)
    else:
        assert execution._configure_regression_target(args) is args
    assert not getattr(args, "_automatic_cortex_gateway", False)
    _automatic_regression_gateway.start.assert_not_called()


@pytest.mark.parametrize("failure", ["exit", "timeout", "shutdown"])
def test_regression_automatic_gateway_startup_and_cleanup_failures_are_saved_and_not_passes(
    _automatic_regression_gateway, monkeypatch, failure
):
    import requests

    state = _automatic_regression_gateway
    if failure == "exit":
        state.process.poll.return_value = 7
        state.process.returncode = 7
    elif failure == "timeout":
        ticks = iter([0, 1, 121])
        monkeypatch.setattr(dss_driver.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(dss_driver.time, "sleep", lambda *a: None)
        monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(status_code=503))
    else:
        state.process.pid = 12345
        state.process.wait.side_effect = [subprocess.TimeoutExpired("gateway", 120), 0]
        monkeypatch.setattr(dss_driver.os, "killpg", Mock())
    if failure in {"exit", "timeout"}:
        with pytest.raises(RuntimeError, match="exited|did not become ready"):
            with dss_driver.local_cortex_gateway(state.out):
                pytest.fail("unready gateway reached target work")
    else:
        result = api.run(
            api.RunRequest(config=_operator_api_CONFIG, tests=("single-step-grads",)),
            api.RunOptions(out=state.out),
            executor=lambda cfg, *a, **k: [Result("single-step-grads", cfg.config_id, Outcome.PASS)],
        )
        assert result.exit_code == 1 and result.stopped_unsafe
        assert result.cases[0]["status"] == "error"
        assert "did not shut down cleanly" in (state.out / "full.log").read_text()
        dss_driver.os.killpg.assert_called_once_with(12345, dss_driver.signal.SIGKILL)
    if failure != "shutdown":
        assert (state.out / "gateway.log").is_file()
    else:
        assert state.start.call_count == 1


@pytest.mark.parametrize("operation", ["terminate", "wait", "killpg"])
def test_regression_gateway_cleanup_exceptions_stop_following_work(
    _automatic_regression_gateway, monkeypatch, operation
):
    state = _automatic_regression_gateway
    state.process.pid = 12345
    if operation == "terminate":
        state.process.terminate.side_effect = ValueError("synthetic termination failure")
    elif operation == "wait":
        state.process.wait.side_effect = FileNotFoundError("synthetic wait failure")
    else:
        state.process.wait.side_effect = subprocess.TimeoutExpired("gateway", 120)
        monkeypatch.setattr(
            dss_driver.os,
            "killpg",
            Mock(side_effect=OSError("synthetic group kill failure")),
        )
    result = api.run(
        api.RunRequest(config=_operator_api_CONFIG, tests=("single-step-grads",)),
        api.RunOptions(out=state.out),
        executor=lambda cfg, *a, **k: [Result("single-step-grads", cfg.config_id, Outcome.PASS)],
    )
    assert result.exit_code == 1 and result.stopped_unsafe
    assert result.cases[0]["status"] == "error"
    assert "UnsafeCleanupError" in (state.out / "full.log").read_text()


def test_regression_gateway_disappearing_group_still_reaps_owned_child(_automatic_regression_gateway, monkeypatch):
    state = _automatic_regression_gateway
    state.process.pid = 12345
    state.process.wait.side_effect = [subprocess.TimeoutExpired("gateway", 120), 0]
    kill = Mock(side_effect=ProcessLookupError("synthetic group already exited"))
    monkeypatch.setattr(dss_driver.os, "killpg", kill)
    with pytest.raises(UnsafeCleanupError, match="did not shut down cleanly"):
        with dss_driver.local_cortex_gateway(state.out):
            pass
    kill.assert_called_once_with(12345, dss_driver.signal.SIGKILL)
    assert state.process.wait.call_count == 2
    assert state.process.wait.call_args_list[0].kwargs == {"timeout": 120}
    assert state.process.wait.call_args_list[1].kwargs == {}


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
@pytest.mark.parametrize("phase", ["startup", "onboarding", "regression"])
def test_owned_gateway_stop_signals_clean_up_and_restore_handlers(
    _automatic_regression_gateway, monkeypatch, capsys, signum, phase
):
    import requests

    from arctic_platform.correctness.onboarding import onboard

    state = _automatic_regression_gateway
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    if phase == "startup":
        monkeypatch.setattr(requests, "get", lambda *a, **k: signal.raise_signal(signum))
        context = dss_driver.local_cortex_gateway(state.out)
    elif phase == "onboarding":
        context = onboard._validation_gateway(Namespace(_automatic_cortex_gateway=True, output_dir=state.out))
    else:
        context = dss_driver.local_cortex_gateway(state.out)
    with pytest.raises(KeyboardInterrupt, match=f"received {signal.Signals(signum).name}"):
        with context:
            signal.raise_signal(signum)
    state.process.terminate.assert_called_once()
    state.process.wait.assert_called_once_with(timeout=120)
    assert {number: signal.getsignal(number) for number in previous} == previous
    output = capsys.readouterr().out
    assert "Stopping owned Cortex-training gateway" in output
    assert "Owned Cortex-training gateway stopped and reaped." in output


def test_owned_gateway_repeated_stop_does_not_interrupt_cleanup(
    _automatic_regression_gateway,
):
    state = _automatic_regression_gateway
    previous = signal.getsignal(signal.SIGTERM)
    state.process.terminate.side_effect = lambda: signal.raise_signal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt, match="received SIGTERM"):
        with dss_driver.local_cortex_gateway(state.out):
            signal.raise_signal(signal.SIGTERM)
    state.process.wait.assert_called_once_with(timeout=120)
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_owned_gateway_real_process_signal_reaps_child(tmp_path, signum):
    script = """
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import requests
import torch
from arctic_platform.correctness.harness import dss_driver

directory = Path(sys.argv[1])
original_popen = subprocess.Popen
def start_gateway(*args, **kwargs):
    return original_popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdout=kwargs["stdout"], stderr=kwargs["stderr"], start_new_session=True, env=kwargs["env"],
    )
dss_driver.subprocess.Popen = start_gateway
dss_driver._allocation_hostfile = lambda: None
dss_driver._gateway_state_directory = lambda: directory / "ownership"
dss_driver._legacy_owned_gateways = lambda: []
torch.cuda.device_count = lambda: 1
requests.get = lambda *args, **kwargs: SimpleNamespace(status_code=200)
with dss_driver.local_cortex_gateway(directory):
    (directory / "ready").write_text(str(os.getpid()))
    signal.pause()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = ""
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "ready").is_file():
            if process.poll() is not None:
                output, _ = process.communicate(timeout=5)
                pytest.fail(f"signal test parent exited before gateway startup:\n{output}")
            assert time.monotonic() < deadline, "signal test gateway startup timed out"
            time.sleep(0.05)
        process.send_signal(signum)
        output, _ = process.communicate(timeout=15)
        assert process.returncode != 0
        assert f"received {signal.Signals(signum).name}" in output
        assert "Owned Cortex-training gateway stopped and reaped." in output
        marker = "Stopping owned Cortex-training gateway (PID "
        child_pid = int(output.split(marker, 1)[1].split(")", 1)[0])
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)


@pytest.mark.parametrize("has_spec", [False, True])
def test_onboarding_numbering_includes_only_queued_checks(_onboarding_prerequisites, monkeypatch, capsys, has_spec):
    state = _onboarding_prerequisites
    state.args.test_id = "single-step-optimizer"
    if has_spec:
        state.args.test_spec.write_text("{}")
    seen = []

    def parent(child):
        seen.append(child.test_id)
        child.test_spec.write_text("{}")
        return 0

    monkeypatch.setattr(state.onboard, "_parent", parent)
    assert state.onboard._onboard_configuration(state.args) == 0
    output = capsys.readouterr().out
    if has_spec:
        assert seen == ["single-step-optimizer"]
        assert "### Onboarding 1/1: Single Step Optimizer ###" in output
        assert "Single Step Grads ###" not in output
    else:
        assert seen == ["single-step-grads", "single-step-optimizer"]
        assert "### Onboarding 1/2: Single Step Grads ###" in output
        assert "### Onboarding 2/2: Single Step Optimizer ###" in output


def test_onboarding_numbering_scales_beyond_two_checks(_onboarding_prerequisites, monkeypatch, capsys):
    from arctic_platform.correctness.harness import registry

    state = _onboarding_prerequisites
    selected = ["single-step-grads", "single-step-optimizer", "synthetic-check"]
    monkeypatch.setattr(registry, "registered_tests", lambda: dict.fromkeys(selected))
    monkeypatch.setattr(registry, "onboarding_inapplicable_reason", lambda *a: None)
    monkeypatch.setattr(state.onboard, "_onboarding_preflight", lambda *a: None)

    def parent(child):
        child.test_spec.write_text("{}")
        return 0

    monkeypatch.setattr(state.onboard, "_parent", parent)
    monkeypatch.setattr(state.onboard, "_prepare_references", lambda *a: 0)
    assert state.onboard._onboard_configuration(state.args) == 0
    output = capsys.readouterr().out
    assert "### Onboarding 1/3: Single Step Grads ###" in output
    assert "### Onboarding 2/3: Single Step Optimizer ###" in output
    assert "### Onboarding 3/3: Synthetic Check ###" in output


@pytest.fixture(autouse=True)
def _isolated_gateway_ownership(tmp_path, monkeypatch):
    scanner = dss_driver._legacy_owned_gateways
    monkeypatch.setattr(dss_driver, "_gateway_state_directory", lambda: tmp_path / "gateway-ownership")
    monkeypatch.setattr(dss_driver, "_legacy_owned_gateways", lambda: [])
    return scanner


def test_onboarding_recovers_orphan_before_reference_prerequisites_and_reports_config(
    _onboarding_prerequisites, monkeypatch, capsys
):
    state = _onboarding_prerequisites
    for name in ("ARCTIC_CORTEX_HOST", "ARCTIC_CORTEX_BASE_URL", "ARCTIC_CORTEX_PAT"):
        monkeypatch.delenv(name, raising=False)
    recovered = []

    def recover():
        assert state.calls == []
        recovered.append(True)

    monkeypatch.setattr(dss_driver, "recover_local_cortex_gateway", recover)
    state.onboard._onboarding_preflight(state.args, state.config)
    assert recovered == [True] and len(state.calls) == 1
    output = capsys.readouterr().out
    assert f"Config file: {state.args.config.resolve()}" in output
    assert "Model source: Qwen/synthetic" in output
    assert "Configured sequence length: 128 tokens" in output


def test_onboarding_external_service_does_not_recover_local_gateway(_onboarding_prerequisites, monkeypatch):
    state = _onboarding_prerequisites
    state.args.backend = "cortex"
    state.args.base_url = "http://127.0.0.1:8003"
    monkeypatch.setattr(
        dss_driver,
        "recover_local_cortex_gateway",
        lambda: pytest.fail("external service recovery"),
    )
    state.onboard._onboarding_preflight(state.args, state.config)
    assert len(state.calls) == 1


@pytest.fixture
def _stale_gateway_record(tmp_path, monkeypatch):
    root = dss_driver._gateway_state_directory()
    root.mkdir()
    path = root / "gateway.json"
    owner = {"pid": 999001, "start_time": "101", "uid": os.getuid()}
    gateway = {"pid": 999002, "start_time": "202", "uid": os.getuid()}
    record = {
        "schema": "correctness-owned-gateway-v1",
        "owner": owner,
        "gateway": gateway,
        "token": "synthetic-owner-token",
        "log": str(tmp_path / "old-gateway.log"),
    }
    path.write_text(json.dumps(record))
    live = {owner["pid"]: None, gateway["pid"]: gateway}
    monkeypatch.setattr(dss_driver, "_gateway_process_identity", lambda pid: live.get(pid))
    read_bytes = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda file: (
            b"DSS_CORRECTNESS_GATEWAY_OWNER=synthetic-owner-token\0"
            if str(file) == f"/proc/{gateway['pid']}/environ"
            else read_bytes(file)
        ),
    )
    monkeypatch.setattr(dss_driver.os, "getpgid", lambda pid: pid)
    kill = Mock(side_effect=lambda pid, signum: live.update({pid: None}))
    monkeypatch.setattr(dss_driver.os, "kill", kill)
    monkeypatch.setattr(dss_driver.os, "killpg", Mock())
    return SimpleNamespace(path=path, record=record, owner=owner, gateway=gateway, live=live, kill=kill)


def test_stale_owned_gateway_is_stopped_before_replacement(_stale_gateway_record, capsys):
    state = _stale_gateway_record
    with dss_driver._owned_gateway_slot() as path:
        assert path == state.path and not path.exists()
        state.kill.assert_called_once_with(state.gateway["pid"], signal.SIGTERM)
    output = capsys.readouterr().out
    assert "Stopping stale owned Cortex-training gateway" in output
    assert "Stale owned Cortex-training gateway stopped." in output


@pytest.mark.parametrize("state_kind", ["gone", "reused-pid"])
def test_stale_gateway_missing_or_reused_pid_is_not_signalled(_stale_gateway_record, state_kind):
    state = _stale_gateway_record
    state.live[state.gateway["pid"]] = None if state_kind == "gone" else {**state.gateway, "start_time": "different"}
    dss_driver.recover_local_cortex_gateway()
    state.kill.assert_not_called()
    assert not state.path.exists()


@pytest.mark.parametrize("fault", ["live-owner", "token", "group", "schema", "user", "json"])
def test_stale_gateway_unproven_ownership_refuses_startup(_stale_gateway_record, monkeypatch, fault):
    state = _stale_gateway_record
    if fault == "live-owner":
        state.live[state.owner["pid"]] = state.owner
    elif fault == "token":
        state.record["token"] = "different"
    elif fault == "group":
        monkeypatch.setattr(dss_driver.os, "getpgid", lambda pid: pid + 1)
    elif fault == "schema":
        state.record["schema"] = "unknown"
    elif fault == "user":
        state.record["gateway"]["uid"] += 1
    state.path.write_text("{" if fault == "json" else json.dumps(state.record))
    monkeypatch.setattr(
        dss_driver.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("unsafe replacement started"),
    )
    with pytest.raises(UnsafeCleanupError, match="still owns|cannot safely recover"):
        with dss_driver.local_cortex_gateway(state.path.parent / "next"):
            pytest.fail("unsafe replacement yielded")
    state.kill.assert_not_called()
    assert state.path.exists()


def test_stale_gateway_forced_shutdown_blocks_replacement(_stale_gateway_record, monkeypatch):
    state = _stale_gateway_record
    state.kill.side_effect = None
    ticks = iter([0, 121])
    monkeypatch.setattr(dss_driver.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        dss_driver.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("unsafe replacement started"),
    )
    with pytest.raises(UnsafeCleanupError, match="required forced shutdown"):
        with dss_driver.local_cortex_gateway(state.path.parent / "next"):
            pytest.fail("unsafe replacement yielded")
    state.kill.assert_called_once_with(state.gateway["pid"], signal.SIGTERM)
    dss_driver.os.killpg.assert_called_once_with(state.gateway["pid"], signal.SIGKILL)
    assert state.path.exists()


def test_owned_gateway_lock_rejects_concurrent_commands():
    with dss_driver._owned_gateway_slot():
        with pytest.raises(UnsafeCleanupError, match="another correctness command"):
            dss_driver.recover_local_cortex_gateway()


@pytest.mark.parametrize("layout", ["onboarding", "regression"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "active-parent",
        "other-command",
        "public-host",
        "other-log",
        "no-skill",
        "no-session",
        "other-user",
    ],
)
def test_legacy_gateway_only_matching_orphan_recovered(
    tmp_path, monkeypatch, _isolated_gateway_ownership, layout, fault
):
    monkeypatch.setattr(dss_driver, "_legacy_owned_gateways", _isolated_gateway_ownership)
    root = tmp_path / "proc"
    process = root / "999003"
    (process / "fd").mkdir(parents=True)
    checkout = tmp_path / "checkout"
    skill = checkout / "skills" / "correctness" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("correctness")
    log_path = (
        checkout / "work" / "logs" / "onboarding-old" / "single-step-grads" / "gateway.log"
        if layout == "onboarding"
        else checkout / "work" / "reports" / "both" / "cases" / "tiny" / "0" / "gateway.log"
    )
    log_path.parent.mkdir(parents=True)
    log_path.write_text("")
    (process / "cwd").symlink_to(checkout, target_is_directory=True)
    (process / "fd" / "1").symlink_to(log_path)
    (process / "fd" / "2").symlink_to(log_path)
    command = [
        sys.executable,
        "-m",
        "dss.ray_dss.gateway_server",
        "--host",
        "127.0.0.1",
        "--port",
        "12345",
        "--backend",
        "gpu",
        "--hostfile",
        "/job/hostfile",
    ]
    fields = ["S", "1", "999003", "999003"] + ["0"] * 15 + ["123456"]
    if fault == "active-parent":
        fields[1] = "555"
    elif fault == "other-command":
        command[2] = "unrelated.service"
    elif fault == "public-host":
        command[4] = "0.0.0.0"
    elif fault == "other-log":
        (process / "fd" / "1").unlink()
        (process / "fd" / "1").symlink_to(tmp_path / "external-gateway.log")
    elif fault == "no-skill":
        skill.unlink()
    elif fault == "other-user":
        uid = os.getuid()
        monkeypatch.setattr(dss_driver.os, "getuid", lambda: uid + 1)
    (process / "cmdline").write_bytes(b"\0".join(value.encode() for value in command) + b"\0")
    (process / "stat").write_text("999003 (python) " + " ".join(fields))
    monkeypatch.setattr(dss_driver, "_GATEWAY_PROC_ROOT", root)
    monkeypatch.setattr(dss_driver.os, "getpgid", lambda pid: pid + 1 if fault == "no-session" else pid)
    identity = dss_driver._gateway_process_identity(999003)
    kill = Mock(side_effect=lambda *a: (process / "stat").unlink())
    monkeypatch.setattr(dss_driver.os, "kill", kill)
    dss_driver.recover_local_cortex_gateway()
    if fault is None:
        kill.assert_called_once_with(identity["pid"], signal.SIGTERM)
    else:
        kill.assert_not_called()


@pytest.mark.parametrize("command", ["onboarding", "regression"])
def test_owned_gateway_registry_records_launch_and_clears_after_cleanup(_automatic_regression_gateway, command):
    from arctic_platform.correctness.onboarding import onboard

    state = _automatic_regression_gateway
    context = (
        onboard._validation_gateway(Namespace(_automatic_cortex_gateway=True, output_dir=state.out))
        if command == "onboarding"
        else dss_driver.local_cortex_gateway(state.out)
    )
    path = dss_driver._gateway_state_directory() / "gateway.json"
    with context:
        record = json.loads(path.read_text())
        assert record["owner"]["pid"] == os.getpid()
        assert record["gateway"]["pid"] == state.process.pid
        assert record["token"] == state.start.call_args.kwargs["env"]["DSS_CORRECTNESS_GATEWAY_OWNER"]
        assert Path(record["log"]).is_file()
    assert not path.exists()


@pytest.mark.parametrize("command", ["onboarding", "regression"])
def test_owned_gateway_real_process_sigkill_recovery_before_replacement(tmp_path, command):
    script = """
import json
import os
import signal
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import requests
import torch
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.onboarding import onboard

root, mode, command = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
dss_driver._gateway_state_directory = lambda: root / "ownership"
dss_driver._allocation_hostfile = lambda: None
dss_driver._legacy_owned_gateways = lambda: []
torch.cuda.device_count = lambda: 1
requests.get = lambda *args, **kwargs: SimpleNamespace(status_code=200)
original_popen = subprocess.Popen
def start_gateway(*args, **kwargs):
    return original_popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdout=kwargs["stdout"], stderr=kwargs["stderr"], start_new_session=True, env=kwargs["env"],
    )
dss_driver.subprocess.Popen = start_gateway
directory = root / mode
context = (onboard._validation_gateway(Namespace(_automatic_cortex_gateway=True, output_dir=directory))
           if command == "onboarding" else dss_driver.local_cortex_gateway(directory))
with context:
    record = json.loads((root / "ownership" / "gateway.json").read_text())
    (root / (mode + "-ready")).write_text(str(record["gateway"]["pid"]))
    if mode == "old":
        signal.pause()
"""
    old = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), "old", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    orphan_pid = None
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "old-ready").is_file():
            if old.poll() is not None:
                output, _ = old.communicate(timeout=5)
                pytest.fail(f"orphan test exited before startup:\n{output}")
            assert time.monotonic() < deadline, "orphan test startup timed out"
            time.sleep(0.05)
        orphan_pid = int((tmp_path / "old-ready").read_text())
        old.kill()
        old.communicate(timeout=5)
        assert dss_driver._gateway_process_identity(orphan_pid) is not None
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), "new", command],
            capture_output=True,
            text=True,
            timeout=40,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Stopping stale owned Cortex-training gateway" in result.stdout
        assert "Stale owned Cortex-training gateway stopped." in result.stdout
        assert "Owned Cortex-training gateway stopped and reaped." in result.stdout
        assert dss_driver._gateway_process_identity(orphan_pid) is None
        replacement_pid = int((tmp_path / "new-ready").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(replacement_pid, 0)
        assert not (tmp_path / "ownership" / "gateway.json").exists()
    finally:
        if old.poll() is None:
            old.kill()
            old.communicate(timeout=5)
        if orphan_pid is not None and dss_driver._gateway_process_identity(orphan_pid) is not None:
            os.kill(orphan_pid, signal.SIGKILL)
