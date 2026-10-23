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

"""Tests for the canonical Cortex Training public surface."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

import cortex_training
import cortex_training._cli as cli_implementation
import cortex_training.cli as cortex_cli
from tests.test_cli import FakeClient
from tests.test_cli import _base_args

_PARSER_CASES = [
    _base_args() + ["submit", "job.json"],
    _base_args() + ["get", "job-1"],
    _base_args() + ["checkpoints", "job-1"],
    _base_args() + ["list", "--status", "running"],
    _base_args() + ["capacity"],
    _base_args() + ["cancel", "job-1"],
    _base_args() + ["wait", "job-1"],
    _base_args() + ["--job", "job-1", "fwd-bwd", "batch.json"],
    _base_args() + ["--job", "job-1", "step", "--lr", "0.001"],
    _base_args() + ["--job", "job-1", "load", "checkpoint-1"],
    _base_args() + ["--job", "job-1", "generate", "generate.json"],
    _base_args() + ["--job", "job-1", "weight-sync"],
    _base_args() + ["download-log", "job-1"],
    ["login", "--config", "config.json"],
]
_COMMANDS = {
    "submit",
    "get",
    "checkpoints",
    "list",
    "capacity",
    "cancel",
    "wait",
    "fwd-bwd",
    "step",
    "load",
    "generate",
    "weight-sync",
    "download-log",
    "login",
}


@pytest.mark.parametrize("argv", _PARSER_CASES)
def test_all_commands_parse(argv):
    expected = next(argument for argument in argv if argument in _COMMANDS)
    assert cortex_cli.parse_args(argv).command == expected


def _run_list():
    captured_args = {}
    client = FakeClient()
    stdout = io.StringIO()
    stderr = io.StringIO()

    def factory(args):
        captured_args.update(vars(args))
        return client

    result = cortex_cli.main(
        _base_args() + ["list", "--status", "running"],
        client_factory=factory,
        stdout=stdout,
        stderr=stderr,
    )
    return result, stdout.getvalue(), stderr.getvalue(), captured_args, client


def test_success_output_exit_code_and_client_arguments():
    result, stdout, stderr, args, client = _run_list()

    assert result == 0
    assert stderr == ""
    assert '"job_id": "j1"' in stdout
    assert args["command"] == "list"
    assert client.status_filter == "running"


def _run_invalid_submit(path: Path):
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = cortex_cli.main(
        _base_args() + ["submit", str(path)],
        client_factory=lambda _args: FakeClient(),
        stdout=stdout,
        stderr=stderr,
    )
    return result, stdout.getvalue(), stderr.getvalue()


def test_error_output_and_exit_code(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")

    result, stdout, stderr = _run_invalid_submit(invalid)

    assert result == 1
    assert stdout == ""
    assert stderr.startswith("error: invalid JSON")


def test_python_package_exports_canonical_objects():
    from cortex_training.client import CortexTrainingClient
    from cortex_training.engine import CortexTrainingEngine

    assert cortex_training.CortexTrainingClient is CortexTrainingClient
    assert cortex_training.CortexTrainingEngine is CortexTrainingEngine
    assert cortex_training.wire.WIRE_FORMAT_VERSION == "DSSST1"


def test_canonical_environment_and_login_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("CORTEX_TRAINING_DATABASE", "TRAINING_DB")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    args = cortex_cli.parse_args(["--base-url", "http://localhost:8084", "list"])

    assert args.database == "TRAINING_DB"
    assert cli_implementation._login_state_path() == tmp_path / "cortex-training" / "login.json"


def test_help_includes_tui():
    choices = next(action.choices for action in cortex_cli.build_parser()._actions if getattr(action, "choices", None))

    assert "tui" in choices


def test_tui_delegates_without_resolving_connection(monkeypatch):
    captured = {}

    def run_tui(argv, *, prog):
        captured["argv"] = argv
        captured["prog"] = prog
        return 17

    monkeypatch.setattr(cortex_cli, "_run_tui", run_tui)

    result = cortex_cli.main(["--config", "config.json", "tui", "job-1", "--sub-job-id", "sub-1"])

    assert result == 17
    assert captured == {
        "argv": ["--config", "config.json", "job-1", "--sub-job-id", "sub-1"],
        "prog": "cortex-training tui",
    }
