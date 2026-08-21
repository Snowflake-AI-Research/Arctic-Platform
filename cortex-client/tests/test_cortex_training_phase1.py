"""Parity tests for the additive Cortex Training public facade."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

import cortex_training
import cortex_training.cli as cortex_cli
import cortex_training.peft as cortex_peft
import dss_client
import dss_client.peft as dss_peft
import dss_neutrino_cli as legacy_cli
from tests.test_neutrino_cli import FakeClient, _base_args


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


@pytest.mark.parametrize("argv", _PARSER_CASES)
def test_all_existing_commands_parse_identically(argv):
    legacy = vars(legacy_cli.parse_args(argv))
    canonical = vars(cortex_cli.parse_args(argv))
    assert canonical == legacy


def _run_list(module):
    captured_args = {}
    client = FakeClient()
    stdout = io.StringIO()
    stderr = io.StringIO()

    def factory(args):
        captured_args.update(vars(args))
        return client

    result = module.main(
        _base_args() + ["list", "--status", "running"],
        client_factory=factory,
        stdout=stdout,
        stderr=stderr,
    )
    return result, stdout.getvalue(), stderr.getvalue(), captured_args, client


def test_success_output_exit_code_and_client_arguments_are_identical():
    legacy = _run_list(legacy_cli)
    canonical = _run_list(cortex_cli)

    assert canonical[:4] == legacy[:4]
    assert canonical[4].status_filter == legacy[4].status_filter == "running"


def _run_invalid_submit(module, path: Path):
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = module.main(
        _base_args() + ["submit", str(path)],
        client_factory=lambda _args: FakeClient(),
        stdout=stdout,
        stderr=stderr,
    )
    return result, stdout.getvalue(), stderr.getvalue()


def test_error_output_and_exit_code_are_identical(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")

    assert _run_invalid_submit(cortex_cli, invalid) == _run_invalid_submit(
        legacy_cli,
        invalid,
    )


def test_python_facade_reexports_existing_objects():
    assert cortex_training.NeutrinoClient is dss_client.NeutrinoClient
    assert cortex_training.NeutrinoTrainingEngine is dss_client.NeutrinoTrainingEngine
    assert cortex_training.CortexTrainingClient is dss_client.NeutrinoClient
    assert cortex_training.CortexTrainingEngine is dss_client.NeutrinoTrainingEngine
    assert cortex_training.wire is dss_client.wire
    assert (
        cortex_peft.normalize_lora_peft_config
        is dss_peft.normalize_lora_peft_config
    )


def test_existing_environment_and_login_paths_remain_canonical(monkeypatch, tmp_path):
    monkeypatch.setenv("NEUTRINO_DATABASE", "LEGACY_DB")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    args = cortex_cli.parse_args(["--base-url", "http://localhost:8084", "list"])

    assert args.database == "LEGACY_DB"
    assert legacy_cli._login_state_path() == tmp_path / "dss-neutrino" / "login.json"


def test_canonical_help_adds_tui_without_changing_legacy_help():
    canonical_choices = next(
        action.choices
        for action in cortex_cli.build_parser()._actions
        if getattr(action, "choices", None)
    )
    legacy_choices = next(
        action.choices
        for action in legacy_cli.build_parser()._actions
        if getattr(action, "choices", None)
    )

    assert "tui" in canonical_choices
    assert "tui" not in legacy_choices


def test_canonical_tui_delegates_without_resolving_connection(monkeypatch):
    captured = {}

    def run_tui(argv, *, prog):
        captured["argv"] = argv
        captured["prog"] = prog
        return 17

    monkeypatch.setattr(cortex_cli, "_run_tui", run_tui)

    result = cortex_cli.main(
        ["--config", "config.json", "tui", "job-1", "--sub-job-id", "sub-1"]
    )

    assert result == 17
    assert captured == {
        "argv": ["--config", "config.json", "job-1", "--sub-job-id", "sub-1"],
        "prog": "cortex-training tui",
    }
