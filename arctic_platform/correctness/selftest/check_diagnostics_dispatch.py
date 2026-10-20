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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU controls for diagnostic discovery, entry-point compatibility and executor isolation."""

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

from arctic_platform.correctness.diagnostics import __main__ as dispatch


def test_catalog_covers_existing_commands_but_not_helper_modules():
    directory = Path(dispatch.__file__).parent
    commands = set()
    for path in directory.glob("*.py"):
        if path.name == "__main__.py":
            continue
        tree = ast.parse(path.read_text())
        if any(
            isinstance(node, ast.If)
            and ast.dump(node.test) == ast.dump(ast.parse('__name__ == "__main__"', mode="eval").body)
            for node in tree.body
        ):
            commands.add(path.stem)
    assert set(dispatch.DIAGNOSTICS) == commands
    assert commands == {"lm_head_chunking"}


def test_list_is_sorted_and_does_not_execute_modules(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("listing must not load an experiment")

    monkeypatch.setattr(dispatch.runpy, "run_module", forbidden)
    assert dispatch.main(["list"]) == 0
    names = [line.split("\t")[0] for line in capsys.readouterr().out.splitlines()]
    assert names == sorted(dispatch.DIAGNOSTICS)


def test_actual_list_entry_needs_only_standard_library():
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
def test_run_preserves_selected_module_and_argument_order(monkeypatch, arguments):
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
def test_invalid_selection_fails_before_module_execution(monkeypatch, arguments):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid selection must not execute")

    monkeypatch.setattr(dispatch.runpy, "run_module", forbidden)
    with pytest.raises(SystemExit) as caught:
        dispatch.main(arguments)
    assert caught.value.code == 2


@pytest.mark.parametrize("code", [None, 0, 7, "diagnostic failure"])
def test_module_exit_is_preserved_and_argv_restored(monkeypatch, code):
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
def test_original_error_and_interruption_propagate(monkeypatch, error):
    original_argv = sys.argv

    def run_module(*args, **kwargs):
        raise error

    monkeypatch.setattr(dispatch.runpy, "run_module", run_module)
    with pytest.raises(type(error)) as caught:
        dispatch.main(["run", "lm_head_chunking"])
    assert caught.value is error
    assert sys.argv is original_argv


def test_real_module_entry_runs_main_with_native_argv(monkeypatch, tmp_path, capsys):
    from arctic_platform.correctness import diagnostics

    source = tmp_path / "dispatch_fixture.py"
    source.write_text(
        "import json\n"
        "import sys\n"
        'if __name__ == "__main__":\n'
        "    print(json.dumps([__name__, sys.argv, sys.modules['__main__'].__file__]))\n"
        "    raise SystemExit(7)\n"
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


def chunking_config():
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


def test_chunking_removes_only_knob_from_copied_training_configs():
    from arctic_platform.correctness.diagnostics.lm_head_chunking import KNOB
    from arctic_platform.correctness.diagnostics.lm_head_chunking import without_chunked_head

    cfg = chunking_config()
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
def test_chunking_uses_injected_api_executor_without_global_mutation(monkeypatch, arguments):
    from arctic_platform.correctness import harness
    from arctic_platform.correctness.diagnostics import lm_head_chunking as diagnostic

    cfg = chunking_config()
    before = copy.deepcopy(cfg)
    request, options = object(), object()
    parse_calls, execute_calls = [], []
    result = object()

    def parse_run_args(argv):
        parse_calls.append(list(argv))
        return request, options

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
    assert all(config is not cfg for config, _, _ in execute_calls)
    assert all(args == ("spec",) and kwargs == {"workdir": "work"} for _, args, kwargs in execute_calls)
    assert cfg == before
    assert runner.execute is execute


def test_chunking_parse_failure_does_not_load_executor(monkeypatch):
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


def test_chunking_executor_failure_keeps_original_config_and_binding(monkeypatch):
    from arctic_platform.correctness import harness
    from arctic_platform.correctness.diagnostics import lm_head_chunking as diagnostic

    cfg = chunking_config()
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
