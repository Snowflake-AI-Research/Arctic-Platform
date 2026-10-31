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
"""Tests for ``arctic_platform.integrations.harbor`` and ``recipes/rl/harbor``."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor_rl.entrypoint import materialize_harbor_task_from_row

from arctic_platform.integrations.harbor.adapter import HarborTrialAdapter
from arctic_platform.integrations.harbor.adapter import grade_harbor
from arctic_platform.integrations.harbor.env import rewrite_harbor_paths
from arctic_platform.integrations.harbor.snowflake_sandbox import shell as sf_shell
from arctic_platform.integrations.harbor.snowflake_sandbox.shell import LOST_EXIT
from arctic_platform.integrations.harbor.snowflake_sandbox.shell import build_argv
from arctic_platform.integrations.harbor.snowflake_sandbox.shell import build_exec_argv
from arctic_platform.integrations.harness.generator import RolloutRequest


def _load_prepare_data():
    pytest.importorskip("pandas")
    spec = importlib.util.spec_from_file_location(
        "harbor_prepare_data",
        Path(__file__).resolve().parents[3] / "recipes" / "rl" / "harbor" / "prepare_data.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_sample_harbor_task(root: Path, name: str = "sample-task") -> Path:
    task_dir = root / name
    env_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    env_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Write 'hello' into /app/output.txt\n", encoding="utf-8")
    (env_dir / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n", encoding="utf-8")
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(
        "#!/usr/bin/env bash\n"
        "mkdir -p /logs/verifier\n"
        "if [ -f /app/output.txt ] && grep -q 'hello' /app/output.txt; then\n"
        "  echo 1.0 > /logs/verifier/reward.txt\n"
        "else\n"
        "  echo 0.0 > /logs/verifier/reward.txt\n"
        "fi\n",
        encoding="utf-8",
    )
    test_sh.chmod(0o755)
    return task_dir


def test_discover_and_build_train_and_eval_harbor_tasks(tmp_path: Path) -> None:
    prep = _load_prepare_data()
    train_root = tmp_path / "train_tasks"
    eval_root = tmp_path / "eval_tasks"
    t1 = _make_sample_harbor_task(train_root, "task-a")
    t2 = _make_sample_harbor_task(train_root, "task-b")
    e1 = _make_sample_harbor_task(eval_root, "task-eval")

    found_train = prep.discover_harbor_tasks(train_root)
    assert [p.name for p in found_train] == ["task-a", "task-b"]

    train_df, val_df = prep.build(train_root, eval_root, repeat=3)
    assert len(train_df) == 6
    assert len(val_df) == 1
    assert set(train_df["harbor_task_dir"]) == {str(t1), str(t2)}
    assert list(val_df["harbor_task_dir"]) == [str(e1)]


def test_materialize_swe_smith_localization_harbor_task(tmp_path: Path) -> None:
    row = dict(
        instance_id="owner__repo.123",
        repo="owner/repo",
        base_commit="abc1234",
        use_patch=True,
        patch="diff --git a/foo.py b/foo.py\n",
        problem_statement="Fix bug in foo.py",
        file_changes=[
            dict(
                file="src/foo.py",
                changes=dict(
                    edited_modules=["src/foo.py:Foo"],
                    edited_entities=["src/foo.py:Foo.bar"],
                ),
            )
        ],
    )
    task_dir = materialize_harbor_task_from_row(row, tmp_path / "tasks")
    assert (task_dir / "task.toml").is_file()
    assert (task_dir / "instruction.md").read_text(encoding="utf-8") == "Fix bug in foo.py"
    assert (task_dir / "environment" / "Dockerfile").is_file()
    assert (task_dir / "environment" / "bug.patch").is_file()
    assert (task_dir / "tests" / "test.sh").is_file()
    truth = json.loads((task_dir / "tests" / "ground_truth.json").read_text(encoding="utf-8"))
    assert truth["instance_id"] == "owner__repo.123"
    assert truth["file_changes"][0]["file"] == "src/foo.py"


def test_swe_smith_export_needs_only_the_standard_library(tmp_path: Path) -> None:
    harbor_recipe = Path(__file__).resolve().parents[3] / "recipes" / "rl" / "harbor"
    row = dict(instance_id="owner__repo.123", repo="owner/repo", base_commit="abc1234", problem_statement="Fix it")
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(harbor_recipe)!r})\n"
        "from harbor_rl.swe_smith_tasks import materialize_harbor_task_from_row\n"
        "from pathlib import Path\n"
        f"task = materialize_harbor_task_from_row(json.loads(sys.argv[1]), Path({str(tmp_path / 'tasks')!r}))\n"
        "print(task)\n"
    )
    # -I -S: no site-packages, so neither Arctic Platform nor the training dependencies are importable,
    # matching prepare_data.py run as a `uv run` script.
    out = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, json.dumps(row)], capture_output=True, text=True, check=True
    )
    task_dir = Path(out.stdout.strip())
    reward = harbor_recipe.parent / "openhands" / "code_localization" / "openhands_localization" / "reward.py"
    assert (task_dir / "tests" / "localization_reward.py").read_text() == reward.read_text()
    assert (task_dir / "tests" / "test.sh").is_file()


def test_harbor_trial_adapter_and_grader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_dir = _make_sample_harbor_task(tmp_path, "sample-task")

    async def fake_run_harbor_trial(task_dir: Path, trials_dir: Path, trial_name: str, **kwargs):
        return dict(
            reward=2.5,
            reward_detail=dict(file_reward=1.0, module_reward=1.0, entity_reward=0.5, reward=2.5),
            exhausted=False,
            messages=[dict(role="assistant", content="done")],
            exception=None,
        )

    monkeypatch.setattr("arctic_platform.integrations.harbor.adapter.run_harbor_trial", fake_run_harbor_trial)

    adapter = HarborTrialAdapter(
        "openai/Qwen/Qwen3.5-4B",
        agent="harbor_rl.agent:OpenHandsHarborAgent",
        trials_root=str(tmp_path / "trials"),
    )
    req = RolloutRequest(
        env_extra=dict(instance_id="sample-task", harbor_task_dir=str(task_dir)),
        trajectory_id=SimpleNamespace(repetition_id=0),
        routing_key="step1-train-0_0",
        base_url="http://127.0.0.1:9999/r/step1-train-0_0/v1",
        api_key="secret",
        phase="train",
        global_step=1,
    )
    rid = adapter.submit(req)
    while adapter.get_status(rid) == "running":
        pass
    assert adapter.get_status(rid) == "completed"
    out = adapter.collect_output(rid)
    grade = grade_harbor([], out, req)
    assert grade.reward == 2.5
    assert grade.reward_detail["file_reward"] == 1.0
    assert grade.exhausted is False


def test_rewrite_harbor_paths(tmp_path: Path) -> None:
    root = tmp_path / "trial"
    src = "pytest /tests/test_outputs.py && cat /app/res.txt > /logs/verifier/reward.txt"
    rewritten = rewrite_harbor_paths(src, root)
    assert f"{root}/tests/test_outputs.py" in rewritten
    assert f"{root}/app/res.txt" in rewritten
    assert f"{root}/logs/verifier/reward.txt" in rewritten


def test_snowflake_sandbox_shell_and_detached_exec_reattach(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    import uuid

    monkeypatch.setattr(sf_shell, "EXEC_ROOT", str(tmp_path / "harbor-exec"))
    argv = build_argv('echo "$FOO|$(pwd)"', cwd=str(tmp_path), env={"FOO": "bar"})
    proc = subprocess.run(argv, capture_output=True, text=True)
    assert proc.returncode == 0
    assert proc.stdout.strip().startswith("bar|")

    marker = tmp_path / "count"
    exec_id = uuid.uuid4().hex
    first = subprocess.run(
        build_exec_argv(f"echo x >> {marker}; echo done; exit 3", exec_id),
        capture_output=True,
        text=True,
    )
    again = subprocess.run(
        build_exec_argv("", exec_id, reattach=True),
        capture_output=True,
        text=True,
    )
    assert (first.returncode, first.stdout) == (3, "done\n")
    assert (again.returncode, again.stdout) == (3, "done\n")
    assert marker.read_text() == "x\n"

    lost = subprocess.run(
        build_exec_argv("", uuid.uuid4().hex, reattach=True),
        capture_output=True,
        text=True,
    )
    assert lost.returncode == LOST_EXIT
    assert "lost track" in lost.stderr


@pytest.mark.parametrize("started_before_the_drop", [False, True])
def test_a_dropped_exec_stream_runs_the_command_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, started_before_the_drop: bool
) -> None:
    pytest.importorskip("snowflake.sandbox")
    import asyncio

    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    from arctic_platform.integrations.harbor.snowflake_sandbox import environment as sf_env

    monkeypatch.setattr(sf_shell, "EXEC_ROOT", str(tmp_path / "harbor-exec"))
    calls: list[list[str]] = []

    class Stream:
        def __init__(self, argv: list[str]) -> None:
            calls.append(argv)
            dropped = len(calls) == 1
            if dropped and not started_before_the_drop:
                self.lines, self.exit_code = [], None
                return
            proc = subprocess.run(argv, capture_output=True, text=True)
            if dropped:
                self.lines, self.exit_code = [], None
                return
            out = [SimpleNamespace(stream="stdout", data=line) for line in proc.stdout.splitlines()]
            err = [SimpleNamespace(stream="stderr", data=line) for line in proc.stderr.splitlines()]
            self.lines, self.exit_code = out + err, proc.returncode

        def __aiter__(self):
            return self._lines()

        async def _lines(self):
            for line in self.lines:
                yield line

    (tmp_path / "environment").mkdir()
    (tmp_path / "trial").mkdir()
    env = sf_env.SnowflakeSandboxEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="task",
        session_id="session",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
        prebuilt_image="/db/schema/repo/image:tag",
    )
    env._sandbox = SimpleNamespace(exec_stream=lambda argv, timeout: Stream(argv))
    marker = tmp_path / "count"

    result = asyncio.run(env.exec(f"echo x >> {marker}; echo done", user="root"))

    assert (result.return_code, result.stdout) == (0, "done\n")
    assert marker.read_text() == "x\n"
    assert len(calls) == (2 if started_before_the_drop else 3)
