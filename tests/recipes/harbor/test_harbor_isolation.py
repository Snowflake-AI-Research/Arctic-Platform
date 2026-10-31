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
"""Isolation, containment, credential, and concurrency tests for the Harbor integration."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.agents.base import BaseAgent  # noqa: E402
from harbor.environments.base import ExecResult  # noqa: E402
from harbor.models.task.config import EnvironmentConfig  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402

from arctic_platform.integrations.harbor.adapter import HarborTrialAdapter  # noqa: E402
from arctic_platform.integrations.harbor.adapter import run_harbor_trial  # noqa: E402
from arctic_platform.integrations.harbor.env import HostEnvironment  # noqa: E402
from arctic_platform.integrations.harbor.env import PathEscapeError  # noqa: E402
from arctic_platform.integrations.harbor.env import host_exec_env  # noqa: E402
from arctic_platform.integrations.harness.generator import RolloutRequest  # noqa: E402

_SECRET = "cortex-pat-must-not-leak"
_ROLLOUT_KEY = "rollout-key-must-not-leak"


def _task(root: Path, dockerfile: str = "FROM python:3.12-slim\nWORKDIR /app\n", name: str = "task") -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text(f'name = "sample/{name}"\nversion = "1.0"\n', encoding="utf-8")
    (task / "instruction.md").write_text("Write 'hello' into /app/output.txt\n", encoding="utf-8")
    (task / "environment" / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    test_sh = task / "tests" / "test.sh"
    test_sh.write_text(
        "#!/usr/bin/env bash\nmkdir -p /logs/verifier\n"
        "if grep -q hello /app/output.txt; then echo 1.0; else echo 0.0; fi > /logs/verifier/reward.txt\n",
        encoding="utf-8",
    )
    test_sh.chmod(0o755)
    return task


def _host_env(tmp_path: Path, task: Path, **kwargs) -> HostEnvironment:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir(exist_ok=True)
    return HostEnvironment(
        environment_dir=task / "environment",
        environment_name="task",
        session_id="session",
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=EnvironmentConfig(),
        **kwargs,
    )


# -- HostEnvironment: opt-in and environment allowlist -----------------------


def test_host_environment_refuses_to_run_without_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("HARBOR_ALLOW_HOST_EXEC", raising=False)
    env = _host_env(tmp_path, _task(tmp_path))
    with pytest.raises(RuntimeError, match="HARBOR_ALLOW_HOST_EXEC=1"):
        asyncio.run(env.start(force_build=False))


def test_host_commands_see_only_allowlisted_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_PAT", _SECRET)
    monkeypatch.setenv("WANDB_API_KEY", _SECRET)
    monkeypatch.setenv("GIT_ASKPASS_TOKEN", _SECRET)
    monkeypatch.setenv("HARBOR_EXTRA_SETTING", "passed-through")
    monkeypatch.setenv("HARBOR_HOST_ENV_PASSTHROUGH", "HARBOR_EXTRA_SETTING")
    env = _host_env(tmp_path, _task(tmp_path), allow_host_exec=True)

    async def run() -> str:
        await env.start(force_build=False)
        try:
            return (await env.exec("env")).stdout
        finally:
            await env.stop(delete=True)

    printed = asyncio.run(run())
    assert _SECRET not in printed
    assert "HARBOR_EXTRA_SETTING=passed-through" in printed
    assert set(host_exec_env()) <= {
        "PATH",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "TERM",
        "TMPDIR",
        "SHELL",
    } | {"HARBOR_EXTRA_SETTING"}


def test_dockerfile_run_lines_get_the_allowlisted_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_PAT", _SECRET)
    task = _task(tmp_path, "FROM python:3.12-slim\nWORKDIR /app\nRUN env > /app/build_env.txt\n")
    env = _host_env(tmp_path, task, allow_host_exec=True)

    async def run() -> str:
        await env.start(force_build=False)
        try:
            return (env.host_root / "app" / "build_env.txt").read_text()
        finally:
            await env.stop(delete=True)

    assert _SECRET not in asyncio.run(run())


# -- HostEnvironment: build-context and trial-root containment ---------------


@pytest.mark.parametrize(
    "copy_line",
    ["COPY ../outside.txt /app/", "COPY /etc/hostname /app/", "COPY link_out /app/"],
    ids=["parent-traversal", "absolute-source", "symlink-out"],
)
def test_copy_sources_must_stay_inside_the_build_context(tmp_path, copy_line):
    (tmp_path / "outside.txt").write_text(_SECRET, encoding="utf-8")
    task = _task(tmp_path, f"FROM python:3.12-slim\nWORKDIR /app\n{copy_line}\n")
    (task / "environment" / "link_out").symlink_to(tmp_path / "outside.txt")
    env = _host_env(tmp_path, task, allow_host_exec=True)
    with pytest.raises(PathEscapeError):
        asyncio.run(env.start(force_build=False))


def test_a_symlink_out_of_the_build_context_is_not_copied_implicitly(tmp_path):
    (tmp_path / "outside.txt").write_text(_SECRET, encoding="utf-8")
    task = _task(tmp_path, "FROM python:3.12-slim\nWORKDIR /app\n")
    (task / "environment" / "data").mkdir()
    (task / "environment" / "data" / "leak").symlink_to(tmp_path / "outside.txt")
    env = _host_env(tmp_path, task, allow_host_exec=True)
    with pytest.raises(PathEscapeError):
        asyncio.run(env.start(force_build=False))


def test_paths_cannot_escape_the_trial_root(tmp_path):
    (tmp_path / "outside.txt").write_text(_SECRET, encoding="utf-8")
    env = _host_env(tmp_path, _task(tmp_path), allow_host_exec=True)

    async def run() -> None:
        await env.start(force_build=False)
        try:
            (env.host_root / "logs" / "agent" / "leak").symlink_to(tmp_path / "outside.txt")
            with pytest.raises(PathEscapeError):
                await env.exec("pwd", cwd="../../../..")
            with pytest.raises(PathEscapeError):
                await env.upload_file(tmp_path / "outside.txt", "/app/../../../planted.txt")
            with pytest.raises(PathEscapeError):
                await env.download_file("/app/../../../outside.txt", tmp_path / "got.txt")
            with pytest.raises(PathEscapeError):
                await env.download_dir("/logs/agent", tmp_path / "got")
        finally:
            await env.stop(delete=True)

    asyncio.run(run())
    assert not (tmp_path / "planted.txt").exists()
    assert not (tmp_path / "got.txt").exists()


# -- Dataset rows: validation before paths and Dockerfile lines --------------


def _row(**overrides):
    row = dict(
        instance_id="owner__repo.abc123",
        repo="owner/repo",
        base_commit="abc1234",
        problem_statement="Fix the bug",
        file_changes=[dict(file="a.py", changes=dict(edited_modules=[], edited_entities=[]))],
    )
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    "overrides",
    [
        dict(instance_id="../escape"),
        dict(instance_id="/abs/path"),
        dict(instance_id="x; rm -rf ~"),
        dict(repo="owner/repo.git /tmp/x && curl evil"),
        dict(repo="../../owner/repo"),
        dict(base_commit="abc1234; curl evil"),
        dict(base_commit="$(id)"),
    ],
    ids=["id-traversal", "id-absolute", "id-shell", "repo-shell", "repo-traversal", "commit-shell", "commit-subst"],
)
def test_malicious_dataset_rows_are_rejected(tmp_path, overrides):
    from harbor_rl.entrypoint import materialize_harbor_task_from_row

    with pytest.raises(ValueError, match="invalid"):
        materialize_harbor_task_from_row(_row(**overrides), tmp_path / "tasks")
    assert not any((tmp_path / "tasks").glob("**/Dockerfile"))


def test_materialized_tasks_carry_their_own_verifier(tmp_path):
    from harbor_rl.entrypoint import materialize_harbor_task_from_row
    from openhands_localization import reward

    task = materialize_harbor_task_from_row(_row(), tmp_path / "tasks")
    assert task.parent == (tmp_path / "tasks").resolve()
    assert (task / "tests" / "localization_reward.py").read_text() == Path(reward.__file__).read_text()
    assert "openhands_localization" not in (task / "tests" / "test.sh").read_text()
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    assert "RUN git clone https://github.com/owner/repo.git /app/owner_repo_owner__repo.abc123" in dockerfile
    assert "RUN git checkout abc1234" in dockerfile


# -- Adapter: one event loop for every rollout -------------------------------


def test_concurrent_rollouts_share_one_event_loop(tmp_path, monkeypatch):
    task = _task(tmp_path)
    shared = {"lock": None}
    loops: set[int] = set()

    async def fake_run_harbor_trial(**kwargs):
        if shared["lock"] is None:
            shared["lock"] = asyncio.Lock()
        loops.add(id(asyncio.get_running_loop()))
        async with shared["lock"]:
            await asyncio.sleep(0.01)
        return dict(reward=1.0, reward_detail=dict(reward=1.0), exhausted=False, messages=[], exception=None)

    monkeypatch.setattr("arctic_platform.integrations.harbor.adapter.run_harbor_trial", fake_run_harbor_trial)
    adapter = HarborTrialAdapter("openai/m", agent="a:B", trials_root=str(tmp_path / "trials"))
    ids = [
        adapter.submit(
            RolloutRequest(
                env_extra=dict(harbor_task_dir=str(task)),
                trajectory_id=SimpleNamespace(repetition_id=i),
                routing_key=f"r{i}",
                base_url="http://127.0.0.1:1/v1",
                api_key="k",
                phase="train",
                global_step=1,
            )
        )
        for i in range(16)
    ]
    for rid in ids:
        while adapter.get_status(rid) == "running":
            pass
    assert {adapter.get_status(rid) for rid in ids} == {"completed"}, [adapter.collect_output(r) for r in ids]
    assert len(loops) == 1


def test_a_row_pointing_outside_a_harbor_task_is_rejected(tmp_path):
    adapter = HarborTrialAdapter("openai/m", agent="a:B", trials_root=str(tmp_path / "trials"))
    with pytest.raises(ValueError, match="not a Harbor task directory"):
        adapter._task_dir(dict(harbor_task_dir=str(tmp_path)))


# -- Snowflake sandbox: loop-safe shared state and registry credentials ------


def test_snowflake_session_is_shared_across_event_loops(monkeypatch):
    pytest.importorskip("snowflake.sandbox")
    from arctic_platform.integrations.harbor.snowflake_sandbox import session

    connects = []

    def fake_connect(connection, role):
        connects.append((connection, role))
        return SimpleNamespace(connection=object())

    monkeypatch.setattr(session, "_connect", fake_connect)
    monkeypatch.setattr(session, "_configs", {})

    async def many() -> set[int]:
        configs = await asyncio.gather(*(session.get_config("conn", "role") for _ in range(8)))
        return {id(c) for c in configs}

    results: list[set[int]] = []
    threads = [threading.Thread(target=lambda: results.append(asyncio.run(many()))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set().union(*results)) == 1
    assert connects == [("conn", "role")]


def test_live_sandbox_slots_are_per_event_loop():
    pytest.importorskip("snowflake.sandbox")
    from arctic_platform.integrations.harbor.snowflake_sandbox.environment import SnowflakeSandboxEnvironment

    holder = SimpleNamespace(_max_live=1)

    async def contend() -> None:
        slots = SnowflakeSandboxEnvironment._slots(holder)
        await slots.acquire()
        waiter = asyncio.ensure_future(slots.acquire())
        await asyncio.sleep(0)
        slots.release()
        await waiter
        slots.release()

    for _ in range(3):
        asyncio.run(contend())


def test_registry_login_never_touches_the_users_docker_config(tmp_path, monkeypatch):
    pytest.importorskip("snowflake.sandbox")
    from arctic_platform.integrations.harbor.snowflake_sandbox import images

    home = tmp_path / "home"
    (home / ".docker").mkdir(parents=True)
    (home / ".docker" / "config.json").write_text('{"auths": {"docker.io": {}}}', encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("DOCKER_CONFIG", raising=False)
    user_config = (home / ".docker" / "config.json").read_text()

    seen: list[dict] = []

    def fake_subprocess_run(argv, *, input, env, **kwargs):
        seen.append(dict(argv=argv, docker_config=env["DOCKER_CONFIG"], token=input))
        config = Path(env["DOCKER_CONFIG"]) / "config.json"
        data = json.loads(config.read_text())
        data.setdefault("auths", {})[argv[3]] = {"auth": "token"}
        config.write_text(json.dumps(data))
        return SimpleNamespace(returncode=0, stderr="")

    async def fake_run(*argv, env=None, **kwargs):
        seen.append(dict(argv=list(argv), docker_config=(env or {}).get("DOCKER_CONFIG")))
        return "sha256:abc"

    monkeypatch.setattr(images.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(images, "_run", fake_run)
    registry = images.ImageRegistry("DB.SCHEMA.REPO", connection=None, role=None)
    monkeypatch.setattr(registry, "_registry_token", lambda: '{"token": "t"}')

    async def fake_host() -> str:
        return "org-acct.registry.snowflakecomputing.com"

    monkeypatch.setattr(registry, "_registry_host", fake_host)
    asyncio.run(registry.ensure_image("python:3.12-slim"))

    private = registry.docker_config
    assert all(call["docker_config"] == str(private) for call in seen)
    assert stat.S_IMODE(private.stat().st_mode) & 0o077 == 0
    assert "org-acct.registry.snowflakecomputing.com" in (private / "config.json").read_text()
    assert "docker.io" in (private / "config.json").read_text()
    assert (home / ".docker" / "config.json").read_text() == user_config

    registry.close()
    assert not private.exists()


def test_registry_logs_in_on_a_freshly_booted_host(monkeypatch):
    pytest.importorskip("snowflake.sandbox")
    from arctic_platform.integrations.harbor.snowflake_sandbox import images

    logins: list[str] = []
    monkeypatch.setattr(images.time, "monotonic", lambda: 5.0)
    registry = images.ImageRegistry("DB.SCHEMA.REPO", connection=None, role=None)
    monkeypatch.setattr(registry, "_login_sync", logins.append)

    async def fake_host() -> str:
        return "org-acct.registry.snowflakecomputing.com"

    monkeypatch.setattr(registry, "_registry_host", fake_host)
    asyncio.run(registry._login())
    assert logins == ["org-acct.registry.snowflakecomputing.com"]


def test_sandbox_is_terminated_and_its_slot_released_when_start_fails(tmp_path, monkeypatch):
    pytest.importorskip("snowflake.sandbox")
    from arctic_platform.integrations.harbor.snowflake_sandbox import environment as sf_env

    terminated: list[str] = []

    class FakeSandbox:
        id = "sb-1"

        async def terminate(self) -> None:
            terminated.append(self.id)

    class FakeAsyncSandbox:
        @staticmethod
        async def create(**kwargs):
            return FakeSandbox()

    async def fake_get_config(connection, role):
        return None

    monkeypatch.setattr(sf_env, "AsyncSandbox", FakeAsyncSandbox)
    monkeypatch.setattr(sf_env, "get_config", fake_get_config)
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (tmp_path / "trial").mkdir()

    def make() -> sf_env.SnowflakeSandboxEnvironment:
        return sf_env.SnowflakeSandboxEnvironment(
            environment_dir=tmp_path / "environment",
            environment_name="task",
            session_id="session",
            trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
            task_env_config=EnvironmentConfig(),
            prebuilt_image="/db/schema/repo/image:tag",
            max_live=1,
        )

    async def run() -> None:
        failing = make()

        async def boom(*args, **kwargs):
            raise RuntimeError("prep failed")

        monkeypatch.setattr(failing, "exec", boom)
        with pytest.raises(RuntimeError, match="prep failed"):
            await failing.start(force_build=False)
        assert terminated == ["sb-1"]
        assert failing._sandbox is None and not failing._slot_held

        ok = make()

        async def fine(*args, **kwargs):
            return ExecResult(stdout="", stderr="", return_code=0)

        monkeypatch.setattr(ok, "exec", fine)
        monkeypatch.setattr(ok, "upload_dir", fine)
        monkeypatch.setattr(ok, "upload_file", fine)
        await asyncio.wait_for(ok.start(force_build=False), timeout=5)
        await ok.stop(delete=True)
        assert terminated == ["sb-1", "sb-1"]
        assert not ok._slot_held

    asyncio.run(run())


# -- The agent runs inside the selected Harbor environment -------------------


class _ScriptedLLM(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint that replays ``script`` as tool calls."""

    script: list[tuple[str, dict]] = []
    calls: list[dict] = []
    auth: list[str | None] = []
    block_on_call: int | None = None
    blocked = threading.Event()
    release = threading.Event()

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append(body)
        type(self).auth.append(self.headers.get("Authorization"))
        if len(self.calls) == self.block_on_call:
            type(self).blocked.set()
            type(self).release.wait(30)
        name, args = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        call = dict(id=f"call{len(self.calls)}", type="function", function=dict(name=name, arguments=json.dumps(args)))
        out = json.dumps(
            dict(
                id="x",
                object="chat.completion",
                created=0,
                model="m",
                prompt_token_ids=[1],
                choices=[
                    dict(
                        index=0,
                        message=dict(role="assistant", content="", tool_calls=[call]),
                        finish_reason="tool_calls",
                        token_ids=[2],
                    )
                ],
                usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def scripted_llm():
    pytest.importorskip("openhands.sdk")
    _ScriptedLLM.calls = []
    _ScriptedLLM.auth = []
    _ScriptedLLM.block_on_call = None
    _ScriptedLLM.blocked = threading.Event()
    _ScriptedLLM.release = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield _ScriptedLLM, f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()


def test_agent_commands_run_in_the_harbor_environment_the_verifier_grades(tmp_path, monkeypatch, scripted_llm):
    llm, base_url = scripted_llm
    llm.script = [
        ("terminal", dict(command="cd /app && echo hello > output.txt && echo pat=${ARCTIC_CORTEX_PAT:-absent}")),
        ("terminal", dict(command="pwd")),
        ("terminal", dict(command="echo keys=${OPENAI_API_KEY:-absent},${LLM_API_KEY:-absent}")),
        ("harbor_finish", dict(summary="done")),
    ]
    monkeypatch.setenv("ARCTIC_CORTEX_PAT", _SECRET)
    monkeypatch.setenv("HARBOR_ALLOW_HOST_EXEC", "1")
    monkeypatch.delenv("OPENHANDS_TOOLS", raising=False)

    out = asyncio.run(
        run_harbor_trial(
            task_dir=_task(tmp_path),
            trials_dir=tmp_path / "trials",
            trial_name="trial",
            agent="harbor_rl.agent:OpenHandsHarborAgent",
            env="host",
            model_name="openai/test-model",
            base_url=base_url,
            api_key=_ROLLOUT_KEY,
            agent_kwargs=dict(max_turns=5, instance={}),
        )
    )

    assert out["exception"] is None
    assert out["reward"] == 1.0
    observations = [
        part["text"]
        for event in out["messages"]
        if isinstance(event, dict) and event.get("kind") == "ObservationEvent"
        for part in event["observation"]["content"]
    ]
    assert observations[0].strip() == "pat=absent"
    assert observations[1].strip().endswith("/host_env/app")
    assert observations[2].strip() == "keys=absent,absent"
    assert _SECRET not in json.dumps(out["messages"])
    assert _ROLLOUT_KEY not in json.dumps(out["messages"])
    assert set(llm.auth) == {f"Bearer {_ROLLOUT_KEY}"}
    tools = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert "terminal" in tools and "file_editor" not in tools


def test_consecutive_general_harbor_rollouts_in_one_process_all_run(tmp_path, monkeypatch, scripted_llm):
    llm, base_url = scripted_llm
    monkeypatch.setenv("HARBOR_ALLOW_HOST_EXEC", "1")
    monkeypatch.delenv("OPENHANDS_TOOLS", raising=False)
    rewards = []
    for i in range(2):
        llm.calls.clear()
        llm.script = [
            ("terminal", dict(command="cd /app && echo hello > output.txt")),
            ("harbor_finish", dict(summary="done")),
        ]
        out = asyncio.run(
            run_harbor_trial(
                task_dir=_task(tmp_path, name=f"task{i}"),
                trials_dir=tmp_path / "trials",
                trial_name=f"trial{i}",
                agent="harbor_rl.agent:OpenHandsHarborAgent",
                env="host",
                model_name="openai/test-model",
                base_url=base_url,
                api_key="k",
                agent_kwargs=dict(max_turns=5, instance={}),
            )
        )
        assert out["exception"] is None, out["exception"]
        rewards.append(out["reward"])
    assert rewards == [1.0, 1.0]


class _ForwardingAgent(BaseAgent):
    """A driver-resident agent that hands Harbor its ``extra_env``, as ``BaseAgent`` intends."""

    seen: list[str] = []

    def __init__(self, logs_dir, model_name=None, *, extra_env=None, api_key=None, base_url=None, **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, extra_env=extra_env)

    @staticmethod
    def name() -> str:
        return "forwarding"

    def version(self) -> str:
        return "0"

    async def setup(self, environment) -> None:
        return

    async def run(self, instruction, environment, context) -> None:
        result = await environment.exec("echo ${OPENAI_API_KEY:-absent},${LLM_API_KEY:-absent}")
        type(self).seen.append((result.stdout or "").strip())


def test_agent_phase_commands_never_see_the_rollout_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HARBOR_ALLOW_HOST_EXEC", "1")
    monkeypatch.setattr(_ForwardingAgent, "seen", [])
    out = asyncio.run(
        run_harbor_trial(
            task_dir=_task(tmp_path),
            trials_dir=tmp_path / "trials",
            trial_name="trial",
            agent=f"{__name__}:_ForwardingAgent",
            env="host",
            model_name="openai/test-model",
            base_url="http://127.0.0.1:1/v1",
            api_key=_ROLLOUT_KEY,
        )
    )
    assert out["exception"] is None, out["exception"]
    assert _ForwardingAgent.seen == ["absent,absent"]


def test_file_editor_is_refused_because_it_bypasses_the_environment(monkeypatch):
    pytest.importorskip("openhands.sdk")
    from harbor_rl.agent import _build_openhands_tools

    monkeypatch.setenv("OPENHANDS_TOOLS", "terminal,file_editor")
    with pytest.raises(ValueError, match="file_editor"):
        _build_openhands_tools("harbor_finish", object())


def test_terminal_executor_tracks_cwd_and_reports_timeouts():
    pytest.importorskip("openhands.sdk")
    from harbor_rl.terminal import HarborTerminalExecutor
    from harbor_rl.terminal import split_cwd
    from openhands.tools.terminal.definition import TerminalAction

    seen: list[tuple[str, str | None]] = []

    class FakeEnvironment:
        async def exec(self, command, cwd=None, timeout_sec=None):
            seen.append((command, cwd))
            if "sleep" in command:
                raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
            return ExecResult(stdout="out\n\n__HARBOR_TERMINAL_CWD__/work/sub\n", stderr="", return_code=0)

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    try:
        executor = HarborTerminalExecutor(FakeEnvironment(), loop, "/work")
        first = executor(TerminalAction(command="cd sub && echo out"))
        assert first.text == "out\n" and first.metadata.working_dir == "/work/sub"
        executor(TerminalAction(command="ls"))
        assert seen[1][1] == "/work/sub"
        timed_out = executor(TerminalAction(command="sleep 999", timeout=1))
        assert timed_out.timeout and timed_out.is_error
        assert executor(TerminalAction(command="", is_input=True)).is_error
    finally:
        loop.call_soon_threadsafe(loop.stop)
    assert split_cwd("no marker") == ("no marker", None)


def test_a_stopped_terminal_runs_no_more_commands_and_abandons_the_one_in_flight():
    pytest.importorskip("openhands.sdk")
    from harbor_rl.terminal import HarborTerminalExecutor
    from openhands.tools.terminal.definition import TerminalAction

    started = threading.Event()
    commands: list[str] = []

    class SlowEnvironment:
        async def exec(self, command, cwd=None, timeout_sec=None):
            commands.append(command)
            started.set()
            await asyncio.sleep(60)

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    stopped = threading.Event()
    try:
        executor = HarborTerminalExecutor(SlowEnvironment(), loop, "/work", stopped=stopped)
        threading.Thread(target=lambda: (started.wait(5), stopped.set()), daemon=True).start()
        t0 = time.monotonic()
        in_flight = executor(TerminalAction(command="sleep 60", timeout=60))
        assert time.monotonic() - t0 < 5
        assert in_flight.is_error and "stopped" in in_flight.text
        assert executor(TerminalAction(command="echo again")).is_error
        assert len(commands) == 1
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_stopping_a_conversation_never_blocks_the_caller():
    from harbor_rl.agent import ConversationStop

    in_step = threading.Lock()
    in_step.acquire()
    paused = threading.Event()

    class ConversationMidStep:
        def pause(self):
            with in_step:
                paused.set()

    stop = ConversationStop()
    assert stop.attach(ConversationMidStep())
    t0 = time.monotonic()
    stop.stop()
    assert time.monotonic() - t0 < 1, "stop() waited for the conversation's step to end"
    assert stop.stopped.is_set() and not paused.is_set()
    in_step.release()
    assert paused.wait(5)
    assert not ConversationStop().stopped.is_set()
    late = ConversationStop()
    late.stop()
    assert not late.attach(ConversationMidStep())


def test_env_kwargs_reach_the_environment_and_snowflake_requires_an_image_repo(monkeypatch):
    from harbor.environments.factory import EnvironmentFactory
    from harbor_rl.entrypoint import check_harbor_env
    from harbor_rl.entrypoint import harbor_env_kwargs

    from arctic_platform.integrations.harbor.adapter import HOST_HARBOR_ENV
    from arctic_platform.integrations.harbor.adapter import SNOWFLAKE_HARBOR_ENV
    from arctic_platform.integrations.harbor.snowflake_sandbox.environment import SnowflakeSandboxEnvironment

    preflights: list[tuple] = []
    monkeypatch.setattr(
        EnvironmentFactory, "run_preflight", lambda type, import_path=None: preflights.append((type, import_path))
    )
    monkeypatch.setattr(
        SnowflakeSandboxEnvironment, "preflight", lambda **kwargs: preflights.append((None, SNOWFLAKE_HARBOR_ENV))
    )
    monkeypatch.delenv("HARBOR_AGENT", raising=False)
    monkeypatch.setenv("HARBOR_ENV", "snowflake")
    monkeypatch.delenv("HARBOR_ENV_KWARGS", raising=False)
    with pytest.raises(SystemExit, match="image_repo"):
        check_harbor_env()
    monkeypatch.setenv("HARBOR_ENV_KWARGS", json.dumps(dict(image_repo="DB.SCHEMA.REPO", role="R")))
    check_harbor_env()
    assert harbor_env_kwargs() == dict(image_repo="DB.SCHEMA.REPO", role="R")

    monkeypatch.setenv("HARBOR_ENV", "host")
    monkeypatch.delenv("HARBOR_ALLOW_HOST_EXEC", raising=False)
    with pytest.raises(SystemExit, match="HARBOR_ALLOW_HOST_EXEC"):
        check_harbor_env()
    monkeypatch.setenv("HARBOR_ALLOW_HOST_EXEC", "1")
    check_harbor_env()

    monkeypatch.setenv("HARBOR_ENV", "docker")
    check_harbor_env()
    assert [p[1] or p[0].value for p in preflights] == [SNOWFLAKE_HARBOR_ENV, HOST_HARBOR_ENV, "docker"]


def test_launch_preflight_catches_a_missing_crane_before_gpus_are_provisioned(monkeypatch):
    pytest.importorskip("snowflake.sandbox")
    from harbor_rl.entrypoint import check_harbor_env

    from arctic_platform.integrations.harbor.snowflake_sandbox import images

    monkeypatch.delenv("HARBOR_AGENT", raising=False)
    monkeypatch.setenv("HARBOR_ENV", "snowflake")
    monkeypatch.setenv("HARBOR_ENV_KWARGS", json.dumps(dict(image_repo="DB.SCHEMA.REPO")))
    monkeypatch.setattr(images.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="crane"):
        check_harbor_env()


def test_a_sandbox_path_prebuilt_image_needs_no_crane(monkeypatch):
    pytest.importorskip("snowflake.sandbox")
    from harbor_rl.entrypoint import check_harbor_env

    from arctic_platform.integrations.harbor.snowflake_sandbox import images

    monkeypatch.delenv("HARBOR_AGENT", raising=False)
    monkeypatch.setenv("HARBOR_ENV", "snowflake")
    monkeypatch.setattr(images.shutil, "which", lambda name: None)
    monkeypatch.setenv("HARBOR_ENV_KWARGS", json.dumps(dict(prebuilt_image="/db/schema/repo/image:tag")))
    check_harbor_env()

    monkeypatch.setenv("HARBOR_ENV_KWARGS", json.dumps(dict(prebuilt_image="python:3.12-slim")))
    with pytest.raises(SystemExit, match="also set image_repo"):
        check_harbor_env()


def test_a_sandbox_path_prebuilt_image_is_used_without_the_registry(tmp_path, monkeypatch):
    pytest.importorskip("snowflake.sandbox")
    from arctic_platform.integrations.harbor.snowflake_sandbox import environment as sf_env

    def no_registry(*args, **kwargs):
        raise AssertionError("a sandbox-path prebuilt image must not touch the image registry")

    monkeypatch.setattr(sf_env, "get_registry", no_registry)
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (tmp_path / "trial").mkdir()
    env = sf_env.SnowflakeSandboxEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="task",
        session_id="session",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
        prebuilt_image="/db/schema/repo/image:tag",
    )
    image, plan = asyncio.run(env._resolve_image(force_build=False))
    assert (image, plan) == ("/db/schema/repo/image:tag", None)


def test_a_base_image_needs_no_crane_or_image_repo(monkeypatch):
    pytest.importorskip("snowflake.sandbox")
    from harbor_rl.entrypoint import check_harbor_env

    from arctic_platform.integrations.harbor.snowflake_sandbox import images

    monkeypatch.delenv("HARBOR_AGENT", raising=False)
    monkeypatch.setenv("HARBOR_ENV", "snowflake")
    monkeypatch.setattr(images.shutil, "which", lambda name: None)
    monkeypatch.setenv("HARBOR_ENV_KWARGS", json.dumps(dict(base_image="sandbox-base")))
    check_harbor_env()

    monkeypatch.setenv("HARBOR_ENV_KWARGS", json.dumps(dict(connection="C")))
    with pytest.raises(SystemExit, match="base_image"):
        check_harbor_env()


@pytest.mark.parametrize("image_repo", [None, "DB.SCHEMA.REPO"])
def test_a_base_image_replaces_from_and_the_dockerfile_still_replays(tmp_path, monkeypatch, image_repo):
    pytest.importorskip("snowflake.sandbox")
    from arctic_platform.integrations.harbor.snowflake_sandbox import environment as sf_env

    def no_registry(*args, **kwargs):
        raise AssertionError("a base_image must not touch the image registry")

    monkeypatch.setattr(sf_env, "get_registry", no_registry)
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "Dockerfile").write_text(
        "FROM python:3.11-slim\nWORKDIR /app\nRUN git clone https://github.com/o/r.git /app/r\n"
    )
    (tmp_path / "trial").mkdir()
    env = sf_env.SnowflakeSandboxEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="task",
        session_id="session",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
        base_image="sandbox-base",
        image_repo=image_repo,
    )
    image, plan = asyncio.run(env._resolve_image(force_build=False))
    assert image == "sandbox-base"
    assert plan is not None
    assert any("git clone https://github.com/o/r.git" in getattr(step, "command", "") for step in plan.steps)


@pytest.mark.parametrize("has_git", [True, False])
def test_swe_smith_tasks_install_git_only_where_it_is_missing(tmp_path, has_git):
    import subprocess

    from harbor_rl.swe_smith_tasks import materialize_harbor_task_from_row

    task = materialize_harbor_task_from_row(
        dict(instance_id="o__r.abc1234", repo="o/r", base_commit="abc1234", problem_statement="p"), tmp_path / "t"
    )
    lines = (task / "environment" / "Dockerfile").read_text().splitlines()
    git_step = next(line for line in lines if "apt-get" in line).removeprefix("RUN ")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "apt-calls"
    stubs = dict(rm="exit 0", **{"apt-get": f'echo "$1" >> {calls}'})
    if has_git:
        stubs["git"] = "exit 0"
    for name, body in stubs.items():
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    shell = subprocess.run(["/bin/sh", "-c", git_step], env=dict(PATH=str(bin_dir)), check=False)
    assert shell.returncode == 0
    assert (calls.read_text().split() if calls.exists() else []) == ([] if has_git else ["update", "install"])


@pytest.mark.parametrize("installed_agent", ["openhands-sdk", "mini-swe-agent", "claude-code"])
def test_agents_installed_in_the_environment_are_refused(tmp_path, monkeypatch, installed_agent):
    from harbor_rl.entrypoint import check_harbor_env

    from arctic_platform.integrations.harbor.adapter import _build_trial_config

    with pytest.raises(ValueError, match="could read the rollout's API key"):
        _build_trial_config(
            _task(tmp_path),
            tmp_path / "trials",
            "trial",
            agent=installed_agent,
            env="host",
            model_name="m",
            base_url="http://127.0.0.1:1/v1",
            api_key=_ROLLOUT_KEY,
        )
    monkeypatch.setenv("HARBOR_ENV", "host")
    monkeypatch.setenv("HARBOR_ALLOW_HOST_EXEC", "1")
    monkeypatch.setenv("HARBOR_AGENT", installed_agent)
    with pytest.raises(SystemExit, match="could read the rollout's API key"):
        check_harbor_env()


def test_driver_resident_agents_get_the_key_only_as_a_constructor_argument(tmp_path):
    from arctic_platform.integrations.harbor.adapter import _build_trial_config

    agent = _build_trial_config(
        _task(tmp_path),
        tmp_path / "trials",
        "trial",
        agent="harbor_rl.agent:OpenHandsHarborAgent",
        env="host",
        model_name="m",
        base_url="http://127.0.0.1:1/v1",
        api_key=_ROLLOUT_KEY,
    ).agent
    assert agent.env == {}
    assert agent.kwargs["api_key"] == _ROLLOUT_KEY


class _SharedStateEnv:
    """Stands in for an environment that keeps state across trials."""

    closed = 0

    @classmethod
    def close_shared_resources(cls) -> None:
        cls.closed += 1


def _request(task: Path, rollout_id: str, base_url: str = "http://127.0.0.1:1/v1") -> RolloutRequest:
    return RolloutRequest(
        env_extra=dict(harbor_task_dir=str(task)),
        trajectory_id=SimpleNamespace(repetition_id=0),
        routing_key=rollout_id,
        base_url=base_url,
        api_key="k",
        phase="train",
        global_step=1,
    )


def test_closing_the_adapter_cancels_rollouts_and_releases_shared_resources(tmp_path, monkeypatch):
    started = threading.Event()
    cleaned_up: list[str] = []

    async def hanging_trial(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.append(kwargs["trial_name"])

    monkeypatch.setattr("arctic_platform.integrations.harbor.adapter.run_harbor_trial", hanging_trial)
    monkeypatch.setattr(_SharedStateEnv, "closed", 0)
    adapter = HarborTrialAdapter(
        "openai/m", agent="a:B", env=f"{__name__}:_SharedStateEnv", trials_root=str(tmp_path / "trials")
    )
    rid = adapter.submit(_request(_task(tmp_path), "r0"))
    assert started.wait(5)
    loop_thread = adapter._loop_thread

    adapter.close()

    assert len(cleaned_up) == 1
    assert adapter.get_status(rid) == "errored"
    assert "CancelledError" in adapter.collect_output(rid)["error"]
    assert loop_thread.loop.is_closed() and not loop_thread._thread.is_alive()
    assert _SharedStateEnv.closed == 1
    adapter.close()
    assert _SharedStateEnv.closed == 2


def test_snowflake_shared_resources_remove_registry_tokens_and_close_sessions(monkeypatch):
    pytest.importorskip("snowflake.sandbox")
    from arctic_platform.integrations.harbor.snowflake_sandbox import images
    from arctic_platform.integrations.harbor.snowflake_sandbox import session
    from arctic_platform.integrations.harbor.snowflake_sandbox.environment import SnowflakeSandboxEnvironment

    monkeypatch.setattr(images, "_registries", {})
    registry = images.get_registry("DB.SCHEMA.REPO", connection=None, role=None)
    token_dir = registry.docker_config
    connection = SimpleNamespace(closed=False)
    connection.close = lambda: setattr(connection, "closed", True)
    monkeypatch.setattr(session, "_configs", {(None, None): SimpleNamespace(connection=connection)})

    SnowflakeSandboxEnvironment.close_shared_resources()

    assert not token_dir.exists()
    assert connection.closed
    assert images._registries == {} and session._configs == {}


def test_close_waits_for_a_blocked_worker_thread_to_finish(tmp_path, monkeypatch):
    started = threading.Event()
    finished: list[bool] = []

    async def trial_blocked_in_a_thread(**kwargs):
        stop = threading.Event()

        def conversation() -> None:
            started.set()
            stop.wait(10)
            time.sleep(0.5)
            finished.append(True)

        try:
            await asyncio.to_thread(conversation)
        except asyncio.CancelledError:
            stop.set()
            raise

    monkeypatch.setattr("arctic_platform.integrations.harbor.adapter.run_harbor_trial", trial_blocked_in_a_thread)
    before = set(threading.enumerate())
    adapter = HarborTrialAdapter("openai/m", agent="a:B", trials_root=str(tmp_path / "trials"))
    adapter.submit(_request(_task(tmp_path), "r0"))
    assert started.wait(5)
    adapter.close(timeout_s=10)
    assert finished == [True]
    assert not [t for t in set(threading.enumerate()) - before if t.name.startswith("harbor-")]


def test_close_gives_up_on_a_stuck_worker_thread_after_its_timeout(tmp_path, monkeypatch, caplog):
    started = threading.Event()
    release = threading.Event()

    async def stuck_trial(**kwargs):
        await asyncio.to_thread(lambda: (started.set(), release.wait(30)))

    monkeypatch.setattr("arctic_platform.integrations.harbor.adapter.run_harbor_trial", stuck_trial)
    adapter = HarborTrialAdapter("openai/m", agent="a:B", trials_root=str(tmp_path / "trials"))
    adapter.submit(_request(_task(tmp_path), "r0"))
    assert started.wait(5)
    t0 = time.monotonic()
    try:
        adapter.close(timeout_s=1)
        assert time.monotonic() - t0 < 5
        assert "still running" in caplog.text
    finally:
        release.set()


def test_a_login_in_progress_is_closed_rather_than_cached_after_cleanup(monkeypatch):
    pytest.importorskip("snowflake.sandbox")
    from arctic_platform.integrations.harbor.snowflake_sandbox import session

    connecting = threading.Event()
    release = threading.Event()
    connection = SimpleNamespace(closed=False)
    connection.close = lambda: setattr(connection, "closed", True)

    def slow_connect(connection_name, role):
        connecting.set()
        release.wait(10)
        return SimpleNamespace(connection=connection)

    monkeypatch.setattr(session, "_connect", slow_connect)
    monkeypatch.setattr(session, "_configs", {})
    monkeypatch.setattr(session, "_locks", {})
    login = threading.Thread(target=session._get_or_connect, args=("conn", "role"))
    login.start()
    assert connecting.wait(5)
    cleanup = threading.Thread(target=session.close_sessions)
    cleanup.start()
    cleanup.join(0.3)
    assert cleanup.is_alive(), "cleanup must wait for the login in progress"
    release.set()
    login.join(5)
    cleanup.join(5)
    assert connection.closed
    assert session._configs == {}


def test_a_closed_registry_refuses_to_write_credentials():
    pytest.importorskip("snowflake.sandbox")
    from arctic_platform.integrations.harbor.snowflake_sandbox import images

    registry = images.ImageRegistry("DB.SCHEMA.REPO", connection=None, role=None)
    token_dir = registry.docker_config
    registry.close()
    assert not token_dir.exists()
    with pytest.raises(images.ImageMirrorError, match="closed"):
        registry.docker_config
    with pytest.raises(images.ImageMirrorError, match="closed"):
        registry._ca_layer()


def test_closing_the_adapter_pauses_a_blocked_openhands_conversation(tmp_path, monkeypatch, scripted_llm):
    llm, base_url = scripted_llm
    llm.script = [("terminal", dict(command="echo step"))] * 9 + [("harbor_finish", dict(summary="done"))]
    llm.block_on_call = 2
    monkeypatch.setenv("HARBOR_ALLOW_HOST_EXEC", "1")
    monkeypatch.delenv("OPENHANDS_TOOLS", raising=False)
    before = set(threading.enumerate())
    adapter = HarborTrialAdapter(
        "openai/test-model",
        agent="harbor_rl.agent:OpenHandsHarborAgent",
        env="host",
        trials_root=str(tmp_path / "trials"),
        max_turns=10,
    )
    adapter.submit(_request(_task(tmp_path), "r0", base_url=base_url))
    assert llm.blocked.wait(30), "the conversation never reached the blocked model call"

    closer = threading.Thread(target=adapter.close, kwargs=dict(timeout_s=30))
    closer.start()
    time.sleep(0.3)
    llm.release.set()
    closer.join(30)

    assert not closer.is_alive()
    assert len(llm.calls) == 2, "the conversation kept calling the model after its rollout was cancelled"
    assert adapter.get_status("r0") == "errored"
    assert not [t for t in set(threading.enumerate()) - before if t.name.startswith("harbor-")]


def test_a_stopped_job_still_closes_the_adapter(monkeypatch):
    from harbor_rl import entrypoint

    closed: list[str] = []

    class StoppedExperiment:
        def __init__(self, cfg, reconnect_config):
            self.harbor_adapter = SimpleNamespace(close=lambda: closed.append("adapter"))
            self.sampling_proxy = SimpleNamespace(stop=lambda: closed.append("proxy"))

        def run(self):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)

    def ray_sigterm_handler(signum, frame):
        raise SystemExit(1)

    monkeypatch.setitem(
        sys.modules, "arctic_platform.integrations.skyrl.entrypoint", SimpleNamespace(patch_upstream=lambda: None)
    )
    monkeypatch.setattr(entrypoint, "_experiment_cls", lambda: StoppedExperiment)
    previous = signal.signal(signal.SIGTERM, ray_sigterm_handler)
    try:
        with pytest.raises(SystemExit):
            entrypoint._run_remote(None, None)
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert closed == ["adapter", "proxy"]


def test_the_default_backend_is_a_sandbox_and_host_is_an_explicit_alias():
    from arctic_platform.integrations.harbor.adapter import DEFAULT_HARBOR_ENV
    from arctic_platform.integrations.harbor.adapter import HOST_HARBOR_ENV
    from arctic_platform.integrations.harbor.adapter import SNOWFLAKE_HARBOR_ENV
    from arctic_platform.integrations.harbor.adapter import resolve_env

    assert DEFAULT_HARBOR_ENV == SNOWFLAKE_HARBOR_ENV
    assert resolve_env("host") == HOST_HARBOR_ENV
    assert resolve_env("snowflake") == SNOWFLAKE_HARBOR_ENV
    assert resolve_env("docker") == "docker"


def test_env_kwargs_are_passed_to_the_selected_environment(tmp_path):
    from arctic_platform.integrations.harbor.adapter import SNOWFLAKE_HARBOR_ENV
    from arctic_platform.integrations.harbor.adapter import _build_trial_config

    kwargs = dict(image_repo="DB.SCHEMA.REPO", connection="CONN", role="ROLE", image_platform="linux/arm64")
    config = _build_trial_config(
        _task(tmp_path),
        tmp_path / "trials",
        "trial",
        agent="harbor_rl.agent:OpenHandsHarborAgent",
        env="snowflake",
        model_name="m",
        base_url="http://127.0.0.1:1/v1",
        api_key="k",
        env_kwargs=kwargs,
    )
    assert config.environment.import_path == SNOWFLAKE_HARBOR_ENV
    assert config.environment.kwargs == kwargs


def test_image_platform_reaches_every_crane_call(monkeypatch):
    pytest.importorskip("snowflake.sandbox")
    from arctic_platform.integrations.harbor.snowflake_sandbox import images

    calls: list[list[str]] = []

    async def fake_run(*argv, env=None, **kwargs):
        calls.append(list(argv))
        return "sha256:abc"

    async def fake_login() -> None:
        return None

    async def fake_host() -> str:
        return "org-acct.registry.snowflakecomputing.com"

    monkeypatch.setattr(images, "_run", fake_run)
    registry = images.ImageRegistry("DB.SCHEMA.REPO", connection=None, role=None, platform="linux/arm64")
    monkeypatch.setattr(registry, "_login", fake_login)
    monkeypatch.setattr(registry, "_registry_host", fake_host)
    monkeypatch.setattr(registry, "_ca_layer", lambda: Path("/dev/null"))
    asyncio.run(registry.ensure_image("python:3.12-slim", force_build=True))

    platform_calls = [c for c in calls if c[1] in ("digest", "append") and "--platform" in c]
    assert platform_calls and all(c[c.index("--platform") + 1] == "linux/arm64" for c in platform_calls)
    registry.close()


def test_a_dropped_transfer_stream_is_retried(tmp_path, monkeypatch):
    pytest.importorskip("snowflake.sandbox")
    from snowflake.sandbox.exceptions import SandboxTransportError

    from arctic_platform.integrations.harbor.snowflake_sandbox import transfer

    async def no_sleep(_):
        return None

    async def exec_fn(cmd: str) -> ExecResult:
        return ExecResult(return_code=0, stdout="5", stderr="")

    class FlakyTransport:
        def __init__(self, drops: int):
            self.drops, self.calls = drops, 0

        async def download_file(self, remote, local):
            self.calls += 1
            if self.calls <= self.drops:
                raise SandboxTransportError("transport error mid-download: ")
            Path(local).write_text("reward", encoding="utf-8")

    monkeypatch.setattr(transfer.asyncio, "sleep", no_sleep)
    flaky = FlakyTransport(drops=1)
    asyncio.run(transfer.download_path(flaky, exec_fn, "/logs/verifier/reward.txt", tmp_path / "reward.txt"))
    assert (tmp_path / "reward.txt").read_text() == "reward" and flaky.calls == 2

    dead = FlakyTransport(drops=99)
    with pytest.raises(SandboxTransportError):
        asyncio.run(transfer.download_path(dead, exec_fn, "/logs/verifier/reward.txt", tmp_path / "other.txt"))
    assert dead.calls == transfer.TRANSFER_ATTEMPTS


def test_the_snowflake_sandbox_can_replay_the_recipes_task_dockerfile(tmp_path):
    from harbor_rl.entrypoint import materialize_harbor_task_from_row

    from arctic_platform.integrations.harbor.snowflake_sandbox.dockerfile_replay import plan_from_dockerfile

    task = materialize_harbor_task_from_row(_row(), tmp_path / "tasks")
    plan = plan_from_dockerfile(task / "environment" / "Dockerfile")
    assert plan.base_image == "python:3.11-slim"
    assert plan.workdir.startswith("/app/")
    assert any("git clone" in str(step) for step in plan.steps)


def test_the_harbor_extra_is_pinned_and_installed_by_the_launcher():
    import tomllib

    root = Path(__file__).resolve().parents[3]
    extras = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert "harbor==0.25.0" in extras["harbor"]
    assert any(dep.startswith("snowflake-sandbox-python==") for dep in extras["harbor"])

    launcher = (root / "recipes" / "rl" / "harbor" / "run_qwen35_4b.sh").read_text(encoding="utf-8")
    assert "[sft,cortex,harbor]" in launcher
    assert "--with harbor " not in launcher
