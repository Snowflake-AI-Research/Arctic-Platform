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
import stat
import threading
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

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

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append(body)
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
            api_key="k",
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
    assert _SECRET not in json.dumps(out["messages"])
    tools = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert "terminal" in tools and "file_editor" not in tools


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


def test_env_kwargs_reach_the_environment_and_snowflake_requires_an_image_repo(monkeypatch):
    from harbor_rl.entrypoint import check_harbor_env
    from harbor_rl.entrypoint import harbor_env_kwargs

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
