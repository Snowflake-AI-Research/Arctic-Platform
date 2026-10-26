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
"""Harbor environment backed by Snowflake Cortex Sandboxes.

Use it from Harbor with ``--env snowflake`` (when installed from upstream
Harbor) or the import-path form::

    harbor run -d terminal-bench@2.0 -a openhands-sdk \\
        --env arctic_platform.integrations.harbor.snowflake_sandbox:SnowflakeSandboxEnvironment \\
        --ek connection=MY_CONNECTION --ek role=MY_ROLE \\
        --ek image_repo=DB.SCHEMA.REPO

Task images are mirrored into ``image_repo`` with ``crane`` (no Docker), and
single-stage Dockerfiles are replayed inside the sandbox after it starts.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
import shlex
import time
import uuid
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from typing import override

from harbor.environments.base import BaseEnvironment
from harbor.environments.base import ExecResult
from harbor.environments.base import SandboxBuildFailedError
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.environments.definition import effective_exec_cwd
from harbor.environments.definition import require_agent_environment_definition
from harbor.environments.definition import should_use_prebuilt_docker_image
from harbor.models.task.config import EnvironmentConfig
from harbor.models.task.config import NetworkMode
from harbor.models.trial.paths import TrialPaths

from arctic_platform.integrations.harbor.snowflake_sandbox import transfer
from arctic_platform.integrations.harbor.snowflake_sandbox.dockerfile_replay import CopyStep
from arctic_platform.integrations.harbor.snowflake_sandbox.dockerfile_replay import ReplayPlan
from arctic_platform.integrations.harbor.snowflake_sandbox.dockerfile_replay import RunStep
from arctic_platform.integrations.harbor.snowflake_sandbox.dockerfile_replay import base_image
from arctic_platform.integrations.harbor.snowflake_sandbox.dockerfile_replay import plan_from_dockerfile
from arctic_platform.integrations.harbor.snowflake_sandbox.images import ImageRegistry
from arctic_platform.integrations.harbor.snowflake_sandbox.images import get_registry
from arctic_platform.integrations.harbor.snowflake_sandbox.shell import LOST_EXIT
from arctic_platform.integrations.harbor.snowflake_sandbox.shell import build_exec_argv
from arctic_platform.integrations.harbor.snowflake_sandbox.shell import is_root

try:
    from snowflake.sandbox import AsyncSandbox
    from snowflake.sandbox import Egress
    from snowflake.sandbox.exceptions import SandboxTransportError

    from arctic_platform.integrations.harbor.snowflake_sandbox.session import get_config

    _HAS_SDK = True
except ImportError:  # pragma: no cover - exercised only without the SDK
    _HAS_SDK = False

    class SandboxTransportError(Exception):  # type: ignore[no-redef]
        pass


logger = logging.getLogger(__name__)

ENVIRONMENT_TYPE = "snowflake-sandbox"

_MEMORY_TIERS_GIB = (1, 4, 8, 16, 32, 64)
_DEFAULT_MAX_LIVE = 90
_DEFAULT_EXEC_TIMEOUT_S = 7200
_SIGKILL_EXIT = 137
_STREAM_SLACK_S = 120
_MAX_REATTACH = 5
_LIVE_SLOTS: dict[int, asyncio.Semaphore] = {}
_warned: set[str] = set()

_SANDBOX_PREP = r"""
if [ -d /etc/apt/apt.conf.d ]; then
  mkdir -p /var/tmp/apt-cache/archives/partial /var/tmp/apt-lists/partial
  cp -r /var/lib/apt/lists/. /var/tmp/apt-lists/ 2>/dev/null || true
  rm -f /etc/apt/apt.conf.d/docker-clean
  cat > /etc/apt/apt.conf.d/99-cortex-sandbox <<'EOF'
APT::Sandbox::User "root";
Dir::Cache "/var/tmp/apt-cache";
Dir::State::Lists "/var/tmp/apt-lists";
EOF
fi
if command -v dpkg-divert >/dev/null 2>&1 && [ ! -e /usr/sbin/update-ca-certificates.harbor ]; then
  mkdir -p /usr/sbin
  dpkg-divert --quiet --local --rename --divert /usr/sbin/update-ca-certificates.harbor \
    --add /usr/sbin/update-ca-certificates 2>/dev/null || true
  cat > /usr/sbin/update-ca-certificates <<'EOF'
#!/bin/sh
# Cortex Sandboxes mount /etc/ssl/certs/ca-certificates.crt read-only.
[ -x /usr/sbin/update-ca-certificates.harbor ] && /usr/sbin/update-ca-certificates.harbor "$@" 2>/dev/null
exit 0
EOF
  chmod 755 /usr/sbin/update-ca-certificates
fi
"""


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def _csv(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return tuple(v.strip() for v in str(value).split(",") if v.strip())


def memory_tier(memory_mb: int | None) -> str:
    """Smallest sandbox memory tier that fits ``memory_mb`` (default 4g)."""
    if not memory_mb:
        return "4g"
    for gib in _MEMORY_TIERS_GIB:
        if memory_mb <= gib * 1024:
            return f"{gib}g"
    raise ValueError(
        f"Task needs {memory_mb} MB of memory; Cortex Sandboxes offer at most {_MEMORY_TIERS_GIB[-1]} GiB."
    )


def _sandbox_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value)[:120]


class SnowflakeSandboxEnvironment(BaseEnvironment):
    """Runs each Harbor trial in its own Cortex Sandbox container."""

    @classmethod
    @override
    def preflight(cls) -> None:
        if not _HAS_SDK:
            raise SystemExit(
                "The Snowflake sandbox environment needs the Cortex Sandboxes SDK: "
                "pip install snowflake-sandbox-python"
            )
        ImageRegistry.preflight()

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args: Any,
        connection: str | None = None,
        role: str | None = None,
        image_repo: str | None = None,
        prebuilt_image: str | None = None,
        egress_eai: str | list[str] | None = None,
        no_network_mode: str = "reject",
        max_live: int | str = _DEFAULT_MAX_LIVE,
        exec_timeout_sec: int | str = _DEFAULT_EXEC_TIMEOUT_S,
        idle_suspend: str = "24h",
        run_entrypoint: bool | str = True,
        **kwargs: Any,
    ) -> None:
        if not _HAS_SDK:
            self.preflight()
        self._no_network_mode = str(no_network_mode).strip().lower()
        if self._no_network_mode not in {"reject", "closed"}:
            raise ValueError("no_network_mode must be 'reject' or 'closed'")
        self._prebuilt_image = prebuilt_image

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        self._connection = connection
        self._role = role
        self._image_repo = image_repo
        self._egress_eais = _csv(egress_eai)
        self._max_live = int(max_live)
        self._exec_timeout_sec = int(exec_timeout_sec)
        self._idle_suspend = idle_suspend

        self._sandbox: AsyncSandbox | None = None
        self._slot_held = False
        self._workdir: str | None = None
        self._image_user: str | None = None
        self._entrypoint: list[str] | None = None
        self._run_entrypoint = str(run_entrypoint).strip().lower() not in {"0", "false", "no", "off"}
        self._finished_execs: list[str] = []

    @staticmethod
    @override
    def type() -> str:
        return ENVIRONMENT_TYPE

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_request=True, memory_request=True)

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=self._no_network_mode == "closed")

    @property
    def _dockerfile(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @override
    def _validate_definition(self) -> None:
        if self._prebuilt_image:
            return
        require_agent_environment_definition(
            self.environment_dir,
            docker_image=self.task_env_config.docker_image,
        )
        if (self.environment_dir / "docker-compose.yaml").exists():
            raise ValueError("docker-compose tasks aren't supported by the Snowflake sandbox environment")

    def _require_sandbox(self) -> AsyncSandbox:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        return self._sandbox

    def _slots(self) -> asyncio.Semaphore:
        return _LIVE_SLOTS.setdefault(self._max_live, asyncio.Semaphore(self._max_live))

    def _egress(self) -> Egress:
        if self.network_policy.network_mode == NetworkMode.NO_NETWORK:
            return Egress.only()
        if self._egress_eais:
            return Egress(external_access_integrations=self._egress_eais)
        return Egress()

    async def _mirror(self, ref: str) -> str:
        if ref.startswith("/"):
            return ref
        if not self._image_repo:
            raise ValueError(
                f"Task image {ref!r} must be mirrored into Snowflake: pass --ek image_repo=DATABASE.SCHEMA.REPOSITORY"
            )
        registry = get_registry(self._image_repo, connection=self._connection, role=self._role)
        return await registry.ensure_image(ref)

    async def _resolve_image(self, force_build: bool) -> tuple[str, ReplayPlan | None]:
        if self._prebuilt_image:
            image = await self._mirror(self._prebuilt_image)
            self._workdir, self._image_user, self._entrypoint = await self._image_defaults(image)
            return image, None
        docker_image = self.task_env_config.docker_image
        if should_use_prebuilt_docker_image(self.environment_dir, docker_image=docker_image, force_build=force_build):
            image = await self._mirror(docker_image)
            self._workdir, self._image_user, self._entrypoint = await self._image_defaults(image)
            return image, None
        image = await self._mirror(base_image(self._dockerfile))
        workdir, user, entrypoint = await self._image_defaults(image)
        plan = plan_from_dockerfile(self._dockerfile, base_workdir=workdir, base_user=user, base_entrypoint=entrypoint)
        self._entrypoint = plan.entrypoint
        return image, plan

    async def _image_defaults(self, image: str) -> tuple[str | None, str | None, list[str] | None]:
        if not self._image_repo:
            return None, None, None
        registry = get_registry(self._image_repo, connection=self._connection, role=self._role)
        try:
            config = await registry.image_config(image)
        except Exception as e:
            logger.warning("Couldn't read image config for %s: %s", image, e)
            return None, None, None
        workdir = config.get("WorkingDir") or None
        user = (config.get("User") or "").split(":")[0] or None
        entrypoint = config.get("Entrypoint") or None
        return workdir, None if user in ("root", "0") else user, entrypoint

    async def _start_entrypoint(self) -> None:
        assert self._entrypoint is not None
        argv = [*self._entrypoint, "sh", "-c", "sleep infinity"]
        launch = (
            "mkdir -p /var/log/harbor && "
            "if command -v setsid >/dev/null 2>&1; then s=setsid; else s=nohup; fi; "
            f"$s {shlex.join(argv)} > /var/log/harbor/entrypoint.log 2>&1 < /dev/null & "
            "sleep 2"
        )
        result = await self.exec(launch, user="root", timeout_sec=60)
        if result.return_code != 0:
            logger.warning("Couldn't start image ENTRYPOINT %s: %s", argv, result.stderr[-300:])

    @override
    async def start(self, force_build: bool) -> None:
        image, plan = await self._resolve_image(force_build)

        if self.network_policy.network_mode == NetworkMode.PUBLIC and not self._egress_eais:
            logger.warning(
                "No egress_eai configured: sandboxes reach Snowflake and public package "
                "registries only, not the general internet."
            )

        await self._slots().acquire()
        self._slot_held = True
        try:
            env = {**(plan.env if plan else {}), **self._startup_env()}
            started = time.monotonic()
            config = await get_config(self._connection, self._role)
            self._sandbox = await AsyncSandbox.create(
                image=image,
                connection=config,
                memory=memory_tier(self._effective_memory_mb),
                cpu=float(self._effective_cpus) if self._effective_cpus else None,
                env=env or None,
                egress=self._egress(),
                idle_suspend=self._idle_suspend,
                tags={
                    "harbor_session": _sandbox_tag(self.session_id),
                    "harbor_environment": _sandbox_tag(self.environment_name),
                },
            )
            logger.debug(
                "Sandbox %s ready from %s in %.1fs",
                self._sandbox.id,
                image,
                time.monotonic() - started,
            )
            prep = await self.exec(_SANDBOX_PREP, user="root")
            if prep.return_code != 0:
                logger.warning("Sandbox prep failed (%s): %s", prep.return_code, prep.stderr[-300:])
            if plan is not None:
                await self._replay(plan)
            await self.ensure_dirs(self._mount_targets(writable_only=True))
            await self._upload_environment_dir_after_start()
            if self._entrypoint and self._run_entrypoint:
                await self._start_entrypoint()
        except BaseException:
            await self.stop(delete=True)
            raise

    async def _replay(self, plan: ReplayPlan) -> None:
        for step in plan.steps:
            if isinstance(step, RunStep):
                result = await self.exec(step.command, cwd=step.workdir, env=step.env, user=step.user or "root")
                if result.return_code != 0:
                    raise SandboxBuildFailedError(
                        f"Dockerfile RUN failed ({result.return_code}): {step.command[:200]}\n"
                        f"{(result.stderr or result.stdout or '')[-2000:]}"
                    )
            elif isinstance(step, CopyStep):
                await self._replay_copy(step)
        self._workdir = plan.workdir if plan.workdir != "/" else None
        self._image_user = plan.user if plan.user not in ("root", "0") else None

    async def _replay_copy(self, step: CopyStep) -> None:
        context = self.environment_dir
        matched: list[Path] = []
        for pattern in step.sources:
            if any(ch in pattern for ch in "*?["):
                found = [
                    p for p in context.glob(pattern) if fnmatch.fnmatch(p.relative_to(context).as_posix(), pattern)
                ]
            else:
                found = [context / pattern]
            if not found or not all(p.exists() for p in found):
                raise SandboxBuildFailedError(f"COPY source not found in build context: {pattern}")
            matched.extend(found)

        dest_is_dir = step.dest_is_dir or len(matched) > 1 or (len(matched) == 1 and matched[0].is_dir())
        for src in matched:
            if src.is_dir():
                await self.upload_dir(src, step.dest)
            else:
                target = str(PurePosixPath(step.dest) / src.name) if dest_is_dir else step.dest
                await self.upload_file(src, target)
        if step.chown:
            _warn_once(
                "chown",
                "COPY --chown is ignored: Cortex Sandboxes can't change file ownership (%s on %s).",
                step.chown,
                step.dest,
            )

    @override
    async def stop(self, delete: bool) -> None:
        if not delete:
            logger.debug("Cortex Sandboxes are ephemeral and are always terminated on stop.")
        try:
            if self._sandbox is not None:
                try:
                    await self._sandbox.terminate()
                except Exception as e:
                    logger.error("Error terminating sandbox %s: %s", self._sandbox.id, e)
                finally:
                    self._sandbox = None
        finally:
            if self._slot_held:
                self._slots().release()
                self._slot_held = False

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        sandbox = self._require_sandbox()
        user = self._resolve_user(user)
        if user is None:
            user = self._image_user
        if not is_root(user):
            _warn_once(
                "user",
                "Cortex Sandboxes can't switch users; commands for user %r run as sandbox root with that user's"
                " HOME/USER.",
                user,
            )
        exec_id = uuid.uuid4().hex
        cleanup, self._finished_execs = self._finished_execs, []
        argv = build_exec_argv(
            command,
            exec_id,
            cwd=effective_exec_cwd(cwd, self.task_env_config.workdir, self._workdir),
            env=self._merge_env(env),
            user=user,
            timeout_sec=timeout_sec,
            cleanup=cleanup,
        )
        budget = timeout_sec + _STREAM_SLACK_S if timeout_sec else self._exec_timeout_sec
        started = time.monotonic()
        for attempt in range(_MAX_REATTACH + 1):
            try:
                stdout, stderr, exit_code = await self._read_exec_stream(sandbox, argv, budget)
                break
            except SandboxTransportError as e:
                if attempt == _MAX_REATTACH:
                    raise
                logger.warning("Exec stream dropped (%s); re-attaching to %s", e, exec_id)
                await asyncio.sleep(min(2**attempt, 10))
                argv = build_exec_argv("", exec_id, reattach=True)
        self._finished_execs.append(exec_id)

        if exit_code == LOST_EXIT and "harbor: lost track of exec" in stderr:
            raise RuntimeError(f"Lost track of command after the exec stream dropped: {command[:200]}")
        if timeout_sec and exit_code == _SIGKILL_EXIT and time.monotonic() - started >= timeout_sec - 1:
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")

        return ExecResult(stdout=stdout, stderr=stderr, return_code=exit_code)

    @staticmethod
    async def _read_exec_stream(sandbox: AsyncSandbox, argv: list[str], budget: float) -> tuple[str, str, int]:
        stream = sandbox.exec_stream(argv, timeout=budget)
        stdout: list[str] = []
        stderr: list[str] = []
        async for line in stream:
            (stdout if line.stream == "stdout" else stderr).append(line.data)
        exit_code = stream.exit_code
        return (
            "\n".join(stdout) + ("\n" if stdout else ""),
            "\n".join(stderr) + ("\n" if stderr else ""),
            exit_code if exit_code is not None else -1,
        )

    async def _root_exec(self, command: str) -> ExecResult:
        return await self.exec(command, user="root")

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await transfer.upload_path(self._require_sandbox(), self._root_exec, Path(source_path), target_path)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await transfer.upload_dir(self._require_sandbox(), self._root_exec, Path(source_dir), target_dir)

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await transfer.download_path(self._require_sandbox(), self._root_exec, source_path, Path(target_path))

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await transfer.download_dir(self._require_sandbox(), self._root_exec, source_dir, Path(target_dir))
