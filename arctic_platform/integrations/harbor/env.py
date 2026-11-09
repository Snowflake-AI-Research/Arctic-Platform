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
"""Host-subprocess Harbor ``BaseEnvironment``.

Runs Harbor trials as host subprocesses under a per-trial root directory
(``{trial_dir}/host_env``). Harbor's canonical container paths (``/app``,
``/workspace``, ``/tests``, ``/logs``, ``/solution``, ``/harbor``) are
rewritten to that per-trial root, and single-stage Dockerfile ``WORKDIR``,
``COPY``, and ``RUN`` directives are replayed on ``start()`` so Harbor tasks
and ``tests/test.sh`` verifiers execute identically on hosts without Docker.

This is not a sandbox: task ``RUN`` lines, verifiers, and agent commands run
as the current user and can read anything that user can. It is meant for
trusted tasks only, so ``start()`` refuses to run unless
``HARBOR_ALLOW_HOST_EXEC=1`` (or ``allow_host_exec=True``) is set. Commands get
an allowlisted environment (``HOST_ENV_ALLOWLIST`` plus any names in
``HARBOR_HOST_ENV_PASSTHROUGH``), never the caller's full ``os.environ``, and
every file path is resolved and must stay inside the trial root or, for
Dockerfile ``COPY`` sources, inside the task's ``environment/`` directory.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    from harbor.environments.base import BaseEnvironment
    from harbor.environments.base import ExecResult
except ImportError:  # pragma: no cover - allows importing helper functions without harbor installed
    BaseEnvironment = object  # type: ignore[assignment,misc]
    ExecResult = None  # type: ignore[assignment,misc]

HOST_ENV_ALLOWLIST = ("PATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "TMPDIR", "SHELL")
ALLOW_HOST_EXEC_VAR = "HARBOR_ALLOW_HOST_EXEC"
_ABS_PATH_TOKEN = re.compile(r"(?<![\w/])(/app|/workspace|/tests|/solution|/logs|/harbor)(?=/|$|[\s'\"\\])")
_REWRITE_SUFFIXES = (".sh", ".py", ".bash", ".zsh", ".txt")


def rewrite_harbor_paths(text: str, root: Path) -> str:
    """Replace Harbor container prefixes with the host trial root."""
    if not text:
        return text
    return _ABS_PATH_TOKEN.sub(lambda m: f"{root}{m.group(1)}", text)


class PathEscapeError(ValueError):
    """A task-supplied path resolves outside the directory it must stay in."""


def contained_path(root: Path, candidate: Path, what: str) -> Path:
    """``candidate`` resolved (symlinks included), or ``PathEscapeError`` if it leaves ``root``."""
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise PathEscapeError(f"{what} {candidate} resolves outside {root}")
    return resolved


def host_exec_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The allowlisted subset of ``os.environ`` that host commands may see, plus ``extra``."""
    names = set(HOST_ENV_ALLOWLIST)
    names.update(n.strip() for n in os.environ.get("HARBOR_HOST_ENV_PASSTHROUGH", "").split(",") if n.strip())
    env = {name: os.environ[name] for name in names if name in os.environ}
    env.update(extra or {})
    return env


def _parse_dockerfile_lines(dockerfile: Path) -> list[str]:
    if not dockerfile.exists():
        return []
    joined: list[str] = []
    buf = ""
    for raw in dockerfile.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.endswith("\\"):
            buf += s[:-1] + " "
        else:
            buf += s
            joined.append(buf.strip())
            buf = ""
    return joined


def _install_shims(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    for tool in ("apt-get", "apt"):
        p = bin_dir / tool
        p.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        p.chmod(0o755)
    curl_shim = bin_dir / "curl"
    curl_shim.write_text(
        '#!/usr/bin/env bash\nif [[ "$*" == *"astral.sh/uv"* ]]; then\n  exit 0\nfi\nexec /usr/bin/curl "$@"\n',
        encoding="utf-8",
    )
    curl_shim.chmod(0o755)
    uvx_shim = bin_dir / "uvx"
    uvx_shim.write_text(
        "#!/usr/bin/env bash\n"
        "args=()\n"
        "while [[ $# -gt 0 ]]; do\n"
        '  if [[ "$1" == "-p" || "$1" == "--python" ]]; then\n'
        "    shift 2\n"
        "  else\n"
        '    args+=("$1")\n'
        "    shift\n"
        "  fi\n"
        "done\n"
        "if command -v uv >/dev/null 2>&1; then\n"
        '  exec uv tool run "${args[@]}"\n'
        "fi\n"
        'exec python3 -m pytest "${args[@]: -3}"\n',
        encoding="utf-8",
    )
    uvx_shim.chmod(0o755)


def host_exec_allowed(flag: bool | str | None) -> bool:
    value = os.environ.get(ALLOW_HOST_EXEC_VAR, "") if flag is None else flag
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class HostEnvironment(BaseEnvironment):
    """Runs trusted Harbor trials as host subprocesses under a per-trial root dir."""

    def __init__(self, *args: Any, allow_host_exec: bool | str | None = None, **kwargs: Any) -> None:
        self._allow_host_exec = allow_host_exec
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "host"

    def _validate_definition(self) -> None:
        return

    def _uses_compose(self) -> bool:  # type: ignore[override]
        return False

    @property
    def host_root(self) -> Path:
        return self._root

    @property
    def working_dir(self) -> Path:
        return self._workdir

    async def start(self, force_build: bool) -> None:  # noqa: ARG002
        if not host_exec_allowed(self._allow_host_exec):
            raise RuntimeError(
                "HostEnvironment runs task and agent commands on this machine without isolation. "
                "Use a sandbox (HARBOR_ENV=snowflake, docker, modal, ...), or set "
                f"{ALLOW_HOST_EXEC_VAR}=1 if every task is trusted."
            )
        self._root = (Path(str(self.trial_paths.trial_dir)) / "host_env").resolve()
        for sub in (
            "app",
            "workspace",
            "tests",
            "solution",
            "logs/agent",
            "logs/verifier",
            "logs/artifacts",
            "harbor/skills",
            "runtime/skills",
            "bin",
        ):
            (self._root / sub).mkdir(parents=True, exist_ok=True)
        _install_shims(self._root / "bin")

        self._workdir = self._root / "app"
        await asyncio.to_thread(self._replay_environment_dir)

    def _replay_environment_dir(self) -> None:
        if not self.environment_dir or not self.environment_dir.exists():
            return
        env_dir = self.environment_dir.resolve()
        dockerfile = env_dir / "Dockerfile"
        lines = _parse_dockerfile_lines(dockerfile)

        env = self._command_env()
        (self._root / ".local" / "bin").mkdir(parents=True, exist_ok=True)
        (self._root / ".local" / "bin" / "env").write_text(
            f'export PATH="{self._root / "bin"}:$PATH"\n', encoding="utf-8"
        )

        has_copy = any(s.startswith("COPY ") and "--from=" not in s for s in lines)
        if not has_copy:
            for item in env_dir.iterdir():
                if item.name == "Dockerfile":
                    continue
                contained_path(env_dir, item, "Build-context entry")
                target = self._workdir / item.name
                if item.is_dir():
                    self._copytree_rewriting(item, target, boundary=env_dir)
                else:
                    self._copy_rewriting(item, target)

        for s in lines:
            if s.startswith("WORKDIR "):
                raw_wd = s.split(None, 1)[1].strip()
                self._workdir = self._to_host(raw_wd)
                self._workdir.mkdir(parents=True, exist_ok=True)
            elif s.startswith("COPY ") and "--from=" not in s:
                parts = shlex.split(s)[1:]
                if len(parts) < 2:
                    continue
                *srcs, dst_raw = parts
                dst_path = self._to_host(dst_raw)
                for src_rel in srcs:
                    clean = src_rel.rstrip("/")
                    if clean.endswith("/."):
                        clean = clean[:-2]
                    if clean.startswith("/"):
                        raise PathEscapeError(f"COPY source {src_rel} is absolute; it must be inside {env_dir}")
                    src_path = contained_path(
                        env_dir, env_dir / clean if clean not in ("", ".", "./") else env_dir, "COPY source"
                    )
                    if not src_path.exists():
                        continue
                    if src_path.is_dir():
                        dst_path.mkdir(parents=True, exist_ok=True)
                        self._copytree_rewriting(src_path, dst_path, boundary=env_dir)
                    elif dst_raw.endswith("/") or dst_raw in (".", "./") or dst_path.is_dir():
                        dst_path.mkdir(parents=True, exist_ok=True)
                        self._copy_rewriting(src_path, dst_path / src_path.name)
                    else:
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        self._copy_rewriting(src_path, dst_path)
            elif s.startswith("RUN ") and "apt" not in s:
                cmd = rewrite_harbor_paths(s[4:].strip(), self._root)
                subprocess.run(
                    ["bash", "-lc", cmd],
                    cwd=str(self._workdir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                )

    async def stop(self, delete: bool) -> None:
        root = getattr(self, "_root", None)
        if delete and root is not None and root.exists():
            shutil.rmtree(root, ignore_errors=True)

    def _to_host(self, container_path: str) -> Path:
        p = container_path.strip()
        if p.startswith("/") and Path(p).resolve().is_relative_to(self._root):
            candidate = Path(p)
        elif p.startswith("/"):
            candidate = self._root / p.lstrip("/")
        else:
            candidate = self._workdir / p
        return contained_path(self._root, candidate, "Path")

    def _command_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = host_exec_env(extra)
        env["PATH"] = f"{self._root / 'bin'}:{env.get('PATH', '/usr/local/bin:/usr/bin:/bin')}"
        env["HOME"] = str(self._root)
        env["HARBOR_HOST_ROOT"] = str(self._root)
        return env

    def _copy_rewriting(self, src: Path, dst: Path) -> None:
        if src.suffix.lower() in _REWRITE_SUFFIXES:
            try:
                text = src.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                shutil.copy2(str(src), str(dst))
                return
            rewritten = rewrite_harbor_paths(text, self._root)
            app_dir = str(self._root / "app")
            rewritten = rewritten.replace(
                "-exec sha256sum {} + | sort | sha256sum",
                f"-exec sha256sum {{}} + | sed 's|{app_dir}|/app|g' | sort | sha256sum",
            )
            dst.write_text(rewritten, encoding="utf-8")
            try:
                dst.chmod(src.stat().st_mode)
            except OSError:
                pass
        else:
            shutil.copy2(str(src), str(dst))

    def _copytree_rewriting(self, src: Path, dst: Path, *, boundary: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            contained_path(boundary, item, "Copied entry")
            target = dst / item.name
            if item.is_dir():
                self._copytree_rewriting(item, target, boundary=boundary)
            else:
                self._copy_rewriting(item, target)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        dst = self._to_host(str(target_path))
        dst.parent.mkdir(parents=True, exist_ok=True)
        self._copy_rewriting(Path(source_path), dst)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir).resolve()
        dst = self._to_host(str(target_dir))
        dst.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            contained_path(source, item, "Uploaded entry")
            target = dst / item.name
            if item.is_dir():
                self._copytree_rewriting(item, target, boundary=source)
            else:
                self._copy_rewriting(item, target)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        src = self._to_host(str(source_path))
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(str(src), str(target_path))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        src = self._to_host(str(source_dir))
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        if not src.is_dir():
            return
        self._copytree_contained(src, Path(target_dir))

    def _copytree_contained(self, src: Path, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            resolved = contained_path(self._root, item, "Downloaded entry")
            if resolved.is_dir():
                self._copytree_contained(resolved, dst / item.name)
            else:
                shutil.copy2(str(resolved), str(dst / item.name))

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,  # noqa: ARG002
    ) -> ExecResult:
        rewritten = rewrite_harbor_paths(command, self._root)
        cwd_host = str(self._to_host(cwd)) if cwd else str(self._workdir)
        Path(cwd_host).mkdir(parents=True, exist_ok=True)

        merged_env = self._command_env(self._merge_env(env))

        proc = await asyncio.create_subprocess_shell(
            rewritten,
            cwd=cwd_host,
            env=merged_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_sec if timeout_sec else None
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult(stdout="", stderr=f"timed out after {timeout_sec}s", return_code=124)

        return ExecResult(
            stdout=stdout_b.decode(errors="replace"),
            stderr=stderr_b.decode(errors="replace"),
            return_code=proc.returncode or 0,
        )
