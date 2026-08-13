# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-subprocess Harbor BaseEnvironment.

Harbor ships environments for Docker/Daytona/Modal/… but not a plain host
runner. This subclass provides one: every ``exec`` is a host subprocess and
every ``upload_dir``/``download_dir`` is a filesystem copy under a per-trial
root. Harbor's canonical container paths (``/tests``, ``/logs``, ``/solution``,
``/harbor``) are rewritten to that root before commands run, so tasks and
verifiers that use those paths behave the same as under Docker.

Not a substitute for a container sandbox — there's no isolation — but it
lets Harbor's trial runner drive tasks on hosts without Docker / podman /
user-namespace access. Development-only.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.task.config import TaskOS


# Harbor's default POSIX container paths (see harbor/models/trial/paths.py).
_ROOTED_PREFIXES = ("/tests", "/solution", "/logs", "/harbor")
_ABS_PATH_TOKEN = re.compile(
    r"(?<![\w/])(/tests|/solution|/logs|/harbor)(?=/|$|[\s'\"\\])"
)


def _rewrite_paths(text: str, root: Path) -> str:
    """Replace Harbor's canonical container prefixes with the host trial root.

    Only touches whole path tokens (word-boundary-ish) — leaves e.g. ``/tests``
    inside a URL or a Python string that happens to share a prefix alone.
    """
    if not text:
        return text
    return _ABS_PATH_TOKEN.sub(lambda m: f"{root}{m.group(1)}", text)


class HostEnvironment(BaseEnvironment):
    """Runs Harbor trials as host subprocesses under a per-trial root dir."""

    @staticmethod
    def type() -> str:
        return "host"

    def _validate_definition(self) -> None:  # nothing to validate — no image build
        return

    def _uses_compose(self) -> bool:  # type: ignore[override]
        return False

    # BaseEnvironment sets self.environment_dir / trial_paths / etc. in __init__.
    # We derive our host root once on start(); on stop(delete=True) we rm it.

    async def start(self, force_build: bool) -> None:  # noqa: ARG002 — no image
        self._root = Path(str(self.trial_paths.trial_dir)) / "host_env"
        for sub in ("tests", "solution", "logs/agent", "logs/verifier", "logs/artifacts", "harbor/skills"):
            (self._root / sub).mkdir(parents=True, exist_ok=True)
        # world-writable so subprocess writes as $USER succeed regardless of umask
        for p in self._root.rglob("*"):
            if p.is_dir():
                try:
                    p.chmod(0o777)
                except PermissionError:
                    pass

    async def stop(self, delete: bool) -> None:
        root = getattr(self, "_root", None)
        if delete and root is not None and root.exists():
            shutil.rmtree(root, ignore_errors=True)

    # ── path translation ────────────────────────────────────────────────────
    def _to_host(self, container_path: str) -> Path:
        """Map a Harbor container-absolute path to its host location."""
        for pref in _ROOTED_PREFIXES:
            if container_path == pref or container_path.startswith(pref + "/"):
                rel = container_path[1:]  # drop leading '/'
                return self._root / rel
        # Non-canonical absolute paths: honor as-is (e.g. /tmp).
        return Path(container_path)

    # ── file transfer (host-only: copy) ─────────────────────────────────────
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        dst = self._to_host(str(target_path))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source_path), str(dst))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        dst = self._to_host(str(target_dir))
        dst.mkdir(parents=True, exist_ok=True)
        # merge copy: don't nuke existing files at dst (e.g. reward.txt from a prior run)
        for item in Path(source_dir).iterdir():
            target = dst / item.name
            if item.is_dir():
                shutil.copytree(str(item), str(target), dirs_exist_ok=True)
            else:
                shutil.copy2(str(item), str(target))

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        src = self._to_host(str(source_path))
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(str(src), str(target_path))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        src = self._to_host(str(source_dir))
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        if not src.exists():
            return
        for item in src.iterdir():
            dst_item = Path(target_dir) / item.name
            if item.is_dir():
                shutil.copytree(str(item), str(dst_item), dirs_exist_ok=True)
            else:
                shutil.copy2(str(item), str(dst_item))

    # ── command execution ───────────────────────────────────────────────────
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,  # noqa: ARG002 — single-user host env
    ) -> ExecResult:
        rewritten = _rewrite_paths(command, self._root)
        cwd_host = str(self._to_host(cwd)) if cwd else str(self._root)
        Path(cwd_host).mkdir(parents=True, exist_ok=True)

        merged_env = os.environ.copy()
        merged = self._merge_env(env)
        if merged:
            merged_env.update(merged)
        # Expose the trial root so tasks that opt in can use it explicitly.
        merged_env["HARBOR_HOST_ROOT"] = str(self._root)

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
