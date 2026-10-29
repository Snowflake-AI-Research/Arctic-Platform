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
"""Mirror task images into a Snowflake image repository for Cortex Sandboxes.

``crane`` authenticates to the Snowflake registry with a session token. Each
registry keeps that token in its own private ``DOCKER_CONFIG`` directory
(seeded from the user's config so source registries still resolve), never in
the user's default container-registry config, and deletes it on ``close()``
or at interpreter exit.
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import weakref
from pathlib import Path
from typing import Any

import certifi

from arctic_platform.integrations.harbor.snowflake_sandbox.session import _get_or_connect
from arctic_platform.integrations.harbor.snowflake_sandbox.session import get_connector

_LAYER_VERSION = "ca1"
_LOGIN_TTL_S = 45 * 60
_SF_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
# Cortex Sandboxes run x86-64 images.
DEFAULT_PLATFORM = "linux/amd64"

logger = logging.getLogger(__name__)


class ImageMirrorError(RuntimeError):
    pass


async def _run(
    *argv: str, stdin: bytes | None = None, timeout: float = 1800, env: dict[str, str] | None = None
) -> str:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
    except TimeoutError:
        proc.kill()
        raise ImageMirrorError(f"{argv[0]} timed out after {timeout:.0f}s")
    if proc.returncode != 0:
        raise ImageMirrorError(
            f"{' '.join(argv[:3])} failed ({proc.returncode}): {err.decode(errors='replace')[-800:]}"
        )
    return out.decode()


def _slug(ref: str) -> str:
    name = ref.split("@")[0].rsplit("/", 1)[-1].split(":")[0]
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-") or "image"


def _private_docker_config() -> Path:
    """A fresh mode-0700 ``DOCKER_CONFIG`` dir holding a copy of the user's config, if any."""
    path = Path(tempfile.mkdtemp(prefix="harbor-sf-crane-"))
    source = Path(os.environ.get("DOCKER_CONFIG") or Path.home() / ".docker") / "config.json"
    if source.is_file():
        shutil.copyfile(source, path / "config.json")
        (path / "config.json").chmod(0o600)
    return path


class ImageRegistry:
    """One Snowflake image repository, shared by every trial in the process."""

    def __init__(
        self,
        image_repo: str,
        *,
        connection: str | None,
        role: str | None,
        platform: str = DEFAULT_PLATFORM,
    ) -> None:
        parts = image_repo.strip().split(".")
        if len(parts) != 3 or not all(_SF_IDENT.match(p) for p in parts):
            raise ValueError(
                f"image_repo must be DATABASE.SCHEMA.REPOSITORY with valid Snowflake identifiers, got {image_repo!r}"
            )
        self.database, self.schema, self.repo = parts
        self.connection = connection
        self.role = role
        self.platform = platform
        self._host: str | None = None
        self._logged_in_at = 0.0
        self._login_guard = threading.RLock()
        self._image_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = (
            weakref.WeakKeyDictionary()
        )
        self._image_locks_guard = threading.Lock()
        self._ca_tar: Path | None = None
        self._docker_config: Path | None = None

    @property
    def docker_config(self) -> Path:
        with self._login_guard:
            if self._docker_config is None:
                self._docker_config = _private_docker_config()
            return self._docker_config

    def _crane_env(self) -> dict[str, str]:
        return {**os.environ, "DOCKER_CONFIG": str(self.docker_config)}

    async def _crane(self, *args: str, timeout: float = 1800) -> str:
        return await _run("crane", *args, timeout=timeout, env=self._crane_env())

    def close(self) -> None:
        """Delete the registry token and the CA layer this registry wrote to disk."""
        with self._login_guard:
            for path in (self._docker_config, self._ca_tar.parent if self._ca_tar else None):
                if path is not None:
                    shutil.rmtree(path, ignore_errors=True)
            self._docker_config = None
            self._ca_tar = None
            self._logged_in_at = 0.0

    def _image_lock(self, tag: str) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        with self._image_locks_guard:
            return self._image_locks.setdefault(loop, {}).setdefault(tag, asyncio.Lock())

    async def _query(self, sql: str) -> list[dict[str, Any]]:
        from snowflake.connector import DictCursor

        conn = await get_connector(self.connection, self.role)

        def run() -> list[dict[str, Any]]:
            with conn.cursor(DictCursor) as cur:
                return cur.execute(sql).fetchall()

        return await asyncio.to_thread(run)

    def _registry_token(self) -> str:
        conn = _get_or_connect(self.connection, self.role).connection
        rest = getattr(conn, "_rest", None) or getattr(conn, "rest", None)
        if rest is None:
            raise ImageMirrorError("Snowflake connection has no REST session for a registry token")
        data = rest._token_request("ISSUE")["data"]
        return json.dumps(
            {
                "token": data["sessionToken"],
                "expires_in": data["validityInSecondsST"],
            }
        )

    async def _registry_host(self) -> str:
        if self._host is None:
            rows = await self._query(
                f"SHOW IMAGE REPOSITORIES LIKE '{self.repo}' IN SCHEMA {self.database}.{self.schema}"
            )
            if not rows:
                raise ImageMirrorError(
                    f"Image repository {self.database}.{self.schema}.{self.repo} not found or not visible to role"
                    f" {self.role}"
                )
            self._host = rows[0]["repository_url"].split("/", 1)[0]
        return self._host

    @property
    def sandbox_prefix(self) -> str:
        return f"/{self.database}/{self.schema}/{self.repo}".lower()

    async def _login(self) -> None:
        if time.monotonic() - self._logged_in_at < _LOGIN_TTL_S:
            return
        host = await self._registry_host()
        await asyncio.to_thread(self._login_sync, host)

    def _login_sync(self, host: str) -> None:
        with self._login_guard:
            if time.monotonic() - self._logged_in_at < _LOGIN_TTL_S:
                return
            proc = subprocess.run(
                ["crane", "auth", "login", host, "-u", "0sessiontoken", "--password-stdin"],
                input=self._registry_token(),
                env=self._crane_env(),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if proc.returncode != 0:
                raise ImageMirrorError(f"crane auth login failed ({proc.returncode}): {proc.stderr[-800:]}")
            self._logged_in_at = time.monotonic()

    def _ca_layer(self) -> Path:
        with self._login_guard:
            return self._ca_layer_locked()

    def _ca_layer_locked(self) -> Path:
        if self._ca_tar is None:
            tmp = Path(tempfile.mkdtemp(prefix="harbor-sf-ca-"))
            tar_path = tmp / "ca-layer.tar"
            with tarfile.open(tar_path, "w") as tar:
                info = tar.gettarinfo(certifi.where(), arcname="etc/ssl/certs/ca-certificates.crt")
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
                info.mode = 0o644
                with open(certifi.where(), "rb") as f:
                    tar.addfile(info, f)
            self._ca_tar = tar_path
        return self._ca_tar

    @staticmethod
    def preflight() -> None:
        if shutil.which("crane") is None:
            raise SystemExit(
                "The Snowflake sandbox environment needs `crane` on PATH to mirror task images "
                "(brew install crane, or see github.com/google/go-containerregistry)."
            )

    async def ensure_image(self, source_ref: str, *, force_build: bool = False) -> str:
        await self._login()
        try:
            source_digest = (await self._crane("digest", "--platform", self.platform, source_ref, timeout=120)).strip()
        except ImageMirrorError:
            source_digest = source_ref
        digest = hashlib.sha256(f"{source_ref}|{source_digest}|{self.platform}|{_LAYER_VERSION}".encode()).hexdigest()[
            :16
        ]
        tag = f"{_slug(source_ref)}:{digest}"
        async with self._image_lock(tag):
            host = await self._registry_host()
            target = f"{host}{self.sandbox_prefix}/{tag}"
            need_mirror = force_build
            if not need_mirror:
                try:
                    await self._crane("digest", target, timeout=120)
                    logger.debug("Image %s already mirrored", target)
                except ImageMirrorError:
                    need_mirror = True
            if need_mirror:
                logger.info("Mirroring %s -> %s", source_ref, target)
                await self._crane(
                    "append", "--platform", self.platform, "-b", source_ref, "-f", str(self._ca_layer()), "-t", target
                )
        return f"{self.sandbox_prefix}/{tag}"

    async def image_config(self, sandbox_path: str) -> dict[str, Any]:
        await self._login()
        host = await self._registry_host()
        raw = await self._crane("config", "--platform", self.platform, f"{host}{sandbox_path}", timeout=120)
        return json.loads(raw).get("config") or {}


_registries: dict[tuple[str, str | None, str | None, str], ImageRegistry] = {}
_registries_guard = threading.Lock()


def get_registry(
    image_repo: str, *, connection: str | None, role: str | None, platform: str = DEFAULT_PLATFORM
) -> ImageRegistry:
    key = (image_repo.upper(), connection, role, platform)
    with _registries_guard:
        if key not in _registries:
            _registries[key] = ImageRegistry(image_repo, connection=connection, role=role, platform=platform)
        return _registries[key]


@atexit.register
def close_registries() -> None:
    """Delete every registry's private token config and CA layer."""
    with _registries_guard:
        registries = list(_registries.values())
        _registries.clear()
    for registry in registries:
        registry.close()
