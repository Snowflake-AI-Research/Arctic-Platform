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
"""One Snowflake session per (connection, role), shared by every trial.

The cache is guarded by thread locks rather than asyncio locks, so trials on
different event loops (or threads) share one session without binding a lock
to whichever loop touched it first.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from snowflake.sandbox.config import Config
from snowflake.sandbox.config import resolve_config

logger = logging.getLogger(__name__)

LOGIN_TIMEOUT_S = 300

_configs: dict[tuple[str | None, str | None], Config] = {}
_locks: dict[tuple[str | None, str | None], threading.Lock] = {}
_locks_guard = threading.Lock()


def _connect(connection: str | None, role: str | None) -> Config:
    import snowflake.connector

    kwargs: dict[str, Any] = {"login_timeout": LOGIN_TIMEOUT_S, "client_session_keep_alive": True}
    if connection:
        kwargs["connection_name"] = connection
    if role:
        kwargs["role"] = role
    logger.info(
        "Logging in to Snowflake%s (SSO connections open a browser once per run)",
        f" with connection {connection!r}" if connection else "",
    )
    conn = snowflake.connector.connect(**kwargs)
    return resolve_config(connection=conn)


def _get_or_connect(connection: str | None, role: str | None) -> Config:
    key = (connection, role)
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        if key not in _configs:
            _configs[key] = _connect(connection, role)
        return _configs[key]


async def get_config(connection: str | None, role: str | None) -> Config:
    """Return the shared sandbox ``Config`` backed by one live connector session."""
    cached = _configs.get((connection, role))
    if cached is not None:
        return cached
    return await asyncio.to_thread(_get_or_connect, connection, role)


def close_sessions() -> None:
    """Close every cached connector session; the next trial logs in again."""
    with _locks_guard:
        configs = list(_configs.values())
        _configs.clear()
    for config in configs:
        if config.connection is None:
            continue
        try:
            config.connection.close()
        except Exception as exc:
            logger.warning("Closing a Snowflake session failed: %s", exc)


async def get_connector(connection: str | None, role: str | None) -> Any:
    """Return the live ``snowflake.connector`` connection behind the shared session."""
    config = await get_config(connection, role)
    if config.connection is None:
        raise RuntimeError("The shared Snowflake session has no live connector connection")
    return config.connection
