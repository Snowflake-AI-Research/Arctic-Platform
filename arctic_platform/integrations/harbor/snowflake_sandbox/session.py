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

A session token lasts about an hour, and the connector's keep-alive heartbeat
can fire after it lapses. Sandbox create then answers ``390112`` with a 200,
which the SDK doesn't treat as a reason to renew. So the token is renewed here
every ``RENEW_AFTER_S``, and on demand after a create reports an expired token.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from snowflake.sandbox.config import REAUTH_GS_CODES
from snowflake.sandbox.config import Config
from snowflake.sandbox.config import refresh_config
from snowflake.sandbox.config import resolve_config

logger = logging.getLogger(__name__)

LOGIN_TIMEOUT_S = 300
RENEW_AFTER_S = 20 * 60

_configs: dict[tuple[str | None, str | None], Config] = {}
_renewed_at: dict[tuple[str | None, str | None], float] = {}
_retired: list[Any] = []
_locks: dict[tuple[str | None, str | None], threading.Lock] = {}
_locks_guard = threading.Lock()


def is_expired_session(exc: BaseException) -> bool:
    """Whether ``exc`` reports a Snowflake session token that needs renewing."""
    text = str(exc)
    return any(code in text for code in REAUTH_GS_CODES)


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


def _renew(config: Config, connection: str | None, role: str | None) -> Config:
    stale = config.resolve_pat()
    try:
        renewed = refresh_config(config, stale)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Renewing the Snowflake session token failed: %s", exc)
        renewed = config
    if renewed.resolve_pat() != stale:
        return renewed
    logger.warning("Couldn't renew the Snowflake session token; logging in again")
    fresh = _connect(connection, role)
    # Sandboxes started on the old session may still be using its token, so it stays open until close_sessions.
    if config.connection is not None:
        with _locks_guard:
            _retired.append(config.connection)
    return fresh


def _get_or_connect(connection: str | None, role: str | None, stale_pat: str | None = None) -> Config:
    key = (connection, role)
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        config = _configs.get(key)
        if config is None:
            config = _connect(connection, role)
        elif (stale_pat is not None and config.resolve_pat() == stale_pat) or time.monotonic() - _renewed_at.get(
            key, 0.0
        ) >= RENEW_AFTER_S:
            config = _renew(config, connection, role)
        else:
            return config
        _configs[key] = config
        _renewed_at[key] = time.monotonic()
        return config


async def get_config(connection: str | None, role: str | None) -> Config:
    """Return the shared sandbox ``Config`` backed by one live connector session."""
    key = (connection, role)
    cached = _configs.get(key)
    if cached is not None and time.monotonic() - _renewed_at.get(key, 0.0) < RENEW_AFTER_S:
        return cached
    return await asyncio.to_thread(_get_or_connect, connection, role)


def get_config_sync(connection: str | None, role: str | None) -> Config:
    """``get_config`` for callers outside an event loop."""
    return _get_or_connect(connection, role)


async def renew_config(connection: str | None, role: str | None, stale_pat: str | None) -> Config:
    """Renew the shared session after a request sent with ``stale_pat`` was refused as expired.

    Concurrent callers that saw the same token renew it once; the rest get the renewed ``Config``.
    """
    return await asyncio.to_thread(_get_or_connect, connection, role, stale_pat)


def close_sessions() -> None:
    """Close every cached connector session; the next trial logs in again.

    Takes each session's lock, so a login still in progress finishes first and
    its connection is closed too rather than cached after cleanup.
    """
    with _locks_guard:
        keys = set(_locks) | set(_configs)
        locks = {key: _locks.setdefault(key, threading.Lock()) for key in keys}
    connections = []
    for key, lock in locks.items():
        with lock:
            config = _configs.pop(key, None)
            _renewed_at.pop(key, None)
        if config is not None and config.connection is not None:
            connections.append(config.connection)
    with _locks_guard:
        connections += _retired
        _retired.clear()
    for conn in connections:
        try:
            conn.close()
        except Exception as exc:
            logger.warning("Closing a Snowflake session failed: %s", exc)


async def get_connector(connection: str | None, role: str | None) -> Any:
    """Return the live ``snowflake.connector`` connection behind the shared session."""
    config = await get_config(connection, role)
    if config.connection is None:
        raise RuntimeError("The shared Snowflake session has no live connector connection")
    return config.connection
