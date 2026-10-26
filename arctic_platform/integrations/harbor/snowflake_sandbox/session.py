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
"""One Snowflake session per (connection, role), shared by every trial."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from snowflake.sandbox.config import Config
from snowflake.sandbox.config import resolve_config

logger = logging.getLogger(__name__)

LOGIN_TIMEOUT_S = 300

_configs: dict[tuple[str | None, str | None], Config] = {}
_locks: dict[tuple[str | None, str | None], asyncio.Lock] = {}


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


async def get_config(connection: str | None, role: str | None) -> Config:
    """Return the shared sandbox ``Config`` backed by one live connector session."""
    key = (connection, role)
    lock = _locks.setdefault(key, asyncio.Lock())
    async with lock:
        if key not in _configs:
            _configs[key] = await asyncio.to_thread(_connect, connection, role)
        return _configs[key]


async def get_connector(connection: str | None, role: str | None) -> Any:
    """Return the live ``snowflake.connector`` connection behind the shared session."""
    config = await get_config(connection, role)
    if config.connection is None:
        raise RuntimeError("The shared Snowflake session has no live connector connection")
    return config.connection
