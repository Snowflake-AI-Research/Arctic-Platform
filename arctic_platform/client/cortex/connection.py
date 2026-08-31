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
"""Where a Cortex connection comes from: one resolver for the CLI, the TUI, and recipes.

Precedence, highest first: explicit overrides (CLI flags) > a connection file
(``--config`` / ``CORTEX_CONFIG``) > the remembered login file > ``CORTEX_*`` env
(falling back to the standard ``SNOWFLAKE_*`` names) > `CortexConfig` defaults.

`login` stores the *path* to a connection file, never its contents, so a PAT is
never copied into a second place on disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from arctic_platform.client.config import CortexConfig

# CortexConfig fields a connection file may set. Anything else is a typo worth
# reporting: silently ignoring it strands the user with a setting that never applied.
_KEYS = ("base_url", "host", "pat", "pat_env_var", "database", "schema", "endpoint", "max_retries")

# Spellings the old dss-neutrino config files used.
_ALIASES = {"url": "base_url", "db": "database"}

# Keys those files carried that this client has no use for: poll/SSL knobs the old
# client owned. Accepted and dropped so an existing config keeps working.
_OBSOLETE = frozenset({"poll_interval", "poll_timeout", "verify_ssl", "no_verify_ssl"})

# CORTEX_* first, then the account-wide Snowflake conventions.
_ENV = {
    "base_url": ("CORTEX_BASE_URL",),
    "host": ("CORTEX_HOST", "SNOWFLAKE_HOST"),
    "pat": ("CORTEX_PAT", "SNOWFLAKE_PAT"),
    "database": ("CORTEX_DATABASE", "SNOWFLAKE_DATABASE"),
    "schema": ("CORTEX_SCHEMA", "SNOWFLAKE_SCHEMA"),
    "endpoint": ("CORTEX_ENDPOINT",),
}

CONFIG_ENV = "CORTEX_CONFIG"
LOGIN_ENV = "CORTEX_LOGIN_FILE"


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def login_path() -> Path:
    """Where the remembered connection-file path lives."""
    if override := _env(LOGIN_ENV):
        return Path(override).expanduser()
    config_home = _env("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "cortex" / "login.json"


def read_connection_file(path: str | Path) -> dict[str, Any]:
    """Parse a connection JSON (bare, or nested under a ``connection`` key)."""
    path = Path(path).expanduser()
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in connection file {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"connection file {path} must be a JSON object")

    connection = parsed.get("connection", parsed)
    if not isinstance(connection, dict):
        raise ValueError(f"connection file {path}: 'connection' must be an object")

    settings: dict[str, Any] = {}
    unknown: list[str] = []
    for raw_key, value in connection.items():
        key = str(_ALIASES.get(raw_key, raw_key))
        if key in _OBSOLETE:
            continue
        if key not in _KEYS:
            unknown.append(key)
            continue
        settings[key] = value
    if unknown:
        raise ValueError(f"connection file {path}: unknown key(s) {', '.join(sorted(unknown))}")
    return settings


def _remembered() -> dict[str, Any]:
    path = login_path()
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid login state {path}: {exc}") from exc
    config_path = state.get("config_path") if isinstance(state, dict) else None
    if not isinstance(config_path, str) or not config_path:
        raise ValueError(f"invalid login state {path}: missing config_path (re-run 'cortex login')")
    return read_connection_file(config_path)


def _from_env() -> dict[str, Any]:
    return {key: value for key, names in _ENV.items() if (value := _env(*names)) is not None}


def resolve(*, config_path: str | Path | None = None, **overrides: Any) -> CortexConfig:
    """Build a `CortexConfig` from overrides, a connection file, login state, and env.

    ``config_path`` (or ``CORTEX_CONFIG``) pins the connection file; without one,
    the file remembered by ``cortex login`` is used. ``overrides`` are CLI flags:
    None-valued entries are dropped so an unset flag never masks a lower layer.

    Raises `ValueError` if the result isn't a usable connection -- `CortexConfig`'s
    own validator decides that (base_url or host, plus database/schema/PAT for host auth).
    """
    config_path = config_path or _env(CONFIG_ENV)
    file_settings = read_connection_file(config_path) if config_path else _remembered()

    settings = {**_from_env(), **file_settings}
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return CortexConfig(**settings)


def login(config_path: str | Path) -> Path:
    """Validate a connection file and remember its path. Returns the resolved path."""
    path = Path(config_path).expanduser().resolve()
    CortexConfig(**read_connection_file(path))  # fail now, not on the next command

    state = login_path()
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"config_path": str(path)}, indent=2) + "\n", encoding="utf-8")
    return path


def logout() -> bool:
    """Forget the remembered connection file. True if there was one."""
    state = login_path()
    if not state.exists():
        return False
    state.unlink()
    return True


def redacted(config: CortexConfig) -> dict[str, Any]:
    """The connection as JSON-able settings, with the PAT masked."""
    settings = config.model_dump(mode="json", by_alias=True, exclude={"type", "protocol"})
    if settings.get("pat"):
        settings["pat"] = "***"
    settings["pat_source"] = "config" if config.pat else config.pat_env_var
    return settings
