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
"""Read a Snowflake connection profile into Cortex host, PAT, database, and schema.

``ARCTIC_CORTEX_*`` stays the explicit connection. When none of those variables
is set, the Tinker client uses the same profile file cortex-training documents:
``~/.snowflake/connections.toml``, or ``$SNOWFLAKE_HOME/connections.toml``.
The profile's ``token`` is a programmatic access token.
"""

from __future__ import annotations

import os
import stat
import tomllib
from pathlib import Path

_ENV_KEYS = (
    "ARCTIC_CORTEX_HOST",
    "ARCTIC_CORTEX_BASE_URL",
    "ARCTIC_CORTEX_PAT",
    "ARCTIC_CORTEX_DATABASE",
    "ARCTIC_CORTEX_SCHEMA",
)
_PAT_AUTHENTICATORS = {"programmatic_access_token"}


def env_configured() -> bool:
    """True when the process set any Cortex connection variable."""
    return any(os.environ.get(key) for key in _ENV_KEYS)


def config_dir() -> Path | None:
    """The Snowflake config directory, or ``None`` when the user has none."""
    home = os.environ.get("SNOWFLAKE_HOME")
    if home:
        return Path(home).expanduser()
    dot = Path.home() / ".snowflake"
    if dot.is_dir():
        return dot
    xdg_root = os.environ.get("XDG_CONFIG_HOME")
    xdg = Path(xdg_root).expanduser() / "snowflake" if xdg_root else Path.home() / ".config" / "snowflake"
    if xdg.is_dir():
        return xdg
    return None


def cortex_fields() -> dict[str, str] | None:
    """Host, PAT, database, and schema from the selected connection profile.

    ``None`` means there is no ``connections.toml``. A profile that is present
    and unusable raises ``ValueError``.
    """
    directory = config_dir()
    if directory is None:
        return None
    path = directory / "connections.toml"
    if not path.is_file():
        return None
    _check_private(path)
    profiles = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"{path} has no connection profiles")
    name = _connection_name(directory, profiles)
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        known = ", ".join(sorted(str(key) for key in profiles))
        raise ValueError(f"Snowflake connection {name!r} is not in {path}; known profiles: {known}")
    return _fields(name, profile)


def _connection_name(directory: Path, profiles: dict) -> str:
    chosen = os.environ.get("SNOWFLAKE_DEFAULT_CONNECTION_NAME")
    if chosen:
        return chosen
    config_path = directory / "config.toml"
    if config_path.is_file():
        parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            named = parsed.get("default_connection_name")
            if isinstance(named, str) and named.strip():
                return named.strip()
    if "default" in profiles:
        return "default"
    raise ValueError(
        f"set SNOWFLAKE_DEFAULT_CONNECTION_NAME, or default_connection_name in {directory / 'config.toml'}"
    )


def _fields(name: str, profile: dict) -> dict[str, str]:
    authenticator = str(profile.get("authenticator") or "programmatic_access_token").strip().lower()
    if authenticator not in _PAT_AUTHENTICATORS:
        raise ValueError(
            f"Snowflake connection {name!r} uses authenticator {authenticator!r}; "
            "this client sends the profile token as a programmatic access token"
        )
    host = _text(profile, "host")
    token = _text(profile, "token")
    database = _text(profile, "database")
    schema = _text(profile, "schema") or "PUBLIC"
    missing = [field for field, value in (("host", host), ("token", token), ("database", database)) if not value]
    if missing:
        raise ValueError(f"Snowflake connection {name!r} needs {', '.join(missing)}")
    return {"host": host, "pat": token, "database": database, "schema": schema}


def _text(profile: dict, key: str) -> str:
    value = profile.get(key)
    if value is None:
        return ""
    return str(value).strip()


def _check_private(path: Path) -> None:
    mode = path.stat().st_mode
    if stat.S_ISREG(mode) and mode & 0o077:
        raise ValueError(f"{path} must be readable only by its owner (chmod 600)")
