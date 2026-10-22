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

from __future__ import annotations

import stat

import pytest

from arctic_platform.integrations.tinker.job import TinkerJobConfig
from arctic_platform.integrations.tinker.job import client_config
from arctic_platform.integrations.tinker.snowflake_profile import cortex_fields

_CORTEX_ENV = (
    "ARCTIC_CORTEX_HOST",
    "ARCTIC_CORTEX_BASE_URL",
    "ARCTIC_CORTEX_PAT",
    "ARCTIC_CORTEX_DATABASE",
    "ARCTIC_CORTEX_SCHEMA",
)

_PROFILE = """\
[training]
account = "ORG-ACCOUNT"
host = "acct.snowflakecomputing.com"
user = "USER"
authenticator = "programmatic_access_token"
token = "pat-from-profile"
database = "CORTEX_TRAINING_DB"
schema = "PUBLIC"

[default]
host = "default.snowflakecomputing.com"
token = "pat-default"
database = "DEFAULT_DB"
schema = "PUBLIC"
"""


def _clear_cortex_env(monkeypatch) -> None:
    for key in _CORTEX_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME", raising=False)


def _write_private(directory, name: str, text: str) -> None:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_profile_supplies_cortex_when_env_is_unset(tmp_path, monkeypatch):
    _clear_cortex_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_HOME", str(tmp_path))
    monkeypatch.setenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME", "training")
    _write_private(tmp_path, "connections.toml", _PROFILE)

    backend = client_config(TinkerJobConfig()).backend

    assert backend.host == "acct.snowflakecomputing.com"
    assert backend.database == "CORTEX_TRAINING_DB"
    assert backend.schema_ == "PUBLIC"
    assert backend.pat.get_secret_value() == "pat-from-profile"
    assert backend.base_url is None


def test_env_wins_over_a_connection_profile(tmp_path, monkeypatch):
    _clear_cortex_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_HOME", str(tmp_path))
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://cortex.test")
    _write_private(tmp_path, "connections.toml", _PROFILE)

    backend = client_config(TinkerJobConfig()).backend

    assert backend.base_url == "http://cortex.test"
    assert backend.host is None
    assert backend.pat is None


def test_config_toml_names_the_profile(tmp_path, monkeypatch):
    _clear_cortex_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_HOME", str(tmp_path))
    _write_private(tmp_path, "connections.toml", _PROFILE)
    _write_private(tmp_path, "config.toml", 'default_connection_name = "training"\n')

    fields = cortex_fields()

    assert fields["host"] == "acct.snowflakecomputing.com"
    assert fields["database"] == "CORTEX_TRAINING_DB"


def test_default_profile_when_nothing_names_one(tmp_path, monkeypatch):
    _clear_cortex_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_HOME", str(tmp_path))
    _write_private(tmp_path, "connections.toml", _PROFILE)

    fields = cortex_fields()

    assert fields["host"] == "default.snowflakecomputing.com"


def test_missing_schema_defaults_to_public(tmp_path, monkeypatch):
    _clear_cortex_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_HOME", str(tmp_path))
    _write_private(
        tmp_path,
        "connections.toml",
        '[default]\nhost = "acct.snowflakecomputing.com"\ntoken = "pat"\ndatabase = "DB"\n',
    )

    assert cortex_fields()["schema"] == "PUBLIC"


def test_world_readable_profile_is_refused(tmp_path, monkeypatch):
    _clear_cortex_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_HOME", str(tmp_path))
    path = tmp_path / "connections.toml"
    path.write_text(_PROFILE, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IROTH)

    with pytest.raises(ValueError, match="chmod 600"):
        cortex_fields()


def test_other_authenticators_are_refused(tmp_path, monkeypatch):
    _clear_cortex_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_HOME", str(tmp_path))
    _write_private(
        tmp_path,
        "connections.toml",
        '[default]\nhost = "acct.snowflakecomputing.com"\ntoken = "pat"\ndatabase = "DB"\nauthenticator ='
        ' "snowflake"\n',
    )

    with pytest.raises(ValueError, match="programmatic access token"):
        cortex_fields()


def test_absent_profile_leaves_cortex_config_to_its_own_check(tmp_path, monkeypatch):
    _clear_cortex_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_HOME", str(tmp_path))

    assert cortex_fields() is None
    with pytest.raises(ValueError, match="base_url"):
        client_config(TinkerJobConfig())


def test_unknown_profile_does_not_echo_the_token(tmp_path, monkeypatch):
    _clear_cortex_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_HOME", str(tmp_path))
    monkeypatch.setenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME", "missing")
    _write_private(tmp_path, "connections.toml", _PROFILE)

    with pytest.raises(ValueError, match="missing") as raised:
        cortex_fields()

    assert "pat-from-profile" not in str(raised.value)
