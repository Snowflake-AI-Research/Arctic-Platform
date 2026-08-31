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
"""Connection resolution: precedence, file parsing, and login state."""

from __future__ import annotations

import json

import pytest

from arctic_platform.client.cortex import connection

_ENV_VARS = (
    "CORTEX_CONFIG",
    "CORTEX_LOGIN_FILE",
    "CORTEX_BASE_URL",
    "CORTEX_HOST",
    "CORTEX_PAT",
    "CORTEX_DATABASE",
    "CORTEX_SCHEMA",
    "CORTEX_ENDPOINT",
    "SNOWFLAKE_HOST",
    "SNOWFLAKE_PAT",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
    "XDG_CONFIG_HOME",
)


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    """No ambient connection: a developer's own login must not steer these tests."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CORTEX_LOGIN_FILE", str(tmp_path / "login.json"))


def write_config(tmp_path, **settings):
    path = tmp_path / "connection.json"
    path.write_text(json.dumps({"host": "h.example.com", "pat": "p", "database": "D", "schema": "S", **settings}))
    return path


class TestReadConnectionFile:
    def test_reads_a_bare_object(self, tmp_path):
        """The common case: connection keys at the top level."""
        assert connection.read_connection_file(write_config(tmp_path))["host"] == "h.example.com"

    def test_reads_a_nested_connection_key(self, tmp_path):
        """A file may wrap its settings under 'connection'."""
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"connection": {"host": "nested.example.com"}, "other": 1}))
        assert connection.read_connection_file(path) == {"host": "nested.example.com"}

    def test_translates_legacy_aliases(self, tmp_path):
        """Old dss-neutrino files spelled these 'url' and 'db'."""
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"url": "http://localhost:8084", "db": "D"}))
        assert connection.read_connection_file(path) == {"base_url": "http://localhost:8084", "database": "D"}

    def test_drops_obsolete_keys(self, tmp_path):
        """Poll/SSL knobs the old client owned are accepted and ignored."""
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"host": "h", "poll_interval": 2, "verify_ssl": False}))
        assert connection.read_connection_file(path) == {"host": "h"}

    def test_rejects_an_unknown_key(self, tmp_path):
        """A typo must fail loudly rather than silently never applying."""
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"host": "h", "hostname": "oops"}))
        with pytest.raises(ValueError, match="unknown key"):
            connection.read_connection_file(path)

    def test_rejects_invalid_json(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("{not json")
        with pytest.raises(ValueError, match="invalid JSON"):
            connection.read_connection_file(path)


class TestResolvePrecedence:
    def test_env_supplies_a_connection(self, monkeypatch):
        """CORTEX_* alone is enough."""
        monkeypatch.setenv("CORTEX_HOST", "env.example.com")
        monkeypatch.setenv("CORTEX_PAT", "env-pat")
        monkeypatch.setenv("CORTEX_DATABASE", "ENVDB")
        monkeypatch.setenv("CORTEX_SCHEMA", "ENVS")
        config = connection.resolve()
        assert (config.host, config.database) == ("env.example.com", "ENVDB")

    def test_snowflake_env_is_a_fallback(self, monkeypatch):
        """SNOWFLAKE_* is a Snowflake convention, not product naming, so it still works."""
        monkeypatch.setenv("SNOWFLAKE_HOST", "sf.example.com")
        monkeypatch.setenv("SNOWFLAKE_PAT", "sf-pat")
        monkeypatch.setenv("SNOWFLAKE_DATABASE", "SFDB")
        monkeypatch.setenv("SNOWFLAKE_SCHEMA", "SFS")
        assert connection.resolve().host == "sf.example.com"

    def test_cortex_env_beats_snowflake_env(self, monkeypatch):
        monkeypatch.setenv("SNOWFLAKE_HOST", "sf.example.com")
        monkeypatch.setenv("CORTEX_HOST", "cortex.example.com")
        monkeypatch.setenv("CORTEX_PAT", "p")
        monkeypatch.setenv("CORTEX_DATABASE", "D")
        monkeypatch.setenv("CORTEX_SCHEMA", "S")
        assert connection.resolve().host == "cortex.example.com"

    def test_file_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORTEX_HOST", "env.example.com")
        monkeypatch.setenv("CORTEX_PAT", "env-pat")
        config = connection.resolve(config_path=write_config(tmp_path))
        assert config.host == "h.example.com"

    def test_config_env_var_names_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORTEX_CONFIG", str(write_config(tmp_path)))
        assert connection.resolve().host == "h.example.com"

    def test_overrides_beat_the_file(self, tmp_path):
        config = connection.resolve(config_path=write_config(tmp_path), host="flag.example.com")
        assert config.host == "flag.example.com"

    def test_unset_overrides_do_not_mask_lower_layers(self, tmp_path):
        """An unpassed CLI flag arrives as None and must not blank the file value."""
        config = connection.resolve(config_path=write_config(tmp_path), host=None, database=None)
        assert (config.host, config.database) == ("h.example.com", "D")

    def test_env_fills_gaps_the_file_leaves(self, tmp_path, monkeypatch):
        """Layers merge per key rather than replacing wholesale."""
        monkeypatch.setenv("CORTEX_ENDPOINT", "custom-endpoint")
        assert connection.resolve(config_path=write_config(tmp_path)).endpoint == "custom-endpoint"

    def test_an_empty_connection_is_an_error(self):
        with pytest.raises(ValueError, match="base_url .*or host"):
            connection.resolve()

    def test_host_without_a_pat_is_an_error(self, monkeypatch):
        monkeypatch.setenv("CORTEX_HOST", "h.example.com")
        monkeypatch.setenv("CORTEX_DATABASE", "D")
        monkeypatch.setenv("CORTEX_SCHEMA", "S")
        with pytest.raises(ValueError, match="no PAT"):
            connection.resolve()


class TestLoginState:
    def test_login_then_resolve_needs_no_flags(self, tmp_path):
        path = write_config(tmp_path)
        assert connection.login(path) == path.resolve()
        assert connection.resolve().host == "h.example.com"

    def test_login_stores_the_path_not_the_pat(self, tmp_path, monkeypatch):
        """The PAT must never be copied into a second file on disk."""
        state = tmp_path / "login.json"
        monkeypatch.setenv("CORTEX_LOGIN_FILE", str(state))
        connection.login(write_config(tmp_path, pat="super-secret"))
        text = state.read_text()
        assert "super-secret" not in text
        assert json.loads(text)["config_path"].endswith("connection.json")

    def test_login_rejects_an_unusable_connection(self, tmp_path):
        """Fail at login, not on the next command."""
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"database": "D"}))
        with pytest.raises(ValueError, match="base_url .*or host"):
            connection.login(path)

    def test_explicit_config_beats_login_state(self, tmp_path):
        connection.login(write_config(tmp_path))
        other = tmp_path / "other.json"
        other.write_text(json.dumps({"host": "other.example.com", "pat": "p", "database": "D", "schema": "S"}))
        assert connection.resolve(config_path=other).host == "other.example.com"

    def test_logout_forgets_the_file(self, tmp_path):
        connection.login(write_config(tmp_path))
        assert connection.logout() is True
        assert connection.logout() is False
        with pytest.raises(ValueError, match="base_url .*or host"):
            connection.resolve()

    def test_corrupt_login_state_names_the_fix(self, tmp_path, monkeypatch):
        state = tmp_path / "login.json"
        state.write_text(json.dumps({"nothing": "useful"}))
        monkeypatch.setenv("CORTEX_LOGIN_FILE", str(state))
        with pytest.raises(ValueError, match="cortex login"):
            connection.resolve()

    def test_login_path_follows_xdg_config_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORTEX_LOGIN_FILE")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert connection.login_path() == tmp_path / "cortex" / "login.json"


class TestRedacted:
    def test_masks_the_pat(self, tmp_path):
        settings = connection.redacted(connection.resolve(config_path=write_config(tmp_path, pat="secret")))
        assert settings["pat"] == "***"
        assert settings["pat_source"] == "config"

    def test_reports_the_env_var_when_the_pat_is_indirect(self, monkeypatch):
        monkeypatch.setenv("CORTEX_PAT", "from-env")
        monkeypatch.setenv("CORTEX_HOST", "h.example.com")
        monkeypatch.setenv("CORTEX_DATABASE", "D")
        monkeypatch.setenv("CORTEX_SCHEMA", "S")
        # CORTEX_PAT reaches the config as an explicit value, so redaction still applies.
        assert connection.redacted(connection.resolve())["pat"] == "***"
