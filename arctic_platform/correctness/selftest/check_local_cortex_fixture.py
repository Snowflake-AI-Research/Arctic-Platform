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

"""Existing local gateways use locally validated weights without hosted publication."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from arctic_platform.client.config import CortexConfig
from arctic_platform.correctness.harness import cortex_fixture
from arctic_platform.correctness.selftest.check_cortex_fixture import _fixture


class Session:
    def __init__(self, health, *, status=200):
        self.health = health
        self.status = status
        self.calls = []
        self.trust_env = True
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return SimpleNamespace(status_code=self.status, json=lambda: self.health)

    def post(self, url, **kwargs):
        pytest.fail("existing gateway must not receive an invented fixture verification POST")


def _session(monkeypatch):
    session = Session({"status": "OK"})
    monkeypatch.setattr(requests, "Session", lambda: session)
    return session


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
def test_local_fixture_binds_local_bytes_with_existing_health_only(tmp_path, monkeypatch, host):
    spec, root = _fixture(tmp_path, legacy=True)
    session = _session(monkeypatch)
    binding = cortex_fixture.load_local_fixture(CortexConfig(base_url=f"http://{host}:8003/"), spec)
    assert binding["model_name"] == str(root.resolve())
    assert binding["distribution"] == "local-gateway"
    assert "publication" not in binding
    assert session.closed
    assert session.trust_env is False
    assert [method for method, _, _ in session.calls] == ["GET"]
    assert session.calls[0][1] == f"http://{host}:8003/health"
    assert all(call[2]["allow_redirects"] is False for call in session.calls)
    assert binding["files"] == [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.iterdir())
    ]
    assert (
        binding["fixture_sha256"]
        == hashlib.sha256(json.dumps(binding["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8003",
        "http://qa6.example:8003",
        "http://127.0.0.1:8003/proxy",
        "http://127.0.0.1:8003?target=qa6",
        "http://127.0.0.1:8003#target",
        "http://user:secret@127.0.0.1:8003",
    ],
)
def test_local_fixture_rejects_nonlocal_or_ambiguous_urls_before_io(tmp_path, monkeypatch, url):
    spec, _ = _fixture(tmp_path)
    monkeypatch.setattr(requests, "Session", lambda: pytest.fail("invalid URL reached network"))
    with pytest.raises(ValueError, match="loopback"):
        cortex_fixture.load_local_fixture(CortexConfig(base_url=url), spec)


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "dss"),
        ("source_root", "/different/AP/checkout"),
        ("supports_local_model_paths", False),
    ],
)
def test_local_fixture_does_not_require_or_attest_backend_identity(tmp_path, monkeypatch, field, value):
    spec, _ = _fixture(tmp_path)
    session = _session(monkeypatch)
    session.health[field] = value
    binding = cortex_fixture.load_local_fixture(CortexConfig(base_url="http://127.0.0.1:8003"), spec)
    assert binding["distribution"] == "local-gateway"
    assert not {"backend", "source_root", "supports_local_model_paths"} & binding.keys()
    assert len(session.calls) == 1
    assert session.closed


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("changed", ["config.json", "generation_config.json", "model.safetensors"])
def test_local_fixture_rejects_protected_asset_size_changes(tmp_path, monkeypatch, legacy, changed):
    spec, root = _fixture(tmp_path, legacy=legacy)
    session = _session(monkeypatch)
    path = root / changed
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file-name/size hash"):
        cortex_fixture.load_local_fixture(CortexConfig(base_url="http://127.0.0.1:8003"), spec)
    assert session.closed


@pytest.mark.parametrize("added", ["extra.safetensors", "extra_config.json"])
def test_local_fixture_rejects_added_unreviewed_assets(tmp_path, monkeypatch, added):
    spec, root = _fixture(tmp_path, legacy=True)
    _session(monkeypatch)
    (root / added).write_text("{}")
    with pytest.raises(ValueError, match="file-name/size hash"):
        cortex_fixture.load_local_fixture(CortexConfig(base_url="http://127.0.0.1:8003"), spec)


def test_local_fixture_requires_reviewed_layer_count(tmp_path, monkeypatch):
    spec, root = _fixture(tmp_path)
    _session(monkeypatch)
    path = root / "config.json"
    config = json.loads(path.read_text())
    config["num_hidden_layers"] = 2
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="layer count"):
        cortex_fixture.load_local_fixture(CortexConfig(base_url="http://127.0.0.1:8003"), spec)


def test_local_fixture_requires_tokenizer_even_with_legacy_hash(tmp_path, monkeypatch):
    spec, root = _fixture(tmp_path, legacy=True)
    _session(monkeypatch)
    (root / "tokenizer_config.json").unlink()
    with pytest.raises(ValueError, match="tokenizer sidecars"):
        cortex_fixture.load_local_fixture(CortexConfig(base_url="http://127.0.0.1:8003"), spec)


@pytest.mark.parametrize("unsafe", ["/tmp", "/code", "/data-fast-shadow", "/data-fast", "symlink"])
def test_local_fixture_rejects_unsafe_model_root_before_io(tmp_path, monkeypatch, unsafe):
    spec, _ = _fixture(tmp_path)
    root = Path(unsafe)
    if unsafe == "symlink":
        root = tmp_path / "escape"
        root.symlink_to("/tmp", target_is_directory=True)
    spec.model.cache_path = str(root)
    monkeypatch.setattr(requests, "Session", lambda: pytest.fail("unsafe root reached network"))
    monkeypatch.setattr(cortex_fixture, "describe_fixture", lambda *args: pytest.fail("unsafe root reached hashing"))
    with pytest.raises(ValueError, match="/data-fast"):
        cortex_fixture.load_local_fixture(CortexConfig(base_url="http://127.0.0.1:8003"), spec)


@pytest.mark.parametrize("credentials", [{"pat": "not-a-real-token"}, {"host": "qa6.example"}])
def test_local_fixture_rejects_authentication_before_io(tmp_path, monkeypatch, credentials):
    spec, _ = _fixture(tmp_path)
    monkeypatch.setattr(requests, "Session", lambda: pytest.fail("credentials reached network"))
    with pytest.raises(ValueError, match="unauthenticated"):
        cortex_fixture.load_local_fixture(CortexConfig(base_url="http://127.0.0.1:8003", **credentials), spec)


@pytest.mark.parametrize("status", [301, 401, 500])
def test_local_fixture_does_not_follow_redirects_or_authenticate(tmp_path, monkeypatch, status):
    spec, _ = _fixture(tmp_path)
    session = _session(monkeypatch)
    session.status = status
    with pytest.raises(ValueError, match="not healthy"):
        cortex_fixture.load_local_fixture(CortexConfig(base_url="http://127.0.0.1:8003"), spec)
    assert len(session.calls) == 1
    assert session.closed


@pytest.mark.parametrize("payload", [None, [], "not an object", {}, {"status": "pending"}])
def test_local_fixture_rejects_malformed_health(tmp_path, monkeypatch, payload):
    spec, _ = _fixture(tmp_path)
    session = _session(monkeypatch)
    session.health = payload
    with pytest.raises(ValueError, match="not healthy"):
        cortex_fixture.load_local_fixture(CortexConfig(base_url="http://127.0.0.1:8003"), spec)


def test_local_fixture_normalizes_connection_failure(tmp_path, monkeypatch):
    spec, _ = _fixture(tmp_path)
    session = _session(monkeypatch)

    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("gateway unavailable")

    monkeypatch.setattr(session, "get", unavailable)
    with pytest.raises(ValueError, match="verification failed: ConnectionError"):
        cortex_fixture.load_local_fixture(CortexConfig(base_url="http://127.0.0.1:8003"), spec)


def test_execution_transport_stays_on_unauthenticated_loopback(monkeypatch):
    from arctic_platform.client.transports.cortex import CortexTransport
    from arctic_platform.correctness.harness.cortex_transport import CorrectnessCortexTransport

    transport = CorrectnessCortexTransport(CortexConfig(base_url="http://127.0.0.1:8003"))
    sent = {}

    async def send(self, method, url, **kwargs):
        assert method == "GET" and url == f"{transport._prefix}/p1"
        sent.update(kwargs)
        return {}

    monkeypatch.setattr(CortexTransport, "_send", send)
    try:
        assert transport._on_loop(transport._send("GET", f"{transport._prefix}/p1")).result() == {}
        session = transport._session
        assert session is not None and session.trust_env is False
        assert "Authorization" not in session.headers
        assert sent["allow_redirects"] is False
    finally:
        transport.shutdown()
    assert session.closed
    assert transport._session is None
