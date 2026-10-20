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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject FAST-path restore before HTTP while durable checkpoint IDs use a separate load contract."""

import asyncio
import json
from types import SimpleNamespace

import aiohttp
import pytest
import requests

from arctic_platform.client import CortexConfig
from arctic_platform.client.requests import load_checkpoint_request
from arctic_platform.client.transport import Request
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import _route
from arctic_platform.correctness.harness import hosted
from arctic_platform.correctness.harness.cortex import CorrectnessClient
from arctic_platform.correctness.harness.cortex import CortexTrainingConfig
from arctic_platform.correctness.harness.cortex_transport import LOCAL_NATIVE_FORWARD_ENV
from arctic_platform.correctness.harness.cortex_transport import CorrectnessCortexTransport
from arctic_platform.correctness.harness.workdir import require_data_fast_path

_LOAD_BODY_ERROR = "^local checkpoint load accepts checkpoint_id/stage_info only; no path, step or source_job_id$"
_LOAD_ID_ERROR = "^local checkpoint load requires a durable cp_<uuid> checkpoint_id, not a FAST path or tag$"
_LOAD_OWNER_ERROR = "^local checkpoint load requires this client's newly owned training child$"
_LOCAL_GATEWAY_ERROR = "^local Cortex gateway requires an unauthenticated HTTP loopback base URL$"
_CHECKPOINT_ID = "cp_00000000-0000-0000-0000-000000000001"


def _response(payload):
    class Reply:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def read(self):
            return json.dumps(payload).encode()

    return Reply()


@pytest.fixture
def checkpoint(tmp_path):
    root = require_data_fast_path(tmp_path / "checkpoint", label="resume control basetemp")
    root.mkdir()
    (root / "latest").write_text("global_step1\n", encoding="ascii")
    (root / "global_step1").mkdir()
    return root


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV, raising=False)
    state = SimpleNamespace(calls=[], roles=[], cancelled=False, clients=[], exchanges=[], closed=[])
    transports = []
    original_start = CortexTransport._start_loop

    def start(transport):
        original_start(transport)
        transports.append(transport)

    def forbidden(*args, **kwargs):
        pytest.fail("checkpoint resume controls must not perform real network requests")

    monkeypatch.setattr(CortexTransport, "_start_loop", start)
    monkeypatch.setattr(requests.Session, "request", forbidden)

    def send(session, method, url, **kwargs):
        assert asyncio.get_running_loop() is session._loop
        assert session.trust_env is False
        if url.startswith("http://127.0.0.1:8003/"):
            assert kwargs["allow_redirects"] is False
        else:
            assert url.startswith(("https://127.0.0.1:8003/", "http://remote.example:8003/"))
            assert "allow_redirects" not in kwargs
        assert "Authorization" not in session.headers
        state.calls.append((method, url, kwargs))
        state.exchanges.append((session, asyncio.get_running_loop()))
        if method == "GET" and url.endswith("/openapi.json"):
            return _response(
                {
                    "paths": {
                        path: {"post": {"parameters": [{"name": "job_id", "in": "query", "required": True}]}}
                        for path in ("/forward", "/log-probs")
                    }
                }
            )
        if method == "POST" and url.endswith("/cortex-training"):
            state.roles = [sub["job_type"] for sub in kwargs["json"]["sub_job_configs"]]
            return _response({"job_id": "owned-resume-parent"})
        if method == "POST" and url.endswith("/owned-resume-parent:cancel"):
            state.cancelled = True
            return _response({})
        if not url.startswith("http://127.0.0.1:8003/"):
            if method == "POST" and url.endswith("/owned-resume-parent/load"):
                return _response({"request_id": "production-load-request"})
            if method == "GET" and url.endswith("/owned-resume-parent/requests/production-load-request"):
                return _response(
                    {
                        "status": "REQUEST_STATE_DONE",
                        "result": {"checkpoint_id": _CHECKPOINT_ID, "version": 1},
                    }
                )
        if method == "GET" and url.endswith("/owned-resume-parent"):
            segments = {"training": "training", "sampling": "sampling", "log_probability": "log_prob"}
            return _response(
                {
                    "job_id": "owned-resume-parent",
                    "status": "cancelled" if state.cancelled else "RUNNING",
                    "sub_jobs": [
                        {
                            "job_type": role,
                            "sub_job_id": f"owned-resume-parent:{segments[role]}:0",
                            "status": "cancelled" if state.cancelled else "running",
                        }
                        for role in state.roles
                    ],
                }
            )
        # No synthetic successful load: the public zone schema cannot carry a FAST restore path.
        pytest.fail(f"unsupported checkpoint restore reached HTTP: {method} {url}")

    monkeypatch.setattr(aiohttp.ClientSession, "request", send)
    original_close = aiohttp.ClientSession.close

    async def close(session):
        loop = asyncio.get_running_loop()
        assert loop is session._loop
        await original_close(session)
        state.closed.append((session, loop))

    monkeypatch.setattr(aiohttp.ClientSession, "close", close)

    def create(*, native=False):
        if native:
            monkeypatch.setenv(LOCAL_NATIVE_FORWARD_ENV, "1")
        config = CortexTrainingConfig(
            model_name="Qwen/Qwen3.6-35B-A3B",
            training_gpus=8,
            sampling_gpus=1,
            log_prob_gpus=1,
            hardware="H200",
            backend=CortexConfig(
                base_url="http://127.0.0.1:8003",
                max_retries=4,
                poll_timeout=1.0,
                request_timeout=1.0,
            ),
        )
        client = CorrectnessClient(config)
        assert isinstance(client.transport, CorrectnessCortexTransport)
        state.clients.append(client)
        state.calls.clear()
        return client

    yield state, create
    for client in state.clients:
        if client.transport.job_id is not None:
            client.transport.config = client.config.backend
            client.shutdown()
    for transport in transports:
        transport._on_loop(transport._close_session()).result()
        transport._loop.call_soon_threadsafe(transport._loop.stop)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_owned_fast_restore_is_unsupported_before_http_and_still_shuts_down(gateway, checkpoint, native, asynchronous):
    state, create = gateway
    client = create(native=native)
    transport = client.transport
    original = checkpoint.joinpath("latest").read_bytes()
    with client:
        with pytest.raises(ValueError, match=_LOAD_BODY_ERROR):
            if asynchronous:
                asyncio.run(transport.acall(load_checkpoint_request(client.jobs, path=str(checkpoint))))
            else:
                client.load_checkpoint(path=str(checkpoint))
        assert state.calls == []
        assert checkpoint.joinpath("latest").read_bytes() == original
        assert transport.job_id == "owned-resume-parent"
    assert [method for method, _, _ in state.calls] == ["POST", "GET"]
    assert state.calls[0][1].endswith("/owned-resume-parent:cancel")
    assert state.calls[1][1].endswith("/owned-resume-parent")
    assert transport.job_id is None and transport._session is None
    assert transport._native_bindings == transport._native_children == {}
    assert transport._local_shutdown_parent is None and transport._local_shutdown_children == {}
    assert len(state.closed) == 1
    session, loop = state.closed[0]
    assert loop is transport._loop and session.closed
    assert all(actual_session is session and actual_loop is loop for actual_session, actual_loop in state.exchanges)


@pytest.mark.parametrize(
    "token",
    [
        None,
        "foreign-parent:training:0",
        "owned-resume-parent",
        "owned-resume-parent:training:1",
        "owned-resume-parent:sampling:0",
        "owned-resume-parent:log_prob:0",
    ],
)
def test_wrong_or_nontraining_token_cannot_dispatch_unsupported_restore(gateway, checkpoint, token):
    state, create = gateway
    client = create()
    request = Request("load-checkpoint", token, {"path": str(checkpoint), "step": None})
    with pytest.raises(ValueError, match=_LOAD_OWNER_ERROR):
        client.transport.call(request)
    assert state.calls == []


@pytest.mark.parametrize(
    "body,message",
    [
        ({}, _LOAD_ID_ERROR),
        ({"path": None, "step": None}, _LOAD_BODY_ERROR),
        ({"path": ""}, _LOAD_BODY_ERROR),
        ({"path": "relative/checkpoint"}, _LOAD_BODY_ERROR),
        ({"path": "/tmp/checkpoint"}, _LOAD_BODY_ERROR),
        ({"path": "/data-fast"}, _LOAD_BODY_ERROR),
        ({"path": 123}, _LOAD_BODY_ERROR),
        ({"checkpoint_path": "/data-fast/tmp/checkpoint"}, _LOAD_BODY_ERROR),
        ({"checkpoint_id": "global_step1"}, _LOAD_ID_ERROR),
        ({"source_job_id": "foreign-parent", "path": "/data-fast/tmp/checkpoint"}, _LOAD_BODY_ERROR),
        ({"source_job_id": "", "path": "/data-fast/tmp/checkpoint"}, _LOAD_BODY_ERROR),
    ],
)
def test_alternate_restore_bodies_never_become_zone_load_requests(gateway, body, message):
    state, create = gateway
    client = create()
    with pytest.raises(ValueError, match=message):
        client.transport.call(Request("load-checkpoint", client.jobs.training, body))
    assert state.calls == []


@pytest.mark.parametrize("step", [0, 1, -1, True, "1"])
def test_explicit_step_does_not_enable_a_local_restore_fallback(gateway, checkpoint, step):
    state, create = gateway
    client = create()
    with pytest.raises(ValueError, match=_LOAD_BODY_ERROR):
        client.load_checkpoint(path=str(checkpoint), step=step)
    assert state.calls == []


@pytest.mark.parametrize(
    "backend",
    [
        {"base_url": "http://127.0.0.1:8003/path"},
        {"base_url": "http://127.0.0.1:8003?query=1"},
        {"base_url": "http://127.0.0.1:8003#fragment"},
        {"base_url": "http://user:secret@127.0.0.1:8003"},
        {"base_url": "http://127.0.0.1:8003", "host": "account.example"},
        {"base_url": "http://127.0.0.1:8003", "pat": "synthetic-control-token"},
    ],
)
def test_authenticated_or_decorated_loopback_cannot_enable_restore(gateway, checkpoint, backend):
    state, create = gateway
    client = create()
    client.transport.config = client.transport.config.model_copy(update=backend)
    with pytest.raises(ValueError, match=_LOCAL_GATEWAY_ERROR):
        client.load_checkpoint(path=str(checkpoint))
    assert state.calls == []


def test_unsupported_restore_never_enters_http_or_polling_even_with_retries(gateway, checkpoint, monkeypatch):
    state, create = gateway
    client = create()
    attempts = []
    existing = aiohttp.ClientSession.request

    def send(session, method, url, **kwargs):
        if url.endswith("/load") or "/requests/" in url:
            attempts.append((method, url))
            pytest.fail("unsupported FAST restore reached HTTP or request polling")
        return existing(session, method, url, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "request", send)
    assert client.transport.config.max_retries == 4
    with pytest.raises(ValueError, match=_LOAD_BODY_ERROR):
        client.load_checkpoint(path=str(checkpoint))
    assert attempts == state.calls == []


@pytest.mark.parametrize("base_url", ["https://127.0.0.1:8003", "http://remote.example:8003"])
def test_nonlocal_resume_uses_production_checkpoint_id_and_source_job(gateway, base_url):
    state, create = gateway
    client = create()
    client.transport.config = client.transport.config.model_copy(update={"base_url": base_url})
    assert not client.transport._is_local_gateway()
    assert client.load_checkpoint(checkpoint_id=_CHECKPOINT_ID, source_job_id="saving-parent") == {
        "checkpoint_id": _CHECKPOINT_ID,
        "version": 1,
    }
    prefix = client.transport._prefix
    assert [(method, url) for method, url, _ in state.calls] == [
        ("POST", f"{prefix}/owned-resume-parent/load"),
        ("GET", f"{prefix}/owned-resume-parent/requests/production-load-request"),
    ]
    assert state.calls[0][2]["json"] == {"checkpoint_id": _CHECKPOINT_ID, "source_job_id": "saving-parent"}


def test_production_cortex_resume_route_preserves_request_body():
    route = _route("load-checkpoint")
    assert route.method == "POST" and route.path == "/{job}/load" and route.poll is True
    request = Request(
        "load-checkpoint",
        "restoring-parent:training:0",
        {"checkpoint_id": _CHECKPOINT_ID, "source_job_id": "saving-parent", "path": None, "step": None},
    )
    assert route.prepare("https://account.example/cortex-training", request) == (
        "https://account.example/cortex-training/restoring-parent/load",
        [{"json": {"checkpoint_id": _CHECKPOINT_ID, "source_job_id": "saving-parent"}}],
    )
    assert route.submitted(request, {"request_id": "production-load-request"}) == "production-load-request"


def test_hosted_training_uses_the_correctness_client():
    assert hosted.ArcticClient is CorrectnessClient


def test_restore_after_owned_shutdown_cannot_reopen_http_session(gateway, checkpoint):
    state, create = gateway
    client = create()
    client.shutdown()
    state.calls.clear()
    with pytest.raises(ValueError, match=_LOAD_OWNER_ERROR):
        client.load_checkpoint(path=str(checkpoint))
    assert state.calls == []
    assert client.transport.job_id is None and client.transport._session is None
