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

"""Static correctness selftests: local transport."""

from __future__ import annotations

import asyncio
import json
import traceback
from copy import deepcopy
from types import SimpleNamespace
from uuid import UUID

import aiohttp
import pytest
import requests
import torch

from arctic_platform import wire
from arctic_platform.client import CortexConfig
from arctic_platform.client.requests import load_checkpoint_request
from arctic_platform.client.transport import Request
from arctic_platform.client.transport import initialize_or_cleanup
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import _route
from arctic_platform.correctness.harness import cortex as cortex_transport
from arctic_platform.correctness.harness import hosted
from arctic_platform.correctness.harness.cortex import LOCAL_NATIVE_FORWARD_ENV
from arctic_platform.correctness.harness.cortex import CorrectnessClient
from arctic_platform.correctness.harness.cortex import CorrectnessCortexTransport
from arctic_platform.correctness.harness.cortex import CortexTrainingConfig
from arctic_platform.correctness.harness.storage import require_data_fast_path
from arctic_platform.correctness.harness.transport import UnsafeCleanupError

# Local cortex checkpoint load.


_local_cortex_checkpoint_load_CHECKPOINT_ID = "cp_00000000-0000-0000-0000-000000000001"

_local_cortex_checkpoint_load_OTHER_ID = "cp_00000000-0000-0000-0000-000000000002"

_local_cortex_checkpoint_load_RESULT = {
    "checkpoint_id": _local_cortex_checkpoint_load_CHECKPOINT_ID,
    "checkpoint_tag": "global_step10",
    "version": 10,
}

_local_cortex_checkpoint_load_STAGE = {
    "location_type": "S3",
    "bucket": "synthetic-control-bucket",
    "prefix": f"jobs/saving-parent/checkpoints/{_local_cortex_checkpoint_load_CHECKPOINT_ID}/",
    "aws": {
        "aws_key_id": "control-key",
        "aws_secret_key": "control-secret",
        "aws_token": "control-token",
    },
}


def _local_cortex_checkpoint_load_response(payload):

    class Reply:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def read(self):
            return json.dumps(payload).encode()

    return Reply()


def _local_cortex_checkpoint_load_load(client, *, stage=None, asynchronous=False):
    if asynchronous:
        return asyncio.run(
            client.transport.acall(
                Request(
                    "load-checkpoint",
                    client.jobs.training,
                    {
                        "checkpoint_id": _local_cortex_checkpoint_load_CHECKPOINT_ID,
                        "stage_info": stage,
                    },
                )
            )
        )
    return client.transport.load_checkpoint(_local_cortex_checkpoint_load_CHECKPOINT_ID, stage_info=stage)


def test_local_cortex_checkpoint_load_production_cortex_load_route_posts_and_polls_parent():
    route = _route("load-checkpoint")
    assert route.method == "POST" and route.path == "/{job}/load" and (route.poll is True)
    request = Request(
        "load-checkpoint",
        "production-parent:training:0",
        {
            "checkpoint_id": _local_cortex_checkpoint_load_CHECKPOINT_ID,
            "stage_info": None,
        },
    )
    assert route.prepare("https://account.example/cortex-training", request) == (
        "https://account.example/cortex-training/production-parent/load",
        [{"json": {"checkpoint_id": _local_cortex_checkpoint_load_CHECKPOINT_ID}}],
    )
    assert route.submitted(request, {"request_id": "production-load-request"}) == "production-load-request"
    assert hosted.ArcticClient is CorrectnessClient


# Local cortex checkpoint resume.

_local_cortex_checkpoint_resume_LOAD_BODY_ERROR = (
    "^local checkpoint load accepts checkpoint_id/stage_info only; no path, step or source_job_id$"
)

_local_cortex_checkpoint_resume_LOAD_ID_ERROR = (
    "^local checkpoint load requires a durable cp_<uuid> checkpoint_id, not a FAST path or tag$"
)

_local_cortex_checkpoint_resume_LOAD_OWNER_ERROR = (
    "^local checkpoint load requires this client's newly owned training child$"
)

_local_cortex_checkpoint_resume_LOCAL_GATEWAY_ERROR = (
    "^local Cortex gateway requires an unauthenticated HTTP loopback base URL$"
)

_local_cortex_checkpoint_resume_CHECKPOINT_ID = "cp_00000000-0000-0000-0000-000000000001"


def _local_cortex_checkpoint_resume_response(payload):

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
def _local_cortex_checkpoint_resume_checkpoint(tmp_path):
    root = require_data_fast_path(tmp_path / "checkpoint", label="resume control basetemp")
    root.mkdir()
    (root / "latest").write_text("global_step1\n", encoding="ascii")
    (root / "global_step1").mkdir()
    return root


@pytest.fixture
def _local_cortex_checkpoint_resume_gateway(monkeypatch):
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
            return _local_cortex_checkpoint_resume_response(
                {
                    "paths": {
                        path: {"post": {"parameters": [{"name": "job_id", "in": "query", "required": True}]}}
                        for path in ("/forward", "/log-probs")
                    }
                }
            )
        if method == "POST" and url.endswith("/cortex-training"):
            state.roles = [sub["job_type"] for sub in kwargs["json"]["sub_job_configs"]]
            return _local_cortex_checkpoint_resume_response({"job_id": "owned-resume-parent"})
        if method == "POST" and url.endswith("/owned-resume-parent:cancel"):
            state.cancelled = True
            return _local_cortex_checkpoint_resume_response({})
        if not url.startswith("http://127.0.0.1:8003/"):
            if method == "POST" and url.endswith("/owned-resume-parent/load"):
                return _local_cortex_checkpoint_resume_response({"request_id": "production-load-request"})
            if method == "GET" and url.endswith("/owned-resume-parent/requests/production-load-request"):
                return _local_cortex_checkpoint_resume_response(
                    {
                        "status": "REQUEST_STATE_DONE",
                        "result": {
                            "checkpoint_id": _local_cortex_checkpoint_resume_CHECKPOINT_ID,
                            "version": 1,
                        },
                    }
                )
        if method == "GET" and url.endswith("/owned-resume-parent"):
            segments = {
                "training": "training",
                "sampling": "sampling",
                "log_probability": "log_prob",
            }
            return _local_cortex_checkpoint_resume_response(
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

    yield (state, create)
    for client in state.clients:
        if client.transport.job_id is not None:
            client.transport.config = client.config.backend
            client.shutdown()
    for transport in transports:
        transport._on_loop(transport._close_session()).result()
        transport._loop.call_soon_threadsafe(transport._loop.stop)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_local_cortex_checkpoint_resume_owned_fast_restore_is_unsupported_before_http_and_still_shuts_down(
    _local_cortex_checkpoint_resume_gateway,
    _local_cortex_checkpoint_resume_checkpoint,
    native,
    asynchronous,
):
    state, create = _local_cortex_checkpoint_resume_gateway
    client = create(native=native)
    transport = client.transport
    original = _local_cortex_checkpoint_resume_checkpoint.joinpath("latest").read_bytes()
    with client:
        with pytest.raises(ValueError, match=_local_cortex_checkpoint_resume_LOAD_BODY_ERROR):
            if asynchronous:
                asyncio.run(
                    transport.acall(
                        load_checkpoint_request(
                            client.jobs,
                            path=str(_local_cortex_checkpoint_resume_checkpoint),
                        )
                    )
                )
            else:
                client.load_checkpoint(path=str(_local_cortex_checkpoint_resume_checkpoint))
        assert state.calls == []
        assert _local_cortex_checkpoint_resume_checkpoint.joinpath("latest").read_bytes() == original
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
    assert all((actual_session is session and actual_loop is loop for actual_session, actual_loop in state.exchanges))


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
def test_local_cortex_checkpoint_resume_wrong_or_nontraining_token_cannot_dispatch_unsupported_restore(
    _local_cortex_checkpoint_resume_gateway,
    _local_cortex_checkpoint_resume_checkpoint,
    token,
):
    state, create = _local_cortex_checkpoint_resume_gateway
    client = create()
    request = Request(
        "load-checkpoint",
        token,
        {"path": str(_local_cortex_checkpoint_resume_checkpoint), "step": None},
    )
    with pytest.raises(ValueError, match=_local_cortex_checkpoint_resume_LOAD_OWNER_ERROR):
        client.transport.call(request)
    assert state.calls == []


@pytest.mark.parametrize(
    "body,message",
    [
        ({}, _local_cortex_checkpoint_resume_LOAD_ID_ERROR),
        ({"path": None, "step": None}, _local_cortex_checkpoint_resume_LOAD_BODY_ERROR),
        ({"path": ""}, _local_cortex_checkpoint_resume_LOAD_BODY_ERROR),
        (
            {"path": "relative/checkpoint"},
            _local_cortex_checkpoint_resume_LOAD_BODY_ERROR,
        ),
        ({"path": "/tmp/checkpoint"}, _local_cortex_checkpoint_resume_LOAD_BODY_ERROR),
        ({"path": "/data-fast"}, _local_cortex_checkpoint_resume_LOAD_BODY_ERROR),
        ({"path": 123}, _local_cortex_checkpoint_resume_LOAD_BODY_ERROR),
        (
            {"checkpoint_path": "/data-fast/tmp/checkpoint"},
            _local_cortex_checkpoint_resume_LOAD_BODY_ERROR,
        ),
        (
            {"checkpoint_id": "global_step1"},
            _local_cortex_checkpoint_resume_LOAD_ID_ERROR,
        ),
        (
            {"source_job_id": "foreign-parent", "path": "/data-fast/tmp/checkpoint"},
            _local_cortex_checkpoint_resume_LOAD_BODY_ERROR,
        ),
        (
            {"source_job_id": "", "path": "/data-fast/tmp/checkpoint"},
            _local_cortex_checkpoint_resume_LOAD_BODY_ERROR,
        ),
    ],
)
def test_local_cortex_checkpoint_resume_alternate_restore_bodies_never_become_zone_load_requests(
    _local_cortex_checkpoint_resume_gateway, body, message
):
    state, create = _local_cortex_checkpoint_resume_gateway
    client = create()
    with pytest.raises(ValueError, match=message):
        client.transport.call(Request("load-checkpoint", client.jobs.training, body))
    assert state.calls == []


@pytest.mark.parametrize("step", [0, 1, -1, True, "1"])
def test_local_cortex_checkpoint_resume_explicit_step_does_not_enable_a_local_restore_fallback(
    _local_cortex_checkpoint_resume_gateway,
    _local_cortex_checkpoint_resume_checkpoint,
    step,
):
    state, create = _local_cortex_checkpoint_resume_gateway
    client = create()
    with pytest.raises(ValueError, match=_local_cortex_checkpoint_resume_LOAD_BODY_ERROR):
        client.load_checkpoint(path=str(_local_cortex_checkpoint_resume_checkpoint), step=step)
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
def test_local_cortex_checkpoint_resume_authenticated_or_decorated_loopback_cannot_enable_restore(
    _local_cortex_checkpoint_resume_gateway,
    _local_cortex_checkpoint_resume_checkpoint,
    backend,
):
    state, create = _local_cortex_checkpoint_resume_gateway
    client = create()
    client.transport.config = client.transport.config.model_copy(update=backend)
    with pytest.raises(ValueError, match=_local_cortex_checkpoint_resume_LOCAL_GATEWAY_ERROR):
        client.load_checkpoint(path=str(_local_cortex_checkpoint_resume_checkpoint))
    assert state.calls == []


def test_local_cortex_checkpoint_resume_unsupported_restore_never_enters_http_or_polling_even_with_retries(
    _local_cortex_checkpoint_resume_gateway,
    _local_cortex_checkpoint_resume_checkpoint,
    monkeypatch,
):
    state, create = _local_cortex_checkpoint_resume_gateway
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
    with pytest.raises(ValueError, match=_local_cortex_checkpoint_resume_LOAD_BODY_ERROR):
        client.load_checkpoint(path=str(_local_cortex_checkpoint_resume_checkpoint))
    assert attempts == state.calls == []


@pytest.mark.parametrize("base_url", ["https://127.0.0.1:8003", "http://remote.example:8003"])
def test_local_cortex_checkpoint_resume_nonlocal_resume_uses_production_checkpoint_id_and_source_job(
    _local_cortex_checkpoint_resume_gateway, base_url
):
    state, create = _local_cortex_checkpoint_resume_gateway
    client = create()
    client.transport.config = client.transport.config.model_copy(update={"base_url": base_url})
    assert not client.transport._is_local_gateway()
    assert client.load_checkpoint(
        checkpoint_id=_local_cortex_checkpoint_resume_CHECKPOINT_ID,
        source_job_id="saving-parent",
    ) == {"checkpoint_id": _local_cortex_checkpoint_resume_CHECKPOINT_ID, "version": 1}
    prefix = client.transport._prefix
    assert [(method, url) for method, url, _ in state.calls] == [
        ("POST", f"{prefix}/owned-resume-parent/load"),
        ("GET", f"{prefix}/owned-resume-parent/requests/production-load-request"),
    ]
    assert state.calls[0][2]["json"] == {
        "checkpoint_id": _local_cortex_checkpoint_resume_CHECKPOINT_ID,
        "source_job_id": "saving-parent",
    }


def test_local_cortex_checkpoint_resume_production_cortex_resume_route_preserves_request_body():
    route = _route("load-checkpoint")
    assert route.method == "POST" and route.path == "/{job}/load" and (route.poll is True)
    request = Request(
        "load-checkpoint",
        "restoring-parent:training:0",
        {
            "checkpoint_id": _local_cortex_checkpoint_resume_CHECKPOINT_ID,
            "source_job_id": "saving-parent",
            "path": None,
            "step": None,
        },
    )
    assert route.prepare("https://account.example/cortex-training", request) == (
        "https://account.example/cortex-training/restoring-parent/load",
        [
            {
                "json": {
                    "checkpoint_id": _local_cortex_checkpoint_resume_CHECKPOINT_ID,
                    "source_job_id": "saving-parent",
                }
            }
        ],
    )
    assert route.submitted(request, {"request_id": "production-load-request"}) == "production-load-request"


def test_local_cortex_checkpoint_resume_hosted_training_uses_the_correctness_client():
    assert hosted.ArcticClient is CorrectnessClient


def test_local_cortex_checkpoint_resume_restore_after_owned_shutdown_cannot_reopen_http_session(
    _local_cortex_checkpoint_resume_gateway, _local_cortex_checkpoint_resume_checkpoint
):
    state, create = _local_cortex_checkpoint_resume_gateway
    client = create()
    client.shutdown()
    state.calls.clear()
    with pytest.raises(ValueError, match=_local_cortex_checkpoint_resume_LOAD_OWNER_ERROR):
        client.load_checkpoint(path=str(_local_cortex_checkpoint_resume_checkpoint))
    assert state.calls == []
    assert client.transport.job_id is None and client.transport._session is None


# Local cortex forward.


def _local_cortex_forward_response(payload=None, *, status=200, data=None):

    class Reply:
        reason = "synthetic gateway response"
        request_info = SimpleNamespace(real_url="http://127.0.0.1:8003/mock")
        history = ()
        headers = {}

        def __init__(self):
            self.status = status
            self.data = data if data is not None else json.dumps(payload).encode()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def read(self):
            return self.data

        async def text(self):
            return self.data.decode()

    return Reply()


def _local_cortex_forward_native_reply(monkeypatch, reply):
    original = aiohttp.ClientSession.request

    def request(session, method, url, **kwargs):
        if any(f"{route}?job_id=" in url for route in ("/forward", "/log-probs", "/save-checkpoint")):
            assert method == "POST"
            assert asyncio.get_running_loop() is session._loop
            assert session.trust_env is False
            assert kwargs["allow_redirects"] is False
            assert "Authorization" not in session.headers
            return reply(session, url, **kwargs)
        return original(session, method, url, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "request", request)


def _local_cortex_forward_client_config(backend=None, **kwargs):
    return CortexTrainingConfig(
        model_name="Qwen/Qwen3-8B",
        training_gpus=kwargs.pop("training_gpus", 4),
        hardware="H200",
        backend=backend or CortexConfig(base_url="http://127.0.0.1:8003", max_retries=0),
        **kwargs,
    )


def _local_cortex_forward_openapi(*paths):
    return {
        "paths": {
            path: {"post": {"parameters": [{"name": "job_id", "in": "query", "required": True}]}} for path in paths
        }
    }


@pytest.fixture
def _local_cortex_forward_gateway(monkeypatch):
    monkeypatch.setenv(LOCAL_NATIVE_FORWARD_ENV, "1")
    calls = []
    schema = _local_cortex_forward_openapi("/forward", "/log-probs")
    state = {"roles": [], "cancelled": False}
    transports = []
    from arctic_platform.client.transports.cortex import CortexTransport

    original_start = CortexTransport._start_loop

    def start(actual):
        original_start(actual)
        transports.append(actual)

    monkeypatch.setattr(CortexTransport, "_start_loop", start)

    def send(session, method, url, **kwargs):
        calls.append((method, url, kwargs))
        assert asyncio.get_running_loop() is session._loop
        assert session.trust_env is False
        assert kwargs["allow_redirects"] is False
        assert "Authorization" not in session.headers
        if url.endswith("/openapi.json"):
            return _local_cortex_forward_response(schema)
        if url.endswith(":cancel"):
            state["cancelled"] = True
            return _local_cortex_forward_response({})
        if method == "POST" and url.endswith("/cortex-training"):
            state["roles"] = [sub["job_type"] for sub in kwargs["json"]["sub_job_configs"]]
            state["cancelled"] = False
            return _local_cortex_forward_response({"job_id": "owned-parent"})
        if method == "GET" and url.endswith("/owned-parent"):
            return _local_cortex_forward_response(
                {
                    "job_id": "owned-parent",
                    "status": "cancelled" if state["cancelled"] else "RUNNING",
                    "sub_jobs": [
                        {
                            "job_type": role,
                            "sub_job_id": f"owned-parent:{('log_prob' if role == 'log_probability' else role)}:0",
                            "status": "cancelled" if state["cancelled"] else "running",
                        }
                        for role in state["roles"]
                    ],
                }
            )
        pytest.fail(f"unexpected request: {method} {url}")

    monkeypatch.setattr(aiohttp.ClientSession, "request", send)
    yield (calls, schema)
    for transport in transports:
        transport._on_loop(transport._close_session()).result()
        transport._loop.call_soon_threadsafe(transport._loop.stop)


def _local_cortex_forward_created(calls):
    return next(
        (kwargs["json"] for method, url, kwargs in calls if method == "POST" and url.endswith("/cortex-training"))
    )


def test_local_cortex_forward_native_binding_uses_only_fresh_public_config_ids(
    _local_cortex_forward_gateway,
):
    calls, _ = _local_cortex_forward_gateway
    config = _local_cortex_forward_client_config(sampling_gpus=1, log_prob_gpus=1)
    original = config.to_cortex()
    transport = CorrectnessCortexTransport(config.backend)
    jobs = transport.initialize(config)
    payload = _local_cortex_forward_created(calls)
    identifiers = [sub["job_id"] for sub in payload["sub_job_configs"]]
    assert len(set(identifiers)) == 3
    assert all((str(UUID(identifier)) == identifier for identifier in identifiers))
    assert payload["hardware"] == "H200"
    assert config.to_cortex() == original
    for sub, expected in zip(payload["sub_job_configs"], original):
        assert {key: value for key, value in sub.items() if key != "job_id"} == expected
        role = "log_prob" if sub["job_type"] == "log_probability" else sub["job_type"]
        token = getattr(jobs, role)
        assert transport._native_bindings[str(token)] == (role, sub["job_id"])
    assert calls[0][1].endswith("/openapi.json")
    transport.shutdown()
    assert calls[-2][1].endswith("/owned-parent:cancel")
    assert calls[-1][0] == "GET" and calls[-1][1].endswith("/owned-parent")
    assert transport._native_bindings == transport._native_children == {}


@pytest.mark.parametrize(
    "backend",
    [
        {"base_url": "https://127.0.0.1"},
        {"base_url": "http://remote.example"},
        {"base_url": "http://127.0.0.1/path"},
        {"base_url": "http://127.0.0.1?query=1"},
        {"base_url": "http://127.0.0.1#fragment"},
        {"base_url": "http://user:secret@127.0.0.1"},
        {"base_url": "http://127.0.0.1", "host": "account.example"},
        {"base_url": "http://127.0.0.1", "pat": "synthetic-test-token"},
    ],
)
def test_local_cortex_forward_native_opt_in_rejects_hosted_or_decorated_gateway_before_request(
    _local_cortex_forward_gateway, backend
):
    calls, _ = _local_cortex_forward_gateway
    config = _local_cortex_forward_client_config(CortexConfig(**backend))
    transport = CorrectnessCortexTransport(config.backend)
    with pytest.raises(ValueError, match="unauthenticated HTTP loopback"):
        initialize_or_cleanup(transport, config)
    assert calls == []
    assert transport.job_id is None
    assert transport._native_bindings == {}


def test_local_cortex_forward_native_reconnect_is_rejected_without_discovery(
    _local_cortex_forward_gateway,
):
    calls, _ = _local_cortex_forward_gateway
    config = _local_cortex_forward_client_config(training_job_id="other-parent:training:0")
    transport = CorrectnessCortexTransport(config.backend)
    with pytest.raises(ValueError, match="reconnect is unsupported"):
        initialize_or_cleanup(transport, config)
    assert calls == []


@pytest.mark.parametrize("broken", ["missing", "no_post", "wrong_query", "optional"])
def test_local_cortex_forward_actual_installed_api_is_checked_before_create(_local_cortex_forward_gateway, broken):
    calls, schema = _local_cortex_forward_gateway
    if broken == "missing":
        schema["paths"].pop("/forward")
    elif broken == "no_post":
        schema["paths"]["/forward"]["get"] = schema["paths"]["/forward"].pop("post")
    elif broken == "wrong_query":
        schema["paths"]["/forward"]["post"]["parameters"][0]["name"] = "sub_job_id"
    else:
        schema["paths"]["/forward"]["post"]["parameters"][0]["required"] = False
    config = _local_cortex_forward_client_config()
    transport = CorrectnessCortexTransport(config.backend)
    with pytest.raises(RuntimeError, match="before job allocation"):
        initialize_or_cleanup(transport, config)
    assert len(calls) == 1
    assert calls[0][0] == "GET"
    assert transport.job_id is None


def test_local_cortex_forward_log_probability_route_checked_for_declared_log_probability_job(
    _local_cortex_forward_gateway,
):
    calls, schema = _local_cortex_forward_gateway
    schema["paths"].pop("/log-probs")
    config = _local_cortex_forward_client_config(training_gpus=0, log_prob_gpus=1)
    transport = CorrectnessCortexTransport(config.backend)
    with pytest.raises(RuntimeError, match="/log-probs"):
        initialize_or_cleanup(transport, config)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "op,role",
    [("forward", "training"), ("log-probs", "log_prob"), ("log-probs", "sampling")],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_local_cortex_forward_native_wire_request_uses_owned_uuid_not_synthetic_token(
    _local_cortex_forward_gateway, monkeypatch, op, role, asynchronous
):
    calls, _ = _local_cortex_forward_gateway
    config = _local_cortex_forward_client_config(sampling_gpus=1, log_prob_gpus=1)
    transport = CorrectnessCortexTransport(config.backend)
    jobs = transport.initialize(config)
    token = getattr(jobs, role)
    child = transport._native_bindings[str(token)][1]
    body = {
        "batch": {"input_ids": torch.tensor([[2, 3]])},
        "processing": {"temperature": 1.0},
        "optional": None,
    }
    expected = {"logprobs": torch.tensor([[-1.2, -2.3]], dtype=torch.float32)}
    captured = []

    def post(session, url, **kwargs):
        captured.append((url, kwargs))
        return _local_cortex_forward_response(data=wire.dumps(expected))

    _local_cortex_forward_native_reply(monkeypatch, post)
    request = Request(op, token, body)
    result = asyncio.run(transport.acall(request)) if asynchronous else transport.call(request)
    assert torch.equal(result["logprobs"], expected["logprobs"])
    url, kwargs = captured[0]
    assert url == f"http://127.0.0.1:8003/{op}?job_id={child}"
    assert "owned-parent" not in url
    assert kwargs["allow_redirects"] is False
    assert kwargs["timeout"].total == config.backend.request_timeout
    if op == "forward":
        decoded = wire.loads(kwargs["data"])
        assert torch.equal(decoded["batch"]["input_ids"], body["batch"]["input_ids"])
        assert decoded["processing"] == body["processing"]
        assert "optional" not in decoded
        assert kwargs["headers"]["Content-Type"] == "application/octet-stream"
    else:
        assert kwargs["json"] == {key: value for key, value in body.items() if value is not None}
    transport.shutdown()


@pytest.mark.parametrize(
    "op,token",
    [
        ("forward", "foreign:training:0"),
        ("forward", "owned-parent:log_probability:0"),
        ("log-probs", "owned-parent:training:0"),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_local_cortex_forward_native_forward_never_discovers_or_reroutes_foreign_or_wrong_role(
    _local_cortex_forward_gateway, op, token, asynchronous
):
    calls, _ = _local_cortex_forward_gateway
    config = _local_cortex_forward_client_config(log_prob_gpus=1)
    transport = CorrectnessCortexTransport(config.backend)
    transport.initialize(config)
    before = len(calls)
    with pytest.raises(ValueError, match="owned child|cannot target"):
        request = Request(op, token, {})
        if asynchronous:
            asyncio.run(transport.acall(request))
        else:
            transport.call(request)
    assert len(calls) == before
    transport.shutdown()
    with pytest.raises(ValueError, match="owned child"):
        request = Request("forward", "owned-parent:training:0", {})
        if asynchronous:
            asyncio.run(transport.acall(request))
        else:
            transport.call(request)


@pytest.mark.parametrize("status", [302, 400, 500])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_local_cortex_forward_native_error_preserves_server_reason_and_cleanup(
    _local_cortex_forward_gateway, monkeypatch, status, asynchronous
):
    calls, _ = _local_cortex_forward_gateway
    config = _local_cortex_forward_client_config()
    transport = CorrectnessCortexTransport(config.backend)
    jobs = transport.initialize(config)
    _local_cortex_forward_native_reply(
        monkeypatch,
        lambda *args, **kwargs: _local_cortex_forward_response({"detail": "actual gateway error"}, status=status),
    )
    with pytest.raises(
        (aiohttp.ClientResponseError, RuntimeError),
        match="actual gateway error|without redirects",
    ):
        request = Request("forward", jobs.training, {})
        if asynchronous:
            asyncio.run(transport.acall(request))
        else:
            transport.call(request)
    transport.shutdown()
    assert calls[-2][1].endswith(":cancel")
    assert transport._native_bindings == {}


def test_local_cortex_forward_native_creation_failure_cleans_parent_and_binding(
    _local_cortex_forward_gateway, monkeypatch
):
    calls, _ = _local_cortex_forward_gateway
    config = _local_cortex_forward_client_config()
    transport = CorrectnessCortexTransport(config.backend)
    monkeypatch.setattr(
        transport,
        "_wait_running",
        lambda: (_ for _ in ()).throw(RuntimeError("actual startup error")),
    )
    with pytest.raises(RuntimeError, match="actual startup error"):
        initialize_or_cleanup(transport, config)
    assert "job_id" in _local_cortex_forward_created(calls)["sub_job_configs"][0]
    assert calls[-2][1].endswith("/owned-parent:cancel")
    assert transport.job_id is None
    assert transport._native_bindings == transport._native_children == {}


@pytest.mark.parametrize(
    "payload",
    [b"invalid wire", wire.dumps({"job_id": "foreign-child", "logprobs": [0.0]})],
    ids=["invalid-wire", "foreign-identity"],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_local_cortex_forward_native_malformed_or_wrong_identity_response_rejected(
    _local_cortex_forward_gateway, monkeypatch, payload, asynchronous
):
    config = _local_cortex_forward_client_config()
    transport = CorrectnessCortexTransport(config.backend)
    jobs = transport.initialize(config)
    _local_cortex_forward_native_reply(
        monkeypatch,
        lambda *args, **kwargs: _local_cortex_forward_response(data=payload),
    )
    with pytest.raises(ValueError):
        request = Request("forward", jobs.training, {})
        if asynchronous:
            asyncio.run(transport.acall(request))
        else:
            transport.call(request)
    transport.shutdown()


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:8003", "http://remote.example"])
def test_local_cortex_forward_no_opt_in_preserves_public_compute_routes(
    _local_cortex_forward_gateway, monkeypatch, base_url
):
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV)
    config = _local_cortex_forward_client_config(CortexConfig(base_url=base_url))
    transport = CorrectnessCortexTransport(config.backend)
    calls = []

    def initialize(actual, cfg):
        actual._on_loop(actual._send("POST", actual._prefix, json={"sub_job_configs": cfg.to_cortex()})).result()
        return actual.jobs

    async def send(actual, method, url, **kwargs):
        assert asyncio.get_running_loop() is actual._loop
        calls.append((method, url, kwargs))
        return {"job_id": "public-parent"}

    from arctic_platform.client.transports.cortex import CortexTransport

    monkeypatch.setattr(CortexTransport, "initialize", initialize)
    monkeypatch.setattr(CortexTransport, "_send", send)
    monkeypatch.setattr(CortexTransport, "call", lambda actual, request: {"public": request.op})
    transport.initialize(config)
    assert len(calls) == 1
    configs = calls[0][2]["json"]["sub_job_configs"]
    if base_url == "http://127.0.0.1:8003":
        assert all(str(UUID(sub["job_id"])) == sub["job_id"] for sub in configs)
    else:
        assert all("job_id" not in sub for sub in configs)
    assert transport.call(Request("forward", "public-training", {})) == {"public": "forward"}
    assert transport.call(Request("log-probs", "public-log-prob", {})) == {"public": "log-probs"}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_local_cortex_save_automatically_uses_explicit_path_and_owned_native_child(
    _local_cortex_forward_gateway, monkeypatch, asynchronous
):
    calls, _ = _local_cortex_forward_gateway
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV)
    config = _local_cortex_forward_client_config()
    transport = CorrectnessCortexTransport(config.backend)
    jobs = transport.initialize(config)
    assert not any(url.endswith("/openapi.json") for _, url, _ in calls)
    child = transport._native_bindings[str(jobs.training)][1]
    root = "/data-fast/tmp/correctness-owned-save-control"
    expected = {
        "job_id": child,
        "checkpoint_path": root,
        "checkpoint_tag": "global_step0",
        "version": 0,
    }
    captured = []

    def post(session, url, **kwargs):
        captured.append((url, kwargs))
        return _local_cortex_forward_response(expected)

    _local_cortex_forward_native_reply(monkeypatch, post)
    request = Request(
        "save",
        jobs.training,
        {"path": root, "checkpoint_type": "resumable", "export_hf": False},
    )
    result = asyncio.run(transport.acall(request)) if asynchronous else transport.call(request)
    assert result == expected
    url, kwargs = captured[0]
    assert url == f"http://127.0.0.1:8003/save-checkpoint?job_id={child}"
    assert kwargs["json"] == {"path": root, "checkpoint_type": "resumable"}
    assert kwargs["allow_redirects"] is False
    transport.shutdown()
    assert transport._native_bindings == {}


@pytest.mark.parametrize(
    "change",
    [
        {"path": "/tmp/not-fast"},
        {"path": "/data-fast/tmp/../control"},
        {"checkpoint_type": "unknown"},
        {"checkpoint_id": "managed-checkpoint"},
        {"stage_info": {"location": "remote"}},
        {"step": 1},
        {"export_hf": True},
        {"save_total_limit": 1},
    ],
)
def test_local_cortex_save_rejects_unsupported_native_options_before_request(
    _local_cortex_forward_gateway, monkeypatch, change
):
    calls, _ = _local_cortex_forward_gateway
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV)
    config = _local_cortex_forward_client_config()
    transport = CorrectnessCortexTransport(config.backend)
    jobs = transport.initialize(config)
    before = len(calls)
    body = {
        "path": "/data-fast/tmp/correctness-owned-save-control",
        "checkpoint_type": "resumable",
        **change,
    }
    with pytest.raises(ValueError):
        transport.call(Request("save", jobs.training, body))
    assert len(calls) == before
    transport.shutdown()


@pytest.mark.parametrize("token", ["foreign:training:0", "owned-parent:sampling:0"])
def test_local_cortex_save_never_reroutes_foreign_or_wrong_role(_local_cortex_forward_gateway, monkeypatch, token):
    calls, _ = _local_cortex_forward_gateway
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV)
    config = _local_cortex_forward_client_config(sampling_gpus=1)
    transport = CorrectnessCortexTransport(config.backend)
    transport.initialize(config)
    before = len(calls)
    body = {
        "path": "/data-fast/tmp/correctness-owned-save-control",
        "checkpoint_type": "resumable",
    }
    with pytest.raises(ValueError, match="owned child|cannot target"):
        transport.call(Request("save", token, body))
    assert len(calls) == before
    transport.shutdown()
    with pytest.raises(ValueError, match="owned child"):
        transport.call(Request("save", "owned-parent:training:0", body))


@pytest.mark.parametrize("payload", [b"invalid-json", b"[]", b'{"job_id": "foreign-child"}'])
def test_local_cortex_save_rejects_malformed_or_wrong_child_response(
    _local_cortex_forward_gateway, monkeypatch, payload
):
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV)
    config = _local_cortex_forward_client_config()
    transport = CorrectnessCortexTransport(config.backend)
    jobs = transport.initialize(config)
    _local_cortex_forward_native_reply(
        monkeypatch,
        lambda *args, **kwargs: _local_cortex_forward_response(data=payload),
    )
    body = {
        "path": "/data-fast/tmp/correctness-owned-save-control",
        "checkpoint_type": "resumable",
    }
    with pytest.raises(ValueError):
        transport.call(Request("save", jobs.training, body))
    transport.shutdown()


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:8003", "http://remote.example"])
def test_local_cortex_save_managed_and_hosted_saves_preserve_public_route(
    _local_cortex_forward_gateway, monkeypatch, base_url
):
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV)
    config = _local_cortex_forward_client_config(CortexConfig(base_url=base_url))
    transport = CorrectnessCortexTransport(config.backend)
    body = {"path": None if base_url.startswith("http://127.") else "/data-fast/tmp/control"}
    request = Request("save", "managed-job", body)
    monkeypatch.setattr(
        CortexTransport,
        "call",
        lambda actual, supplied: {"public": supplied is request},
    )
    assert transport.call(request) == {"public": True}


def test_local_cortex_forward_native_sync_and_async_forward_share_owned_loop_session_binding_and_wire(
    _local_cortex_forward_gateway, monkeypatch
):
    config = _local_cortex_forward_client_config()
    transport = CorrectnessCortexTransport(config.backend)
    jobs = transport.initialize(config)
    expected = {"logprobs": torch.tensor([[-0.5]])}
    calls = []
    caller_loops = []

    def post(session, url, **kwargs):
        calls.append((session, asyncio.get_running_loop(), url, kwargs))
        return _local_cortex_forward_response(data=wire.dumps(expected))

    _local_cortex_forward_native_reply(monkeypatch, post)
    request = Request("forward", jobs.training, {"batch": {"input_ids": [1]}})
    sync_result = transport.call(request)

    async def call():
        caller_loops.append(asyncio.get_running_loop())
        return await transport.acall(request)

    result = asyncio.run(call())
    assert torch.equal(sync_result["logprobs"], expected["logprobs"])
    assert torch.equal(result["logprobs"], expected["logprobs"])
    assert len(calls) == 2
    session = transport._session
    assert session is not None
    assert all((actual_session is session and loop is transport._loop for actual_session, loop, _, _ in calls))
    assert caller_loops[0] is not transport._loop
    assert all((url.endswith(f"/forward?job_id={transport._native_children['training']}") for _, _, url, _ in calls))
    assert all((kwargs["allow_redirects"] is False for _, _, _, kwargs in calls))
    assert all((kwargs["timeout"].total == config.backend.request_timeout for _, _, _, kwargs in calls))
    assert wire.loads(calls[0][3]["data"]) == wire.loads(calls[1][3]["data"]) == request.body
    transport.shutdown()
    assert session.closed and transport._session is None


# Local cortex shutdown.


@pytest.fixture
def _local_cortex_shutdown_gateway(monkeypatch):
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV, raising=False)
    state = SimpleNamespace(
        calls=[],
        polls=[],
        closed=False,
        cancelled=False,
        elapsed=0.0,
        slept=[],
        error=None,
        roles=[],
        exchanges=[],
        closes=[],
    )
    transports = []

    def sleep(seconds):
        state.slept.append(seconds)
        state.elapsed += seconds

    monkeypatch.setattr(
        cortex_transport,
        "time",
        SimpleNamespace(monotonic=lambda: state.elapsed, sleep=sleep),
    )

    def job(status="cancelled", child_status="cancelled"):
        segments = {
            "training": "training",
            "sampling": "sampling",
            "log_probability": "log_prob",
        }
        return {
            "job_id": "owned-parent",
            "status": status,
            "sub_jobs": [
                {
                    "job_type": role,
                    "sub_job_id": f"owned-parent:{segments[role]}:0",
                    "status": child_status,
                }
                for role in state.roles
            ],
        }

    def create(
        *,
        base_url="http://127.0.0.1:8003",
        native=False,
        reconnect=False,
        roles=("training",),
    ):
        if native:
            monkeypatch.setenv(LOCAL_NATIVE_FORWARD_ENV, "1")
        config = CortexTrainingConfig(
            model_name="Qwen/Qwen3-8B",
            training_gpus=4 if "training" in roles else 0,
            sampling_gpus=1 if "sampling" in roles else 0,
            log_prob_gpus=1 if "log_probability" in roles else 0,
            training_job_id="owned-parent:training:0" if reconnect else None,
            backend=CortexConfig(base_url=base_url, poll_timeout=1.0, request_timeout=10.0, max_retries=3),
        )
        transport = CorrectnessCortexTransport(config.backend)
        transports.append(transport)
        transport.poll_interval = 0.4
        original_config = transport.config

        def send(session, method, url, **kwargs):
            assert not state.closed
            assert asyncio.get_running_loop() is transport._loop is session._loop
            assert "Authorization" not in session.headers
            if transport._is_local_gateway():
                assert kwargs["allow_redirects"] is False
                assert session.trust_env is False
            state.calls.append((method, url, kwargs))
            state.exchanges.append((session, asyncio.get_running_loop()))
            if url.endswith(":cancel"):
                if state.error:
                    raise state.error
                state.cancelled = True
                payload = {}
            elif method == "POST":
                state.roles = [sub["job_type"] for sub in kwargs["json"]["sub_job_configs"]]
                payload = {"job_id": "owned-parent"}
            elif state.cancelled:
                assert transport.job_id == transport._local_shutdown_parent == "owned-parent"
                assert transport._local_shutdown_children
                assert 0 < kwargs["timeout"].total <= original_config.poll_timeout - state.elapsed
                if native:
                    assert transport._native_bindings and transport._native_children
                payload = state.polls.pop(0) if state.polls else job()
                if callable(payload):
                    payload = payload()
            else:
                if reconnect:
                    state.roles = list(roles)
                payload = job("running", "running")

            class Reply:
                status = 200

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    return False

                async def read(self):
                    return json.dumps(payload).encode()

            return Reply()

        monkeypatch.setattr(aiohttp.ClientSession, "request", send)
        monkeypatch.setattr(transport, "_validate_native_api", lambda actual: None)
        original_close = aiohttp.ClientSession.close

        async def close(session):
            assert asyncio.get_running_loop() is transport._loop is session._loop
            state.closes.append((session, asyncio.get_running_loop()))
            await original_close(session)
            state.closed = True

        monkeypatch.setattr(aiohttp.ClientSession, "close", close)
        transport.initialize(config)
        state.calls.clear()
        return (transport, original_config)

    yield (state, job, create)
    for transport in transports:
        transport._on_loop(transport._close_session()).result()
        transport._loop.call_soon_threadsafe(transport._loop.stop)


@pytest.mark.parametrize("native", [False, True])
def test_local_cortex_shutdown_cancel_ack_waits_for_every_owned_child_before_closing(
    _local_cortex_shutdown_gateway, native
):
    state, job, create = _local_cortex_shutdown_gateway
    transport, original = create(native=native, roles=("training", "sampling", "log_probability"))
    mixed = job(child_status="cancelled")
    mixed["sub_jobs"][1]["status"] = "pending"
    state.polls = [job(child_status="running"), mixed, job()]
    transport.shutdown()
    assert [method for method, _, _ in state.calls] == ["POST", "GET", "GET", "GET"]
    assert all((url.startswith(f"{transport._prefix}/owned-parent") for _, url, _ in state.calls))
    assert state.calls[0][1].endswith(":cancel")
    assert all((kwargs["allow_redirects"] is False for _, _, kwargs in state.calls))
    assert state.slept == [0.4, 0.4]
    assert state.closed and transport.job_id is None
    assert transport._native_bindings == transport._native_children == {}
    assert transport._local_shutdown_parent is None and transport._local_shutdown_children == {}
    assert transport.config is original
    assert len(state.closes) == 1
    session, loop = state.closes[0]
    assert loop is transport._loop and session.closed and (transport._session is None)
    assert all((actual_session is session and actual_loop is loop for actual_session, actual_loop in state.exchanges))
    before = len(state.calls)
    transport.shutdown()
    assert len(state.calls) == before


@pytest.mark.parametrize(
    "broken",
    [
        "parent",
        "parent-failed",
        "parent-reason",
        "missing",
        "extra",
        "duplicate",
        "token",
        "role",
        "child-failed",
        "child-reason",
        "child-unknown",
        "child-nonfinite",
        "malformed",
        "no-list",
    ],
)
def test_local_cortex_shutdown_invalid_or_failed_owned_cancellation_fails_closed(
    _local_cortex_shutdown_gateway, broken
):
    state, job, create = _local_cortex_shutdown_gateway
    transport, original = create(roles=("training", "sampling"))
    payload = job()
    children = payload["sub_jobs"]
    if broken == "parent":
        payload["job_id"] = "foreign-parent"
    elif broken == "parent-failed":
        payload["status"] = "failed"
    elif broken == "parent-reason":
        payload["reason"] = "actual placement destroy failed"
    elif broken == "missing":
        children.pop()
    elif broken == "extra":
        children.append(deepcopy(children[0]))
    elif broken == "duplicate":
        children[1] = deepcopy(children[0])
    elif broken == "token":
        children[0]["sub_job_id"] = "foreign-parent:training:0"
    elif broken == "role":
        children[0]["job_type"] = "log_probability"
    elif broken.startswith("child-"):
        key, value = {
            "child-failed": ("status", "failed"),
            "child-reason": ("reason", "actual owned destroy failed"),
            "child-unknown": ("status", "destroying"),
            "child-nonfinite": ("status", float("nan")),
        }[broken]
        children[0][key] = value
    elif broken == "malformed":
        children[0] = None
    else:
        payload["sub_jobs"] = {}
    state.polls = [payload]
    with pytest.raises(UnsafeCleanupError, match="local gateway"):
        transport.shutdown()
    assert state.closed and transport.job_id is None and (transport.config is original)
    assert state.slept == []


def test_local_cortex_shutdown_pending_child_reaches_configured_monotonic_deadline(
    _local_cortex_shutdown_gateway,
):
    state, job, create = _local_cortex_shutdown_gateway
    transport, original = create()
    state.polls = [job(child_status="pending")] * 3
    with pytest.raises(UnsafeCleanupError, match="cancellation deadline") as caught:
        transport.shutdown()
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert state.elapsed == original.poll_timeout
    assert len(state.calls) == 4
    assert state.closed and transport.config is original


@pytest.mark.parametrize("field", ["poll_timeout", "request_timeout", "poll_interval"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_local_cortex_shutdown_nonfinite_or_nonpositive_timing_is_rejected_without_cancel(
    _local_cortex_shutdown_gateway, field, value
):
    state, _, create = _local_cortex_shutdown_gateway
    transport, _ = create()
    if field == "poll_interval":
        transport.poll_interval = value
    else:
        transport.config = transport.config.model_copy(update={field: value})
    with pytest.raises(UnsafeCleanupError, match="finite and positive") as caught:
        transport.shutdown()
    assert isinstance(caught.value.__cause__, ValueError)
    assert state.calls == [] and state.closed


def test_local_cortex_shutdown_cancel_network_failure_is_not_suppressed_or_retried(
    _local_cortex_shutdown_gateway,
):
    state, _, create = _local_cortex_shutdown_gateway
    transport, original = create()
    state.error = aiohttp.ClientConnectionError("actual cancellation request failed")
    with pytest.raises(UnsafeCleanupError, match="actual cancellation request failed") as caught:
        transport.shutdown()
    assert caught.value.__cause__ is state.error
    assert len(state.calls) == 1 and state.closed
    assert transport.config is original


def test_local_cortex_shutdown_poll_network_failure_restores_config_and_closes(
    _local_cortex_shutdown_gateway,
):
    state, _, create = _local_cortex_shutdown_gateway
    transport, original = create()

    def failure():
        raise aiohttp.ClientConnectionError("actual cancellation poll failed")

    state.polls = [failure]
    with pytest.raises(UnsafeCleanupError, match="actual cancellation poll failed") as caught:
        transport.shutdown()
    assert isinstance(caught.value.__cause__, aiohttp.ClientConnectionError)
    assert len(state.calls) == 2 and state.closed
    assert transport.config is original


def test_local_cortex_shutdown_close_failure_preserves_cancellation_error_and_owned_identity(
    _local_cortex_shutdown_gateway, monkeypatch
):
    state, _, create = _local_cortex_shutdown_gateway
    transport, original = create(native=True, roles=("training", "sampling", "log_probability"))
    primary = aiohttp.ClientConnectionError("owned cancellation failed")
    state.error = primary
    close = transport._close_session

    async def failing_close():
        await close()
        raise RuntimeError("owned session close failed")

    with monkeypatch.context() as patch:
        patch.setattr(transport, "_close_session", failing_close)
        with pytest.raises(UnsafeCleanupError) as caught:
            transport.shutdown()

    assert caught.value.__cause__ is caught.value.shutdown_error is primary
    assert len(state.calls) == 1 and state.closed
    assert transport.config is original and transport.job_id is None
    assert transport._native_bindings == transport._native_children == {}
    assert transport._local_shutdown_parent is None and transport._local_shutdown_children == {}
    notes = str(caught.value) + "\n" + "\n".join(caught.value.__notes__)
    assert "cancellation was not verified" in notes and "owned-parent" in notes
    for role in ("training", "sampling", "log_prob"):
        assert f"owned-parent:{role}:0" in notes
    assert "HTTP session closure also failed: RuntimeError: owned session close failed" in notes


def test_local_cortex_shutdown_close_failure_after_cancellation_still_raises(
    _local_cortex_shutdown_gateway, monkeypatch
):
    state, _, create = _local_cortex_shutdown_gateway
    transport, _ = create()
    close = transport._close_session
    failure = RuntimeError("owned session close failed")

    async def failing_close():
        await close()
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(transport, "_close_session", failing_close)
        with pytest.raises(RuntimeError) as caught:
            transport.shutdown()

    assert caught.value is failure
    assert not getattr(failure, "__notes__", [])
    assert len(state.calls) == 2 and state.closed
    assert transport.job_id is None and transport._local_shutdown_parent is None


def test_local_cortex_shutdown_interruption_preserves_ownership_when_session_close_fails(
    _local_cortex_shutdown_gateway, monkeypatch
):
    state, _, create = _local_cortex_shutdown_gateway
    transport, _ = create()
    primary = KeyboardInterrupt("owned cancellation interrupted")
    close = transport._close_session

    def interrupted_cancel(parent):
        assert parent == "owned-parent"
        raise primary

    async def failing_close():
        await close()
        raise RuntimeError("owned session close failed")

    with monkeypatch.context() as patch:
        patch.setattr(transport, "_wait_local_cancelled", interrupted_cancel)
        patch.setattr(transport, "_close_session", failing_close)
        with pytest.raises(KeyboardInterrupt) as caught:
            transport.shutdown()

    assert caught.value is primary
    assert state.calls == [] and state.closed
    notes = "\n".join(primary.__notes__)
    assert "cancellation was not verified" in notes and "owned-parent:training:0" in notes
    assert "HTTP session closure also failed" in notes


def test_local_cortex_shutdown_late_cancelled_response_does_not_override_deadline(
    _local_cortex_shutdown_gateway,
):
    state, job, create = _local_cortex_shutdown_gateway
    transport, original = create()

    def late():
        state.elapsed = original.poll_timeout
        return job()

    state.polls = [late]
    with pytest.raises(UnsafeCleanupError, match="cancellation deadline") as caught:
        transport.shutdown()
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert state.closed and transport.config is original


def test_local_cortex_shutdown_changed_parent_identity_never_cancels_another_parent(
    _local_cortex_shutdown_gateway,
):
    state, _, create = _local_cortex_shutdown_gateway
    transport, _ = create()
    transport.job_id = "foreign-parent"
    with pytest.raises(UnsafeCleanupError, match="inconsistent owned parent"):
        transport.shutdown()
    assert state.calls == [] and state.closed


@pytest.mark.parametrize(
    "base_url,reconnect",
    [
        ("http://remote.example", False),
        ("https://127.0.0.1:8003", False),
        ("http://127.0.0.1:8003", True),
    ],
)
def test_local_cortex_shutdown_hosted_and_reconnected_shutdown_keep_ack_only_behavior(
    _local_cortex_shutdown_gateway, base_url, reconnect
):
    state, _, create = _local_cortex_shutdown_gateway
    transport, original = create(base_url=base_url, reconnect=reconnect)
    transport.shutdown()
    assert [method for method, _, _ in state.calls] == ["POST"]
    assert state.calls[0][2]["timeout"].total == original.request_timeout
    assert state.slept == [] and state.closed


def test_local_cortex_shutdown_startup_failure_still_cancels_owned_parent_with_child_fence(
    _local_cortex_shutdown_gateway, monkeypatch
):
    state, job, create = _local_cortex_shutdown_gateway
    transport, _ = create()
    state.cancelled = False
    state.closed = False
    state.polls = [job()]
    monkeypatch.setattr(
        transport,
        "_wait_running",
        lambda: (_ for _ in ()).throw(RuntimeError("actual startup failure")),
    )
    config = CortexTrainingConfig(model_name="Qwen/Qwen3-8B", training_gpus=4, backend=transport.config)
    with pytest.raises(RuntimeError, match="actual startup failure"):
        initialize_or_cleanup(transport, config)
    assert [method for method, _, _ in state.calls] == ["POST", "POST", "GET"]
    assert state.closed and transport.job_id is None


@pytest.mark.parametrize("workload_fails", [False, True])
def test_correctness_client_unsafe_shutdown_preserves_failures_and_owned_ids(
    _local_cortex_shutdown_gateway, workload_fails
):
    state, _, create = _local_cortex_shutdown_gateway
    transport, _ = create()
    client = object.__new__(CorrectnessClient)
    client.transport = transport
    cancellation = aiohttp.ClientConnectionError("owned cancellation failed")
    workload = RuntimeError("checkpoint create failed: rc=0, receipts={}")
    state.error = cancellation
    with pytest.raises(UnsafeCleanupError) as caught:
        with client:
            if workload_fails:
                raise workload
    error = caught.value
    assert error.__cause__ is error.shutdown_error is cancellation
    assert error.workload_error is (workload if workload_fails else None)
    assert error.owned_parent == "owned-parent"
    assert error.owned_children == {"training": "owned-parent:training:0"}
    rendered = "".join(traceback.format_exception(error))
    assert "owned cancellation failed" in rendered and "owned-parent:training:0" in str(error)
    if workload_fails:
        assert "checkpoint create failed: rc=0, receipts={}" in str(error)
        assert "Original workload traceback" in rendered
    assert len(state.calls) == 1 and state.closed


@pytest.mark.parametrize("workload_fails", [False, True])
def test_correctness_client_verified_shutdown_preserves_workload_behavior(
    _local_cortex_shutdown_gateway, workload_fails
):
    state, _, create = _local_cortex_shutdown_gateway
    transport, _ = create()
    client = object.__new__(CorrectnessClient)
    client.transport = transport
    workload = RuntimeError("original workload failure")

    def run():
        with client:
            if workload_fails:
                raise workload

    if workload_fails:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is workload
        assert not getattr(workload, "__notes__", [])
    else:
        run()
    assert [method for method, _, _ in state.calls] == ["POST", "GET"] and state.closed


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_correctness_client_workload_interrupt_remains_primary_after_unsafe_shutdown(
    _local_cortex_shutdown_gateway, exception
):
    state, _, create = _local_cortex_shutdown_gateway
    transport, _ = create()
    client = object.__new__(CorrectnessClient)
    client.transport = transport
    state.error = aiohttp.ClientConnectionError("owned cancellation failed")
    interruption = exception("original workload interrupted")
    with pytest.raises(exception) as caught:
        with client:
            raise interruption
    assert caught.value is interruption
    unsafe = interruption.__cause__
    assert isinstance(unsafe, UnsafeCleanupError) and unsafe.__cause__ is state.error
    assert "owned-parent:training:0" in "\n".join(interruption.__notes__)
    assert state.closed


def test_correctness_client_shutdown_interrupt_retains_original_workload(_local_cortex_shutdown_gateway, monkeypatch):
    state, _, create = _local_cortex_shutdown_gateway
    transport, _ = create()
    client = object.__new__(CorrectnessClient)
    client.transport = transport
    interruption = KeyboardInterrupt("shutdown interrupted")
    workload = RuntimeError("original checkpoint failure")

    def cancel(parent):
        raise interruption

    monkeypatch.setattr(transport, "_wait_local_cancelled", cancel)
    with pytest.raises(KeyboardInterrupt) as caught:
        with client:
            raise workload
    assert caught.value is interruption
    rendered = "".join(traceback.format_exception(interruption))
    assert "original checkpoint failure" in rendered and "owned-parent:training:0" in rendered
    assert state.closed


@pytest.mark.parametrize("cancel_fails", [False, True])
def test_correctness_client_startup_failure_does_not_suppress_unsafe_shutdown(
    _local_cortex_shutdown_gateway, monkeypatch, cancel_fails
):
    state, _, create = _local_cortex_shutdown_gateway
    transport, _ = create()
    startup = RuntimeError("original startup failure")

    def initialize(config):
        raise startup

    monkeypatch.setattr(transport, "initialize", initialize)
    monkeypatch.setattr(cortex_transport, "CorrectnessCortexTransport", lambda backend: transport)
    config = CortexTrainingConfig(model_name="Qwen/Qwen3-8B", training_gpus=4, backend=transport.config)
    if cancel_fails:
        state.error = aiohttp.ClientConnectionError("owned cancellation failed")
        with pytest.raises(UnsafeCleanupError) as caught:
            CorrectnessClient(config)
        assert caught.value.workload_error is startup and caught.value.shutdown_error is state.error
        assert "original startup failure" in str(caught.value)
    else:
        with pytest.raises(RuntimeError) as caught:
            CorrectnessClient(config)
        assert caught.value is startup
    assert state.closed


@pytest.mark.parametrize("base_url,reconnect", [("http://remote.example", False), ("http://127.0.0.1:8003", True)])
def test_correctness_client_hosted_or_reconnected_shutdown_preserves_existing_behavior(
    _local_cortex_shutdown_gateway, base_url, reconnect
):
    state, _, create = _local_cortex_shutdown_gateway
    transport, original = create(base_url=base_url, reconnect=reconnect)
    client = object.__new__(CorrectnessClient)
    client.transport = transport
    state.error = aiohttp.ClientConnectionError("ACK-only cancellation failed")
    workload = RuntimeError("original hosted workload failure")
    with pytest.raises(RuntimeError) as caught:
        with client:
            raise workload
    assert caught.value is workload and not getattr(workload, "__notes__", [])
    assert len(state.calls) == (0 if reconnect else original.max_retries + 1) and state.closed


def test_correctness_cleanup_contract_reexport_is_the_same_class():
    from arctic_platform.correctness.harness.dss_driver import UnsafeCleanupError as DriverError

    assert DriverError is cortex_transport.UnsafeCleanupError is UnsafeCleanupError
