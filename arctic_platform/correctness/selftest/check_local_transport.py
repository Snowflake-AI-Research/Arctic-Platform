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
from copy import deepcopy
from types import SimpleNamespace

import aiohttp
import pytest

from arctic_platform.client import CortexConfig
from arctic_platform.client.requests import Request
from arctic_platform.client.transport import initialize_or_cleanup
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.correctness.harness import cortex as cortex_transport
from arctic_platform.correctness.harness.cortex import LOCAL_NATIVE_FORWARD_ENV
from arctic_platform.correctness.harness.cortex import CorrectnessCortexTransport
from arctic_platform.correctness.harness.cortex import CortexTrainingConfig


@pytest.mark.parametrize("asynchronous", [False, True])
def test_local_cortex_save_automatically_uses_explicit_path_and_owned_native_child(
    _local_cortex_shutdown_gateway, monkeypatch, asynchronous
):
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV, raising=False)
    state, _, create = _local_cortex_shutdown_gateway
    transport, _ = create()
    assert transport._native_gateway is False
    token = transport.jobs.training
    child = transport._native_bindings[str(token)][1]
    root = "/data-fast/tmp/correctness-owned-save-control"
    expected = {"job_id": child, "checkpoint_path": root, "checkpoint_tag": "global_step0", "version": 0}
    captured = []

    async def native(method, url, **kwargs):
        assert asyncio.get_running_loop() is transport._loop
        captured.append((method, url, kwargs))
        return json.dumps(expected).encode()

    monkeypatch.setattr(transport, "_native_http", native)
    request = Request("save", token, {"path": root, "checkpoint_type": "resumable", "export_hf": False})
    result = asyncio.run(transport.acall(request)) if asynchronous else transport.call(request)
    assert result == expected
    assert captured == [
        (
            "POST",
            f"http://127.0.0.1:8003/save-checkpoint?job_id={child}",
            {"json": {"path": root, "checkpoint_type": "resumable"}},
        )
    ]
    assert state.calls == []
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
    _local_cortex_shutdown_gateway, monkeypatch, change
):
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV, raising=False)
    _, _, create = _local_cortex_shutdown_gateway
    transport, _ = create()

    async def unexpected(*args, **kwargs):
        pytest.fail("invalid save reached HTTP")

    monkeypatch.setattr(transport, "_native_http", unexpected)
    body = {"path": "/data-fast/tmp/correctness-owned-save-control", "checkpoint_type": "resumable", **change}
    with pytest.raises(ValueError):
        transport.call(Request("save", transport.jobs.training, body))
    transport.shutdown()


@pytest.mark.parametrize("token", ["foreign:training:0", "owned-parent:sampling:0"])
def test_local_cortex_save_never_reroutes_foreign_or_wrong_role(_local_cortex_shutdown_gateway, monkeypatch, token):
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV, raising=False)
    _, _, create = _local_cortex_shutdown_gateway
    transport, _ = create()
    transport._native_bindings["owned-parent:sampling:0"] = ("sampling", "sampling-child")

    async def unexpected(*args, **kwargs):
        pytest.fail("nonowned or wrong-role save reached HTTP")

    monkeypatch.setattr(transport, "_native_http", unexpected)
    body = {"path": "/data-fast/tmp/correctness-owned-save-control", "checkpoint_type": "resumable"}
    with pytest.raises(ValueError, match="owned child|cannot target"):
        transport.call(Request("save", token, body))
    transport.shutdown()
    with pytest.raises(ValueError, match="owned child"):
        transport.call(Request("save", "owned-parent:training:0", body))


@pytest.mark.parametrize("payload", [b"invalid-json", b"[]", b'{"job_id": "foreign-child"}'])
def test_local_cortex_save_rejects_malformed_or_wrong_child_response(
    _local_cortex_shutdown_gateway, monkeypatch, payload
):
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV, raising=False)
    _, _, create = _local_cortex_shutdown_gateway
    transport, _ = create()

    async def native(*args, **kwargs):
        return payload

    monkeypatch.setattr(transport, "_native_http", native)
    body = {"path": "/data-fast/tmp/correctness-owned-save-control", "checkpoint_type": "resumable"}
    with pytest.raises(ValueError):
        transport.call(Request("save", transport.jobs.training, body))
    transport.shutdown()


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:8003", "http://remote.example"])
def test_local_cortex_save_managed_and_hosted_saves_preserve_public_route(
    _local_cortex_shutdown_gateway, monkeypatch, base_url
):
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV, raising=False)
    _, _, create = _local_cortex_shutdown_gateway
    transport, _ = create(base_url=base_url)
    body = {"path": None if base_url.startswith("http://127.") else "/data-fast/tmp/control"}
    request = Request("save", transport.jobs.training, body)
    monkeypatch.setattr(CortexTransport, "call", lambda actual, supplied: {"public": supplied is request})
    assert transport.call(request) == {"public": True}
    transport.shutdown()


@pytest.fixture
def _local_cortex_shutdown_gateway(monkeypatch):
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

    monkeypatch.setattr(cortex_transport, "time", SimpleNamespace(monotonic=lambda: state.elapsed, sleep=sleep))

    def job(status="cancelled", child_status="cancelled"):
        segments = {"training": "training", "sampling": "sampling", "log_probability": "log_prob"}
        return {
            "job_id": "owned-parent",
            "status": status,
            "sub_jobs": [
                {"job_type": role, "sub_job_id": f"owned-parent:{segments[role]}:0", "status": child_status}
                for role in state.roles
            ],
        }

    def create(*, base_url="http://127.0.0.1:8003", reconnect=False, roles=("training",)):
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


def test_local_cortex_shutdown_cancel_ack_waits_for_every_owned_child_before_closing(
    _local_cortex_shutdown_gateway,
):
    state, job, create = _local_cortex_shutdown_gateway
    transport, original = create()
    mixed = job(child_status="cancelled")
    mixed["sub_jobs"][0]["status"] = "pending"
    state.polls = [job(child_status="running"), mixed, job()]
    transport.shutdown()
    assert [method for method, _, _ in state.calls] == ["POST", "GET", "GET", "GET"]
    assert all((url.startswith(f"{transport._prefix}/owned-parent") for _, url, _ in state.calls))
    assert state.calls[0][1].endswith(":cancel")
    assert all((kwargs["allow_redirects"] is False for _, _, kwargs in state.calls))
    assert state.slept == [0.4, 0.4]
    assert state.closed and transport.job_id is None
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
    transport, original = create()
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
        children.append(deepcopy(children[0]))
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
    with pytest.raises(RuntimeError, match="local gateway"):
        transport.shutdown()
    assert state.closed and transport.job_id is None and (transport.config is original)
    assert state.slept == []


def test_local_cortex_shutdown_pending_child_reaches_configured_monotonic_deadline(_local_cortex_shutdown_gateway):
    state, job, create = _local_cortex_shutdown_gateway
    transport, original = create()
    state.polls = [job(child_status="pending")] * 3
    with pytest.raises(TimeoutError, match="cancellation deadline"):
        transport.shutdown()
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
    with pytest.raises(ValueError, match="finite and positive"):
        transport.shutdown()
    assert state.calls == [] and state.closed


def test_local_cortex_shutdown_cancel_network_failure_is_not_suppressed_or_retried(_local_cortex_shutdown_gateway):
    state, _, create = _local_cortex_shutdown_gateway
    transport, original = create()
    state.error = aiohttp.ClientConnectionError("actual cancellation request failed")
    with pytest.raises(aiohttp.ClientConnectionError, match="actual cancellation request failed"):
        transport.shutdown()
    assert len(state.calls) == 1 and state.closed
    assert transport.config is original


def test_local_cortex_shutdown_poll_network_failure_restores_config_and_closes(_local_cortex_shutdown_gateway):
    state, _, create = _local_cortex_shutdown_gateway
    transport, original = create()

    def failure():
        raise aiohttp.ClientConnectionError("actual cancellation poll failed")

    state.polls = [failure]
    with pytest.raises(aiohttp.ClientConnectionError, match="actual cancellation poll failed"):
        transport.shutdown()
    assert len(state.calls) == 2 and state.closed
    assert transport.config is original


def test_local_cortex_shutdown_late_cancelled_response_does_not_override_deadline(_local_cortex_shutdown_gateway):
    state, job, create = _local_cortex_shutdown_gateway
    transport, original = create()

    def late():
        state.elapsed = original.poll_timeout
        return job()

    state.polls = [late]
    with pytest.raises(TimeoutError, match="cancellation deadline"):
        transport.shutdown()
    assert state.closed and transport.config is original


def test_local_cortex_shutdown_changed_parent_identity_never_cancels_another_parent(_local_cortex_shutdown_gateway):
    state, _, create = _local_cortex_shutdown_gateway
    transport, _ = create()
    transport.job_id = "foreign-parent"
    with pytest.raises(RuntimeError, match="inconsistent owned parent"):
        transport.shutdown()
    assert state.calls == [] and state.closed


@pytest.mark.parametrize(
    "base_url,reconnect",
    [("http://remote.example", False), ("https://127.0.0.1:8003", False), ("http://127.0.0.1:8003", True)],
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
        transport, "_wait_running", lambda: (_ for _ in ()).throw(RuntimeError("actual startup failure"))
    )
    config = CortexTrainingConfig(model_name="Qwen/Qwen3-8B", training_gpus=4, backend=transport.config)
    with pytest.raises(RuntimeError, match="actual startup failure"):
        initialize_or_cleanup(transport, config)
    assert [method for method, _, _ in state.calls] == ["POST", "POST", "GET"]
    assert state.closed and transport.job_id is None


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
        with pytest.raises(aiohttp.ClientConnectionError) as caught:
            transport.shutdown()

    assert caught.value is primary
    assert len(state.calls) == 1 and state.closed
    assert transport.config is original and transport.job_id is None
    assert transport._native_bindings == transport._native_children == {}
    assert transport._local_shutdown_parent is None and transport._local_shutdown_children == {}
    notes = "\n".join(primary.__notes__)
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
