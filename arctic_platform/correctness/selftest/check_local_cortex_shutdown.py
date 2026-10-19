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

"""Fence owned local child cancellation through the existing public SnowAPI."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import aiohttp
import pytest

from arctic_platform.client import CortexConfig
from arctic_platform.client.transport import initialize_or_cleanup
from arctic_platform.correctness.harness import cortex_transport
from arctic_platform.correctness.harness.cortex import CortexTrainingConfig
from arctic_platform.correctness.harness.cortex_transport import LOCAL_NATIVE_FORWARD_ENV
from arctic_platform.correctness.harness.cortex_transport import CorrectnessCortexTransport


@pytest.fixture
def gateway(monkeypatch):
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

    def create(*, base_url="http://127.0.0.1:8003", native=False, reconnect=False, roles=("training",)):
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
        return transport, original_config

    yield state, job, create
    for transport in transports:
        transport._on_loop(transport._close_session()).result()
        transport._loop.call_soon_threadsafe(transport._loop.stop)


@pytest.mark.parametrize("native", [False, True])
def test_cancel_ack_waits_for_every_owned_child_before_closing(gateway, native):
    state, job, create = gateway
    transport, original = create(native=native, roles=("training", "sampling", "log_probability"))
    mixed = job(child_status="cancelled")
    mixed["sub_jobs"][1]["status"] = "pending"
    state.polls = [job(child_status="running"), mixed, job()]
    transport.shutdown()
    assert [method for method, _, _ in state.calls] == ["POST", "GET", "GET", "GET"]
    assert all(url.startswith(f"{transport._prefix}/owned-parent") for _, url, _ in state.calls)
    assert state.calls[0][1].endswith(":cancel")
    assert all(kwargs["allow_redirects"] is False for _, _, kwargs in state.calls)
    assert state.slept == [0.4, 0.4]
    assert state.closed and transport.job_id is None
    assert transport._native_bindings == transport._native_children == {}
    assert transport._local_shutdown_parent is None and transport._local_shutdown_children == {}
    assert transport.config is original
    assert len(state.closes) == 1
    session, loop = state.closes[0]
    assert loop is transport._loop and session.closed and transport._session is None
    assert all(actual_session is session and actual_loop is loop for actual_session, actual_loop in state.exchanges)
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
def test_invalid_or_failed_owned_cancellation_fails_closed(gateway, broken):
    state, job, create = gateway
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
    with pytest.raises(RuntimeError, match="local gateway"):
        transport.shutdown()
    assert state.closed and transport.job_id is None and transport.config is original
    assert state.slept == []


def test_pending_child_reaches_configured_monotonic_deadline(gateway):
    state, job, create = gateway
    transport, original = create()
    state.polls = [job(child_status="pending")] * 3
    with pytest.raises(TimeoutError, match="cancellation deadline"):
        transport.shutdown()
    assert state.elapsed == original.poll_timeout
    assert len(state.calls) == 4
    assert state.closed and transport.config is original


@pytest.mark.parametrize("field", ["poll_timeout", "request_timeout", "poll_interval"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_nonfinite_or_nonpositive_timing_is_rejected_without_cancel(gateway, field, value):
    state, _, create = gateway
    transport, _ = create()
    if field == "poll_interval":
        transport.poll_interval = value
    else:
        transport.config = transport.config.model_copy(update={field: value})
    with pytest.raises(ValueError, match="finite and positive"):
        transport.shutdown()
    assert state.calls == [] and state.closed


def test_cancel_network_failure_is_not_suppressed_or_retried(gateway):
    state, _, create = gateway
    transport, original = create()
    state.error = aiohttp.ClientConnectionError("actual cancellation request failed")
    with pytest.raises(aiohttp.ClientConnectionError, match="actual cancellation request failed"):
        transport.shutdown()
    assert len(state.calls) == 1 and state.closed
    assert transport.config is original


def test_poll_network_failure_restores_config_and_closes(gateway):
    state, _, create = gateway
    transport, original = create()

    def failure():
        raise aiohttp.ClientConnectionError("actual cancellation poll failed")

    state.polls = [failure]
    with pytest.raises(aiohttp.ClientConnectionError, match="actual cancellation poll failed"):
        transport.shutdown()
    assert len(state.calls) == 2 and state.closed
    assert transport.config is original


def test_late_cancelled_response_does_not_override_deadline(gateway):
    state, job, create = gateway
    transport, original = create()

    def late():
        state.elapsed = original.poll_timeout
        return job()

    state.polls = [late]
    with pytest.raises(TimeoutError, match="cancellation deadline"):
        transport.shutdown()
    assert state.closed and transport.config is original


def test_changed_parent_identity_never_cancels_another_parent(gateway):
    state, _, create = gateway
    transport, _ = create()
    transport.job_id = "foreign-parent"
    with pytest.raises(RuntimeError, match="inconsistent owned parent"):
        transport.shutdown()
    assert state.calls == [] and state.closed


@pytest.mark.parametrize(
    "base_url,reconnect",
    [("http://remote.example", False), ("https://127.0.0.1:8003", False), ("http://127.0.0.1:8003", True)],
)
def test_hosted_and_reconnected_shutdown_keep_ack_only_behavior(gateway, base_url, reconnect):
    state, _, create = gateway
    transport, original = create(base_url=base_url, reconnect=reconnect)
    transport.shutdown()
    assert [method for method, _, _ in state.calls] == ["POST"]
    assert state.calls[0][2]["timeout"].total == original.request_timeout
    assert state.slept == [] and state.closed


def test_startup_failure_still_cancels_owned_parent_with_child_fence(gateway, monkeypatch):
    state, job, create = gateway
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
