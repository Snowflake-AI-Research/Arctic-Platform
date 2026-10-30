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
"""OnPremTransport's private loop must shut down cleanly and must not deadlock itself."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import CancelledError

import pytest

from arctic_platform.client.transport import Request
from arctic_platform.client.transports.onprem import OnPremTransport


def _transport() -> OnPremTransport:
    # Skip Ray startup. The loop thread and shutdown path under test are the real ones.
    transport = object.__new__(OnPremTransport)
    transport._reconnect = True
    transport._loop = asyncio.new_event_loop()
    transport._loop_thread = OnPremTransport._start_loop_thread(transport._loop)
    return transport


def test_run_on_loop_returns_from_another_thread():
    transport = _transport()
    try:

        async def _value():
            return 7

        assert transport._run_on_loop(_value()) == 7
    finally:
        transport.shutdown()


def test_run_on_loop_from_the_loop_thread_raises():
    transport = _transport()
    box: dict = {}
    done = threading.Event()

    def _from_loop():
        async def _value():
            return 1

        try:
            transport._run_on_loop(_value())
        except Exception as exc:
            box["exc"] = exc
        finally:
            done.set()

    try:
        transport._loop.call_soon_threadsafe(_from_loop)
        assert done.wait(5)
        assert isinstance(box["exc"], RuntimeError)
        # The loop is still usable from a caller thread after the rejected call.

        async def _value():
            return 7

        assert transport._run_on_loop(_value()) == 7
    finally:
        transport.shutdown()


def test_acall_runs_on_the_private_loop_not_the_caller_loop():
    transport = _transport()

    class _Server:
        def __init__(self):
            self.loop = None
            self.thread = None

        async def wake_inference(self, job_id, body):
            self.loop = asyncio.get_running_loop()
            self.thread = threading.current_thread()
            return {"job": job_id, "body": body}

    server = _Server()
    transport._server = server
    caller_loop = None

    async def _go():
        nonlocal caller_loop
        caller_loop = asyncio.get_running_loop()
        return await transport.acall(Request("wake-inference", "job-1", {"n": 1}))

    try:
        assert asyncio.run(_go()) == {"job": "job-1", "body": {"n": 1}}
        assert caller_loop is not transport._loop
        assert server.loop is transport._loop
        assert server.thread is transport._loop_thread
    finally:
        transport.shutdown()


def test_call_and_acall_see_the_same_loop():
    transport = _transport()
    seen = []

    class _Server:
        async def wake_inference(self, job_id, body):
            del job_id, body
            seen.append(asyncio.get_running_loop())
            return {}

    transport._server = _Server()
    try:
        transport.call(Request("wake-inference", "job-1", {}))
        asyncio.run(transport.acall(Request("wake-inference", "job-1", {})))
        assert seen == [transport._loop, transport._loop]
    finally:
        transport.shutdown()


def test_cancel_pending_propagates_errors_other_than_timeout():
    transport = _transport()

    class _Future:
        def result(self, timeout=None):
            del timeout
            raise ValueError("cancel failed")

    def _submit(coro):
        coro.close()
        return _Future()

    transport._submit = _submit
    try:
        with pytest.raises(ValueError, match="cancel failed"):
            transport._cancel_pending()
    finally:
        del transport._submit
        transport.shutdown()


def test_shutdown_cancels_an_in_flight_op_and_closes_the_loop():
    transport = _transport()
    started = threading.Event()

    async def _long():
        started.set()
        await asyncio.sleep(3600)

    fut = asyncio.run_coroutine_threadsafe(_long(), transport._loop)
    assert started.wait(5)
    transport.shutdown()
    assert not transport._loop_thread.is_alive()
    assert transport._loop.is_closed()
    with pytest.raises(CancelledError):
        fut.result(timeout=5)
    transport.shutdown()
