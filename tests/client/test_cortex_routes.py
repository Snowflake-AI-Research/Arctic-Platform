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
"""Cortex `_ROUTES`: the HTTP each op sends, its retry policy, and its normalized reply.

`_send` / `_asend` are stubbed, so these tests pin what the transport sends and
how it reads replies without a live GS.
"""

from __future__ import annotations

import asyncio

import pytest
import requests
import torch

from arctic_platform import wire
from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import CortexConfig
from arctic_platform.client import JobHandles
from arctic_platform.client import Request
from arctic_platform.client import requests as client_requests
from arctic_platform.client.transports import cortex as cortex_module
from arctic_platform.client.transports.cortex import CortexTransport

PREFIX = "http://gs.test/api/v2/databases/db/schemas/schema/cortex-training"
JOB = "job-1"
TRAINING = f"{JOB}:training:0"
SAMPLING = f"{JOB}:sampling:0"
RESTART = {"code": "chunk_group_restart_required"}


class FakeSend:
    """Records every send and answers from `replies`, keyed by (method, url suffix)."""

    def __init__(self, replies: dict | None = None) -> None:
        self.sent: list[dict] = []
        self.replies = replies or {}

    def __call__(self, method: str, url: str, *, retry_on=None, **kwargs):
        self.sent.append({"method": method, "url": url, "retry_on": retry_on, **kwargs})
        for (reply_method, suffix), reply in self.replies.items():
            if method == reply_method and url.endswith(suffix):
                return reply(len(self.sent)) if callable(reply) else reply
        if method == "GET":
            return {"status": "REQUEST_STATE_DONE", "result": {}}
        return {"request_id": "req-1"}


def _transport(replies: dict | None = None) -> tuple[CortexTransport, FakeSend]:
    transport = CortexTransport(CortexConfig(base_url="http://gs.test", database="db", schema="schema"))
    transport.job_id = JOB
    transport.jobs = JobHandles(training=TRAINING, sampling=SAMPLING)
    send = FakeSend(replies)
    transport._send = send  # type: ignore[method-assign]
    return transport, send


class TestLifecycleRoutes:
    def test_initialize_creates_without_replay_and_captures_sub_jobs(self):
        """Create retries only undelivered sends, so a retry cannot spawn a second job."""
        job = {
            "job_id": JOB,
            "status": "JOB_STATE_RUNNING",
            "sub_jobs": [{"job_type": "JOB_TYPE_TRAINING", "sub_job_id": TRAINING}],
        }
        transport, send = _transport({("POST", "cortex-training"): {"job_id": JOB}, ("GET", f"/{JOB}"): job})
        transport.job_id = None
        transport.jobs = JobHandles()

        jobs = transport.initialize(ArcticClientConfig(model_name="m", backend=transport.config, training_gpus=1))

        create = send.sent[0]
        assert (create["method"], create["url"], create["retry_on"]) == (
            "POST",
            PREFIX,
            cortex_module._is_connect_error,
        )
        assert "sub_job_configs" in create["json"]
        assert jobs == JobHandles(training=TRAINING)

    def test_shutdown_cancel_is_replay_safe(self):
        """Cancel is idempotent, so teardown retries transient errors instead of leaking the job."""
        transport, send = _transport()
        transport.shutdown()

        (sent,) = send.sent
        assert (sent["url"], sent["retry_on"]) == (f"{PREFIX}/{JOB}:cancel", cortex_module._is_transient)

    def test_shutdown_tolerates_a_failed_cancel(self):
        transport, _ = _transport()

        def fail(*args, **kwargs):
            raise requests.exceptions.ConnectionError("gone")

        transport._send = fail  # type: ignore[method-assign]
        transport.shutdown()


def _fwd_bwd_request() -> Request:
    batch = {"batch": {"input_ids": torch.arange(4).reshape(1, -1)}, "meta": {}, "processing": {}}
    return client_requests.fwd_bwd_request(JobHandles(training=TRAINING), batch)


def _fail_first_post(error: dict):
    def reply(n_sent: int):
        if n_sent == 1:
            raise cortex_module._ChunkGroupError(error)
        return {"request_id": "req-1"}

    return reply


class TestChunkGroupRestart:
    def test_fwd_bwd_reposts_the_whole_group_once(self):
        transport, send = _transport({("POST", "/forward-backward"): _fail_first_post(RESTART)})
        transport.call(_fwd_bwd_request())

        posts = [sent for sent in send.sent if sent["method"] == "POST"]
        assert len(posts) == 2
        assert posts[0]["data"] == posts[1]["data"]

    def test_fwd_bwd_gives_up_after_one_restart(self):
        def always(n_sent: int):
            raise cortex_module._ChunkGroupError(RESTART)

        transport, _ = _transport({("POST", "/forward-backward"): always})
        with pytest.raises(cortex_module._ChunkGroupError, match="restart_required"):
            transport.call(_fwd_bwd_request())

    def test_other_chunk_group_errors_are_not_restarted(self):
        transport, _ = _transport({("POST", "/forward-backward"): _fail_first_post({"code": "chunk_group_conflict"})})
        with pytest.raises(cortex_module._ChunkGroupError, match="conflict"):
            transport.call(_fwd_bwd_request())

    def test_generate_is_not_restartable(self):
        transport, _ = _transport({("POST", "/generate"): _fail_first_post(RESTART)})
        with pytest.raises(cortex_module._ChunkGroupError):
            transport.call(client_requests.generate_request(JobHandles(sampling=SAMPLING), ["hi"]))

    def test_async_fwd_bwd_reposts_the_whole_group_once(self):
        transport, send = _transport({("POST", "/forward-backward"): _fail_first_post(RESTART)})

        async def asend(method, url, *, retry_on=None, **kwargs):
            return send(method, url, retry_on=retry_on, **kwargs)

        transport._asend = asend  # type: ignore[method-assign]
        asyncio.run(transport.acall(_fwd_bwd_request()))

        posts = [sent for sent in send.sent if sent["method"] == "POST"]
        assert len(posts) == 2
        assert posts[1]["retry_on"] is cortex_module._is_transient_async

    def test_error_body_raises_chunk_group_error_that_is_never_retried(self):
        """GS wraps the chunk-group error as JSON inside a string field."""
        resp = requests.Response()
        resp.status_code = 409
        resp._content = b'{"message": "{\\"code\\": \\"chunk_group_restart_required\\"}"}'

        with pytest.raises(cortex_module._ChunkGroupError) as exc_info:
            cortex_module._raise_for_status(resp)
        assert exc_info.value.restart_required
        assert not cortex_module._is_transient(exc_info.value)


class TestOctetEncoding:
    def test_small_fwd_bwd_frame_is_still_chunk_wrapped(self):
        """A small fwd-bwd frame still travels in a chunk envelope (its idempotency key)."""
        transport, send = _transport()
        transport.call(_fwd_bwd_request())

        (post,) = [sent for sent in send.sent if sent["method"] == "POST"]
        assert wire.read_byte_chunk_metadata(post["data"])["operation"] == "fwd-bwd"
