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
"""`forward` (no-grad log-probs) over the Cortex transport.

forward lowers like forward-backward -- a DSSST1 octet body -- with one
difference that matters: forward-backward is always training, while forward runs
on the training sub-job for current-policy log-probs and on the log_prob sub-job
for reference log-probs. Direct octet endpoints carry no routing envelope, so
the sub-job token has to ride on the URL.

`_send` is stubbed, so these tests pin the request the transport builds and the
result it decodes without a live GS.
"""

from __future__ import annotations

import base64
from urllib.parse import unquote

import pytest
import torch

from arctic_platform import wire
from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import CortexConfig
from arctic_platform.client import JobHandles
from arctic_platform.client.requests import fwd_bwd_request
from arctic_platform.client.requests import fwd_no_grad_request
from arctic_platform.client.transport import Request
from arctic_platform.client.transports import cortex as cortex_module
from arctic_platform.client.transports.cortex import CortexTransport

JOB = "job-1"
TRAINING = f"{JOB}:training:0"
LOG_PROB = f"{JOB}:log_probability:0"
BATCH_IDS = [[1, 2, 3]]


class FakeSend:
    """Stands in for `CortexTransport._send`: records POSTs, answers GETs as done."""

    def __init__(self, result: dict) -> None:
        self.posts: list[dict] = []
        self.result = result

    def __call__(self, method: str, url: str, *, retry_on=None, **kwargs):
        if method == "GET":
            return {"status": "REQUEST_STATE_DONE", "result": self.result}
        self.posts.append({"url": url, **kwargs})
        return {"request_id": "req-1"}

    @property
    def urls(self) -> list[str]:
        """POSTed urls, percent-decoding so sub-job tokens read as `job:type:0`."""
        return [unquote(post["url"]) for post in self.posts]

    @property
    def frames(self) -> list[bytes]:
        return [post["data"] for post in self.posts]


def _transport(result: dict | None = None) -> tuple[CortexTransport, FakeSend]:
    config = ArcticClientConfig(
        model_name="m",
        backend=CortexConfig(base_url="http://gs.test"),
        training_gpus=1,
        log_prob_gpus=1,
    )
    transport = CortexTransport(config)
    transport.job_id = JOB
    transport.jobs = JobHandles(training=TRAINING, log_prob=LOG_PROB)
    send = FakeSend(result or {})
    transport._send = send  # type: ignore[method-assign]
    return transport, send


def _batch() -> dict:
    return {"batch": {"input_ids": torch.tensor(BATCH_IDS)}, "meta": {}, "processing": {}}


def _dssst1_result(payload: dict) -> dict:
    """An inline (unchunked) DSSST1 result, as `_decode_result` expects it."""
    return {
        "wire_format": wire.WIRE_FORMAT_VERSION,
        "payload_b64": base64.b64encode(wire.dumps(payload)).decode(),
    }


class TestForwardWire:
    def test_posts_dssst1_octet_body(self):
        """forward goes out as DSSST1 octet, not a base64 JSON /operation envelope."""
        transport, send = _transport()
        transport.call(fwd_no_grad_request(transport.jobs, _batch()))

        (post,) = send.posts
        assert post["headers"]["Content-Type"] == "application/octet-stream"
        assert wire.loads(post["data"])["batch"]["input_ids"].tolist() == BATCH_IDS

    def test_hits_the_forward_endpoint(self):
        transport, send = _transport()
        transport.call(fwd_no_grad_request(transport.jobs, _batch()))

        assert send.urls[0].split("?")[0].endswith(f"/{JOB}/forward")

    def test_frame_under_the_cap_posts_bare(self):
        """forward has no idempotency key to carry, so chunking stays size-driven.

        forward-backward force-wraps every frame because its chunk_group_id is
        the server's at-most-once key. A no-grad forward mutates nothing, so a
        small batch posts as raw DSSST1 -- what the backend /forward reads.
        """
        transport, send = _transport()
        transport.call(fwd_no_grad_request(transport.jobs, _batch()))

        assert wire.read_byte_chunk_metadata(send.frames[0]) is None

    def test_oversized_frame_chunks_under_the_fwd_label(self, monkeypatch):
        """Log-prob batches can exceed the 60 MiB cap, so the split must round-trip."""
        monkeypatch.setattr(cortex_module, "_MAX_OCTET_BYTES", 64 * 1024)
        transport, send = _transport()
        batch = {"batch": {"input_ids": torch.arange(20_000).reshape(1, -1)}, "meta": {}, "processing": {}}
        transport.call(fwd_no_grad_request(transport.jobs, batch))

        assert len(send.frames) > 1
        assert {wire.read_byte_chunk_metadata(f)["operation"] for f in send.frames} == {"fwd"}
        frame = wire.decode_byte_chunks(send.frames, kind="request")
        assert wire.loads(frame)["batch"]["input_ids"].shape == (1, 20_000)


class TestForwardSubJobRouting:
    def test_current_policy_routes_to_training(self):
        transport, send = _transport()
        transport.call(fwd_no_grad_request(transport.jobs, _batch()))

        assert f"sub_job_id={TRAINING}" in send.urls[0]

    def test_reference_model_routes_to_log_prob(self):
        """Losing this hint returns current-policy log-probs where reference
        log-probs were asked for: a silent numerical error, not a failure."""
        transport, send = _transport()
        transport.call(fwd_no_grad_request(transport.jobs, _batch(), reference_model=True))

        assert f"sub_job_id={LOG_PROB}" in send.urls[0]

    def test_forward_backward_carries_no_routing_hint(self):
        """forward-backward is unambiguously training; its URL is unchanged."""
        transport, send = _transport()
        transport.call(fwd_bwd_request(transport.jobs, _batch()))

        assert "sub_job_id" not in send.urls[0]


class TestForwardResult:
    def test_tensors_survive_the_result_decode(self):
        """forward returns log-prob tensors; only generate is lowered to lists."""
        result = _dssst1_result({"batch": {"logprobs": torch.tensor([[-0.5, -1.5]])}, "avg_loss": 1.0})
        transport, _ = _transport(result)

        decoded = transport.call(fwd_no_grad_request(transport.jobs, _batch()))

        assert torch.is_tensor(decoded["batch"]["logprobs"])
        assert decoded["batch"]["logprobs"].tolist() == [[-0.5, -1.5]]


class TestStillUnsupported:
    def test_log_probs_raises(self):
        """The text-in `log-probs` op has no Cortex route; forward is the tensor path."""
        transport, _ = _transport()
        with pytest.raises(NotImplementedError, match="log-probs"):
            transport.call(Request("log-probs", TRAINING, {"prompts": ["hi"]}))
