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

Unlike forward-backward and generate, `forward` has no octet endpoint of its own
on SnowAPI. It rides the generic `/operation` RPC instead, which carries a struct
rather than bytes, so the DSSST1 frame travels base64'd in the envelope's payload
and the sub-job token is a body field rather than a query param.

Three things these tests exist to pin, all of which were wrong at first:

- the route. An earlier revision posted an octet body to /{job}/forward. That
  endpoint exists on the zone but not through the control plane, so it never
  reached a GPU.
- chunking. The budget applies to the JSON body, and base64 inflates the frame
  4/3, so the frame slice must be the smaller 3/4 share of what the envelope
  leaves. Only the final chunk starts execution and returns a request_id.
- the result shape. The zone wraps its DSSST1 frame in a JSON envelope that
  lacks the `wire_format` marker the generic decoder keys off, so `forward`
  needs its own unwrap or the caller gets base64 text instead of tensors.

`_send` is stubbed, so these tests pin the request the transport builds and the
result it decodes without a live GS.
"""

from __future__ import annotations

import asyncio
import base64
import json
from copy import deepcopy
from urllib.parse import unquote

import pytest
import torch

from arctic_platform import wire
from arctic_platform.client import CortexConfig
from arctic_platform.client import JobHandles
from arctic_platform.client.requests import bootstrap_router_replay_request
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

    async def __call__(self, method: str, url: str, *, retry_on=None, **kwargs):
        if method == "GET":
            return {"status": "REQUEST_STATE_DONE", "result": self.result}
        self.posts.append({"url": url, **kwargs})
        frame = kwargs.get("data")
        if frame is None and "json" in kwargs:
            frame = base64.b64decode(kwargs["json"]["payload"]["payload_b64"])
        chunk = wire.read_byte_chunk_metadata(frame) if frame is not None else None
        if chunk is not None and chunk["chunk_idx"] < chunk["total_chunks"] - 1:
            return {"chunk_cached": True}
        return {"request_id": "req-1"}

    @property
    def urls(self) -> list[str]:
        """POSTed urls, percent-decoding so sub-job tokens read as `job:type:0`."""
        return [unquote(post["url"]) for post in self.posts]

    @property
    def frames(self) -> list[bytes]:
        return [post["data"] for post in self.posts]

    @property
    def envelopes(self) -> list[dict]:
        return [post["json"] for post in self.posts]

    @property
    def operation_frame(self) -> bytes:
        """The frame the zone will reassemble from every payload we posted."""
        frames = [base64.b64decode(e["payload"]["payload_b64"]) for e in self.envelopes]
        if len(frames) == 1:
            return frames[0]
        return wire.decode_byte_chunks(frames, kind="request")


def _transport(result: dict | None = None) -> tuple[CortexTransport, FakeSend]:
    transport = CortexTransport(CortexConfig(base_url="http://gs.test"))
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
    def test_posts_the_operation_envelope(self):
        """forward goes out as an /operation struct with the frame base64'd inside."""
        transport, send = _transport()
        transport.call(fwd_no_grad_request(transport.jobs, _batch()))

        (envelope,) = send.envelopes
        assert envelope["operation_type"] == "forward"
        assert envelope["payload"]["content_type"] == "application/octet-stream"
        assert wire.loads(send.operation_frame)["batch"]["input_ids"].tolist() == BATCH_IDS

    def test_hits_the_operation_endpoint(self):
        """There is no /forward through the control plane -- only /operation.

        Posting to /{job}/forward reaches the zone's route in isolation but never
        gets past GS, so the op has to go out as an operation.
        """
        transport, send = _transport()
        transport.call(fwd_no_grad_request(transport.jobs, _batch()))

        assert send.urls[0].endswith(f"/{JOB}/operation")

    def test_frame_under_the_cap_posts_one_envelope(self):
        """A frame that fits travels whole: one POST, no chunk wrapper.

        forward-backward force-wraps every frame because its chunk_group_id is
        the server's at-most-once key. A no-grad forward mutates nothing, so
        chunking here stays purely size-driven.
        """
        transport, send = _transport()
        transport.call(fwd_no_grad_request(transport.jobs, _batch()))

        assert len(send.posts) == 1
        assert wire.read_byte_chunk_metadata(send.operation_frame) is None

    def test_oversized_frame_splits_across_envelopes(self, monkeypatch):
        """Each POST carries a request-chunk frame the zone can reassemble."""
        monkeypatch.setattr(cortex_module, "_OPERATION_MAX_JSON_BYTES", 16 * 1024)
        transport, send = _transport()
        batch = {"batch": {"input_ids": torch.arange(20_000).reshape(1, -1)}, "meta": {}, "processing": {}}
        transport.call(fwd_no_grad_request(transport.jobs, batch))

        assert len(send.posts) > 1
        assert {e["operation_type"] for e in send.envelopes} == {"forward"}
        assert wire.loads(send.operation_frame)["batch"]["input_ids"].shape == (1, 20_000)

    def test_chunk_slices_respect_the_json_budget(self, monkeypatch):
        """The cap is on the JSON body, which base64 inflates 4/3.

        Sizing the slice against the raw budget would overshoot by a third and be
        rejected by GS, so this pins the encoded payload under the cap.
        """
        budget = 16 * 1024
        monkeypatch.setattr(cortex_module, "_OPERATION_MAX_JSON_BYTES", budget)
        transport, send = _transport()
        batch = {"batch": {"input_ids": torch.arange(20_000).reshape(1, -1)}, "meta": {}, "processing": {}}
        transport.call(fwd_no_grad_request(transport.jobs, batch))

        assert max(len(json.dumps(envelope).encode()) for envelope in send.envelopes) <= budget

    def test_missing_request_id_on_the_last_chunk_is_an_error(self):
        """Only the final chunk starts execution; without its id there is nothing to poll."""
        transport, send = _transport()

        async def cached(*args, **kwargs):
            return {"status": "chunk_cached"}

        transport._send = cached  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="carried no request_id"):
            transport.call(fwd_no_grad_request(transport.jobs, _batch()))

    def test_forward_backward_still_chunks(self, monkeypatch):
        """The cap is forward's alone -- fwd-bwd's chunk group must keep working."""
        monkeypatch.setattr(cortex_module, "_MAX_OCTET_BYTES", 64 * 1024)
        transport, send = _transport()
        batch = {"batch": {"input_ids": torch.arange(20_000).reshape(1, -1)}, "meta": {}, "processing": {}}
        transport.call(fwd_bwd_request(transport.jobs, batch))

        assert len(send.frames) > 1
        assert {wire.read_byte_chunk_metadata(f)["operation"] for f in send.frames} == {"fwd-bwd"}
        frame = wire.decode_byte_chunks(send.frames, kind="request")
        assert wire.loads(frame)["batch"]["input_ids"].shape == (1, 20_000)


class TestForwardSubJobRouting:
    def test_current_policy_routes_to_training(self):
        transport, send = _transport()
        transport.call(fwd_no_grad_request(transport.jobs, _batch()))

        assert send.envelopes[0]["sub_job_id"] == TRAINING

    def test_reference_model_routes_to_log_prob(self):
        """Reference log-probs must name the log_prob sub-job.

        Cortex answers that with Unimplemented today, since its zone serves
        forward on training only. Sending the hint anyway is the point: dropping
        it would quietly score against the current policy instead, which reads as
        a plausible number rather than an error.
        """
        transport, send = _transport()
        transport.call(fwd_no_grad_request(transport.jobs, _batch(), reference_model=True))

        assert send.envelopes[0]["sub_job_id"] == LOG_PROB

    def test_forward_backward_carries_no_routing_hint(self):
        """forward-backward is unambiguously training; its URL is unchanged."""
        transport, send = _transport()
        transport.call(fwd_bwd_request(transport.jobs, _batch()))

        assert "target_sub_job_id" not in send.urls[0]


class TestRouterReplayWire:
    @pytest.mark.parametrize("operation", ["forward-backward", "forward"])
    @pytest.mark.parametrize("chunked", [False, True])
    def test_replay_coordination_is_metadata_not_a_model_batch_field(self, monkeypatch, operation, chunked):
        if chunked:
            monkeypatch.setattr(cortex_module, "_MAX_OCTET_BYTES", 16 * 1024)
            monkeypatch.setattr(cortex_module, "_OPERATION_MAX_JSON_BYTES", 16 * 1024)
        replay = {"sampling_job_id": "sampler-1", "identity_version": 1}
        width = 20_000 if chunked else 3
        batch = {
            "args": (),
            "kwargs": {
                "input_ids": torch.arange(width).reshape(1, -1),
                "rr_sample_ids": ["rr1:sample-1"],
                "rr_replay_meta": [{"replay_id": "rr1:sample-1"}],
                "rr_discard": True,
            },
            "context": {"temperature": 1.0},
            "processing": {"loss_fn": "grpo"},
        }
        before = deepcopy(batch)
        jobs = JobHandles(training=TRAINING)
        if operation == "forward-backward":
            request = fwd_bwd_request(jobs, batch, router_replay=replay)
        else:
            request = fwd_no_grad_request(jobs, {**batch, "router_replay": replay})
        request_body = deepcopy(request.body)

        _, sends = cortex_module._route(operation).prepare("http://gs.test", request)

        frames = [
            (
                send["data"]
                if operation == "forward-backward"
                else base64.b64decode(send["json"]["payload"]["payload_b64"])
            )
            for send in sends
        ]
        if operation == "forward-backward":
            frame = wire.decode_byte_chunks(frames, kind="request")
        else:
            frame = wire.decode_byte_chunks(frames, kind="request") if len(frames) > 1 else frames[0]
        if chunked:
            assert len(sends) > 1
        assert wire.read_metadata(frame) == {**cortex_module._CHUNKED_RESULTS, "router_replay": replay}
        payload = wire.loads(frame)
        assert set(payload) == set(batch)
        assert payload["kwargs"]["rr_sample_ids"] == ["rr1:sample-1"]
        assert payload["kwargs"]["rr_replay_meta"] == [{"replay_id": "rr1:sample-1"}]
        assert payload["kwargs"]["rr_discard"] is True
        assert payload["context"] == batch["context"]
        assert payload["processing"] == batch["processing"]
        assert torch.equal(payload["kwargs"]["input_ids"], before["kwargs"]["input_ids"])
        assert torch.equal(batch["kwargs"]["input_ids"], before["kwargs"]["input_ids"])
        assert request.body["router_replay"] == request_body["router_replay"]
        assert set(request.body) == set(request_body)

    @pytest.mark.parametrize("operation", ["forward-backward", "forward"])
    def test_absent_replay_keeps_the_ordinary_batch_and_response_metadata(self, operation):
        jobs = JobHandles(training=TRAINING)
        batch = _batch()
        request = fwd_bwd_request(jobs, batch) if operation == "forward-backward" else fwd_no_grad_request(jobs, batch)
        _, sends = cortex_module._route(operation).prepare("http://gs.test", request)
        frame = (
            wire.decode_byte_chunks([send["data"] for send in sends], kind="request")
            if operation == "forward-backward"
            else base64.b64decode(sends[0]["json"]["payload"]["payload_b64"])
        )
        assert wire.read_metadata(frame) == cortex_module._CHUNKED_RESULTS
        payload = wire.loads(frame)
        assert set(payload) == set(batch)
        assert payload["batch"]["input_ids"].tolist() == BATCH_IDS


class TestRouterReplayJobIdentity:
    @pytest.mark.parametrize("asynchronous", [False, True])
    @pytest.mark.parametrize("operation", ["forward-backward", "forward"])
    @pytest.mark.parametrize("chunked", [False, True])
    def test_bootstrap_sampler_identity_reaches_training_metadata(self, monkeypatch, asynchronous, operation, chunked):
        if chunked:
            monkeypatch.setattr(cortex_module, "_MAX_OCTET_BYTES", 16 * 1024)
            monkeypatch.setattr(cortex_module, "_OPERATION_MAX_JSON_BYTES", 16 * 1024)
        transport, send = _transport()
        sampling = f"{JOB}:sampling:0"
        transport.jobs.sampling = sampling
        bootstrap = {"status": "ok", "training_job_id": "internal-training", "sampling_job_id": "internal-sampling"}
        bootstraps = []

        async def reply(method, url, **kwargs):
            body = kwargs.get("json") or {}
            if body.get("operation_type") == "bootstrap-router-replay":
                bootstraps.append(body)
                return bootstrap
            return await send(method, url, **kwargs)

        transport._send = reply

        def call(request):
            return asyncio.run(transport.acall(request)) if asynchronous else transport.call(request)

        assert call(bootstrap_router_replay_request(transport.jobs)) == bootstrap
        assert bootstraps[0]["payload"]["source_sub_job_id"] == sampling
        assert bootstraps[0]["payload"]["target_sub_job_id"] == TRAINING
        replay = {"sampling_job_id": sampling, "identity_version": 1}
        batch = {"batch": {"input_ids": torch.arange(20_000 if chunked else 3).reshape(1, -1)}}
        request = (
            fwd_bwd_request(transport.jobs, batch, router_replay=replay)
            if operation == "forward-backward"
            else fwd_no_grad_request(transport.jobs, {**batch, "router_replay": replay})
        )
        call(request)
        frame = (
            wire.decode_byte_chunks(send.frames, kind="request")
            if operation == "forward-backward"
            else send.operation_frame
        )
        assert wire.read_metadata(frame)["router_replay"] == {
            "sampling_job_id": "internal-sampling",
            "identity_version": 1,
        }
        assert len(send.posts) > 1 if chunked else len(send.posts) == 1
        assert request.body["router_replay"] == replay
        assert replay["sampling_job_id"] == sampling
        assert transport.jobs.training == TRAINING
        assert transport.jobs.sampling == sampling

    @pytest.mark.parametrize("unmapped", ["different-training", "different-sampling", "already-internal"])
    def test_bootstrap_translation_is_scoped_to_the_public_job_pair(self, unmapped):
        transport, send = _transport()
        sampling = f"{JOB}:sampling:0"
        transport.jobs.sampling = sampling

        async def reply(method, url, **kwargs):
            if (kwargs.get("json") or {}).get("operation_type") == "bootstrap-router-replay":
                return {"status": "ok", "sampling_job_id": "internal-sampling"}
            return await send(method, url, **kwargs)

        transport._send = reply
        transport.call(bootstrap_router_replay_request(transport.jobs))
        jobs = JobHandles(training="another:training:0" if unmapped == "different-training" else TRAINING)
        sampler = {
            "different-training": sampling,
            "different-sampling": "another:sampling:0",
            "already-internal": "internal-sampling",
        }[unmapped]
        transport.call(fwd_bwd_request(jobs, _batch(), router_replay={"sampling_job_id": sampler}))
        frame = wire.decode_byte_chunks(send.frames, kind="request")
        assert wire.read_metadata(frame)["router_replay"]["sampling_job_id"] == sampler

    def test_failed_bootstrap_does_not_translate_later_requests(self):
        transport, send = _transport()
        transport.jobs.sampling = f"{JOB}:sampling:0"

        async def reply(method, url, **kwargs):
            if (kwargs.get("json") or {}).get("operation_type") == "bootstrap-router-replay":
                raise RuntimeError("bootstrap failed")
            return await send(method, url, **kwargs)

        transport._send = reply
        with pytest.raises(RuntimeError, match="bootstrap failed"):
            transport.call(bootstrap_router_replay_request(transport.jobs))
        replay = {"sampling_job_id": transport.jobs.sampling}
        transport.call(fwd_bwd_request(transport.jobs, _batch(), router_replay=replay))
        frame = wire.decode_byte_chunks(send.frames, kind="request")
        assert wire.read_metadata(frame)["router_replay"] == replay


class TestForwardResult:
    def test_tensors_survive_the_result_decode(self):
        """forward returns log-prob tensors; only generate is lowered to lists."""
        result = _dssst1_result({"batch": {"logprobs": torch.tensor([[-0.5, -1.5]])}, "avg_loss": 1.0})
        transport, _ = _transport(result)

        decoded = transport.call(fwd_no_grad_request(transport.jobs, _batch()))

        assert torch.is_tensor(decoded["batch"]["logprobs"])
        assert decoded["batch"]["logprobs"].tolist() == [[-0.5, -1.5]]

    def test_result_without_a_payload_passes_through(self):
        """An error/plain-JSON result must not be mistaken for an envelope."""
        transport, _ = _transport({"job_id": JOB, "status": "weird"})

        assert transport.call(fwd_no_grad_request(transport.jobs, _batch())) == {
            "job_id": JOB,
            "status": "weird",
        }


class TestStillUnsupported:
    def test_log_probs_raises(self):
        """The text-in `log-probs` op has no Cortex route; forward is the tensor path."""
        transport, _ = _transport()
        with pytest.raises(NotImplementedError, match="log-probs"):
            transport.call(Request("log-probs", TRAINING, {"prompts": ["hi"]}))
