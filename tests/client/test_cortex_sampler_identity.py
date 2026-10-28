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

import asyncio
from concurrent.futures import Future

import pytest

from arctic_platform.client.transports.base import Request
from arctic_platform.client.transports.cortex import CortexTransport


@pytest.fixture
def transport():
    result = object.__new__(CortexTransport)
    result._router_replay_sampling_ids = {}
    return result


def _request(op, body, job_id="training"):
    return Request(op=op, body=body, job_id=job_id)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_bootstrap_response_translates_only_the_matching_pair(transport, monkeypatch, asynchronous):
    bootstrap = _request(
        "operation",
        {
            "operation_type": "bootstrap-router-replay",
            "payload": {"target_sub_job_id": "training", "source_sub_job_id": "public-sampling"},
        },
    )
    future = Future()
    reply = {"status": "ok", "sampling_job_id": "zone-sampling"}
    future.set_result(([], reply))

    class Route:
        @staticmethod
        def finish(chunks, response):
            assert chunks == []
            return response

    monkeypatch.setattr(transport, "_submit", lambda *args: (Route(), future))
    assert (asyncio.run(transport.acall(bootstrap)) if asynchronous else transport._call(bootstrap)) == reply
    original = _request("forward", {"router_replay": {"sampling_job_id": "public-sampling", "identity_version": 1}})
    translated = transport._router_replay_request(original)
    assert translated.body["router_replay"] == {"sampling_job_id": "zone-sampling", "identity_version": 1}
    assert original.body["router_replay"]["sampling_job_id"] == "public-sampling"
    other_job = _request("forward", original.body, job_id="different-training")
    assert transport._router_replay_request(other_job) is other_job
    other_sampler = _request("forward-backward", {"router_replay": {"sampling_job_id": "different-sampling"}})
    assert transport._router_replay_request(other_sampler) is other_sampler


@pytest.mark.parametrize("operation", ["forward", "forward-backward", "optimizer-step"])
def test_submit_prepares_the_translated_request_only_for_forward(transport, monkeypatch, operation):
    transport._router_replay_sampling_ids[("training", "public-sampling")] = "zone-sampling"
    transport._prefix = "test-prefix"
    request = _request(operation, {"router_replay": {"sampling_job_id": "public-sampling"}})
    prepared = []

    class Route:
        @staticmethod
        def prepare(prefix, forwarded):
            assert prefix == "test-prefix"
            prepared.append(forwarded)
            return "test-url", []

    monkeypatch.setattr(transport, "_deliver", lambda *args: args)
    monkeypatch.setattr(transport, "_on_loop", lambda value: value)
    transport._submit(request, Route())
    assert prepared[0].body["router_replay"]["sampling_job_id"] == (
        "public-sampling" if operation == "optimizer-step" else "zone-sampling"
    )
    assert request.body["router_replay"]["sampling_job_id"] == "public-sampling"
