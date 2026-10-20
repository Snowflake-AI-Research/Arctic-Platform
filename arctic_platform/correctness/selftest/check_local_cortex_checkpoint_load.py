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
"""Owned local cp/stage loads through genuine client delivery; FAST resumes remain closed."""

import asyncio
import base64
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
import requests

from arctic_platform import wire
from arctic_platform.client import CortexConfig
from arctic_platform.client.transport import Request
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import _route
from arctic_platform.correctness.harness import hosted
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.cortex import CorrectnessClient
from arctic_platform.correctness.harness.cortex import CortexTrainingConfig
from arctic_platform.correctness.harness.cortex_transport import LOCAL_NATIVE_FORWARD_ENV
from arctic_platform.correctness.harness.transport import Checkpoint
from arctic_platform.correctness.harness.workdir import require_data_fast_path

CHECKPOINT_ID = "cp_00000000-0000-0000-0000-000000000001"
OTHER_ID = "cp_00000000-0000-0000-0000-000000000002"
RESULT = {"checkpoint_id": CHECKPOINT_ID, "checkpoint_tag": "global_step10", "version": 10}
STAGE = {
    "location_type": "S3",
    "bucket": "synthetic-control-bucket",
    "prefix": f"jobs/saving-parent/checkpoints/{CHECKPOINT_ID}/",
    "aws": {"aws_key_id": "control-key", "aws_secret_key": "control-secret", "aws_token": "control-token"},
}
CONFIG = Path(__file__).resolve().parents[1] / "configs/qwen3.6-35b-a3b/h200/train-sft-full-8gpus-sp8-64k.config"


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
def gateway(monkeypatch):
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV, raising=False)
    state = SimpleNamespace(
        calls=[],
        exchanges=[],
        closed=[],
        clients=[],
        roles={},
        cancelled=set(),
        reply="polled",
        result=deepcopy(RESULT),
        submitted={"request_id": "owned-load-request"},
        failure=None,
        save_extra={},
        saved=[],
    )
    transports = []
    start_loop = CortexTransport._start_loop

    def start(transport):
        start_loop(transport)
        transports.append(transport)

    def forbidden(*args, **kwargs):
        pytest.fail("load controls must not perform real network requests")

    monkeypatch.setattr(CortexTransport, "_start_loop", start)
    monkeypatch.setattr(requests.Session, "request", forbidden)

    def send(session, method, url, **kwargs):
        loop = asyncio.get_running_loop()
        assert loop is session._loop
        assert session.trust_env is False
        if url.startswith("http://127.0.0.1:8003/"):
            assert kwargs["allow_redirects"] is False
        else:
            assert url.startswith(("https://127.0.0.1:8003/", "http://remote.example:8003/"))
            assert "allow_redirects" not in kwargs
        assert "Authorization" not in session.headers
        state.calls.append((method, url, deepcopy(kwargs)))
        state.exchanges.append((session, loop))
        if method == "GET" and url.endswith("/openapi.json"):
            return _response(
                {
                    "paths": {
                        path: {"post": {"parameters": [{"name": "job_id", "in": "query", "required": True}]}}
                        for path in ("/forward", "/log-probs")
                    },
                }
            )
        if method == "POST" and url.endswith("/cortex-training"):
            parent = f"owned-load-parent-{len(state.roles)}"
            state.roles[parent] = [sub["job_type"] for sub in kwargs["json"]["sub_job_configs"]]
            return _response({"job_id": parent})
        if method == "POST" and url.endswith(":cancel"):
            state.cancelled.add(url.rsplit("/", 1)[-1].removesuffix(":cancel"))
            return _response({})
        if method == "GET" and "/requests/" in url:
            if "save-request" in url:
                return _response({"status": "REQUEST_STATE_DONE", "result": state.save_result})
            if state.reply == "chunked":
                frame = wire.dumps(state.result)
                return _response(
                    {
                        "status": "REQUEST_STATE_DONE",
                        "events": [{"type": "result_chunk", "payload_b64": base64.b64encode(frame).decode()}],
                    }
                )
            return _response({"status": "REQUEST_STATE_DONE", "result": state.result})
        if method == "GET":
            parent = url.rsplit("/", 1)[-1]
            roles = state.roles[parent]
            segments = {"training": "training", "sampling": "sampling", "log_probability": "log_prob"}
            cancelled = parent in state.cancelled
            return _response(
                {
                    "job_id": parent,
                    "status": "cancelled" if cancelled else "RUNNING",
                    "sub_jobs": [
                        {
                            "job_type": role,
                            "sub_job_id": f"{parent}:{segments[role]}:0",
                            "status": "cancelled" if cancelled else "running",
                        }
                        for role in roles
                    ],
                }
            )
        if method == "POST" and url.endswith("/load"):
            if state.failure is not None:
                raise state.failure
            return _response(state.result if state.reply == "inline" else state.submitted)
        if method == "POST" and url.endswith("/save"):
            body = kwargs["json"]
            assert body["checkpoint_type"] == "resumable"
            root = require_data_fast_path(body["path"], label="save control basetemp")
            root.mkdir()
            (root / "latest").write_text("global_step10\n", encoding="ascii")
            state.saved.append(root)
            state.save_result = {
                "checkpoint_path": str(root),
                "checkpoint_tag": "global_step10",
                "version": 10,
                "stage_path": "",
                **state.save_extra,
            }
            return _response({"request_id": "save-request"})
        pytest.fail(f"load controls reached an unexpected HTTP route: {method} {url}")

    monkeypatch.setattr(aiohttp.ClientSession, "request", send)
    original_close = aiohttp.ClientSession.close

    async def close(session):
        loop = asyncio.get_running_loop()
        assert loop is session._loop
        await original_close(session)
        state.closed.append((session, loop))

    monkeypatch.setattr(aiohttp.ClientSession, "close", close)

    def create(*, native=False, training_gpus=8):
        monkeypatch.setenv(LOCAL_NATIVE_FORWARD_ENV, "1" if native else "0")
        client = CorrectnessClient(
            CortexTrainingConfig(
                model_name="Qwen/Qwen3.6-35B-A3B",
                training_gpus=training_gpus,
                sampling_gpus=1,
                log_prob_gpus=1,
                hardware="H200",
                backend=CortexConfig(
                    base_url="http://127.0.0.1:8003",
                    max_retries=4,
                    request_timeout=1.0,
                    poll_timeout=1.0,
                ),
            )
        )
        state.clients.append(client)
        state.calls.clear()
        return client

    yield state, create
    for transport in transports:
        transport.config = CortexConfig(base_url="http://127.0.0.1:8003", poll_timeout=1.0)
        if transport.job_id is not None:
            transport.shutdown()
        transport._on_loop(transport._close_session()).result()
        transport._loop.call_soon_threadsafe(transport._loop.stop)


def _load(client, *, stage=None, asynchronous=False):
    if asynchronous:
        return asyncio.run(
            client.transport.acall(
                Request(
                    "load-checkpoint",
                    client.jobs.training,
                    {"checkpoint_id": CHECKPOINT_ID, "stage_info": stage},
                )
            )
        )
    return client.transport.load_checkpoint(CHECKPOINT_ID, stage_info=stage)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reply", ["inline", "polled", "chunked"])
@pytest.mark.parametrize("stage", [None, STAGE])
def test_owned_cp_load_uses_real_delivery_polling_loop_and_owned_shutdown(
    gateway,
    native,
    asynchronous,
    reply,
    stage,
):
    state, create = gateway
    client = create(native=native)
    state.reply = reply
    transport = client.transport
    before = deepcopy(stage)
    with client:
        assert _load(client, stage=stage, asynchronous=asynchronous) == RESULT
        assert stage == before
        sends = list(state.calls)
        assert sends[0][0] == "POST" and sends[0][1].endswith(f"/{transport.job_id}/load")
        assert sends[0][2]["json"] == {
            "checkpoint_id": CHECKPOINT_ID,
            **({} if stage is None else {"stage_info": stage}),
        }
        assert len(sends) == (1 if reply == "inline" else 2)
        if reply != "inline":
            assert sends[1][0] == "GET"
            assert sends[1][1].endswith(f"/{transport.job_id}/requests/owned-load-request")
    assert [method for method, _, _ in state.calls[-2:]] == ["POST", "GET"]
    assert state.calls[-2][1].endswith(":cancel")
    assert transport.job_id is None and transport._session is None
    assert transport._native_bindings == transport._native_children == {}
    assert transport._local_shutdown_parent is None and transport._local_shutdown_children == {}
    assert len(state.closed) == 1
    session, loop = state.closed[0]
    assert loop is transport._loop and session.closed
    assert all(actual_session is session and actual_loop is loop for actual_session, actual_loop in state.exchanges)


@pytest.mark.parametrize("role", [None, "foreign", "parent", "other-rank", "sampling", "log_prob"])
def test_load_rejects_nonowned_or_nontraining_tokens_before_http(gateway, role):
    state, create = gateway
    client = create()
    parent = client.transport.job_id
    tokens = {
        None: None,
        "foreign": "foreign-parent:training:0",
        "parent": parent,
        "other-rank": f"{parent}:training:1",
        "sampling": client.jobs.sampling,
        "log_prob": client.jobs.log_prob,
    }
    with pytest.raises(ValueError, match="newly owned training"):
        client.transport.call(Request("load-checkpoint", tokens[role], {"checkpoint_id": CHECKPOINT_ID}))
    assert state.calls == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"path": "/data-fast/tmp/retained"},
        {"path": CHECKPOINT_ID},
        {"checkpoint_path": "/data-fast/tmp/retained"},
        {"checkpoint_id": CHECKPOINT_ID, "step": 10},
        {"checkpoint_id": CHECKPOINT_ID, "source_job_id": "saving-parent"},
        {"checkpoint_id": CHECKPOINT_ID, "source_job_id": None},
        {"checkpoint_id": CHECKPOINT_ID, "path": None},
        {"checkpoint_id": "global_step10"},
        {"checkpoint_id": "/data-fast/tmp/retained"},
        {"checkpoint_id": "../cp_other"},
        {"checkpoint_id": None},
        {"checkpoint_id": 10},
    ],
)
def test_load_body_never_translates_paths_tags_step_or_source_job(gateway, body):
    state, create = gateway
    client = create()
    with pytest.raises(ValueError):
        client.transport.call(Request("load-checkpoint", client.jobs.training, body))
    assert state.calls == []


@pytest.mark.parametrize(
    "stage",
    [
        {},
        [],
        {"location_type": []},
        {**STAGE, "location_type": "LOCAL"},
        {**STAGE, "bucket": "/data-fast/tmp/checkpoint"},
        {**STAGE, "prefix": f"jobs/other/checkpoints/{OTHER_ID}/"},
        {**STAGE, "prefix": f"../checkpoints/{CHECKPOINT_ID}/"},
        {**STAGE, "checkpoint_path": "/data-fast/tmp/checkpoint"},
        {**STAGE, "aws": object()},
    ],
)
def test_invalid_or_misbound_stage_fails_before_http_without_leaking_secrets(gateway, stage):
    state, create = gateway
    client = create()
    with pytest.raises(ValueError) as error:
        _load(client, stage=stage)
    assert "control-secret" not in str(error.value) and "control-token" not in str(error.value)
    assert state.calls == []


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"checkpoint_id": OTHER_ID, "checkpoint_tag": "global_step10", "version": 10},
        {**RESULT, "checkpoint_tag": "../global_step10"},
        {**RESULT, "checkpoint_tag": "global_step010"},
        {**RESULT, "version": 11},
        {**RESULT, "version": True},
        {**RESULT, "version": "10"},
    ],
)
@pytest.mark.parametrize("reply", ["inline", "polled"])
def test_load_rejects_missing_or_inconsistent_completed_metadata(gateway, result, reply):
    state, create = gateway
    client = create()
    state.reply, state.result = reply, result
    with pytest.raises(RuntimeError, match="local checkpoint load returned"):
        _load(client)
    assert sum(method == "POST" and url.endswith("/load") for method, url, _ in state.calls) == 1


@pytest.mark.parametrize(
    "failure",
    [
        aiohttp.ServerDisconnectedError("synthetic lost load reply"),
        asyncio.TimeoutError("synthetic load timeout"),
        aiohttp.ClientResponseError(None, (), status=503, message="synthetic load failure"),
    ],
)
def test_mutating_load_is_not_replayed_even_with_configured_retries(gateway, failure):
    state, create = gateway
    client = create()
    state.failure = failure
    with pytest.raises(type(failure)):
        _load(client)
    assert client.transport.config.max_retries == 4
    assert len(state.calls) == 1 and state.calls[0][1].endswith("/load")


@pytest.mark.parametrize("submission", [{"request_id": ""}, {"request_id": 123}])
def test_load_rejects_invalid_async_submission_without_polling(gateway, submission):
    state, create = gateway
    client = create()
    state.submitted = submission
    with pytest.raises(RuntimeError, match="valid request_id"):
        _load(client)
    assert len(state.calls) == 1


def test_load_rejects_binary_request_before_http(gateway):
    state, create = gateway
    client = create()
    with pytest.raises(ValueError, match="checkpoint_id/stage_info only"):
        client.transport.call(
            Request(
                "load-checkpoint",
                client.jobs.training,
                {"checkpoint_id": CHECKPOINT_ID},
                binary=True,
            )
        )
    assert state.calls == []


@pytest.mark.parametrize("base_url", ["https://127.0.0.1:8003", "http://remote.example:8003"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_nonlocal_backend_uses_production_load_delivery(gateway, base_url, asynchronous):
    state, create = gateway
    client = create()
    client.transport.config = client.transport.config.model_copy(update={"base_url": base_url})
    assert not client.transport._is_local_gateway()
    assert _load(client, stage=STAGE, asynchronous=asynchronous) == RESULT
    prefix = client.transport._prefix
    parent = client.transport.job_id
    assert [(method, url) for method, url, _ in state.calls] == [
        ("POST", f"{prefix}/{parent}/load"),
        ("GET", f"{prefix}/{parent}/requests/owned-load-request"),
    ]
    assert state.calls[0][2]["json"] == {"checkpoint_id": CHECKPOINT_ID, "stage_info": STAGE}


@pytest.mark.parametrize(
    "backend",
    [
        {"base_url": "http://127.0.0.1:8003/path"},
        {"base_url": "http://127.0.0.1:8003?query=1"},
        {"base_url": "http://127.0.0.1:8003#fragment"},
        {"base_url": "http://user:secret@127.0.0.1:8003"},
        {"base_url": "http://127.0.0.1:8003", "host": "account.example"},
        {"base_url": "http://127.0.0.1:8003", "pat": "synthetic-pat"},
    ],
)
def test_authenticated_or_decorated_loopback_cannot_enable_local_load(gateway, backend):
    state, create = gateway
    client = create()
    client.transport.config = client.transport.config.model_copy(update=backend)
    with pytest.raises(ValueError, match="unauthenticated HTTP loopback base URL"):
        _load(client)
    assert state.calls == []


def test_load_after_owned_shutdown_cannot_reopen_a_session(gateway):
    state, create = gateway
    client = create()
    client.shutdown()
    state.calls.clear()
    with pytest.raises(ValueError, match="newly owned training"):
        _load(client)
    assert state.calls == [] and client.transport._session is None


def test_sampling_only_client_cannot_load_a_training_checkpoint(gateway):
    state, create = gateway
    client = create(training_gpus=0)
    with pytest.raises(ValueError, match="newly owned training"):
        _load(client)
    assert state.calls == []


def test_uninitialized_transport_cannot_open_a_load_session(gateway):
    state, _ = gateway
    from arctic_platform.correctness.harness.cortex_transport import CorrectnessCortexTransport

    transport = CorrectnessCortexTransport(CortexConfig(base_url="http://127.0.0.1:8003"))
    with pytest.raises(ValueError, match="newly owned training"):
        transport.call(Request("load-checkpoint", "unowned:training:0", {"checkpoint_id": CHECKPOINT_ID}))
    assert state.calls == [] and transport._session is None


def test_production_cortex_load_route_posts_and_polls_parent():
    route = _route("load-checkpoint")
    assert route.method == "POST" and route.path == "/{job}/load" and route.poll is True
    request = Request(
        "load-checkpoint",
        "production-parent:training:0",
        {"checkpoint_id": CHECKPOINT_ID, "stage_info": None},
    )
    assert route.prepare("https://account.example/cortex-training", request) == (
        "https://account.example/cortex-training/production-parent/load",
        [{"json": {"checkpoint_id": CHECKPOINT_ID}}],
    )
    assert route.submitted(request, {"request_id": "production-load-request"}) == "production-load-request"
    assert hosted.ArcticClient is CorrectnessClient


@pytest.mark.parametrize(
    "identity",
    [
        Checkpoint("/data-fast/tmp/retained", "saving-parent"),
        Checkpoint(CHECKPOINT_ID, "saving-parent"),
        hosted.LocalCortexCheckpoint(CHECKPOINT_ID, "saving-parent", "/data-fast/tmp/retained", "global_step10", 10),
    ],
)
def test_hosted_cross_job_without_typed_stage_identity_rejects_before_allocation(gateway, identity):
    state, _ = gateway
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    with hosted.HostedTransport(backend, load_config(CONFIG)) as transport:
        with pytest.raises((ValueError, RuntimeError), match="stage_info"):
            with transport.job(resume_from=identity):
                pytest.fail("unresolvable cross-job resume yielded a training job")
    assert state.calls == [] and state.roles == {}


def test_actual_public_save_path_only_receipt_never_mints_a_durable_identity(gateway, tmp_path, monkeypatch):
    state, create = gateway
    client = create()
    root = require_data_fast_path(tmp_path / "retained-save", label="save control basetemp")
    monkeypatch.setattr(hosted, "correctness_workdir", lambda prefix: root)
    with pytest.raises(RuntimeError, match="only a FAST path/tag"):
        hosted.HostedTrainingJob(client, client.transport.job_id, local_gateway=True).save_resumable()
    assert root.joinpath("latest").read_text() == "global_step10\n"
    assert state.saved == [root]
    assert all("/checkpoints" not in url and not url.endswith("/load") for _, url, _ in state.calls)


def test_typed_identity_comes_only_from_explicit_save_metadata_and_hides_credentials(
    gateway,
    tmp_path,
    monkeypatch,
):
    state, create = gateway
    client = create()
    root = require_data_fast_path(tmp_path / "explicit-save", label="save control basetemp")
    monkeypatch.setattr(hosted, "correctness_workdir", lambda prefix: root)
    state.save_extra = {"checkpoint_id": CHECKPOINT_ID, "stage_info": deepcopy(STAGE)}
    checkpoint = hosted.HostedTrainingJob(client, client.transport.job_id, local_gateway=True).save_resumable()
    assert isinstance(checkpoint, hosted.LocalCortexCheckpoint)
    assert checkpoint.reference == CHECKPOINT_ID and checkpoint.checkpoint_path == str(root)
    assert checkpoint.stage_info == STAGE and checkpoint.stage_info is not state.save_extra["stage_info"]
    assert "control-secret" not in repr(checkpoint) and "control-token" not in repr(checkpoint)
    # This branch's synthetic receipt is a contract control, not proof current MAIN saves cp/stage.


def test_hosted_explicit_stage_resume_uses_real_client_without_source_job_rewrite(gateway):
    state, _ = gateway
    identity = hosted.LocalCortexCheckpoint(
        CHECKPOINT_ID,
        "saving-parent",
        "/data-fast/tmp/retained",
        "global_step10",
        10,
        stage_info=deepcopy(STAGE),
    )
    with hosted.HostedTransport(CortexConfig(base_url="http://127.0.0.1:8003"), load_config(CONFIG)) as transport:
        with transport.job(resume_from=identity) as job:
            assert isinstance(job.client, CorrectnessClient)
            state.clients.append(job.client)
            creates = [
                kwargs["json"]
                for method, url, kwargs in state.calls
                if method == "POST" and url.endswith("/cortex-training")
            ]
            assert len(creates) == 1
            assert "source_checkpoint_info" not in creates[0]["sub_job_configs"][0]
            loads = [
                kwargs["json"] for method, url, kwargs in state.calls if method == "POST" and url.endswith("/load")
            ]
            assert loads == [{"checkpoint_id": CHECKPOINT_ID, "stage_info": STAGE}]
    assert len(state.closed) == 1 and state.clients[0].transport._session is None


def test_hosted_wrong_saved_step_releases_owned_children_and_session(gateway):
    state, _ = gateway
    state.result = {**RESULT, "checkpoint_tag": "global_step11", "version": 11}
    identity = hosted.LocalCortexCheckpoint(
        CHECKPOINT_ID,
        "saving-parent",
        "/data-fast/tmp/retained",
        "global_step10",
        10,
        stage_info=deepcopy(STAGE),
    )
    with hosted.HostedTransport(CortexConfig(base_url="http://127.0.0.1:8003"), load_config(CONFIG)) as transport:
        with pytest.raises(RuntimeError, match="different global step"):
            with transport.job(resume_from=identity):
                pytest.fail("wrong-step restore yielded a training job")
    assert state.cancelled == {"owned-load-parent-0"}
    assert len(state.closed) == 1 and state.closed[0][0].closed
