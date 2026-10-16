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

"""Exercise owned public child IDs and native wire routes without allocating jobs."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
import requests
import torch

from arctic_platform import wire
from arctic_platform.client import CortexConfig
from arctic_platform.client.transport import Request
from arctic_platform.client.transport import initialize_or_cleanup
from arctic_platform.correctness.harness.cortex import CortexTrainingConfig
from arctic_platform.correctness.harness.cortex_transport import LOCAL_NATIVE_FORWARD_ENV
from arctic_platform.correctness.harness.cortex_transport import CorrectnessCortexTransport


def response(payload=None, *, status=200, data=None):
    reply = requests.Response()
    reply.status_code = status
    reply.url = "http://127.0.0.1:8003/mock"
    reply._content = data if data is not None else json.dumps(payload).encode()
    return reply


def client_config(backend=None, **kwargs):
    return CortexTrainingConfig(
        model_name="Qwen/Qwen3-8B",
        training_gpus=kwargs.pop("training_gpus", 4),
        hardware="H200",
        backend=backend or CortexConfig(base_url="http://127.0.0.1:8003", max_retries=0),
        **kwargs,
    )


def openapi(*paths):
    return {
        "paths": {
            path: {"post": {"parameters": [{"name": "job_id", "in": "query", "required": True}]}} for path in paths
        }
    }


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setenv(LOCAL_NATIVE_FORWARD_ENV, "1")
    calls = []
    schema = openapi("/forward", "/log-probs")

    def send(session, method, url, **kwargs):
        calls.append((method, url, kwargs))
        assert session.trust_env is False
        assert kwargs["allow_redirects"] is False
        if url.endswith("/openapi.json"):
            return response(schema)
        if url.endswith(":cancel"):
            return response({})
        if method == "POST" and url.endswith("/cortex-training"):
            return response({"job_id": "owned-parent"})
        if method == "GET" and url.endswith("/owned-parent"):
            return response(
                {
                    "status": "RUNNING",
                    "sub_jobs": [
                        {"job_type": role, "sub_job_id": f"owned-parent:{role}:0"}
                        for role in ("training", "sampling", "log_probability")
                    ],
                }
            )
        pytest.fail(f"unexpected request: {method} {url}")

    monkeypatch.setattr(requests.Session, "request", send)
    return calls, schema


def created(calls):
    return next(
        kwargs["json"] for method, url, kwargs in calls if method == "POST" and url.endswith("/cortex-training")
    )


def test_native_binding_uses_only_fresh_public_config_ids(gateway):
    calls, _ = gateway
    config = client_config(sampling_gpus=1, log_prob_gpus=1)
    original = config.to_cortex()
    transport = CorrectnessCortexTransport(config.backend)
    jobs = transport.initialize(config)
    payload = created(calls)
    identifiers = [sub["job_id"] for sub in payload["sub_job_configs"]]
    assert len(set(identifiers)) == 3
    assert all(str(UUID(identifier)) == identifier for identifier in identifiers)
    assert payload["hardware"] == "H200"
    assert config.to_cortex() == original
    for sub, expected in zip(payload["sub_job_configs"], original):
        assert {key: value for key, value in sub.items() if key != "job_id"} == expected
        role = "log_prob" if sub["job_type"] == "log_probability" else sub["job_type"]
        token = getattr(jobs, role)
        assert transport._native_bindings[str(token)] == (role, sub["job_id"])
    assert calls[0][1].endswith("/openapi.json")
    transport.shutdown()
    assert calls[-1][1].endswith("/owned-parent:cancel")
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
def test_native_opt_in_rejects_hosted_or_decorated_gateway_before_request(gateway, backend):
    calls, _ = gateway
    config = client_config(CortexConfig(**backend))
    transport = CorrectnessCortexTransport(config.backend)
    with pytest.raises(ValueError, match="unauthenticated HTTP loopback"):
        initialize_or_cleanup(transport, config)
    assert calls == []
    assert transport.job_id is None
    assert transport._native_bindings == {}


def test_native_reconnect_is_rejected_without_discovery(gateway):
    calls, _ = gateway
    config = client_config(training_job_id="other-parent:training:0")
    transport = CorrectnessCortexTransport(config.backend)
    with pytest.raises(ValueError, match="reconnect is unsupported"):
        initialize_or_cleanup(transport, config)
    assert calls == []


@pytest.mark.parametrize("broken", ["missing", "no_post", "wrong_query", "optional"])
def test_actual_installed_api_is_checked_before_create(gateway, broken):
    calls, schema = gateway
    if broken == "missing":
        schema["paths"].pop("/forward")
    elif broken == "no_post":
        schema["paths"]["/forward"]["get"] = schema["paths"]["/forward"].pop("post")
    elif broken == "wrong_query":
        schema["paths"]["/forward"]["post"]["parameters"][0]["name"] = "sub_job_id"
    else:
        schema["paths"]["/forward"]["post"]["parameters"][0]["required"] = False
    config = client_config()
    transport = CorrectnessCortexTransport(config.backend)
    with pytest.raises(RuntimeError, match="before job allocation"):
        initialize_or_cleanup(transport, config)
    assert len(calls) == 1
    assert calls[0][0] == "GET"
    assert transport.job_id is None


def test_log_probability_route_checked_for_declared_log_probability_job(gateway):
    calls, schema = gateway
    schema["paths"].pop("/log-probs")
    config = client_config(training_gpus=0, log_prob_gpus=1)
    transport = CorrectnessCortexTransport(config.backend)
    with pytest.raises(RuntimeError, match="/log-probs"):
        initialize_or_cleanup(transport, config)
    assert len(calls) == 1


@pytest.mark.parametrize("op,role", [("forward", "training"), ("log-probs", "log_prob"), ("log-probs", "sampling")])
def test_native_wire_request_uses_owned_uuid_not_synthetic_token(gateway, monkeypatch, op, role):
    calls, _ = gateway
    config = client_config(sampling_gpus=1, log_prob_gpus=1)
    transport = CorrectnessCortexTransport(config.backend)
    jobs = transport.initialize(config)
    token = getattr(jobs, role)
    child = transport._native_bindings[str(token)][1]
    body = {"batch": {"input_ids": torch.tensor([[2, 3]])}, "processing": {"temperature": 1.0}, "optional": None}
    expected = {"logprobs": torch.tensor([[-1.2, -2.3]], dtype=torch.float32)}
    captured = []

    def post(url, **kwargs):
        captured.append((url, kwargs))
        return response(data=wire.dumps(expected))

    monkeypatch.setattr(transport.session, "post", post)
    result = transport.call(Request(op, token, body))
    assert torch.equal(result["logprobs"], expected["logprobs"])
    url, kwargs = captured[0]
    assert url == f"http://127.0.0.1:8003/{op}?job_id={child}"
    assert "owned-parent" not in url
    assert kwargs["allow_redirects"] is False
    assert kwargs["timeout"] == config.backend.request_timeout
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
def test_native_forward_never_discovers_or_reroutes_foreign_or_wrong_role(gateway, op, token):
    calls, _ = gateway
    config = client_config(log_prob_gpus=1)
    transport = CorrectnessCortexTransport(config.backend)
    transport.initialize(config)
    before = len(calls)
    with pytest.raises(ValueError, match="owned child|cannot target"):
        transport.call(Request(op, token, {}))
    assert len(calls) == before
    transport.shutdown()
    with pytest.raises(ValueError, match="owned child"):
        transport.call(Request("forward", "owned-parent:training:0", {}))


@pytest.mark.parametrize("status", [302, 400, 500])
def test_native_error_preserves_server_reason_and_cleanup(gateway, monkeypatch, status):
    calls, _ = gateway
    config = client_config()
    transport = CorrectnessCortexTransport(config.backend)
    jobs = transport.initialize(config)
    monkeypatch.setattr(
        transport.session, "post", lambda *args, **kwargs: response({"detail": "actual gateway error"}, status=status)
    )
    with pytest.raises((requests.HTTPError, RuntimeError), match="actual gateway error|without redirects"):
        transport.call(Request("forward", jobs.training, {}))
    transport.shutdown()
    assert calls[-1][1].endswith(":cancel")
    assert transport._native_bindings == {}


def test_native_creation_failure_cleans_parent_and_binding(gateway, monkeypatch):
    calls, _ = gateway
    config = client_config()
    transport = CorrectnessCortexTransport(config.backend)
    monkeypatch.setattr(
        transport, "_wait_running", lambda: (_ for _ in ()).throw(RuntimeError("actual startup error"))
    )
    with pytest.raises(RuntimeError, match="actual startup error"):
        initialize_or_cleanup(transport, config)
    assert "job_id" in created(calls)["sub_job_configs"][0]
    assert calls[-1][1].endswith("/owned-parent:cancel")
    assert transport.job_id is None
    assert transport._native_bindings == transport._native_children == {}


@pytest.mark.parametrize(
    "payload",
    [b"invalid wire", wire.dumps({"job_id": "foreign-child", "logprobs": [0.0]})],
    ids=["invalid-wire", "foreign-identity"],
)
def test_native_malformed_or_wrong_identity_response_rejected(gateway, monkeypatch, payload):
    config = client_config()
    transport = CorrectnessCortexTransport(config.backend)
    jobs = transport.initialize(config)
    monkeypatch.setattr(transport.session, "post", lambda *args, **kwargs: response(data=payload))
    with pytest.raises(ValueError):
        transport.call(Request("forward", jobs.training, {}))
    transport.shutdown()


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:8003", "http://remote.example"])
def test_no_opt_in_leaves_public_cortex_unchanged(gateway, monkeypatch, base_url):
    monkeypatch.delenv(LOCAL_NATIVE_FORWARD_ENV)
    config = client_config(CortexConfig(base_url=base_url))
    transport = CorrectnessCortexTransport(config.backend)
    calls = []

    def initialize(actual, cfg):
        actual._send("POST", actual._prefix, json={"sub_job_configs": cfg.to_cortex()})
        return actual.jobs

    def send(actual, method, url, **kwargs):
        calls.append((method, url, kwargs))
        return {"job_id": "public-parent"}

    from arctic_platform.client.transports.cortex import CortexTransport

    monkeypatch.setattr(CortexTransport, "initialize", initialize)
    monkeypatch.setattr(CortexTransport, "_send", send)
    monkeypatch.setattr(CortexTransport, "call", lambda actual, request: {"public": request.op})
    transport.initialize(config)
    assert len(calls) == 1
    assert all("job_id" not in sub for sub in calls[0][2]["json"]["sub_job_configs"])
    assert transport.call(Request("forward", "public-training", {})) == {"public": "forward"}
    assert transport.call(Request("log-probs", "public-log-prob", {})) == {"public": "log-probs"}


def test_native_async_forward_same_binding_and_wire(gateway, monkeypatch):
    config = client_config()
    transport = CorrectnessCortexTransport(config.backend)
    jobs = transport.initialize(config)
    expected = {"logprobs": torch.tensor([[-0.5]])}
    calls = []

    class Reply:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def read(self):
            return wire.dumps(expected)

    class Session:
        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return Reply()

    async def session():
        return Session()

    monkeypatch.setattr(transport, "_ensure_asession", session)
    result = asyncio.run(transport.acall(Request("forward", jobs.training, {"batch": {"input_ids": [1]}})))
    assert torch.equal(result["logprobs"], expected["logprobs"])
    assert calls[0][0].endswith(f"/forward?job_id={transport._native_children['training']}")
    assert calls[0][1]["allow_redirects"] is False
    transport.shutdown()


@pytest.mark.parametrize("failure", [False, True])
def test_full_model_e2e_existing_replay_routes_held_out_to_owned_native_child(gateway, monkeypatch, tmp_path, failure):
    from arctic_platform.correctness.checks import e2e_sft_train_validate as check
    from arctic_platform.correctness.harness import dss_driver
    from arctic_platform.correctness.harness import gsm8k
    from arctic_platform.correctness.harness.config import load_config

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config")
    ctx = SimpleNamespace(
        cfg=config,
        client_backend=client_config().backend,
        local_gateway_full_model=True,
        local_full_model_path="/data-fast/tmp/test-full-model/hf/hub/snapshots/pinned",
        attn_implementation=config.attention_implementation,
    )
    examples = [
        gsm8k.Example(
            question=f"row-{index}", prompt="q:", completion=" a", input_ids=(10 + index, 30, 40), n_prompt=1
        )
        for index in range(6)
    ]
    replay = check.materialize_replay(
        check.replay_batches(examples, 2, 3, pad_id=0),
        gsm8k.build_batch("validation", examples[:2], pad_id=0),
        tmp_path / "replay",
    )
    _, validation = check.request_bodies(replay, model_provider="huggingface")
    validation["processing"].pop("return_logprobs")
    expected = dss_driver.cortex_training_body(validation)
    original_send = requests.Session.request
    calls, _ = gateway
    replies, backward, steps, forwards = {}, [], [], []

    def send(session, method, url, **kwargs):
        if method == "GET" and "/requests/" in url:
            return response({"status": "REQUEST_STATE_DONE", "result": replies[url.rsplit("/", 1)[-1]]})
        if method == "POST" and url.endswith("/forward-backward"):
            backward.append(wire.loads(wire.decode_byte_chunks([kwargs["data"]], kind="request")))
            result = {"avg_loss": float(len(backward))}
        elif method == "POST" and url.endswith("/step"):
            steps.append(kwargs["json"])
            result = {}
        elif method == "POST" and "/forward?job_id=" in url:
            forwards.append((url, wire.loads(kwargs["data"])))
            if failure:
                return response({"detail": "native forward failed"}, status=400)
            return response(data=wire.dumps({"logprobs": torch.tensor([[1.0, 2.0, 3.0]])}))
        else:
            return original_send(session, method, url, **kwargs)
        identifier = f"request-{len(replies)}"
        replies[identifier] = result
        return response({"request_id": identifier})

    monkeypatch.setattr(requests.Session, "request", send)
    monkeypatch.setattr(dss_driver, "_gateway_runtime", lambda *args, **kwargs: pytest.fail("no new gateway"))
    if failure:
        with pytest.raises(requests.HTTPError, match="native forward failed"):
            check.run_dss_trajectory(ctx, replay, tmp_path / "target")
    else:
        losses, scores = check.run_dss_trajectory(ctx, replay, tmp_path / "target")
        assert losses == [1.0, 2.0, 3.0]
        assert torch.equal(scores, torch.tensor([[1.0, 2.0, 3.0]]))
    payload = created(calls)
    assert len(payload["sub_job_configs"]) == 1
    sub = payload["sub_job_configs"][0]
    assert sub["model_name"] == config.model_name == "Qwen/Qwen3-8B"
    assert sub["training_config"]["n_gpus"] == 4
    assert len(backward) == len(steps) == 3
    assert all(step["learning_rate"] == check.configured_learning_rate(config) for step in steps)
    assert forwards[0][0] == f"http://127.0.0.1:8003/forward?job_id={sub['job_id']}"
    actual = forwards[0][1]
    for key, value in expected["kwargs"].items():
        assert torch.equal(actual["kwargs"][key], value)
    assert actual["context"] == expected["context"]
    assert calls[-1][1].endswith("/owned-parent:cancel")
