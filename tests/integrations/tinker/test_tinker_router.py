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

"""Integration tests for the tinker_router FastAPI router.

Each test drives the router through ``httpx.AsyncClient`` with the app-level
Arctic backend mocked (see ``conftest.py::mock_backend``). Runs CPU-only,
in-process, without Ray / DeepSpeed / vLLM.
"""

from __future__ import annotations

import math

import pytest
import tinker
from tinker.proto import request_conv
from tinker.proto import response_conv

from arctic_platform.integrations.tinker.proto_wire import PROTO_CONTENT_TYPE

pytestmark = pytest.mark.asyncio


def _proto_forward_request(loss_fn: str = "cross_entropy") -> bytes:
    tokens = [1, 2, 3]
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": [2, 3, 4],
            "weights": [0.0, 1.0, 1.0],
            "advantages": [0.0, 0.5, 0.5],
            "logprobs": [-1.0, -1.0, -1.0],
        },
    )
    request = tinker.types.ForwardBackwardRequest(
        model_id="main",
        seq_id=1,
        forward_backward_input=tinker.types.ForwardBackwardInput(
            data=[datum],
            loss_fn=loss_fn,
            loss_fn_config={},
        ),
    )
    message = request_conv.forward_backward_request_to_proto(request)
    message.forward_only = True
    return message.SerializeToString()


# ---------------------------------------------------------------------------
# Bootstrap verbs
# ---------------------------------------------------------------------------


async def test_create_session_issues_session_id(client):
    r = await client.post("/api/v1/create_session", json={"tags": ["rl", "smoke"], "sdk_version": "0.42.0"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "create_session"
    assert body["session_id"].startswith("sess-")


async def test_session_heartbeat_no_op(client):
    r = await client.post("/api/v1/session_heartbeat", json={"session_id": "sess-anything"})
    assert r.status_code == 200
    assert r.json() == {}


async def test_client_config_forces_json_path(client):
    r = await client.post("/api/v1/client/config", json={"sdk_version": "0.42.0"})
    assert r.status_code == 200
    body = r.json()
    assert body["proto_write_fwdbwd"] is False
    assert body["proto_compress_fwdbwd"] is False


async def test_auth_token_returns_dummy_jwt(client):
    r = await client.post("/api/v1/auth/token", json={})
    assert r.status_code == 200
    assert r.json() == {"jwt": "tml-dummy"}


async def test_telemetry_no_op(client):
    r = await client.post("/api/v1/telemetry", json={"events": [{"name": "test"}]})
    assert r.status_code == 200
    assert r.json()["status"] == "accepted"


async def test_get_server_capabilities_returns_base_model(client):
    r = await client.get("/api/v1/get_server_capabilities")
    assert r.status_code == 200
    body = r.json()
    assert body["supported_models"] == [{"model_name": "Qwen/Qwen3-8B"}]


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------


async def test_create_model_full_weight_accepted(client):
    r = await client.post(
        "/api/v1/create_model",
        json={
            "session_id": "sess-1",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-8B",
            "lora_config": {"rank": 0},
        },
    )
    assert r.status_code == 200, r.text
    fut = r.json()
    assert fut["type"] == "future"
    assert fut["request_id"] == "0"
    assert fut["model_id"] == "main"


async def test_create_model_no_lora_config_accepted(client):
    r = await client.post(
        "/api/v1/create_model",
        json={
            "session_id": "sess-1",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-8B",
        },
    )
    assert r.status_code == 200, r.text


async def test_create_model_lora_rank_positive_rejected(client):
    r = await client.post(
        "/api/v1/create_model",
        json={
            "session_id": "sess-1",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-8B",
            "lora_config": {"rank": 32},
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "rank=32" in detail
    assert "lora_rank=0" in detail


async def test_create_model_wrong_base_model_rejected(client):
    r = await client.post(
        "/api/v1/create_model",
        json={
            "session_id": "sess-1",
            "model_seq_id": 0,
            "base_model": "meta-llama/Llama-3-8B",  # server was started with Qwen3-8B
            "lora_config": {"rank": 0},
        },
    )
    assert r.status_code == 400
    assert "base_model" in r.json()["detail"]


async def test_get_info_after_create_model(client):
    await client.post(
        "/api/v1/create_model",
        json={
            "session_id": "sess-1",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-8B",
            "lora_config": {"rank": 0},
        },
    )
    r = await client.post("/api/v1/get_info", json={"model_id": "main"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_id"] == "main"
    assert body["model_data"]["base_model"] == "Qwen/Qwen3-8B"


async def test_get_info_missing_model_404(client):
    r = await client.post("/api/v1/get_info", json={"model_id": "nonexistent"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Training verbs
# ---------------------------------------------------------------------------


def _mk_datum_dict(tokens=(1, 2, 3), advantages=(0.5, 0.5, 0.5), logprobs=(-1.0, -1.1, -1.2), mask=(1.0, 1.0, 1.0)):
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": list(tokens)}]},
        "loss_fn_inputs": {
            "advantages": {"dtype": "float32", "data": list(advantages), "shape": [len(advantages)]},
            "logprobs": {"dtype": "float32", "data": list(logprobs), "shape": [len(logprobs)]},
            "mask": {"dtype": "float32", "data": list(mask), "shape": [len(mask)]},
        },
    }


async def test_forward_backward_happy_path(client, mock_backend):
    r = await client.post(
        "/api/v1/forward_backward",
        json={
            "forward_backward_input": {
                "data": [_mk_datum_dict()],
                "loss_fn": "ppo",
                "loss_fn_config": {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1},
            },
            "model_id": "main",
        },
    )
    assert r.status_code == 200, r.text
    fut = r.json()
    assert fut["type"] == "future"

    # Retrieve the future — should resolve immediately in v1.
    r = await client.post("/api/v1/retrieve_future", json={"request_id": fut["request_id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metrics"]["loss:mean"] == 0.5

    # Confirm the backend received a properly-mapped batch.
    call = mock_backend["calls"]["fwd_bwd"][-1]
    assert call["processing"]["loss_fn"] == "verl_grpo"
    assert call["processing"]["ratio_clip"] == (0.9, 1.1)


async def test_forward_backward_importance_sampling(client, mock_backend):
    r = await client.post(
        "/api/v1/forward_backward",
        json={
            "forward_backward_input": {
                "data": [_mk_datum_dict()],
                "loss_fn": "importance_sampling",
            },
            "model_id": "main",
        },
    )
    assert r.status_code == 200
    call = mock_backend["calls"]["fwd_bwd"][-1]
    assert call["processing"]["loss_fn"] == "verl_grpo"
    assert call["processing"]["ratio_clip"] == (0.0, math.inf)
    assert call["batch"]["old_log_probs_shifted"].any()


@pytest.mark.parametrize(
    ("loss_fn", "loss_fn_config"),
    [
        ("ppo", {"kl_coef": 0.01}),
        ("ppo", {"clip_low_threshold": 1.1}),
        ("importance_sampling", {"clip_low_threshold": 0.9}),
    ],
)
async def test_forward_backward_refuses_loss_fn_config_it_cannot_apply(client, mock_backend, loss_fn, loss_fn_config):
    r = await client.post(
        "/api/v1/forward_backward",
        json={
            "forward_backward_input": {
                "data": [_mk_datum_dict()],
                "loss_fn": loss_fn,
                "loss_fn_config": loss_fn_config,
            },
            "model_id": "main",
        },
    )
    assert r.status_code == 400, r.text
    assert not mock_backend["calls"]["fwd_bwd"]


@pytest.mark.parametrize("loss_fn", ["cispo", "dro"])
async def test_forward_backward_unsupported_loss_400(client, loss_fn):
    r = await client.post(
        "/api/v1/forward_backward",
        json={
            "forward_backward_input": {
                "data": [_mk_datum_dict()],
                "loss_fn": loss_fn,
            },
            "model_id": "main",
        },
    )
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert loss_fn in detail
    assert "supported" in detail


async def test_proto_forward_only_returns_logprobs(client, mock_backend):
    r = await client.post(
        "/api/v1/forward_backward",
        content=_proto_forward_request(),
        headers={"Content-Type": PROTO_CONTENT_TYPE},
    )
    assert r.status_code == 200
    fut = r.json()
    r = await client.post(
        "/api/v1/retrieve_future",
        json={"request_id": fut["request_id"]},
        headers={"Accept": PROTO_CONTENT_TYPE},
    )
    assert r.status_code == 200
    output = response_conv.deserialize_forward_backward_output(r.content)
    assert output.loss_fn_output_type == "ArrayRecord"
    assert len(output.loss_fn_outputs) == 1
    assert output.loss_fn_outputs[0]["logprobs"].tolist() == pytest.approx([-1.5, -1.5, -1.5])
    assert len(mock_backend["calls"]["fwd_no_grad"]) == 1


async def test_optim_step_threads_overrides(client, mock_backend):
    r = await client.post(
        "/api/v1/optim_step",
        json={
            "adam_params": {"learning_rate": 5e-5, "beta1": 0.85, "beta2": 0.99, "eps": 1e-8, "weight_decay": 0.01},
            "model_id": "main",
        },
    )
    assert r.status_code == 200
    fut = r.json()
    r = await client.post("/api/v1/retrieve_future", json={"request_id": fut["request_id"]})
    assert r.status_code == 200
    assert r.json()["metrics"]["last_lr:mean"] == pytest.approx(5e-5)

    call = mock_backend["calls"]["step"][-1]
    assert call["lr"] == pytest.approx(5e-5)
    assert call["betas"] == (0.85, 0.99)
    assert call["weight_decay"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Weight sync + sampling
# ---------------------------------------------------------------------------


async def test_save_weights_bumps_gen_and_issues_session_id(client, mock_backend, app):
    assert app.state.tinker_weight_gen == 0
    r = await client.post("/api/v1/save_weights_for_sampler", json={"model_id": "main"})
    assert r.status_code == 200
    fut = r.json()
    r = await client.post("/api/v1/retrieve_future", json={"request_id": fut["request_id"]})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "save_weights_for_sampler"
    assert body["path"] == "tinker://main/sampler_weights/1"
    assert body["sampling_session_id"] == "ss@1"
    assert app.state.tinker_weight_gen == 1
    assert len(mock_backend["calls"]["sync_weights"]) == 1

    # Second call bumps to 2.
    r = await client.post("/api/v1/save_weights_for_sampler", json={"model_id": "main"})
    fut = r.json()
    r = await client.post("/api/v1/retrieve_future", json={"request_id": fut["request_id"]})
    assert r.json()["sampling_session_id"] == "ss@2"


async def test_create_sampling_session_reflects_current_gen(client, app):
    app.state.tinker_weight_gen = 3
    r = await client.post(
        "/api/v1/create_sampling_session",
        json={"session_id": "sess-1", "sampling_session_seq_id": 0, "model_path": "tinker://main/sampler_weights/3"},
    )
    assert r.status_code == 200
    assert r.json()["sampling_session_id"] == "ss@3"


async def test_base_model_session_is_the_untrained_weights(client, app):
    """The sampler holds the base weights only until the first sync, so a
    base-model session must not quietly follow training."""
    app.state.tinker_weight_gen = 3
    r = await client.post(
        "/api/v1/create_sampling_session",
        json={"session_id": "sess-1", "sampling_session_seq_id": 0, "base_model": "Qwen/Qwen3-8B"},
    )
    assert r.status_code == 200
    session_id = r.json()["sampling_session_id"]
    assert session_id == "ss@0"

    for target in ({"sampling_session_id": session_id}, {"base_model": "Qwen/Qwen3-8B"}):
        r = await client.post(
            "/api/v1/asample",
            json={
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
                "sampling_params": {"max_tokens": 4},
                **target,
            },
        )
        assert r.status_code == 409, target


async def test_asample_serves_current_gen(client, mock_backend):
    r = await client.post(
        "/api/v1/asample",
        json={
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
            "sampling_params": {"max_tokens": 4, "temperature": 1.0, "top_p": 0.9},
            "num_samples": 2,
            "sampling_session_id": "ss@0",
        },
    )
    assert r.status_code == 200
    fut = r.json()
    r = await client.post("/api/v1/retrieve_future", json={"request_id": fut["request_id"]})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "sample"
    assert len(body["sequences"]) == 2
    assert body["sequences"][0]["tokens"] == [100, 101, 102, 103]
    assert body["sequences"][0]["stop_reason"] == "stop"
    assert body["sequences"][1]["stop_reason"] == "length"

    # Confirm sampling params flowed through.
    (prompt, sp) = mock_backend["calls"]["generate"][-1]
    assert prompt == [1, 2, 3]
    assert sp["n"] == 2
    assert sp["temperature"] == pytest.approx(1.0)
    # RL loops always want logprobs.
    assert sp["logprobs"] == 1


async def test_asample_stale_snapshot_409(client, app):
    # Advance server-side gen so ss@0 becomes stale.
    app.state.tinker_weight_gen = 2
    r = await client.post(
        "/api/v1/asample",
        json={
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
            "sampling_params": {"max_tokens": 4},
            "num_samples": 1,
            "sampling_session_id": "ss@0",
        },
    )
    # The 409 comes from the future's runner. In v1 execution is inline, so
    # the error surfaces on the submit call itself.
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "stale sampling_session_id" in detail
    assert "E1" in detail


async def test_asample_without_session_id_serves(client):
    """No sampling_session_id → sample against current weights (base_model path)."""
    r = await client.post(
        "/api/v1/asample",
        json={
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
            "sampling_params": {"max_tokens": 4},
            "num_samples": 1,
            "base_model": "Qwen/Qwen3-8B",
        },
    )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Teacher sampling (on-policy distillation)
# ---------------------------------------------------------------------------


async def _teacher_session(client, teacher_model: str) -> str:
    r = await client.post(
        "/api/v1/create_sampling_session",
        json={"session_id": "sess-1", "sampling_session_seq_id": 1, "base_model": teacher_model},
    )
    assert r.status_code == 200, r.text
    return r.json()["sampling_session_id"]


def _logprobs_request(session_id: str, tokens: list[int]) -> dict:
    """What `SamplingClient.compute_logprobs` sends."""
    return {
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "sampling_params": {"max_tokens": 1},
        "num_samples": 1,
        "sampling_session_id": session_id,
        "prompt_logprobs": True,
    }


async def test_teacher_session_scores_on_the_teacher(
    teacher_client, teacher_app, teacher_model, mock_backend, teacher_calls
):
    # The teacher keeps its base weights however far the student has trained.
    teacher_app.state.tinker_weight_gen = 5
    session_id = await _teacher_session(teacher_client, teacher_model)

    r = await teacher_client.post("/api/v1/asample", json=_logprobs_request(session_id, [10, 20, 30]))
    assert r.status_code == 200, r.text
    r = await teacher_client.post("/api/v1/retrieve_future", json={"request_id": r.json()["request_id"]})
    assert r.json()["prompt_logprobs"] == pytest.approx([None, -2.0, -3.0])

    assert mock_backend["calls"]["generate"] == []
    ((prompt, params),) = teacher_calls
    assert prompt == [10, 20, 30]
    assert params["prompt_logprobs"] == 0


async def test_teacher_prompt_logprobs_survive_the_proto_reply(teacher_client, teacher_model):
    session_id = await _teacher_session(teacher_client, teacher_model)
    r = await teacher_client.post("/api/v1/asample", json=_logprobs_request(session_id, [10, 20]))
    r = await teacher_client.post(
        "/api/v1/retrieve_future",
        json={"request_id": r.json()["request_id"]},
        headers={"Accept": PROTO_CONTENT_TYPE},
    )
    assert r.status_code == 200
    reply = response_conv.deserialize_sample_response(r.content)
    assert reply.prompt_logprobs[0] is None
    assert reply.prompt_logprobs[1:] == pytest.approx([-2.0])


async def test_sampler_for_an_unserved_model_is_refused(teacher_client, teacher_model):
    """Answering it from the trained model would score the student against itself."""
    r = await teacher_client.post(
        "/api/v1/create_sampling_session",
        json={"session_id": "sess-1", "sampling_session_seq_id": 1, "base_model": "Qwen/Qwen3-4B"},
    )
    assert r.status_code == 400
    assert teacher_model in r.json()["detail"]

    r = await teacher_client.post(
        "/api/v1/asample",
        json={
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
            "sampling_params": {"max_tokens": 1},
            "base_model": "Qwen/Qwen3-4B",
        },
    )
    assert r.status_code == 400


async def test_teacher_checkpoint_is_refused(teacher_client, teacher_model):
    r = await teacher_client.post(
        "/api/v1/create_sampling_session",
        json={
            "session_id": "sess-1",
            "sampling_session_seq_id": 1,
            "base_model": teacher_model,
            "model_path": "tinker://elsewhere/sampler_weights/final",
        },
    )
    assert r.status_code == 400
    assert "model_path" in r.json()["detail"]


async def test_capabilities_list_the_teacher(teacher_client, teacher_model):
    r = await teacher_client.get("/api/v1/get_server_capabilities")
    assert [m["model_name"] for m in r.json()["supported_models"]] == ["Qwen/Qwen3-8B", teacher_model]


async def test_topk_prompt_logprobs_refused(teacher_client, teacher_model):
    session_id = await _teacher_session(teacher_client, teacher_model)
    r = await teacher_client.post(
        "/api/v1/asample", json={**_logprobs_request(session_id, [1, 2]), "topk_prompt_logprobs": 5}
    )
    assert r.status_code == 400
    assert "topk_prompt_logprobs" in r.json()["detail"]


async def test_prompt_logprobs_the_sampler_did_not_return_fail_loud(client):
    r = await client.post("/api/v1/asample", json=_logprobs_request("ss@0", [1, 2, 3]))
    assert r.status_code == 502
    assert "prompt log-probs" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Futures
# ---------------------------------------------------------------------------


async def test_retrieve_unknown_future_try_again(client):
    r = await client.post("/api/v1/retrieve_future", json={"request_id": "does-not-exist"})
    assert r.status_code == 200
    assert r.json() == {"type": "try_again"}


async def test_future_store_pop_semantics(client):
    """v1 pops on read — a second retrieve returns TryAgainResponse."""
    r = await client.post(
        "/api/v1/create_model",
        json={
            "session_id": "sess-1",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-8B",
            "lora_config": {"rank": 0},
        },
    )
    fut_id = r.json()["request_id"]
    r1 = await client.post("/api/v1/retrieve_future", json={"request_id": fut_id})
    assert r1.json()["type"] == "create_model"
    r2 = await client.post("/api/v1/retrieve_future", json={"request_id": fut_id})
    assert r2.json() == {"type": "try_again"}


# ---------------------------------------------------------------------------
# Custom loss (forward_backward_custom's two passes)
# ---------------------------------------------------------------------------


def _mk_ce_datum_dict(tokens=(1, 2, 3), weights=(0.0, 1.0, 1.0)):
    """The datum shape ``forward_backward_custom`` sends: targets + weights."""
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": list(tokens)}]},
        "loss_fn_inputs": {
            "target_tokens": {"dtype": "int64", "data": list(tokens[1:]) + [tokens[-1] + 1], "shape": [len(tokens)]},
            "weights": {"dtype": "float32", "data": list(weights), "shape": [len(weights)]},
        },
    }


async def test_cross_entropy_accepted_on_forward_backward(client, mock_backend):
    """Pass 2. Previously a 400, which killed forward_backward_custom outright."""
    r = await client.post(
        "/api/v1/forward_backward",
        json={
            "forward_backward_input": {
                "data": [_mk_ce_datum_dict()],
                "loss_fn": "cross_entropy",
            },
            "model_id": "main",
        },
    )
    assert r.status_code == 200, r.text
    batch = mock_backend["calls"]["fwd_bwd"][-1]
    assert batch["processing"]["loss_fn"] == "weighted_logprob_sum"
    assert batch["batch"]["logprob_weights_shifted"].any()


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------


async def test_sample_refuses_temperature_a_backend_cannot_score(client):
    """A backend with no temperature post-processor trains at 1.0 whatever the
    sampler did, so anything else is a silent sampler/trainer mismatch."""
    r = await client.post(
        "/api/v1/asample",
        json={
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
            "num_samples": 2,
            "sampling_params": {"temperature": 0.7, "max_tokens": 4},
        },
    )
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "temperature=1.0" in detail


async def test_sample_allows_unit_temperature(client):
    r = await client.post(
        "/api/v1/asample",
        json={
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
            "num_samples": 1,
            "sampling_params": {"temperature": 1.0, "max_tokens": 4},
        },
    )
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# Misconfiguration
# ---------------------------------------------------------------------------


async def test_unwired_layer_returns_500():
    """When the app has the router mounted but no backend wired,
    calls surface a clear 500 instead of an obscure attribute error."""
    import httpx
    from fastapi import FastAPI

    from arctic_platform.integrations.tinker.router import router as tinker_router

    app = FastAPI()
    app.include_router(tinker_router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/v1/get_server_capabilities")
    assert r.status_code == 500
    assert "init_tinker_state" in r.json()["detail"]
