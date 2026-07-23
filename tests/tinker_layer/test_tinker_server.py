# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the tinker_server FastAPI router.

Each test drives the router through ``httpx.AsyncClient`` with the app-level
Arctic backend mocked (see ``conftest.py::mock_backend``). Runs CPU-only,
in-process, without Ray / DeepSpeed / vLLM.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Bootstrap verbs
# ---------------------------------------------------------------------------


async def test_create_session_issues_session_id(client):
    r = await client.post("/api/v1/create_session",
                          json={"tags": ["rl", "smoke"],
                                "sdk_version": "0.42.0"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "create_session"
    assert body["session_id"].startswith("sess-")


async def test_session_heartbeat_no_op(client):
    r = await client.post("/api/v1/session_heartbeat",
                          json={"session_id": "sess-anything"})
    assert r.status_code == 200
    assert r.json() == {}


async def test_client_config_forces_json_path(client):
    r = await client.post("/api/v1/client/config",
                          json={"sdk_version": "0.42.0"})
    assert r.status_code == 200
    body = r.json()
    assert body["proto_write_fwdbwd"] is False
    assert body["proto_compress_fwdbwd"] is False


async def test_auth_token_returns_dummy_jwt(client):
    r = await client.post("/api/v1/auth/token", json={})
    assert r.status_code == 200
    assert r.json() == {"jwt": "tml-dummy"}


async def test_telemetry_no_op(client):
    r = await client.post("/api/v1/telemetry",
                          json={"events": [{"name": "test"}]})
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
    r = await client.post("/api/v1/create_model", json={
        "session_id": "sess-1",
        "model_seq_id": 0,
        "base_model": "Qwen/Qwen3-8B",
        "lora_config": {"rank": 0},
    })
    assert r.status_code == 200, r.text
    fut = r.json()
    assert fut["type"] == "future"
    assert fut["request_id"] == "0"
    assert fut["model_id"] == "main"


async def test_create_model_no_lora_config_accepted(client):
    r = await client.post("/api/v1/create_model", json={
        "session_id": "sess-1",
        "model_seq_id": 0,
        "base_model": "Qwen/Qwen3-8B",
    })
    assert r.status_code == 200, r.text


async def test_create_model_lora_rank_positive_rejected(client):
    r = await client.post("/api/v1/create_model", json={
        "session_id": "sess-1",
        "model_seq_id": 0,
        "base_model": "Qwen/Qwen3-8B",
        "lora_config": {"rank": 32},
    })
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "LoraConfig(rank=0)" in detail
    assert "SkyRL-tx" in detail


async def test_create_model_wrong_base_model_rejected(client):
    r = await client.post("/api/v1/create_model", json={
        "session_id": "sess-1",
        "model_seq_id": 0,
        "base_model": "meta-llama/Llama-3-8B",  # server was started with Qwen3-8B
        "lora_config": {"rank": 0},
    })
    assert r.status_code == 400
    assert "base_model" in r.json()["detail"]


async def test_get_info_after_create_model(client):
    await client.post("/api/v1/create_model", json={
        "session_id": "sess-1",
        "model_seq_id": 0,
        "base_model": "Qwen/Qwen3-8B",
        "lora_config": {"rank": 0},
    })
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


def _mk_datum_dict(tokens=(1, 2, 3),
                   advantages=(0.5, 0.5, 0.5),
                   logprobs=(-1.0, -1.1, -1.2),
                   mask=(1.0, 1.0, 1.0)):
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": list(tokens)}]},
        "loss_fn_inputs": {
            "advantages": {"dtype": "float32", "data": list(advantages), "shape": [len(advantages)]},
            "logprobs": {"dtype": "float32", "data": list(logprobs), "shape": [len(logprobs)]},
            "mask": {"dtype": "float32", "data": list(mask), "shape": [len(mask)]},
        },
    }


async def test_forward_backward_happy_path(client, mock_backend):
    r = await client.post("/api/v1/forward_backward", json={
        "forward_backward_input": {
            "data": [_mk_datum_dict()],
            "loss_fn": "ppo",
            "loss_fn_config": {"clip_low_threshold": 0.9,
                               "clip_high_threshold": 1.1,
                               "kl_coef": 0.01},
        },
        "model_id": "main",
    })
    assert r.status_code == 200, r.text
    fut = r.json()
    assert fut["type"] == "future"

    # Retrieve the future — should resolve immediately in v1.
    r = await client.post("/api/v1/retrieve_future",
                          json={"request_id": fut["request_id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metrics"]["loss:mean"] == 0.5

    # Confirm the backend received a properly-mapped batch.
    call = mock_backend["calls"]["fwd_bwd"][-1]
    assert call["processing"]["loss_fn"] == "verl_grpo"
    actor_cfg = call["meta"]["actor_config"]
    assert actor_cfg["eps_clip"] == pytest.approx(0.1)
    assert actor_cfg["kl_loss_coef"] == pytest.approx(0.01)
    assert actor_cfg["use_kl_loss"] is True


async def test_forward_backward_importance_sampling(client, mock_backend):
    r = await client.post("/api/v1/forward_backward", json={
        "forward_backward_input": {
            "data": [_mk_datum_dict()],
            "loss_fn": "importance_sampling",
        },
        "model_id": "main",
    })
    assert r.status_code == 200
    call = mock_backend["calls"]["fwd_bwd"][-1]
    assert call["meta"]["actor_config"]["eps_clip"] > 1e6


@pytest.mark.parametrize("loss_fn", ["cross_entropy", "cispo", "dro"])
async def test_forward_backward_unsupported_loss_400(client, loss_fn):
    r = await client.post("/api/v1/forward_backward", json={
        "forward_backward_input": {
            "data": [_mk_datum_dict()],
            "loss_fn": loss_fn,
        },
        "model_id": "main",
    })
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert loss_fn in detail
    assert "supported" in detail


async def test_forward_only_returns_logprobs(client, mock_backend):
    r = await client.post("/api/v1/forward", json={
        "forward_input": {
            "data": [_mk_datum_dict()],
            "loss_fn": "ppo",
        },
        "model_id": "main",
    })
    assert r.status_code == 200
    fut = r.json()
    r = await client.post("/api/v1/retrieve_future",
                          json={"request_id": fut["request_id"]})
    assert r.status_code == 200
    body = r.json()
    assert body["loss_fn_output_type"] == "ArrayRecord"
    assert len(body["loss_fn_outputs"]) == 1
    logprobs_td = body["loss_fn_outputs"][0]["logprobs"]
    assert logprobs_td["dtype"] == "float32"
    # Mock returns -1.5 everywhere.
    assert logprobs_td["data"][0] == pytest.approx(-1.5)

    # Confirm forward_only=True was threaded through.
    call = mock_backend["calls"]["fwd_no_grad"][-1]
    assert call["meta"]["forward_only"] is True


async def test_optim_step_threads_overrides(client, mock_backend):
    r = await client.post("/api/v1/optim_step", json={
        "adam_params": {"learning_rate": 5e-5, "beta1": 0.85,
                        "beta2": 0.99, "eps": 1e-8, "weight_decay": 0.01},
        "model_id": "main",
    })
    assert r.status_code == 200
    fut = r.json()
    r = await client.post("/api/v1/retrieve_future",
                          json={"request_id": fut["request_id"]})
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
    r = await client.post("/api/v1/save_weights_for_sampler",
                          json={"model_id": "main"})
    assert r.status_code == 200
    fut = r.json()
    r = await client.post("/api/v1/retrieve_future",
                          json={"request_id": fut["request_id"]})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "save_weights_for_sampler"
    assert body["path"] == "tinker://main/sampler_weights/1"
    assert body["sampling_session_id"] == "ss@1"
    assert app.state.tinker_weight_gen == 1
    assert len(mock_backend["calls"]["sync_weights"]) == 1

    # Second call bumps to 2.
    r = await client.post("/api/v1/save_weights_for_sampler",
                          json={"model_id": "main"})
    fut = r.json()
    r = await client.post("/api/v1/retrieve_future",
                          json={"request_id": fut["request_id"]})
    assert r.json()["sampling_session_id"] == "ss@2"


async def test_create_sampling_session_reflects_current_gen(client, app):
    app.state.tinker_weight_gen = 3
    r = await client.post("/api/v1/create_sampling_session",
                          json={"session_id": "sess-1", "sampling_session_seq_id": 0,
                                "base_model": "Qwen/Qwen3-8B"})
    assert r.status_code == 200
    assert r.json()["sampling_session_id"] == "ss@3"


async def test_asample_serves_current_gen(client, mock_backend):
    r = await client.post("/api/v1/asample", json={
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
        "sampling_params": {"max_tokens": 4, "temperature": 0.7, "top_p": 0.9},
        "num_samples": 2,
        "sampling_session_id": "ss@0",
    })
    assert r.status_code == 200
    fut = r.json()
    r = await client.post("/api/v1/retrieve_future",
                          json={"request_id": fut["request_id"]})
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
    assert sp["temperature"] == pytest.approx(0.7)
    # RL loops always want logprobs.
    assert sp["logprobs"] == 1


async def test_asample_stale_snapshot_409(client, app):
    # Advance server-side gen so ss@0 becomes stale.
    app.state.tinker_weight_gen = 2
    r = await client.post("/api/v1/asample", json={
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
        "sampling_params": {"max_tokens": 4},
        "num_samples": 1,
        "sampling_session_id": "ss@0",
    })
    # The 409 comes from the future's runner. In v1 execution is inline, so
    # the error surfaces on the submit call itself.
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "stale sampling_session_id" in detail
    assert "E1" in detail


async def test_asample_without_session_id_serves(client):
    """No sampling_session_id → sample against current weights (base_model path)."""
    r = await client.post("/api/v1/asample", json={
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
        "sampling_params": {"max_tokens": 4},
        "num_samples": 1,
        "base_model": "Qwen/Qwen3-8B",
    })
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Futures
# ---------------------------------------------------------------------------


async def test_retrieve_unknown_future_try_again(client):
    r = await client.post("/api/v1/retrieve_future",
                          json={"request_id": "does-not-exist"})
    assert r.status_code == 200
    assert r.json() == {"type": "try_again"}


async def test_future_store_pop_semantics(client):
    """v1 pops on read — a second retrieve returns TryAgainResponse."""
    r = await client.post("/api/v1/create_model", json={
        "session_id": "sess-1",
        "model_seq_id": 0,
        "base_model": "Qwen/Qwen3-8B",
        "lora_config": {"rank": 0},
    })
    fut_id = r.json()["request_id"]
    r1 = await client.post("/api/v1/retrieve_future",
                           json={"request_id": fut_id})
    assert r1.json()["type"] == "create_model"
    r2 = await client.post("/api/v1/retrieve_future",
                           json={"request_id": fut_id})
    assert r2.json() == {"type": "try_again"}


# ---------------------------------------------------------------------------
# Misconfiguration
# ---------------------------------------------------------------------------


async def test_unwired_layer_returns_500():
    """When the app has the router mounted but no backend wired,
    calls surface a clear 500 instead of an obscure attribute error."""
    from fastapi import FastAPI
    import httpx

    from arctic_platform.rl.tinker_server import router as tinker_router

    app = FastAPI()
    app.include_router(tinker_router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/v1/get_server_capabilities")
    assert r.status_code == 500
    assert "init_tinker_state" in r.json()["detail"]
