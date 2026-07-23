# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for ``POST /tinker/bind``.

The endpoint lives in ``http_server`` (not the Tinker router itself) because
it wires two Arctic ``/initialize``-created jobs into the Tinker adapter.
Tests use ``AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")`` — the same
tokenizer used elsewhere in the tinker_layer suite — so no GPU is touched.
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
def bindable_app():
    """Return the real ``http_server.app`` with just enough state faked
    for ``/tinker/bind`` to run: two jobs registered, colocate=True.

    ``init_tinker_state`` may already be wired from a prior test in the
    same process; clear the sentinel so the 409 guard doesn't trip."""
    from arctic_platform.rl import http_server as srv

    app = srv.app
    app.state.jobs = {
        1: {"job_type": "training", "model_name": "Qwen/Qwen3-0.6B"},
        2: {"job_type": "sampling", "model_name": "Qwen/Qwen3-0.6B"},
    }
    app.state.colocate = True
    app.state.training_gpus = 1
    app.state.sampling_gpus = 1
    if getattr(app.state, "tinker_base_model", None) is not None:
        app.state.tinker_base_model = None
    return app


@pytest.fixture
async def client(bindable_app):
    transport = httpx.ASGITransport(app=bindable_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


BODY = {
    "training_job_id": 1,
    "sampling_job_id": 2,
    "base_model": "Qwen/Qwen3-0.6B",
    "max_prompt_length": 128,
    "max_response_length": 32,
}


async def test_bind_happy_path(client, bindable_app):
    r = await client.post("/tinker/bind", json=BODY)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {
        "training_job_id": 1,
        "sampling_job_id": 2,
        "base_model": "Qwen/Qwen3-0.6B",
        "max_prompt_length": 128,
        "max_response_length": 32,
    }
    # Verify the Tinker verbs now see the bound base_model.
    caps = await client.get("/api/v1/get_server_capabilities")
    assert caps.status_code == 200
    assert caps.json()["supported_models"][0]["model_name"] == "Qwen/Qwen3-0.6B"


async def test_bind_rejects_second_bind(client):
    assert (await client.post("/tinker/bind", json=BODY)).status_code == 200
    r = await client.post("/tinker/bind", json=BODY)
    assert r.status_code == 409
    assert "already bound" in r.json()["detail"]


async def test_bind_unknown_job_404(client):
    r = await client.post("/tinker/bind", json={**BODY, "training_job_id": 99})
    assert r.status_code == 404


async def test_bind_wrong_job_type_400(client):
    # Job 2 is a sampling job — passing it as training_job_id must 400.
    r = await client.post("/tinker/bind", json={**BODY, "training_job_id": 2})
    assert r.status_code == 400
