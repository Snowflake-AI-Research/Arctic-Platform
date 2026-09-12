# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Shared fixtures for the Tinker HTTP layer tests. Every fixture wires the
# router against a mocked backend so the entire test module can run
# CPU-only, in-process, with no dependency on Ray / DeepSpeed / vLLM.

from __future__ import annotations

import inspect
from typing import Any

import pytest
import pytest_asyncio


def pytest_collection_modifyitems(items) -> None:
    """Mark this directory's coroutine tests for pytest-asyncio.

    They are written in ``asyncio_mode=auto`` style. Setting that mode in the
    root ``pyproject.toml`` would change collection for the whole repo, so the
    marker is applied here instead.
    """
    for item in items:
        if inspect.iscoroutinefunction(getattr(item, "function", None)):
            item.add_marker(pytest.mark.asyncio)


@pytest.fixture
def mock_backend() -> dict[str, Any]:
    """Track calls made to Arctic handlers so assertions can inspect them."""
    calls: dict[str, list[Any]] = {
        "fwd_bwd": [],
        "fwd_no_grad": [],
        "step": [],
        "sync_weights": [],
        "generate": [],
    }

    async def fwd_bwd_handler(batch):
        calls["fwd_bwd"].append(batch)
        return {
            "job_id": 1,
            "avg_loss": 0.5,
            "metrics": {"loss": 0.5, "grad_norm": 1.0, "kl": 0.01},
        }

    async def fwd_no_grad_handler(batch):
        calls["fwd_no_grad"].append(batch)
        import numpy as np

        bsz = batch["batch"]["input_ids"].shape[0]
        seqlen = batch["batch"]["input_ids"].shape[1]
        return {
            "job_id": 1,
            "batch": {"logprobs": np.full((bsz, seqlen), -1.5, dtype=np.float32)},
            "metrics": {"tokens": float(bsz * seqlen)},
        }

    async def step_handler(overrides):
        calls["step"].append(overrides)
        return {
            "job_id": 1,
            "metrics": {"last_lr": overrides["lr"] if overrides else 1e-4,
                        "grad_norm": 0.9},
            "batch": {},
        }

    async def sync_weights_handler():
        calls["sync_weights"].append(True)
        return {"ok": True}

    async def generate_handler(prompt_tokens, sampling_params):
        calls["generate"].append((list(prompt_tokens), dict(sampling_params)))
        n = sampling_params.get("n", 1)
        max_tokens = sampling_params.get("max_tokens", 4)
        return {
            "outputs": [
                {
                    # Deterministic mock rollouts: tokens 100, 101, 102, ...
                    "token_ids": list(range(100, 100 + max_tokens)),
                    "logprobs": [-0.5] * max_tokens,
                    "finish_reason": "stop" if i == 0 else "length",
                }
                for i in range(n)
            ]
        }

    return {"calls": calls, "handlers": dict(
        fwd_bwd_handler=fwd_bwd_handler,
        fwd_no_grad_handler=fwd_no_grad_handler,
        step_handler=step_handler,
        sync_weights_handler=sync_weights_handler,
        generate_handler=generate_handler,
    )}


@pytest.fixture
def app(mock_backend):
    """Build a FastAPI app with only the Tinker router mounted + backend wired."""
    from fastapi import FastAPI

    from arctic_platform.integrations.tinker.router import init_tinker_state
    from arctic_platform.integrations.tinker.router import router as tinker_router

    app = FastAPI()
    app.include_router(tinker_router)
    init_tinker_state(
        app,
        base_model="Qwen/Qwen3-8B",
        max_prompt_length=16,
        max_response_length=8,
        pad_token_id=0,
        **mock_backend["handlers"],
    )
    return app


@pytest_asyncio.fixture
async def client(app):
    """Async httpx client rooted at the test app."""
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
