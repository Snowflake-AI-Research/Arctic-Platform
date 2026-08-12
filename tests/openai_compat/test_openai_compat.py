# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compat HTTP surface: contract tests.

Exercises ``arctic_platform.rl.openai_compat`` in isolation from vLLM /
Ray / GPUs by mounting the router on a fresh FastAPI app with a fake
``ReplicaPool`` and fake tokenizer under ``app.state``. The fake pool
mimics ``ReplicaPool.generate``'s contract (``list[dict]`` with ``text``,
``token_ids``, ``finish_reason``, ``prompt_len``, ``generation_len``).

Covers:

* ``GET /v1/models`` before + after initialize.
* ``POST /v1/chat/completions`` — happy path, ``n>1``, ``max_completion_tokens``,
  ``stop`` list, request-body validation, tokenizer missing / no chat_template.
* ``POST /v1/completions`` — string prompt, token-id prompt, batched prompts.
* ``stream=True`` — SSE frame structure for both chat + text completions.
* 503 when the sampling job hasn't been initialized.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from arctic_platform.openai_compat import router


class _FakeTokenizer:
    """Minimal ``AutoTokenizer`` stand-in that returns a deterministic
    rendering of the message list. Real chat templates are model-specific;
    the router doesn't care about the exact text, only that a string comes
    back."""

    chat_template = "<template>"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True,
    ) -> str:
        assert tokenize is False, "router must render text, not tokenize"
        assert add_generation_prompt is True
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


class _NoTemplateTokenizer:
    chat_template = None  # models with no template

    def apply_chat_template(self, *a, **k):  # pragma: no cover — router shouldn't call this
        raise AssertionError("router must reject before calling apply_chat_template")


class _FakePool:
    """Fake ``ReplicaPool``. Records every call and returns a scripted
    completion. ``_config`` is set to a truthy sentinel so
    ``_get_pool_and_tokenizer`` treats the pool as initialized."""

    def __init__(self, *, text: str = "Hello world", finish: str = "stop") -> None:
        self._config = object()
        self._text = text
        self._finish = finish
        self.calls: list[tuple[list[Any], dict[str, Any]]] = []

    async def generate(
        self,
        prompts: list[Any],
        sampling_params: dict[str, Any] | None = None,
        model_id: str | None = None,
        routing_key: Any = None,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        self.calls.append((list(prompts), dict(sampling_params or {})))
        out = []
        for p in prompts:
            plen = len(p) if isinstance(p, list) else len(str(p).split())
            out.append({
                "text": self._text,
                "token_ids": [1, 2, 3, 4],
                "finish_reason": self._finish,
                "prompt_len": plen,
                "generation_len": 4,
                "prefix_cache_len": 0,
            })
        return out


def _make_app(pool: _FakePool | None, tokenizer: Any = None,
              model_name: str | None = "Qwen/Qwen3-0.6B") -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.sampling_pool = pool if pool is not None else _FakePool()
    app.state.sampling_tokenizer = tokenizer or _FakeTokenizer()
    app.state.sampling_model_name = model_name
    app.state.sampling_created = 12345
    return app


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


def test_models_after_initialize_lists_the_model():
    app = _make_app(_FakePool())
    with TestClient(app) as c:
        resp = c.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == "Qwen/Qwen3-0.6B"
    assert body["data"][0]["object"] == "model"


def test_models_before_initialize_returns_empty_list():
    """Empty list, not 503 — LiteLLM polls /v1/models for capability discovery
    and a 503 causes the client to give up instead of retrying."""
    app = FastAPI()
    app.include_router(router)
    app.state.sampling_pool = None
    app.state.sampling_tokenizer = None
    app.state.sampling_model_name = None
    with TestClient(app) as c:
        resp = c.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json() == {"object": "list", "data": []}


# ---------------------------------------------------------------------------
# /v1/chat/completions — happy paths + parameter mapping
# ---------------------------------------------------------------------------


def _chat_body(**overrides: Any) -> dict[str, Any]:
    base = {
        "model": "hosted_vllm/Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hi."},
        ],
    }
    base.update(overrides)
    return base


def test_chat_completions_happy_path():
    pool = _FakePool(text="hi there")
    app = _make_app(pool)
    with TestClient(app) as c:
        resp = c.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "Qwen/Qwen3-0.6B"
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"] == {"role": "assistant", "content": "hi there"}
    assert choice["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 4
    assert body["id"].startswith("chatcmpl-")

    prompts, params = pool.calls[0]
    assert len(prompts) == 1
    # Chat template was applied by the fake tokenizer.
    assert "user: Say hi." in prompts[0]
    assert prompts[0].endswith("assistant:")
    assert params["n"] == 1


def test_chat_completions_forwards_sampling_params():
    pool = _FakePool()
    app = _make_app(pool)
    with TestClient(app) as c:
        resp = c.post("/v1/chat/completions", json=_chat_body(
            temperature=0.3, top_p=0.9, max_tokens=32, seed=42,
            stop=["\n\n"], presence_penalty=0.1, frequency_penalty=0.2,
        ))
    assert resp.status_code == 200
    _, params = pool.calls[0]
    assert params == {
        "n": 1,
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 32,
        "seed": 42,
        "stop": ["\n\n"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
    }


def test_chat_completions_prefers_max_completion_tokens():
    """OpenAI renamed ``max_tokens`` -> ``max_completion_tokens``; if both are
    sent, the newer name wins."""
    pool = _FakePool()
    app = _make_app(pool)
    with TestClient(app) as c:
        c.post("/v1/chat/completions", json=_chat_body(
            max_tokens=8, max_completion_tokens=64,
        ))
    _, params = pool.calls[0]
    assert params["max_tokens"] == 64


def test_chat_completions_n_greater_than_one():
    """``n=k`` fans out to ``k`` batched prompts so each hits ``generate``
    with ``n=1``. The pool sees ``k`` prompts in one call."""
    pool = _FakePool(text="sample")
    app = _make_app(pool)
    with TestClient(app) as c:
        resp = c.post("/v1/chat/completions", json=_chat_body(n=3))
    body = resp.json()
    assert len(body["choices"]) == 3
    assert {c["index"] for c in body["choices"]} == {0, 1, 2}
    prompts, params = pool.calls[0]
    assert len(prompts) == 3
    assert params["n"] == 1  # never leaked upstream


def test_chat_completions_string_stop_wrapped_in_list():
    pool = _FakePool()
    app = _make_app(pool)
    with TestClient(app) as c:
        c.post("/v1/chat/completions", json=_chat_body(stop="END"))
    _, params = pool.calls[0]
    assert params["stop"] == ["END"]


def test_chat_completions_rejects_missing_messages():
    pool = _FakePool()
    app = _make_app(pool)
    with TestClient(app) as c:
        resp = c.post("/v1/chat/completions", json={"model": "x"})
    assert resp.status_code == 400
    assert "invalid request body" in resp.json()["detail"]


def test_chat_completions_rejects_model_without_chat_template():
    pool = _FakePool()
    app = _make_app(pool, tokenizer=_NoTemplateTokenizer())
    with TestClient(app) as c:
        resp = c.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "chat_template" in detail
    assert "/v1/completions" in detail


def test_chat_completions_before_initialize_returns_503():
    app = FastAPI()
    app.include_router(router)
    app.state.sampling_pool = None
    app.state.sampling_tokenizer = None
    app.state.sampling_model_name = None
    with TestClient(app) as c:
        resp = c.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 503
    assert "not initialized" in resp.json()["detail"]


def test_chat_completions_finish_reason_length_and_abort_mapping():
    """vLLM's ``length`` passes through; ``abort`` maps to ``content_filter``."""
    for vllm_reason, expected in [("length", "length"), ("abort", "content_filter"), (None, "stop")]:
        pool = _FakePool(finish=vllm_reason)
        app = _make_app(pool)
        with TestClient(app) as c:
            resp = c.post("/v1/chat/completions", json=_chat_body())
        assert resp.json()["choices"][0]["finish_reason"] == expected


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------


def test_text_completions_string_prompt():
    pool = _FakePool(text="42")
    app = _make_app(pool)
    with TestClient(app) as c:
        resp = c.post("/v1/completions", json={
            "model": "any", "prompt": "2 + 2 =", "max_tokens": 4,
        })
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "42"
    prompts, params = pool.calls[0]
    assert prompts == ["2 + 2 ="]
    assert params["max_tokens"] == 4


def test_text_completions_token_id_prompt():
    pool = _FakePool()
    app = _make_app(pool)
    with TestClient(app) as c:
        c.post("/v1/completions", json={
            "model": "any", "prompt": [1, 2, 3, 4], "max_tokens": 4,
        })
    prompts, _ = pool.calls[0]
    assert prompts == [[1, 2, 3, 4]]  # list[int] passes through as one prompt


def test_text_completions_batched_string_prompts():
    """OpenAI legacy supports ``prompt: list[str]``. We take the first
    element (documented, since fanning out per-prompt across ``n>1`` gets
    messy in the OpenAI ``index`` scheme)."""
    pool = _FakePool()
    app = _make_app(pool)
    with TestClient(app) as c:
        resp = c.post("/v1/completions", json={
            "model": "any", "prompt": ["hello", "world"], "max_tokens": 2,
        })
    assert resp.status_code == 200
    prompts, _ = pool.calls[0]
    assert prompts == ["hello"]


def test_text_completions_batched_token_id_prompts():
    """``prompt: list[list[int]]`` fans out one call per element."""
    pool = _FakePool()
    app = _make_app(pool)
    with TestClient(app) as c:
        resp = c.post("/v1/completions", json={
            "model": "any", "prompt": [[1, 2], [3, 4, 5]], "max_tokens": 2,
        })
    assert resp.status_code == 200
    assert len(pool.calls) == 2
    assert pool.calls[0][0] == [[1, 2]]
    assert pool.calls[1][0] == [[3, 4, 5]]
    assert len(resp.json()["choices"]) == 2


# ---------------------------------------------------------------------------
# Streaming (SSE)
# ---------------------------------------------------------------------------


def _parse_sse(raw: str) -> list[dict[str, Any] | str]:
    """Parse ``data: ...`` lines. Returns dicts, or the literal string
    ``"[DONE]"`` for the terminator."""
    events: list[dict[str, Any] | str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(payload))
    return events


def test_chat_completions_streaming_structure():
    """First frame has role delta; middle frames have content deltas;
    last non-DONE frame carries ``finish_reason``. Terminated by [DONE]."""
    pool = _FakePool(text="ab" * 40)  # 80 chars -> >1 content chunks
    app = _make_app(pool)
    with TestClient(app) as c:
        resp = c.post("/v1/chat/completions", json=_chat_body(stream=True))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    assert events[-1] == "[DONE]"

    frames = [e for e in events if isinstance(e, dict)]
    assert frames[0]["object"] == "chat.completion.chunk"
    assert frames[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert frames[0]["choices"][0]["finish_reason"] is None

    content_frames = [f for f in frames if "content" in f["choices"][0].get("delta", {})]
    assert content_frames, "expected at least one content delta"
    # Content deltas re-assemble the original text.
    joined = "".join(f["choices"][0]["delta"]["content"] for f in content_frames)
    assert joined == "ab" * 40

    finish_frame = frames[-1]
    assert finish_frame["choices"][0]["finish_reason"] == "stop"
    assert finish_frame["choices"][0]["delta"] == {}


def test_chat_completions_streaming_n_greater_than_one_emits_per_choice():
    pool = _FakePool(text="hi")
    app = _make_app(pool)
    with TestClient(app) as c:
        resp = c.post("/v1/chat/completions", json=_chat_body(stream=True, n=2))
    events = _parse_sse(resp.text)
    frames = [e for e in events if isinstance(e, dict)]
    indices = {f["choices"][0]["index"] for f in frames}
    assert indices == {0, 1}


def test_text_completions_streaming_structure():
    pool = _FakePool(text="hello")
    app = _make_app(pool)
    with TestClient(app) as c:
        resp = c.post("/v1/completions", json={
            "model": "any", "prompt": "hi", "stream": True,
        })
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[-1] == "[DONE]"
    frames = [e for e in events if isinstance(e, dict)]
    assert all(f["object"] == "text_completion" for f in frames)
    joined = "".join(f["choices"][0]["text"] for f in frames if f["choices"][0]["finish_reason"] is None)
    assert joined == "hello"
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Content-type + extras (unknown fields tolerated, per OpenAI compatibility)
# ---------------------------------------------------------------------------


def test_chat_completions_tolerates_unknown_openai_fields():
    """Newer OpenAI SDK versions add fields we don't implement (e.g.
    ``response_format``, ``tools``). Accept + ignore rather than 422."""
    pool = _FakePool()
    app = _make_app(pool)
    with TestClient(app) as c:
        resp = c.post("/v1/chat/completions", json=_chat_body(
            response_format={"type": "text"},
            tools=[{"type": "function", "function": {"name": "f"}}],
            parallel_tool_calls=False,
        ))
    assert resp.status_code == 200
