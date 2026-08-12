# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible HTTP surface for the sampling sub-job.

The sampling sub-job already runs vLLM's ``AsyncLLM`` in-process via
``arctic_inference.server.replica_pool.ReplicaPool``. This module exposes
a thin OpenAI-compatible HTTP router (``/v1/models``,
``/v1/chat/completions``, ``/v1/completions``) that translates OpenAI
requests to ``ReplicaPool.generate()`` calls and formats the results into
OpenAI's response shape. No vLLM HTTP subprocess is spawned; scheduling,
prefix caching, and tensor-parallel routing all continue to go through
``ReplicaPool``.

Mounted by ``arctic_platform.rl.http_server`` alongside the RL-shaped
``/generate`` route. Lives outside ``arctic_platform.rl`` so it can be
imported by tests without dragging in the training kernel (tensordict /
Ray / DeepSpeed).

Design notes:

* Zero coupling to the RL wire (no ``arctic_platform.wire``, no
  ``GenerateRequest``). The router only talks OpenAI-shaped JSON.
* Sits alongside ``/generate``: the RL-shaped path keeps working
  byte-for-byte for training frameworks that already speak it (SkyRL,
  verl, our own ``CortexRLAgent``).
* Streaming (``stream=True``) is implemented by running the full
  generation and replaying it as SSE deltas. Client contract (SSE frames
  in OpenAI's shape) is preserved; first-token latency is not.
  Incremental streaming needs a delta-yielding surface on
  ``ReplicaPool``.
* Chat template rendering uses the tokenizer loaded at ``/initialize``
  time (``app.state.sampling_tokenizer``). Models without a chat
  template return HTTP 400.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any
from typing import AsyncIterator
from typing import Iterable
from typing import Literal

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/v1", tags=["openai-compat"])


# ---------------------------------------------------------------------------
# Wire schemas — a strict subset of OpenAI's shape that our translation layer
# actually reads / writes. Unknown fields are accepted (``model_config`` sets
# ``extra="allow"``) so future OpenAI additions don't 422 the request; we
# just don't act on them.
# ---------------------------------------------------------------------------


class _AllowExtra(BaseModel):
    model_config = {"extra": "allow"}


class ChatMessage(_AllowExtra):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(_AllowExtra):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    user: str | None = None


class CompletionRequest(_AllowExtra):
    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    max_tokens: int | None = 16
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    logprobs: int | None = None
    echo: bool = False
    user: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_pool_and_tokenizer(request: Request) -> tuple[Any, Any, str]:
    """Resolve ``(sampling_pool, tokenizer, model_name)`` from ``app.state``.

    A sampling job must have completed ``/initialize`` first (which sets
    ``app.state.sampling_pool._config`` and ``app.state.sampling_tokenizer``).
    We surface a specific 503 instead of a generic 500 so clients can retry
    against a still-warming endpoint.
    """
    state = request.app.state
    pool = getattr(state, "sampling_pool", None)
    tokenizer = getattr(state, "sampling_tokenizer", None)
    model_name = getattr(state, "sampling_model_name", None)
    if pool is None or getattr(pool, "_config", None) is None:
        raise HTTPException(
            status_code=503,
            detail="Sampling job is not initialized. POST /initialize with "
                   "job_type='sampling' first, or wait for it to finish "
                   "warming up.",
        )
    if tokenizer is None or model_name is None:
        raise HTTPException(
            status_code=503,
            detail="Sampling tokenizer is not loaded yet.",
        )
    return pool, tokenizer, model_name


def _to_sampling_params(
    *,
    n: int,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    stop: str | list[str] | None,
    presence_penalty: float | None,
    frequency_penalty: float | None,
    seed: int | None,
    logprobs_topk: int | None = None,
) -> dict[str, Any]:
    """Map OpenAI-shaped fields onto vLLM ``SamplingParams`` kwargs.

    Only fields the client actually provided are forwarded, so we don't
    override vLLM defaults (e.g. temperature=1.0) when the caller omitted
    them. ``top_k`` is a vLLM extension over the OpenAI spec — accepted if
    the client sent it, ignored otherwise.
    """
    params: dict[str, Any] = {"n": max(1, int(n))}
    if max_tokens is not None:
        params["max_tokens"] = int(max_tokens)
    if temperature is not None:
        params["temperature"] = float(temperature)
    if top_p is not None:
        params["top_p"] = float(top_p)
    if top_k is not None:
        params["top_k"] = int(top_k)
    if stop is not None:
        params["stop"] = [stop] if isinstance(stop, str) else list(stop)
    if presence_penalty is not None:
        params["presence_penalty"] = float(presence_penalty)
    if frequency_penalty is not None:
        params["frequency_penalty"] = float(frequency_penalty)
    if seed is not None:
        params["seed"] = int(seed)
    if logprobs_topk is not None and logprobs_topk > 0:
        params["logprobs"] = int(logprobs_topk)
    return params


def _render_chat_prompt(tokenizer: Any, messages: list[ChatMessage]) -> str:
    """Apply the tokenizer's chat template.

    Rejects with 400 rather than silently falling back to a hand-rolled
    role concatenation, because the wrong template produces plausible-
    looking but subtly broken generations (misaligned system prompt,
    missing generation prefix) — the exact class of bug that's hardest to
    diagnose in an RL loop.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Loaded tokenizer has no chat_template; /v1/chat/completions "
                "requires an instruct/chat model. Use /v1/completions for "
                "raw-text prompts, or load a model whose tokenizer defines "
                "a chat_template."
            ),
        )
    payload = [
        {"role": m.role, "content": m.content if isinstance(m.content, str) else json.dumps(m.content)}
        for m in messages
    ]
    try:
        return tokenizer.apply_chat_template(
            payload, tokenize=False, add_generation_prompt=True,
        )
    except Exception as exc:  # noqa: BLE001 — surface template rendering errors
        raise HTTPException(
            status_code=400,
            detail=f"Chat template failed to render: {type(exc).__name__}: {exc}",
        ) from exc


def _finish_reason_to_openai(reason: str | None) -> str:
    """vLLM emits ``stop``/``length``/``abort``; OpenAI expects
    ``stop``/``length``/``content_filter``/``tool_calls``. Map best-effort
    and fall back to ``stop``.
    """
    if reason in ("stop", "length", "tool_calls", "content_filter"):
        return reason
    if reason == "abort":
        return "content_filter"
    return "stop"


async def _generate_n(
    pool: Any,
    prompt: str | list[int],
    sampling_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Call ``ReplicaPool.generate`` and normalize to a list of ``n`` results.

    vLLM's ``SamplingParams(n=k)`` returns a *single* ``RequestOutput`` with
    ``k`` sub-outputs, but ``InferenceWorker.generate`` only surfaces
    ``outputs[0]``. To make ``n>1`` behave, we invoke the pool ``n`` times
    with ``n=1`` per call — this is O(n) requests but every request goes
    through the same batching+prefix-cache path, so the pool amortizes.
    """
    n = int(sampling_params.get("n", 1))
    if n <= 1:
        params = dict(sampling_params)
        params["n"] = 1
        results = await pool.generate([prompt], params)
        return list(results)
    per_call_params = dict(sampling_params)
    per_call_params["n"] = 1
    # Re-issue as a single batch — the scheduler dedups the prompt via
    # prefix cache so the marginal cost of an extra sample is ~KV only.
    prompts = [prompt] * n
    results = await pool.generate(prompts, per_call_params)
    return list(results)


def _usage_from_results(results: Iterable[dict[str, Any]]) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    for r in results:
        prompt_tokens = max(prompt_tokens, int(r.get("prompt_len") or 0))
        completion_tokens += int(r.get("generation_len") or len(r.get("token_ids") or []))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _now() -> int:
    return int(time.time())


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/models")
async def list_models(request: Request) -> dict[str, Any]:
    """OpenAI ``/v1/models``. Returns the single sampling model.

    Clients like LiteLLM occasionally call this for capability discovery;
    returning a well-formed empty list before initialize is more useful
    than a 503, since the client can then retry.
    """
    state = request.app.state
    model_name = getattr(state, "sampling_model_name", None)
    data: list[dict[str, Any]] = []
    if model_name:
        data.append({
            "id": model_name,
            "object": "model",
            "created": getattr(state, "sampling_created", _now()),
            "owned_by": "arctic-cortex",
        })
    return {"object": "list", "data": data}


@router.post("/chat/completions")
async def chat_completions(request: Request) -> Any:
    payload = await request.json()
    try:
        req = ChatCompletionRequest.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid request body: {exc}") from exc

    pool, tokenizer, model_name = _get_pool_and_tokenizer(request)
    prompt_text = _render_chat_prompt(tokenizer, req.messages)

    # OpenAI renamed ``max_tokens`` to ``max_completion_tokens`` — accept
    # both, prefer the newer field if both are set.
    max_tokens = req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens

    sampling_params = _to_sampling_params(
        n=req.n,
        max_tokens=max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        stop=req.stop,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        seed=req.seed,
        logprobs_topk=req.top_logprobs if req.logprobs else None,
    )

    results = await _generate_n(pool, prompt_text, sampling_params)
    completion_id = _new_id("chatcmpl")
    created = _now()

    if req.stream:
        return StreamingResponse(
            _stream_chat_completion(
                completion_id=completion_id,
                created=created,
                model_name=model_name,
                results=results,
                include_logprobs=bool(req.logprobs),
            ),
            media_type="text/event-stream",
        )

    choices = []
    for idx, r in enumerate(results):
        choice: dict[str, Any] = {
            "index": idx,
            "message": {"role": "assistant", "content": r.get("text", "")},
            "finish_reason": _finish_reason_to_openai(r.get("finish_reason")),
        }
        if req.logprobs and r.get("logprobs") is not None:
            choice["logprobs"] = {"content": _format_chat_logprobs(r["logprobs"])}
        choices.append(choice)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": choices,
        "usage": _usage_from_results(results),
    }


@router.post("/completions")
async def completions(request: Request) -> Any:
    payload = await request.json()
    try:
        req = CompletionRequest.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid request body: {exc}") from exc

    pool, _tokenizer, model_name = _get_pool_and_tokenizer(request)

    # OpenAI's /v1/completions is single-prompt for chat models but the
    # legacy contract allows either a raw string or a batched list. We
    # take the first prompt; batched completions on one call would need a
    # per-choice `index` scheme that most agent clients don't use anyway.
    prompt = req.prompt
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
        # list[str] batched form — pick the first, warn via header
        prompt = prompt[0]
    # list[int] and list[list[int]] fall through to vLLM directly.

    sampling_params = _to_sampling_params(
        n=req.n,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        stop=req.stop,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        seed=req.seed,
        logprobs_topk=req.logprobs,
    )

    if isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
        # list[list[int]] — batched token-id prompts. We fan out one call
        # per element so each keeps its own ``n`` samples.
        all_results: list[list[dict[str, Any]]] = []
        for p in prompt:
            all_results.append(await _generate_n(pool, list(p), sampling_params))
        flat_results: list[dict[str, Any]] = [r for group in all_results for r in group]
    else:
        flat_results = await _generate_n(pool, prompt, sampling_params)

    completion_id = _new_id("cmpl")
    created = _now()

    if req.stream:
        return StreamingResponse(
            _stream_text_completion(
                completion_id=completion_id,
                created=created,
                model_name=model_name,
                results=flat_results,
            ),
            media_type="text/event-stream",
        )

    choices = []
    for idx, r in enumerate(flat_results):
        choices.append({
            "index": idx,
            "text": r.get("text", ""),
            "finish_reason": _finish_reason_to_openai(r.get("finish_reason")),
            "logprobs": None,
        })

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": choices,
        "usage": _usage_from_results(flat_results),
    }


# ---------------------------------------------------------------------------
# Streaming (SSE)
# ---------------------------------------------------------------------------
#
# vLLM's own OpenAI server yields token-by-token deltas because the engine
# yields incremental ``RequestOutput`` objects. Our pool surface only
# returns the final ``dict``, so we synthesize a chunked SSE stream from
# the completed generation. That still preserves the wire contract:
# clients receive a series of ``data: {...}\n\n`` frames terminated by
# ``data: [DONE]\n\n``. Rolling this out to true streaming is a
# scheduler-side change — see PLAN.md.


_SSE_CHUNK_SIZE_CHARS = 64


def _chunk_text(text: str, size: int) -> list[str]:
    if not text:
        return [""]
    return [text[i : i + size] for i in range(0, len(text), size)]


async def _stream_chat_completion(
    *,
    completion_id: str,
    created: int,
    model_name: str,
    results: list[dict[str, Any]],
    include_logprobs: bool = False,
) -> AsyncIterator[bytes]:
    for idx, r in enumerate(results):
        text = r.get("text", "") or ""
        # 1) role delta first (OpenAI clients expect the assistant role
        #    on the first chunk of each choice).
        role_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": idx, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield _sse(role_chunk)

        # 2) content deltas.
        for piece in _chunk_text(text, _SSE_CHUNK_SIZE_CHARS):
            content_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": idx, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield _sse(content_chunk)

        # 3) terminal chunk with finish_reason.
        finish_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": idx,
                "delta": {},
                "finish_reason": _finish_reason_to_openai(r.get("finish_reason")),
            }],
        }
        yield _sse(finish_chunk)

    yield b"data: [DONE]\n\n"


async def _stream_text_completion(
    *,
    completion_id: str,
    created: int,
    model_name: str,
    results: list[dict[str, Any]],
) -> AsyncIterator[bytes]:
    for idx, r in enumerate(results):
        text = r.get("text", "") or ""
        for piece in _chunk_text(text, _SSE_CHUNK_SIZE_CHARS):
            chunk = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": idx,
                    "text": piece,
                    "finish_reason": None,
                    "logprobs": None,
                }],
            }
            yield _sse(chunk)
        final = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": idx,
                "text": "",
                "finish_reason": _finish_reason_to_openai(r.get("finish_reason")),
                "logprobs": None,
            }],
        }
        yield _sse(final)
    yield b"data: [DONE]\n\n"


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _format_chat_logprobs(vllm_logprobs: list[Any]) -> list[dict[str, Any]]:
    """Convert vLLM's per-position logprob dict into OpenAI chat-shape.

    OpenAI's schema is:

        [{"token": "...", "logprob": -0.1, "bytes": [...],
          "top_logprobs": [{"token": "...", "logprob": -0.2, "bytes": [...]}, ...]}]

    vLLM's ``choice.logprobs`` (via the worker's
    ``_serialize_logprobs_position``) is a list of per-position dicts
    keyed by token id. We surface the sampled token's logprob and (if
    present) the top-K alternatives.
    """
    out: list[dict[str, Any]] = []
    for pos in vllm_logprobs or []:
        if not isinstance(pos, dict) or not pos:
            continue
        # Pick the sampled entry: the one with rank==1, or the highest logprob.
        sampled_key = None
        sampled = None
        for k, v in pos.items():
            if isinstance(v, dict) and v.get("rank") == 1:
                sampled_key, sampled = k, v
                break
        if sampled is None:
            sampled_key, sampled = max(
                pos.items(),
                key=lambda kv: (kv[1].get("logprob", float("-inf")) if isinstance(kv[1], dict) else float("-inf")),
            )
        top: list[dict[str, Any]] = []
        for k, v in pos.items():
            if not isinstance(v, dict):
                continue
            top.append({
                "token": v.get("decoded_token") or str(k),
                "logprob": v.get("logprob"),
                "bytes": None,
            })
        out.append({
            "token": sampled.get("decoded_token") if isinstance(sampled, dict) else str(sampled_key),
            "logprob": sampled.get("logprob") if isinstance(sampled, dict) else None,
            "bytes": None,
            "top_logprobs": top,
        })
    return out
