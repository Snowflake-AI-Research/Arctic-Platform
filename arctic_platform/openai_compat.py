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
import re
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
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None


class ChatCompletionRequest(_AllowExtra):
    model: str
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
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


def _tokenize_for_return_ids(tokenizer: Any, prompt: str | list[int]) -> list[int]:
    """Best-effort ``prompt_token_ids`` for a rendered prompt.

    Harbor's ``LiteLLM._extract_token_ids`` (and any other client that
    consumes vLLM's ``return_token_ids`` extension) reads
    ``response.prompt_token_ids`` — a flat token id list. vLLM's OpenAI
    server derives it by tokenizing the rendered prompt with the
    engine's tokenizer. We do the same here so the round-trip matches
    vLLM's own OpenAI surface byte-for-byte for downstream RL.

    If tokenization fails (tokenizer is a mock in a unit test, or the
    prompt is already token ids), fall back to what we already have:
    integer prompts are already the ids we want; otherwise return ``[]``
    so the client sees an explicit "not available" instead of a lie.
    """
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
        return list(prompt)
    if tokenizer is None:
        return []
    try:
        encode = getattr(tokenizer, "encode", None)
        if encode is None:
            return []
        ids = encode(prompt, add_special_tokens=False)
        return list(ids) if isinstance(ids, (list, tuple)) else []
    except Exception:  # noqa: BLE001 — token-id echo is best-effort
        return []


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
    # ``>= 0``, not ``> 0``: zero is meaningful to vLLM and to OpenAI alike —
    # return the log-prob of each sampled token and no alternatives. Rejecting
    # it made ``logprobs: true`` without ``top_logprobs`` a silent no-op, so
    # callers who only want the sampled token's log-prob (RL drivers replaying
    # a batch off-policy, say) got a response with no log-probs and no error.
    if logprobs_topk is not None and logprobs_topk >= 0:
        params["logprobs"] = int(logprobs_topk)
    return params


def _tool_call_for_template(tc: dict[str, Any]) -> dict[str, Any]:
    """Re-shape one OpenAI tool call for a Jinja chat template.

    On the wire ``function.arguments`` is a JSON *string*, but Qwen's template
    iterates it with ``.items()`` — handing the string straight through raises
    "Can only get item pairs from a mapping" mid-render.
    """
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return tc
    args = fn.get("arguments")
    if not isinstance(args, str):
        return tc
    try:
        parsed = json.loads(args)
    except (json.JSONDecodeError, ValueError):
        return tc
    if not isinstance(parsed, dict):
        return tc
    return {**tc, "function": {**fn, "arguments": parsed}}


def _render_chat_prompt(
    tokenizer: Any,
    messages: list[ChatMessage],
    template_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Apply the tokenizer's chat template.

    Rejects with 400 rather than silently falling back to a hand-rolled
    role concatenation, because the wrong template produces plausible-
    looking but subtly broken generations (misaligned system prompt,
    missing generation prefix) — the exact class of bug that's hardest
    to diagnose in an RL loop.

    ``enable_thinking=False`` is our default. Qwen3 emits a
    ``<think>...</think>`` scratchpad ahead of the answer under its
    default chat template, which burns ``max_tokens`` before the model
    ever gets to the answer. RL sampling recipes typically want the
    short-form answer path (native ``CortexRLAgent`` does this too),
    and non-Qwen3 tokenizers ignore the kwarg. Clients that want
    thinking back on can send
    ``extra_body.chat_template_kwargs.enable_thinking=True``.
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
    payload = []
    for m in messages:
        entry: dict[str, Any] = {
            "role": m.role,
            "content": m.content if isinstance(m.content, str) else json.dumps(m.content),
        }
        # Tool-call round-trip. A tool-using agent replays its own prior
        # assistant turns back to us, so dropping these fields would render
        # a prompt in which the model appears to have called nothing and the
        # ``tool`` replies answer no one — the template then either errors or
        # silently produces a transcript the model was never trained on.
        if m.tool_calls:
            entry["tool_calls"] = [_tool_call_for_template(tc) for tc in m.tool_calls]
        if m.tool_call_id:
            entry["tool_call_id"] = m.tool_call_id
        if m.name:
            entry["name"] = m.name
        if m.reasoning_content:
            entry["reasoning_content"] = m.reasoning_content
        payload.append(entry)
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    if tools:
        kwargs["tools"] = tools
    if template_kwargs:
        kwargs.update(template_kwargs)
    # Older tokenizers reject kwargs they don't know. Shed them one at a time,
    # most-optional first, so a tokenizer that *does* support tools never has
    # them silently dropped — a dropped tool list yields a model that answers
    # in prose while the harness waits for a call it can dispatch. Only an
    # *unexpected-kwarg* TypeError is retryable; a TypeError raised inside the
    # template is a real bug in the payload and must surface, not be masked by
    # falling back to a tool-less prompt.
    for drop in (None, "enable_thinking", "tools"):
        if drop is not None:
            if drop not in kwargs:
                continue
            kwargs.pop(drop)
        try:
            return tokenizer.apply_chat_template(payload, **kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" in str(exc):
                continue
            raise HTTPException(
                status_code=400,
                detail=f"Chat template failed to render: TypeError: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001 — surface template rendering errors
            raise HTTPException(
                status_code=400,
                detail=f"Chat template failed to render: {type(exc).__name__}: {exc}",
            ) from exc
    raise HTTPException(
        status_code=400,
        detail="Chat template rejected every supported kwarg combination.",
    )


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>\s*(.*?)\s*</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>", re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"^(.*?)</think>", re.DOTALL)
_THINK_PAIR_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


def _split_reasoning(text: str, think_open: bool = False) -> tuple[str, str | None]:
    """Peel Qwen3's thinking scratchpad off the visible answer.

    Qwen3.5's generation prompt *opens* ``<think>`` itself, so a completion
    typically carries only the closing tag; handling the paired form alone
    would leave the entire scratchpad sitting in ``content``. Harnesses that
    score format correctness want the analysis as ``reasoning_content``, and
    an answer polluted with reasoning reads to them as a protocol violation.

    ``think_open`` says the prompt left a ``<think>`` block open. If the model
    then never closes it, every token it produced is inside that block, so the
    whole completion is reasoning and there is no visible answer. Reading it as
    ``content`` instead is what makes a well-formed tool call look like a
    format violation: the model follows the template's own instruction to
    "provide optional reasoning ... BEFORE the function call", and the prose it
    was invited to write lands in the field the harness requires to be empty.
    """
    if _THINK_PAIR_RE.search(text):
        blocks = _THINK_PAIR_RE.findall(text)
        return _THINK_PAIR_RE.sub("", text).strip(), "\n".join(blocks).strip()
    m = _THINK_CLOSE_RE.match(text)
    if m:
        return text[m.end():].strip(), m.group(1).strip()
    if think_open and text.strip():
        return "", text.strip()
    return text, None


def _parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Convert Qwen3.5 tool-call spans into OpenAI ``tool_calls``.

    Qwen3.5's template specifies a nested XML envelope, not JSON::

        <tool_call><function=execute_bash><parameter=cmd>
        ls -la
        </parameter></function></tool_call>

    Values are untyped text, so JSON scalars are recovered where they parse
    and everything else stays a string — the tool schema, not this parser, is
    what a harness validates against. A malformed span is deliberately left in
    ``content`` rather than raised: a model that emits a broken call should be
    scored as a format failure by the harness, not 500 the gateway.
    """
    calls: list[dict[str, Any]] = []
    for raw in _TOOL_CALL_RE.findall(text):
        fn = _FUNCTION_RE.search(raw)
        if fn is None:
            continue
        name, body = fn.group(1), fn.group(2)
        args: dict[str, Any] = {}
        for key, value in _PARAMETER_RE.findall(body):
            try:
                args[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                args[key] = value
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "index": len(calls),
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    if not calls:
        return text, []
    return _TOOL_CALL_RE.sub("", text).strip(), calls


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
    # Optional per-request chat-template override for tokenizers that
    # take extra kwargs (e.g. Qwen3's ``enable_thinking``).
    template_kwargs: dict[str, Any] = {}
    # vLLM's OpenAI server accepts ``chat_template_kwargs`` at the top level;
    # the ``extra_body`` nesting is how the OpenAI *SDK* smuggles it there.
    # Accept both, so a caller that already works against vLLM works here.
    for container in (payload, payload.get("extra_body")):
        if isinstance(container, dict):
            ct = container.get("chat_template_kwargs")
            if isinstance(ct, dict):
                template_kwargs.update(ct)
    prompt_text = _render_chat_prompt(
        tokenizer, req.messages, template_kwargs=template_kwargs, tools=req.tools
    )

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
        # In the chat schema these are two separate switches: ``logprobs`` asks
        # for the sampled tokens' log-probs, ``top_logprobs`` additionally asks
        # for K alternatives. Defaulting the count to 0 keeps the first usable
        # on its own instead of depending on the second.
        logprobs_topk=(req.top_logprobs or 0) if req.logprobs else None,
    )

    results = await _generate_n(pool, prompt_text, sampling_params)
    completion_id = _new_id("chatcmpl")
    created = _now()

    # ``prompt_token_ids`` (top-level) + per-choice ``token_ids`` are
    # vLLM's OpenAI-server extensions. Harbor's LiteLLM backend reads them
    # to populate ``RolloutDetail`` when ``collect_rollout_details=True``
    # is set (see ``harbor.llms.lite_llm.LiteLLM._extract_token_ids``).
    # Emitted unconditionally: clients that don't consume the fields
    # (plain OpenAI SDK) ignore them, per the OpenAI spec's
    # forward-compatibility rule.
    prompt_token_ids = _tokenize_for_return_ids(tokenizer, prompt_text)

    if req.stream:
        return StreamingResponse(
            _stream_chat_completion(
                completion_id=completion_id,
                created=created,
                model_name=model_name,
                results=results,
                include_logprobs=bool(req.logprobs),
                prompt_token_ids=prompt_token_ids,
            ),
            media_type="text/event-stream",
        )

    # Whether the rendered prompt left a ``<think>`` block open decides how an
    # unclosed completion is read, so it has to be measured on the prompt the
    # template actually produced rather than assumed from the model name.
    think_open = prompt_text.rstrip().endswith("<think>")

    choices = []
    for idx, r in enumerate(results):
        text = r.get("text", "")
        text, reasoning = _split_reasoning(text, think_open=think_open)
        # Tool calls are extracted from whichever side they landed on. Inside an
        # unclosed think block there is no visible answer to scan, but the
        # template invites a call there, so the reasoning is what carries it.
        if req.tools and reasoning and not text:
            reasoning, tool_calls = _parse_tool_calls(reasoning)
        else:
            text, tool_calls = _parse_tool_calls(text) if req.tools else (text, [])
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        choice: dict[str, Any] = {
            "index": idx,
            "message": message,
            # A parsed call outranks the raw stop reason: the model stopped on
            # the tool-call end token, which OpenAI clients expect to see
            # reported as ``tool_calls`` so they dispatch instead of finishing.
            "finish_reason": (
                "tool_calls" if tool_calls else _finish_reason_to_openai(r.get("finish_reason"))
            ),
            # vLLM-compat: completion token ids alongside the message so
            # Harbor / any RL client can build a batch without a second
            # tokenize pass.
            "token_ids": list(r.get("token_ids") or []),
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
        "prompt_token_ids": prompt_token_ids,
    }


@router.post("/completions")
async def completions(request: Request) -> Any:
    payload = await request.json()
    try:
        req = CompletionRequest.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid request body: {exc}") from exc

    pool, tokenizer, model_name = _get_pool_and_tokenizer(request)

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
        # Record every prompt's token ids for the client (parallel to the
        # per-choice ``token_ids`` in the flat response).
        prompt_ids_per_batch = [_tokenize_for_return_ids(tokenizer, list(p)) for p in prompt]
        prompt_token_ids: list[int] = prompt_ids_per_batch[0] if prompt_ids_per_batch else []
    else:
        flat_results = await _generate_n(pool, prompt, sampling_params)
        prompt_token_ids = _tokenize_for_return_ids(tokenizer, prompt)

    completion_id = _new_id("cmpl")
    created = _now()

    if req.stream:
        return StreamingResponse(
            _stream_text_completion(
                completion_id=completion_id,
                created=created,
                model_name=model_name,
                results=flat_results,
                prompt_token_ids=prompt_token_ids,
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
            "token_ids": list(r.get("token_ids") or []),
        })

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": choices,
        "usage": _usage_from_results(flat_results),
        "prompt_token_ids": prompt_token_ids,
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
    prompt_token_ids: list[int] | None = None,
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

        # 3) terminal chunk with finish_reason. Carry ``token_ids`` on the
        # per-choice terminal chunk and ``prompt_token_ids`` on the
        # top-level payload for the same vLLM-compat reason as non-stream.
        finish_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": idx,
                "delta": {},
                "finish_reason": _finish_reason_to_openai(r.get("finish_reason")),
                "token_ids": list(r.get("token_ids") or []),
            }],
        }
        if prompt_token_ids:
            finish_chunk["prompt_token_ids"] = list(prompt_token_ids)
        yield _sse(finish_chunk)

    yield b"data: [DONE]\n\n"


async def _stream_text_completion(
    *,
    completion_id: str,
    created: int,
    model_name: str,
    results: list[dict[str, Any]],
    prompt_token_ids: list[int] | None = None,
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
                "token_ids": list(r.get("token_ids") or []),
            }],
        }
        if prompt_token_ids:
            final["prompt_token_ids"] = list(prompt_token_ids)
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
