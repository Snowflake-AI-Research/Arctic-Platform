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
"""OpenAI request/response translation. No I/O, no framework, no client."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import ValidationError

# Hermes/Qwen tool-call syntax, which is what the templates we serve emit.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter"})


class OpenAIError(Exception):
    """An error that serializes to OpenAI's ``{"error": {...}}`` envelope.

    FastAPI's default ``{"detail": ...}`` reaches the openai SDK as a bare
    "Error code: 400" with the message dropped.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        err_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.err_type = err_type
        self.param = param
        self.code = code
        self.headers = headers or {}

    def body(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.err_type, "param": self.param, "code": self.code}}


# Accepted and dropped: none of these can change the sampled text.
_IGNORED = frozenset(
    {
        "user",
        "store",
        "metadata",
        "service_tier",
        "prompt_cache_key",
        "safety_identifier",
        "stream_options",
        "parallel_tool_calls",
    }
)

# Rejected: OpenAI accepts these and they change what the model should produce,
# but we can't honor them. Silently dropping one returns a plausible 200 that is
# simply wrong -- the most expensive failure mode to diagnose.
_REJECTED: dict[str, str] = {
    "response_format": (
        "Constrained decoding is not wired through the sampling job, so the reply would be"
        " unconstrained text. Prompt for JSON and parse it yourself."
    ),
    "logit_bias": "Per-token bias is not forwarded to the sampler.",
    "functions": "Deprecated by OpenAI; use 'tools'.",
    "function_call": "Deprecated by OpenAI; use 'tool_choice'.",
    "audio": "This endpoint serves text-only models.",
    "modalities": "This endpoint serves text-only models.",
    "prediction": "Predicted outputs are not supported.",
    "web_search_options": "Server-side tools are not supported.",
    "reasoning_effort": (
        "Not forwarded to the sampler. Use extra_body.chat_template_kwargs (e.g. enable_thinking)"
        " if the model's chat template supports it."
    ),
    "suffix": "Infilling is not supported.",
    "best_of": "Server-side candidate selection is not supported; use 'n' and pick client-side.",
}


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    refusal: str | None = None


class _Request(BaseModel):
    # extra="forbid" so a parameter in neither _IGNORED nor _REJECTED is a 400
    # rather than a silent drop.
    model_config = ConfigDict(extra="forbid")

    model: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    stream: bool | None = None
    # vLLM extensions, accepted because callers reach for them against any
    # vLLM-backed endpoint.
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None


class ChatCompletionRequest(_Request):
    messages: list[ChatMessage]
    max_completion_tokens: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class CompletionRequest(_Request):
    prompt: str | list[str] | list[int] | list[list[int]]
    logprobs: int | None = None
    echo: bool | None = None


def parse_request(payload: Any, model_cls: type[BaseModel]) -> Any:
    """Validate one request body, applying the ignore/reject policy first."""
    if not isinstance(payload, dict):
        raise OpenAIError(400, "Request body must be a JSON object.")

    # The openai SDK merges extra_body into the body before sending; LiteLLM and
    # hand-rolled callers sometimes nest it.
    nested = payload.pop("extra_body", None)
    if isinstance(nested, dict):
        payload = {**nested, **payload}

    for field, reason in _REJECTED.items():
        if payload.get(field) is not None:
            raise OpenAIError(400, f"{field!r} is not supported by this endpoint. {reason}", param=field)
    if payload.get("stream"):
        raise OpenAIError(400, "'stream' is not supported: this endpoint is non-streaming.", param="stream")
    if payload.get("echo"):
        raise OpenAIError(400, "'echo' is not supported.", param="echo")

    try:
        return model_cls.model_validate({k: v for k, v in payload.items() if k not in _IGNORED})
    except ValidationError as exc:
        first = exc.errors()[0]
        param = ".".join(str(p) for p in first.get("loc", ())) or None
        raise OpenAIError(400, first.get("msg", "Invalid request"), param=param) from exc


def flatten_content(content: str | list[dict[str, Any]] | None, *, where: str) -> str:
    """Normalize message content to the string a chat template expects.

    ``null`` (a tool-only assistant turn) and content-part arrays are both legal
    and both common. json.dumps'ing either puts "null" or a JSON blob into the
    prompt, and the model answers the wrong question with a 200.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            raise OpenAIError(400, f"{where}: content parts must be objects.", param="messages")
        kind = part.get("type")
        if kind == "text":
            parts.append(str(part.get("text") or ""))
        else:
            raise OpenAIError(
                400,
                f"{where}: content of type {kind!r} is not supported; this endpoint serves text-only models.",
                param="messages",
            )
    return "".join(parts)


def render_chat_prompt(tokenizer: Any, req: ChatCompletionRequest) -> str:
    """Apply the model's own chat template.

    No hand-rolled fallback: the wrong template produces fluent, plausible,
    subtly wrong output, which is much harder to notice than a 400.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        raise OpenAIError(
            400,
            "This model's tokenizer defines no chat template. Use /v1/completions, or serve an instruct model.",
            param="messages",
        )

    tools = req.tools
    if req.tool_choice == "none":
        tools = None
    elif isinstance(req.tool_choice, dict) or req.tool_choice == "required":
        raise OpenAIError(
            400,
            "Only tool_choice='auto' and 'none' are supported; forcing a tool needs constrained decoding.",
            param="tool_choice",
        )

    # tool_calls/tool_call_id are preserved because templates render tool turns
    # from them.
    messages = []
    for i, m in enumerate(req.messages):
        entry = {"role": m.role, "content": flatten_content(m.content, where=f"messages[{i}]")}
        for key in ("name", "tool_calls", "tool_call_id"):
            if getattr(m, key) is not None:
                entry[key] = getattr(m, key)
        messages.append(entry)

    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    kwargs.update(req.chat_template_kwargs or {})
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except OpenAIError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise OpenAIError(400, f"Chat template failed to render: {type(exc).__name__}: {exc}") from exc


def resolve_max_tokens(requested: int | None, *, prompt_tokens: int, max_model_len: int) -> int:
    """Omitted means "until the model stops", per OpenAI.

    Not forwarding the field instead lets vLLM's SamplingParams default of 16
    apply, truncating every reply from a client that omits it.
    """
    remaining = max_model_len - prompt_tokens
    if remaining <= 0:
        raise OpenAIError(
            400,
            f"This model's maximum context length is {max_model_len} tokens, but the prompt is"
            f" {prompt_tokens} tokens. Shorten the prompt.",
            param="messages",
            code="context_length_exceeded",
        )
    return remaining if requested is None else min(int(requested), remaining)


def sampling_params(req: _Request, *, max_tokens: int, logprobs_topk: int | None) -> dict[str, Any]:
    """Map the request onto vLLM SamplingParams kwargs.

    The sampling job builds SamplingParams(**params) with no allowlist of its
    own, so a field missing here is dropped while the request still succeeds.
    """
    params: dict[str, Any] = {"n": 1, "max_tokens": max_tokens}
    for field in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "seed",
    ):
        value = getattr(req, field, None)
        if value is not None:
            params[field] = value
    if req.stop is not None:
        params["stop"] = [req.stop] if isinstance(req.stop, str) else list(req.stop)
    if logprobs_topk:
        params["logprobs"] = int(logprobs_topk)
    return params


def parse_tool_calls(text: str) -> list[dict[str, Any]] | None:
    """Recover OpenAI tool_calls from the model's tool-call markup."""
    calls: list[dict[str, Any]] = []
    for raw in _TOOL_CALL_RE.findall(text or ""):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Far likelier to be the model rambling about tools than a real
            # call; leave it as content.
            return None
        for payload in parsed if isinstance(parsed, list) else [parsed]:
            if not isinstance(payload, dict):
                return None
            fn = payload.get("function") if isinstance(payload.get("function"), dict) else payload
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                return None
            args = fn.get("arguments")
            args = "{}" if args is None else args if isinstance(args, str) else json.dumps(args)
            calls.append(
                {
                    "id": payload.get("id") or f"call_{uuid.uuid4().hex[:20]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            )
    return calls or None


def _logprobs(token_ids: list[int], raw: Any, tokenizer: Any) -> dict[str, Any] | None:
    """Build OpenAI's logprobs.content.

    The schema wants the token string and its bytes; the sampler reports ids, so
    decode rather than emit the empty strings that would validate but say nothing.
    """
    if not isinstance(raw, list) or not token_ids:
        return None
    content = []
    for i, token_id in enumerate(token_ids):
        position = raw[i] if i < len(raw) else None
        # A position is either a bare float or a dict keyed by token id.
        if isinstance(position, (int, float)):
            logprob = float(position)
        elif isinstance(position, dict):
            entry = position.get(token_id, position.get(str(token_id), position.get("logprob")))
            logprob = float(entry["logprob"]) if isinstance(entry, dict) else entry
        else:
            return None
        if logprob is None:
            return None
        try:
            piece = tokenizer.decode([int(token_id)])
        except Exception:  # noqa: BLE001
            piece = ""
        content.append({"token": piece, "logprob": float(logprob), "bytes": list(piece.encode()), "top_logprobs": []})
    return {"content": content}


def _usage(results: list[dict[str, Any]], prompt_tokens: int) -> dict[str, int]:
    completion = sum(r.get("generation_len") or len(r.get("token_ids") or []) for r in results)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "total_tokens": prompt_tokens + completion,
    }


def _finish(raw: Any, *, tool_calls: bool) -> str:
    if tool_calls:
        return "tool_calls"
    return raw if raw in _FINISH_REASONS else "stop"


def _envelope(prefix: str, obj: str, model: str) -> dict[str, Any]:
    return {
        "id": f"{prefix}-{uuid.uuid4().hex[:24]}",
        "object": obj,
        "created": int(time.time()),
        "model": model,
    }


def chat_completion(
    results: list[dict[str, Any]],
    *,
    model: str,
    prompt_token_ids: list[int],
    tokenizer: Any,
    want_logprobs: bool,
    tools_offered: bool,
) -> dict[str, Any]:
    choices = []
    for index, result in enumerate(results):
        text = result.get("text") or ""
        calls = parse_tool_calls(text) if tools_offered else None
        message: dict[str, Any] = {"role": "assistant", "content": text, "refusal": None}
        if calls:
            message["content"] = _TOOL_CALL_RE.sub("", text).strip() or None
            message["tool_calls"] = calls
        token_ids = [int(t) for t in result.get("token_ids") or []]
        choice: dict[str, Any] = {
            "index": index,
            "message": message,
            "finish_reason": _finish(result.get("finish_reason"), tool_calls=bool(calls)),
            "logprobs": _logprobs(token_ids, result.get("logprobs"), tokenizer) if want_logprobs else None,
        }
        # vLLM's OpenAI-server extension: RL harnesses read these to turn an eval
        # transcript into rollouts. Clients that don't know the field ignore it.
        if token_ids:
            choice["token_ids"] = token_ids
        choices.append(choice)

    return {
        **_envelope("chatcmpl", "chat.completion", model),
        "choices": choices,
        "usage": _usage(results, len(prompt_token_ids)),
        "prompt_token_ids": prompt_token_ids,
    }


def text_completion(
    results_per_prompt: list[list[dict[str, Any]]],
    *,
    model: str,
    prompt_tokens: int,
    tokenizer: Any,
    want_logprobs: bool,
) -> dict[str, Any]:
    choices, flat = [], []
    for results in results_per_prompt:
        for result in results:
            built = (
                _logprobs([int(t) for t in result.get("token_ids") or []], result.get("logprobs"), tokenizer)
                if want_logprobs
                else None
            )
            choices.append(
                {
                    "index": len(choices),
                    "text": result.get("text") or "",
                    "finish_reason": _finish(result.get("finish_reason"), tool_calls=False),
                    # The legacy schema names these differently from the chat one.
                    "logprobs": (
                        built
                        and {
                            "tokens": [e["token"] for e in built["content"]],
                            "token_logprobs": [e["logprob"] for e in built["content"]],
                            "top_logprobs": None,
                            "text_offset": [],
                        }
                    ),
                }
            )
            flat.append(result)

    return {
        **_envelope("cmpl", "text_completion", model),
        "choices": choices,
        "usage": _usage(flat, prompt_tokens),
    }
