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
"""Pure OpenAI <-> sampling-job translation. No I/O, no framework, no client.

Everything here is a function of its arguments so the wire contract can be
tested without a GPU, a job, or a server.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from arctic_platform.openai_compat.errors import OpenAIError
from arctic_platform.openai_compat.schemas import ChatCompletionRequest
from arctic_platform.openai_compat.schemas import ChatMessage
from arctic_platform.openai_compat.schemas import CompletionRequest

# Hermes/Qwen-style tool-call syntax, which is what the chat templates we serve
# emit. Models that use a different convention need their own extractor; the
# alternative -- not parsing at all -- means `tools` silently never works.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter"})


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def now() -> int:
    return int(time.time())


def flatten_content(content: str | list[dict[str, Any]] | None, *, where: str) -> str:
    """Normalize OpenAI message content down to the string a template expects.

    Three shapes are legal in a request and only one is a plain string:
    ``null`` (an assistant turn carrying only tool calls) and the content-part
    array (``[{"type": "text", "text": "hi"}]``, which plenty of clients emit
    even for pure text). Serializing either one with ``json.dumps`` puts
    ``null`` or a JSON blob into the prompt and the model answers the wrong
    question, with a 200 and no warning.
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
        elif kind in ("image_url", "input_audio", "file", "video_url"):
            raise OpenAIError(
                400,
                f"{where}: content of type {kind!r} is not supported; this endpoint serves text-only models.",
                param="messages",
            )
        else:
            raise OpenAIError(400, f"{where}: unsupported content part type {kind!r}.", param="messages")
    return "".join(parts)


def to_template_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Shape messages the way ``apply_chat_template`` expects them.

    ``tool_calls`` and ``tool_call_id`` are preserved because chat templates
    render tool turns from them; dropping them silently reshapes a multi-turn
    tool transcript into an unrelated conversation.
    """
    out: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        entry: dict[str, Any] = {
            "role": message.role,
            "content": flatten_content(message.content, where=f"messages[{index}]"),
        }
        if message.name is not None:
            entry["name"] = message.name
        if message.tool_calls:
            entry["tool_calls"] = message.tool_calls
        if message.tool_call_id is not None:
            entry["tool_call_id"] = message.tool_call_id
        out.append(entry)
    return out


def render_chat_prompt(
    tokenizer: Any,
    messages: list[ChatMessage],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Apply the model's own chat template.

    We never hand-roll a role concatenation fallback: the wrong template
    produces fluent, plausible, subtly wrong output, which is far harder to
    notice than a 400.

    ``tools`` are passed through to the template rather than dropped, so the
    model is actually told the tools exist. ``tool_choice="none"`` withholds
    them, matching OpenAI's semantics.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        raise OpenAIError(
            400,
            "This model's tokenizer defines no chat template, so /v1/chat/completions cannot render a prompt."
            " Use /v1/completions with a raw-text prompt, or serve an instruct model.",
            param="messages",
        )

    if tool_choice == "none":
        tools = None
    elif isinstance(tool_choice, dict) or tool_choice == "required":
        # Honoring these means constraining decoding to a specific tool, which
        # the sampling path can't do. Saying so beats emitting a free-form
        # answer the caller will treat as a guaranteed tool call.
        raise OpenAIError(
            400,
            "Only tool_choice='auto' and tool_choice='none' are supported; forcing a specific tool requires"
            " constrained decoding, which is not wired through the sampling job.",
            param="tool_choice",
        )

    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    if chat_template_kwargs:
        kwargs.update(chat_template_kwargs)

    try:
        return tokenizer.apply_chat_template(to_template_messages(messages), **kwargs)
    except OpenAIError:
        raise
    except Exception as exc:  # noqa: BLE001 -- surfaced verbatim; template errors are user-actionable
        raise OpenAIError(400, f"Chat template failed to render: {type(exc).__name__}: {exc}") from exc


def resolve_max_tokens(requested: int | None, *, prompt_tokens: int, max_model_len: int) -> int:
    """Default ``max_tokens`` to the rest of the context window.

    OpenAI's default is "generate until the model stops or runs out of
    context". vLLM's ``SamplingParams`` default is 16 tokens, so simply not
    forwarding the field -- the obvious way to "not override engine defaults"
    -- truncates every reply from a client that omits it.
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
    if requested is None:
        return remaining
    return min(int(requested), remaining)


def sampling_params(
    req: ChatCompletionRequest | CompletionRequest,
    *,
    max_tokens: int,
    logprobs_topk: int | None,
) -> dict[str, Any]:
    """Map the request onto vLLM ``SamplingParams`` kwargs.

    Only fields the caller actually set are forwarded, so engine defaults
    stand for everything else -- except ``max_tokens``, which
    :func:`resolve_max_tokens` always resolves for the reason documented there.

    The sampling job builds ``SamplingParams(**params)`` verbatim with no
    allowlist of its own, so a field missing from this function is dropped
    silently while the request still succeeds.
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
    if logprobs_topk is not None and logprobs_topk > 0:
        params["logprobs"] = int(logprobs_topk)
    return params


def parse_tool_calls(text: str) -> list[dict[str, Any]] | None:
    """Recover OpenAI ``tool_calls`` from the model's tool-call markup."""
    matches = _TOOL_CALL_RE.findall(text or "")
    if not matches:
        return None

    calls: list[dict[str, Any]] = []
    for raw in matches:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Malformed markup is far likelier to be the model rambling about
            # tools than a real call; returning None leaves the text as content.
            return None
        for payload in parsed if isinstance(parsed, list) else [parsed]:
            if not isinstance(payload, dict):
                return None
            function = payload.get("function") if isinstance(payload.get("function"), dict) else payload
            name = function.get("name")
            if not isinstance(name, str) or not name:
                return None
            arguments = function.get("arguments")
            if arguments is None:
                arguments = "{}"
            elif not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            calls.append(
                {
                    "id": payload.get("id") or f"call_{uuid.uuid4().hex[:20]}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
    return calls or None


def strip_tool_markup(text: str) -> str | None:
    remainder = _TOOL_CALL_RE.sub("", text or "").strip()
    return remainder or None


def _finish_reason(raw: Any, *, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    return raw if raw in _FINISH_REASONS else "stop"


def _logprob_at(position: Any, token_id: int) -> float | None:
    """Pull the sampled token's logprob out of one position entry.

    The sampling job serializes a position either as a bare float or as a dict
    keyed by token id (with ``rank``/``logprob`` inside); accept both.
    """
    if isinstance(position, (int, float)):
        return float(position)
    if not isinstance(position, dict):
        return None
    if "logprob" in position and not isinstance(position.get("logprob"), dict):
        return float(position["logprob"])
    for key in (token_id, str(token_id)):
        entry = position.get(key)
        if entry is None:
            continue
        if isinstance(entry, dict):
            return float(entry["logprob"]) if "logprob" in entry else None
        return float(entry)
    return None


def chat_logprobs(token_ids: list[int], raw_logprobs: Any, tokenizer: Any) -> dict[str, Any] | None:
    """Build OpenAI's ``logprobs.content`` array.

    OpenAI's schema requires the token *string* and its UTF-8 bytes, which the
    sampler only reports as ids -- so decode them here rather than emitting the
    empty strings that would technically validate but tell the caller nothing.
    """
    if not isinstance(raw_logprobs, list) or not token_ids:
        return None
    content: list[dict[str, Any]] = []
    for index, token_id in enumerate(token_ids):
        logprob = _logprob_at(raw_logprobs[index] if index < len(raw_logprobs) else None, token_id)
        if logprob is None:
            return None
        piece = _decode_token(tokenizer, token_id)
        content.append(
            {
                "token": piece,
                "logprob": logprob,
                "bytes": list(piece.encode("utf-8")),
                "top_logprobs": [],
            }
        )
    return {"content": content}


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:  # noqa: BLE001 -- a tokenizer that can't decode shouldn't fail the request
        return ""


def _completion_tokens(result: dict[str, Any]) -> int:
    reported = result.get("generation_len")
    if isinstance(reported, int):
        return reported
    token_ids = result.get("token_ids")
    return len(token_ids) if isinstance(token_ids, list) else 0


def usage(results: list[dict[str, Any]], *, prompt_tokens: int) -> dict[str, int]:
    completion_tokens = sum(_completion_tokens(r) for r in results)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
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
    """Shape a ``chat.completion`` from the sampling job's results."""
    choices: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        text = result.get("text") or ""
        calls = parse_tool_calls(text) if tools_offered else None
        message: dict[str, Any] = {"role": "assistant", "content": text, "refusal": None}
        if calls:
            message["content"] = strip_tool_markup(text)
            message["tool_calls"] = calls

        choice: dict[str, Any] = {
            "index": index,
            "message": message,
            "finish_reason": _finish_reason(result.get("finish_reason"), has_tool_calls=bool(calls)),
            "logprobs": None,
        }
        token_ids = [int(t) for t in result.get("token_ids") or []]
        if want_logprobs:
            choice["logprobs"] = chat_logprobs(token_ids, result.get("logprobs"), tokenizer)
        # vLLM's OpenAI-server extension. Harbor and other RL harnesses read
        # these to turn an eval transcript into trainable rollouts without a
        # second pass; clients that don't know the field ignore it.
        if token_ids:
            choice["token_ids"] = token_ids
        choices.append(choice)

    return {
        "id": new_id("chatcmpl"),
        "object": "chat.completion",
        "created": now(),
        "model": model,
        "choices": choices,
        "usage": usage(results, prompt_tokens=len(prompt_token_ids)),
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
    """Shape a legacy ``text_completion`` response.

    Choices are numbered across all prompts, which is what OpenAI does for a
    batched ``prompt`` array.
    """
    choices: list[dict[str, Any]] = []
    flat: list[dict[str, Any]] = []
    for results in results_per_prompt:
        for result in results:
            token_ids = [int(t) for t in result.get("token_ids") or []]
            choice: dict[str, Any] = {
                "index": len(choices),
                "text": result.get("text") or "",
                "finish_reason": _finish_reason(result.get("finish_reason"), has_tool_calls=False),
                "logprobs": None,
            }
            if want_logprobs:
                built = chat_logprobs(token_ids, result.get("logprobs"), tokenizer)
                if built is not None:
                    # The legacy completions schema names things differently
                    # from the chat schema.
                    choice["logprobs"] = {
                        "tokens": [entry["token"] for entry in built["content"]],
                        "token_logprobs": [entry["logprob"] for entry in built["content"]],
                        "top_logprobs": None,
                        "text_offset": [],
                    }
            choices.append(choice)
            flat.append(result)

    return {
        "id": new_id("cmpl"),
        "object": "text_completion",
        "created": now(),
        "model": model,
        "choices": choices,
        "usage": usage(flat, prompt_tokens=prompt_tokens),
    }
