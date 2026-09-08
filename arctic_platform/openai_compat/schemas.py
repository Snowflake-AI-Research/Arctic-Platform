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
"""Request models, and an explicit policy for every field we don't implement.

Three buckets, and which bucket a field lands in is the whole design:

- **implemented** — a typed field below.
- **ignored** — accepted and dropped, because it cannot change the sampled
  text (``user``, ``store``, telemetry hints).
- **rejected** — accepted by OpenAI, changes what the model should produce,
  and we can't honor it. These 400 with the parameter named.

Models are ``extra="forbid"`` so a field in none of the three buckets is a
400 rather than a silent drop. The tempting alternative (``extra="allow"``,
so future OpenAI additions never 422) is what turns an unimplemented feature
into a wrong answer: the request succeeds, the parameter does nothing, and
the caller has no way to tell.
"""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import ValidationError

from arctic_platform.openai_compat.errors import OpenAIError
from arctic_platform.openai_compat.errors import unsupported_param

# Accepted and dropped: none of these can change the sampled text.
_IGNORED_FIELDS = frozenset(
    {
        "user",
        "store",
        "metadata",
        "service_tier",
        "prompt_cache_key",
        "safety_identifier",
        # Only meaningful while streaming, which we reject outright below.
        "stream_options",
        # We never emit more than one tool call per turn, so "may the model
        # emit several" is vacuously satisfied either way.
        "parallel_tool_calls",
    }
)

# Rejected, with the reason the caller needs to act on.
_REJECTED_FIELDS: dict[str, str] = {
    "response_format": (
        "Constrained decoding (json_object / json_schema) is not wired through the sampling job, so the"
        " response would be unconstrained text. Prompt for JSON and parse it yourself."
    ),
    "logit_bias": "Per-token bias is not forwarded to the sampler.",
    "functions": "Deprecated by OpenAI; use 'tools'.",
    "function_call": "Deprecated by OpenAI; use 'tool_choice'.",
    "audio": "This endpoint serves text-only models.",
    "modalities": "This endpoint serves text-only models.",
    "prediction": "Predicted outputs are not supported.",
    "web_search_options": "Server-side tools are not supported.",
    "reasoning_effort": (
        "Not forwarded to the sampler. Control thinking with"
        " extra_body.chat_template_kwargs (e.g. enable_thinking) if the model's chat template supports it."
    ),
    "suffix": "Infilling is not supported.",
    "best_of": "Server-side candidate selection is not supported; use 'n' and pick client-side.",
}


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    # OpenAI allows content parts (``[{"type": "text", ...}]``) everywhere a
    # string is allowed, and requires ``null`` on an assistant turn that only
    # carries tool calls. Both shapes are normalized in ``translation``.
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    refusal: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stream: bool | None = None

    # vLLM extensions, accepted because callers reach for them against any
    # vLLM-backed endpoint.
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    logprobs: int | None = None
    echo: bool | None = None
    stream: bool | None = None

    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None


def _flatten_extra_body(payload: dict[str, Any]) -> dict[str, Any]:
    """Hoist ``extra_body`` to the top level.

    The OpenAI SDK merges ``extra_body`` into the JSON body before sending, so
    a real client never nests it. LiteLLM and hand-rolled callers sometimes do,
    and a nested vendor extension that silently does nothing is exactly the
    failure this module exists to prevent.
    """
    nested = payload.get("extra_body")
    if not isinstance(nested, dict):
        return payload
    merged = {k: v for k, v in payload.items() if k != "extra_body"}
    for key, value in nested.items():
        merged.setdefault(key, value)
    return merged


def parse_request(payload: Any, model_cls: type[BaseModel]) -> Any:
    """Validate one request body, applying the ignore / reject policy first.

    Policy runs before pydantic so the caller gets "response_format is not
    supported because ..." instead of "extra fields not permitted".
    """
    if not isinstance(payload, dict):
        raise OpenAIError(400, "Request body must be a JSON object.")

    payload = _flatten_extra_body(payload)

    for field, reason in _REJECTED_FIELDS.items():
        if payload.get(field) is not None:
            raise unsupported_param(field, reason)

    if payload.get("stream"):
        raise unsupported_param(
            "stream",
            "This endpoint is non-streaming. Set stream=false (or omit it); the full response is returned at once.",
        )
    if payload.get("echo"):
        raise unsupported_param("echo", "Echoing the prompt back is not supported.")

    cleaned = {k: v for k, v in payload.items() if k not in _IGNORED_FIELDS}

    try:
        return model_cls.model_validate(cleaned)
    except ValidationError as exc:
        first = exc.errors()[0]
        param = ".".join(str(p) for p in first.get("loc", ())) or None
        raise OpenAIError(400, f"{first.get('msg', 'Invalid request')}", param=param) from exc
