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
"""OpenAI Responses requests as chat completion requests, and chat completions as Responses objects.

Non-streaming text and function calls only. A request this cannot express as chat raises ``ValueError``.
"""

from __future__ import annotations

import uuid
from typing import Any

_PASSED_THROUGH = ("model", "temperature", "top_p", "seed", "tool_choice", "stream")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if part["type"] not in ("input_text", "output_text"):
            raise ValueError(f"content part type {part['type']!r} is not supported")
        parts.append(part["text"])
    return "".join(parts)


def chat_body_from_responses(body: dict[str, Any]) -> dict[str, Any]:
    if body.get("previous_response_id") is not None:
        raise ValueError("previous_response_id is not supported; send the whole conversation as input")
    if body.get("background"):
        raise ValueError("background mode is not supported; the harness answers each request before returning")
    if body.get("tool_choice") not in (None, "auto", "none"):
        raise ValueError("the harness supports only auto and none for tool_choice")
    if body.get("tools") and body.get("tool_choice") != "none" and body.get("parallel_tool_calls") is False:
        raise ValueError("the harness cannot limit the model to one function call")
    text_format = (body.get("text") or {}).get("format") or {}
    if text_format.get("type", "text") != "text":
        raise ValueError("only plain text output is supported")
    messages: list[dict[str, Any]] = []
    if body.get("instructions"):
        messages.append(dict(role="system", content=body["instructions"]))
    items = body["input"]
    for item in [dict(role="user", content=items)] if isinstance(items, str) else items:
        kind = item.get("type", "message")
        if kind == "message":
            role = "system" if item["role"] == "developer" else item["role"]
            messages.append(dict(role=role, content=_text(item["content"])))
        elif kind == "function_call":
            call = dict(
                id=item["call_id"], type="function", function=dict(name=item["name"], arguments=item["arguments"])
            )
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append(dict(role="assistant", content="", tool_calls=[call]))
        elif kind == "function_call_output":
            messages.append(dict(role="tool", tool_call_id=item["call_id"], content=_text(item["output"])))
        else:
            raise ValueError(f"input item type {kind!r} is not supported")
    chat: dict[str, Any] = dict(messages=messages)
    for key in _PASSED_THROUGH:
        if body.get(key) is not None:
            chat[key] = body[key]
    if body.get("tools"):
        if any(tool["type"] != "function" for tool in body["tools"]):
            raise ValueError("only function tools are supported")
        if any(tool.get("strict") for tool in body["tools"]):
            raise ValueError("strict function schemas are not supported")
        chat["tools"] = [
            dict(
                type="function",
                function={key: tool[key] for key in ("name", "description", "parameters") if key in tool},
            )
            for tool in body["tools"]
        ]
    if body.get("max_output_tokens") is not None:
        chat["max_completion_tokens"] = body["max_output_tokens"]
    return chat


def responses_from_chat(
    completion: dict[str, Any], request: dict[str, Any], generation: dict[str, Any]
) -> dict[str, Any]:
    choice = completion["choices"][0]
    message = choice["message"]
    reason = {"length": "max_output_tokens", "content_filter": "content_filter"}.get(choice["finish_reason"])
    status = "incomplete" if reason else "completed"
    output: list[dict[str, Any]] = []
    if message["content"] or not message["tool_calls"]:
        output.append(
            dict(
                type="message",
                id=f"msg_{uuid.uuid4().hex}",
                role="assistant",
                status=status,
                content=[dict(type="output_text", text=message["content"] or "", annotations=[])],
            )
        )
    for call in message["tool_calls"]:
        output.append(
            dict(
                type="function_call",
                id=f"fc_{uuid.uuid4().hex}",
                call_id=call["id"],
                name=call["function"]["name"],
                arguments=call["function"]["arguments"],
                status=status,
            )
        )
    usage = dict(
        input_tokens=completion["usage"]["prompt_tokens"],
        output_tokens=completion["usage"]["completion_tokens"],
        total_tokens=completion["usage"]["total_tokens"],
    )
    if generation.get("prefix_cache_len") is not None:
        usage["input_tokens_details"] = dict(cached_tokens=generation["prefix_cache_len"])
    if generation.get("reasoning_tokens") is not None:
        usage["output_tokens_details"] = dict(reasoning_tokens=generation["reasoning_tokens"])
    return dict(
        id=f"resp_{uuid.uuid4().hex}",
        object="response",
        created_at=completion["created"],
        model=completion["model"],
        status=status,
        incomplete_details=dict(reason=reason) if reason else None,
        instructions=request.get("instructions"),
        max_output_tokens=request.get("max_output_tokens"),
        parallel_tool_calls=request.get("parallel_tool_calls") is not False,
        tools=request.get("tools") or [],
        tool_choice=request.get("tool_choice") or "auto",
        temperature=request.get("temperature"),
        top_p=request.get("top_p"),
        output=output,
        usage=usage,
        prompt_token_ids=completion["prompt_token_ids"],
        output_token_ids=choice["token_ids"],
    )
