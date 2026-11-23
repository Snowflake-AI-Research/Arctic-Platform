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
            call = dict(id=item["call_id"], type="function", function=dict(name=item["name"], arguments=item["arguments"]))
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append(dict(role="assistant", content="", tool_calls=[call]))
        elif kind == "function_call_output":
            messages.append(dict(role="tool", tool_call_id=item["call_id"], content=item["output"]))
        else:
            raise ValueError(f"input item type {kind!r} is not supported")
    chat: dict[str, Any] = dict(messages=messages)
    for key in _PASSED_THROUGH:
        if body.get(key) is not None:
            chat[key] = body[key]
    if isinstance(chat.get("tool_choice"), dict):
        chat["tool_choice"] = dict(type="function", function=dict(name=chat["tool_choice"]["name"]))
    if body.get("tools"):
        if any(tool["type"] != "function" for tool in body["tools"]):
            raise ValueError("only function tools are supported")
        chat["tools"] = [
            dict(type="function", function={key: tool[key] for key in ("name", "description", "parameters") if key in tool})
            for tool in body["tools"]
        ]
    if body.get("max_output_tokens") is not None:
        chat["max_completion_tokens"] = body["max_output_tokens"]
    return chat


def responses_from_chat(completion: dict[str, Any]) -> dict[str, Any]:
    choice = completion["choices"][0]
    message = choice["message"]
    output: list[dict[str, Any]] = []
    if message["content"]:
        output.append(
            dict(
                type="message", id=f"msg_{uuid.uuid4().hex}", role="assistant", status="completed",
                content=[dict(type="output_text", text=message["content"], annotations=[])],
            )
        )
    for call in message["tool_calls"]:
        output.append(
            dict(
                type="function_call", id=f"fc_{uuid.uuid4().hex}", call_id=call["id"],
                name=call["function"]["name"], arguments=call["function"]["arguments"], status="completed",
            )
        )
    truncated = choice["finish_reason"] == "length"
    usage = completion["usage"]
    return dict(
        id=f"resp_{uuid.uuid4().hex}",
        object="response",
        created_at=completion["created"],
        model=completion["model"],
        status="incomplete" if truncated else "completed",
        incomplete_details=dict(reason="max_output_tokens") if truncated else None,
        output=output,
        usage=dict(
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
        ),
        prompt_token_ids=completion["prompt_token_ids"],
        output_token_ids=choice["token_ids"],
    )
