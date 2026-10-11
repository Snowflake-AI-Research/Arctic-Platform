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
"""Tool calls in a decoded Qwen completion.

Hermes JSON matches vLLM's ``hermes`` parser, which is what
https://github.com/18jeffreyma/codescout/blob/abab719e08a55dde78c6da864cd24d84fd47bdf2/src/cortex/tool_parser.py
implements. Qwen3.5's chat template emits XML instead. That shape is from
the same repo at 8184b42. A block that is neither JSON nor that XML fails
the whole response, so a half-parsed call is never executed.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _argument_value(raw: str) -> Any:
    """Parameter text is a string, unless it is itself a JSON object or list.

    ``localization_finish`` takes a list of locations. Leaving that as a string
    makes the tool schema reject the call.
    """
    text = raw.strip()
    if len(text) > 0 and text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def _parse_call(body: str) -> dict[str, Any]:
    body = body.strip()
    if body.startswith("{"):
        return json.loads(body)
    match = _FUNCTION_RE.search(body)
    if match is None:
        raise ValueError("tool call is neither JSON nor a Qwen3.5 function block")
    arguments = {name: _argument_value(value) for name, value in _PARAMETER_RE.findall(match.group(2))}
    return dict(name=match.group(1), arguments=arguments)


def parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Split a completion into ``(content, tool_calls)``.

    ``tool_calls`` are OpenAI chat-completion tool calls. ``arguments`` is a
    JSON string. Text with no ``<tool_call>`` is returned unchanged.
    """
    if TOOL_CALL_START not in text:
        return text, []

    tool_calls = []
    try:
        for closed, unclosed in _TOOL_CALL_RE.findall(text):
            call = _parse_call(closed or unclosed)
            tool_calls.append(
                dict(
                    id=f"chatcmpl-tool-{uuid.uuid4().hex}",
                    type="function",
                    function=dict(
                        name=call["name"],
                        arguments=json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                    ),
                )
            )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return text, []

    content = text[: text.find(TOOL_CALL_START)]
    if content == "":
        return None, tool_calls
    return content, tool_calls
