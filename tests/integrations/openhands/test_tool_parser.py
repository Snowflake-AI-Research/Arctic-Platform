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
"""Hermes JSON and Qwen3.5 XML tool calls."""

from __future__ import annotations

import json

from arctic_platform.integrations.openhands.tool_parser import parse_tool_calls


def test_plain_text_has_no_tool_calls():
    assert parse_tool_calls("just thinking out loud") == ("just thinking out loud", [])


def test_hermes_json_call():
    text = 'Let me look.\n<tool_call>\n{"name": "terminal", "arguments": {"command": "rg foo"}}\n</tool_call>'
    content, calls = parse_tool_calls(text)
    assert content == "Let me look.\n"
    assert calls[0]["function"]["name"] == "terminal"
    assert json.loads(calls[0]["function"]["arguments"]) == dict(command="rg foo")


def test_tool_call_only_has_null_content():
    content, calls = parse_tool_calls('<tool_call>\n{"name": "localization_finish", "arguments": {}}\n</tool_call>')
    assert content is None
    assert [call["function"]["name"] for call in calls] == ["localization_finish"]


def test_malformed_tool_call_falls_back_to_content():
    text = "<tool_call>{not json}</tool_call>"
    assert parse_tool_calls(text) == (text, [])


def test_qwen35_xml_tool_call():
    text = (
        "I'll start by listing the repo.\n"
        "<tool_call>\n"
        "<function=terminal>\n"
        "<parameter=command>\n"
        "ls -la\n"
        "</parameter>\n"
        "<parameter=security_risk>\n"
        "LOW\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    content, calls = parse_tool_calls(text)
    assert content == "I'll start by listing the repo.\n"
    assert calls[0]["function"]["name"] == "terminal"
    assert json.loads(calls[0]["function"]["arguments"]) == dict(command="ls -la", security_risk="LOW")


def test_qwen35_json_parameter_is_decoded():
    text = (
        "<tool_call>\n"
        "<function=localization_finish>\n"
        "<parameter=locations>\n"
        '[{"file": "src/a.py", "class_name": null, "function_name": "f"}]\n'
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    content, calls = parse_tool_calls(text)
    assert content is None
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["locations"] == [dict(file="src/a.py", class_name=None, function_name="f")]
