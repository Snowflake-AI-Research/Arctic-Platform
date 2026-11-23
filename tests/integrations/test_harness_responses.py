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
"""The harness proxy's Responses route, end to end over HTTP, with a stub tokenizer and generate."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from arctic_platform.integrations.harness.proxy import ChatCompletionsProxy


class _Tokenizer:
    """Renders ``<role>content</role>`` and records the messages it rendered."""

    eos_token_id = 7

    def __init__(self):
        self.rendered = []

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False, **kwargs):
        self.rendered.append((messages, tools))
        text = "".join(f"<{m['role']}>{m.get('content') or ''}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=True):
        return "text"


@pytest.fixture
def served():
    tokenizer = _Tokenizer()

    async def generate(prompt_ids, params, routing_key):
        return dict(token_ids=[42, 43], finish_reason="stop")

    def parser(text):
        return "calling", [dict(id="call_2", type="function", function=dict(name="ls", arguments="{}"))]

    proxy = ChatCompletionsProxy(generate, tokenizer, "m", max_model_len=4096, default_max_tokens=16, parse_tool_calls=parser)
    proxy.start()
    yield proxy, tokenizer
    proxy.stop()


def _post(proxy, body):
    request = urllib.request.Request(
        f"{proxy.url_for('k')}/responses", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_a_responses_request_runs_as_chat_and_returns_items_and_token_ids(served):
    proxy, tokenizer = served
    tools = [dict(type="function", name="ls", description="list", parameters=dict(type="object"))]
    status, body = _post(
        proxy,
        dict(
            model="m",
            instructions="be brief",
            input=[
                dict(role="user", content=[dict(type="input_text", text="hi")]),
                dict(type="function_call", call_id="call_1", name="ls", arguments="{}"),
                dict(type="function_call_output", call_id="call_1", output="a.py"),
            ],
            tools=tools,
            max_output_tokens=8,
        ),
    )

    assert status == 200
    messages, rendered_tools = tokenizer.rendered[-1]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2]["tool_calls"][0]["function"]["name"] == "ls" and messages[3]["tool_call_id"] == "call_1"
    assert rendered_tools[0]["function"]["parameters"] == dict(type="object")
    assert body["object"] == "response" and body["status"] == "completed"
    assert [item["type"] for item in body["output"]] == ["message", "function_call"]
    assert body["output"][0]["content"][0]["text"] == "calling"
    assert (body["output"][1]["call_id"], body["output"][1]["name"]) == ("call_2", "ls")
    assert body["output_token_ids"] == [42, 43]
    assert body["usage"]["input_tokens"] == len(body["prompt_token_ids"]) > 0


@pytest.mark.parametrize(
    "request_body",
    [
        dict(input=[dict(type="reasoning", summary=[])]),
        dict(input="hi", previous_response_id="resp_1"),
        dict(input="hi", tools=[dict(type="web_search")]),
        dict(input="hi", stream=True),
    ],
)
def test_requests_chat_cannot_express_are_refused(served, request_body):
    proxy, _ = served
    status, body = _post(proxy, request_body)
    assert status == 400 and body["error"]["message"]
