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
        return "" if ids == [self.eos_token_id] else "text"


@pytest.fixture
def generation():
    return dict(token_ids=[42, 43], finish_reason="stop", prefix_cache_len=3)


@pytest.fixture
def served(generation):
    tokenizer = _Tokenizer()

    async def generate(prompt_ids, params, routing_key):
        return generation

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
        data = error.read()
        return error.code, json.loads(data) if error.headers.get_content_type() == "application/json" else data


def test_a_responses_request_runs_as_chat_and_returns_items_and_token_ids(served, generation):
    proxy, tokenizer = served
    generation["reasoning_tokens"] = 1
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
    assert body["usage"]["input_tokens_details"]["cached_tokens"] == 3
    assert body["usage"]["output_tokens_details"]["reasoning_tokens"] == 1
    assert body["tools"] == tools and body["tool_choice"] == "auto" and body["parallel_tool_calls"]
    assert body["instructions"] == "be brief" and body["max_output_tokens"] == 8


@pytest.mark.parametrize("empty", [False, True], ids=["tool-result", "eos-only"])
def test_replaying_output_keeps_the_sampled_ids(served, generation, empty):
    proxy, _ = served
    options = {}
    if empty:
        generation["token_ids"] = [7]
    else:
        options["tools"] = [dict(type="function", name="ls")]
    history = [dict(role="user", content="hi")]
    status, first = _post(proxy, dict(input=history, **options))
    assert status == 200
    if empty:
        followup = dict(role="user", content="again")
    else:
        followup = dict(
            type="function_call_output",
            call_id=first["output"][-1]["call_id"],
            output=[dict(type="input_text", text="a.py")],
        )
    status, second = _post(proxy, dict(input=[*history, *first["output"], followup], **options))
    prefix = first["prompt_token_ids"] + first["output_token_ids"]
    assert status == 200 and second["prompt_token_ids"][: len(prefix)] == prefix
    assert proxy.episodes("k")[-1].extended


def test_length_limited_message_is_incomplete(served, generation):
    proxy, _ = served
    generation["finish_reason"] = "length"
    status, body = _post(proxy, dict(input="hi"))
    assert status == 200 and body["status"] == body["output"][0]["status"] == "incomplete"
    assert body["incomplete_details"] == dict(reason="max_output_tokens")


def test_aborted_generation_returns_an_error(served, generation):
    proxy, _ = served
    generation["finish_reason"] = "abort"
    status, body = _post(proxy, dict(input="hi"))
    assert status == 502 and body["error"]["message"]


@pytest.mark.parametrize(
    "request_body",
    [
        dict(input=[dict(type="reasoning", summary=[])]),
        dict(input="hi", previous_response_id="resp_1"),
        dict(input="hi", tools=[dict(type="web_search")]),
        dict(input="hi", stream=True),
        dict(input="hi", tools=[dict(type="function", name="ls")], tool_choice=dict(type="function", name="ls")),
        dict(input="hi", tools=[dict(type="function", name="ls")], tool_choice="required"),
        dict(input="hi", tools=[dict(type="function", name="ls")], parallel_tool_calls=False),
        dict(input=[dict(type="function_call_output", call_id="c", output=[dict(type="input_image", file_id="f")])]),
        dict(input="hi", tools=[dict(type="function", name="ls", strict=True)]),
        dict(input="hi", text=dict(format=dict(type="json_schema", name="result", schema=dict(type="object")))),
    ],
)
def test_requests_chat_cannot_express_are_refused(served, request_body):
    proxy, _ = served
    status, body = _post(proxy, request_body)
    assert status == 400 and body["error"]["message"]
