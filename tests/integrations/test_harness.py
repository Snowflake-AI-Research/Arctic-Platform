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
"""The chat endpoint, the training row, and the Qwen tool parser. No harness."""

from __future__ import annotations

import json

from arctic_platform.integrations.harness.generator import RolloutRequest
from arctic_platform.integrations.harness.generator import _detail_means
from arctic_platform.integrations.harness.generator import _invoke
from arctic_platform.integrations.harness.generator import trajectory_payload
from arctic_platform.integrations.harness.loss_mask import assistant_loss_mask
from arctic_platform.integrations.harness.loss_mask import mask_if_exhausted
from arctic_platform.integrations.harness.proxy import ChatCompletionsProxy
from arctic_platform.integrations.harness.tokens import buffer_after_role
from arctic_platform.integrations.harness.tokens import stitch_response
from arctic_platform.integrations.harness.tokens import training_row
from arctic_platform.integrations.harness.tool_parser import parse_tool_calls

START = 1
ASSISTANT = 2


class _Tokenizer:
    eos_token_id = 7

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_start|>": START, "assistant": ASSISTANT}[token]


def _events() -> list[dict]:
    return [
        dict(prompt_token_ids=[1, 2], response_token_ids=[3]),
        dict(prompt_token_ids=[1, 2, 3, 4], response_token_ids=[5]),
    ]


def test_plain_text_has_no_tool_calls():
    assert parse_tool_calls("just thinking out loud") == ("just thinking out loud", [])


def test_qwen_xml_and_hermes_json_both_parse():
    xml = (
        "<tool_call><function=terminal><parameter=command>ls</parameter>"
        "<parameter=security_risk>LOW</parameter></function></tool_call>"
    )
    hermes = '<tool_call>{"name": "terminal", "arguments": {"command": "rg foo"}}</tool_call>'
    _, xml_calls = parse_tool_calls(xml)
    _, hermes_calls = parse_tool_calls(hermes)
    assert json.loads(xml_calls[0]["function"]["arguments"])["command"] == "ls"
    assert json.loads(hermes_calls[0]["function"]["arguments"]) == dict(command="rg foo")
    assert parse_tool_calls("<tool_call>{not json}</tool_call>")[1] == []


def test_json_list_parameter_stays_structured():
    text = (
        "<tool_call><function=finish><parameter=locations>" '[{"file": "src/a.py"}]</parameter></function></tool_call>'
    )
    _, calls = parse_tool_calls(text)
    assert json.loads(calls[0]["function"]["arguments"])["locations"] == [dict(file="src/a.py")]


def test_mask_trains_on_assistant_content_only():
    response = [9, START, ASSISTANT, 4, 5]
    mask = assistant_loss_mask(response, start_token_id=START, assistant_token_id=ASSISTANT, buffer_succeed=1)
    assert mask == [0, 0, 0, 0, 1]
    assert mask_if_exhausted(mask, True) == [0, 0, 0, 0, 0]


def test_stitch_drops_the_shared_prefix():
    prompt, response = stitch_response(_events())
    assert prompt == [1, 2]
    assert response == [3, 4, 5]
    assert buffer_after_role("Qwen/Qwen3-4B-Instruct-2507") == 1
    assert buffer_after_role("Qwen/Qwen3.5-4B") == 5


def test_training_row_uses_the_harness_reward():
    row = training_row(
        dict(token_messages=_events(), reward=2.0, reward_detail=dict(reward=2.0), exhausted=False),
        _Tokenizer(),
        "Qwen/Qwen3.5-4B",
    )
    assert row["reward"] == 2.0
    assert row["response_ids"] == [3, 4, 5]
    exhausted = training_row(
        dict(token_messages=_events(), reward=2.0, reward_detail=dict(reward=2.0), exhausted=True),
        _Tokenizer(),
        "Qwen/Qwen3.5-4B",
    )
    assert exhausted["loss_mask"] == [0] * len(exhausted["response_ids"])


def _request() -> RolloutRequest:
    return RolloutRequest(
        env_extra=dict(instance_id="inst"),
        trajectory_id=None,
        routing_key="route-1",
        base_url="http://127.0.0.1:9/r/route-1/v1",
        phase="train",
        global_step=1,
    )


def test_one_failed_call_does_not_drop_the_batch():
    def boom(_request):
        raise RuntimeError("sandbox")

    row = _invoke(boom, dict(reward=0.0), _request())
    assert row["reward"] == 0.0
    assert row["routing_key"] == "route-1"
    assert "sandbox" in row["error"]


def test_rollout_receives_the_request():
    def rollout(request: RolloutRequest):
        assert request.base_url.endswith("/r/route-1/v1")
        return dict(messages=[], token_messages=[], reward=1.0, reward_detail=dict(score=1.0), exhausted=False)

    row = _invoke(rollout, dict(reward=0.0), _request())
    assert row["reward"] == 1.0


def test_trajectory_keeps_harness_fields():
    payload = trajectory_payload(
        "inst",
        dict(reward=2.0, reward_detail=dict(reward=2.0)),
        dict(messages=[dict(content="hi", prompt_token_ids=[1])], locations=[dict(file="a.py")], reward=2.0),
    )
    assert payload["locations"] == [dict(file="a.py")]
    assert payload["messages"] == [dict(content="hi")]
    assert "reward" not in payload


def test_reward_means_follow_the_keys_the_harness_returns():
    means = _detail_means([dict(reward=1.0, passed=1.0), dict(reward=0.0, passed=0.0)])
    assert means == {"reward/passed": 0.5}


def test_each_rollout_gets_its_own_url_and_parser():
    def parser(text: str):
        return f"parsed:{text}", []

    proxy = ChatCompletionsProxy(
        generate_fn=None,
        tokenizer=None,
        model_name="m",
        max_model_len=8,
        default_max_tokens=4,
        parse_tool_calls=parser,
    )
    proxy.base_url = "http://127.0.0.1:9"
    assert proxy.url_for("step1-train-0") == "http://127.0.0.1:9/r/step1-train-0/v1"
    assert proxy._parse_tool_calls("ls") == ("parsed:ls", [])
