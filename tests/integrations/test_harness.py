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

import pytest

from arctic_platform.integrations.harness.episodes import Episode
from arctic_platform.integrations.harness.episodes import Session
from arctic_platform.integrations.harness.episodes import plan_prompt
from arctic_platform.integrations.harness.episodes import remember
from arctic_platform.integrations.harness.episodes import training_sequences
from arctic_platform.integrations.harness.generator import Grade
from arctic_platform.integrations.harness.generator import _detail_means
from arctic_platform.integrations.harness.generator import rows_for_rollout
from arctic_platform.integrations.harness.generator import trajectory_payload
from arctic_platform.integrations.harness.proxy import ChatCompletionsProxy
from arctic_platform.integrations.harness.tokens import masked_row
from arctic_platform.integrations.harness.tool_parser import parse_tool_calls


class _Tokenizer:
    eos_token_id = 7


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
    text = '<tool_call><function=finish><parameter=locations>[{"file": "src/a.py"}]</parameter></function></tool_call>'
    _, calls = parse_tool_calls(text)
    assert json.loads(calls[0]["function"]["arguments"])["locations"] == [dict(file="src/a.py")]


def test_masked_row_rejects_a_mask_of_the_wrong_length():
    with pytest.raises(ValueError, match="loss mask"):
        masked_row([1], [2, 3], [1], 0.0, {}, False)


class _ChatTokenizer(_Tokenizer):
    """Generation prompt ``<a>`` is a prefix of an assistant turn, as in the Qwen template."""

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False, **kwargs):
        del tools, tokenize, kwargs
        parts = []
        for message in messages:
            content = message.get("content") or ""
            if message["role"] == "assistant":
                parts.append(f"<a>{content}</a>")
            else:
                parts.append(f"<{message['role']}>{content}</{message['role']}>")
        text = "".join(parts)
        if add_generation_prompt:
            text += "<a>"
        return text

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=True):
        del ids, skip_special_tokens
        return "ls"


def _body(messages):
    return dict(messages=messages, chat_template_kwargs=dict(enable_thinking=False))


def test_continuation_keeps_the_sampled_tokens():
    tokenizer = _ChatTokenizer()
    session = Session()
    first = _body([dict(role="system", content="s"), dict(role="user", content="hi")])
    prompt, extended = plan_prompt(session, tokenizer, first)
    assert extended is False
    remember(
        session,
        first,
        dict(role="assistant", content="ls", tool_calls=[]),
        prompt,
        [42, 43],
        dict(temperature=1),
        extended,
        [-0.2, -0.3],
    )
    second = _body(
        [
            dict(role="system", content="s"),
            dict(role="user", content="hi"),
            dict(role="assistant", content="rewritten"),
            dict(role="tool", content="out", tool_call_id="call_1"),
        ]
    )
    spliced, extended = plan_prompt(session, tokenizer, second)
    fresh = tokenizer.encode(tokenizer.apply_chat_template(second["messages"], add_generation_prompt=True))
    assert extended is True
    assert spliced[len(prompt) : len(prompt) + 2] == [42, 43]
    assert spliced != fresh
    remember(session, second, dict(role="assistant", content="done"), spliced, [7], {}, extended, None)
    sequences = training_sequences(session.episodes, step_wise=False)
    assert len(sequences) == 1
    _, response, mask = sequences[0]
    assert response[:2] == [42, 43]
    # Only the two calls' sampled tokens train; the end marker and tool turn between them do not.
    assert mask == [1, 1] + [0] * (len(response) - 3) + [1]


def test_a_new_system_message_starts_a_new_episode():
    tokenizer = _ChatTokenizer()
    session = Session()
    first = _body([dict(role="user", content="hi")])
    prompt, extended = plan_prompt(session, tokenizer, first)
    remember(session, first, dict(role="assistant", content="ls"), prompt, [42], {}, extended, None)
    other = _body([dict(role="user", content="other task")])
    prompt, extended = plan_prompt(session, tokenizer, other)
    assert extended is False
    assert 42 not in prompt


def test_rows_use_the_grader_and_keep_one_row_per_trajectory():
    episodes = [
        Episode([1, 2], [3], None, {}, False),
        Episode([1, 2, 3, 4], [5], None, {}, False),
    ]
    grade = Grade(reward=2.0, reward_detail=dict(file_reward=1.0, reward=2.0), exhausted=False)
    rows = rows_for_rollout(episodes, grade, _Tokenizer(), step_wise=False)
    assert len(rows) == 1
    assert rows[0]["reward"] == 2.0
    assert rows[0]["response_ids"] == [3, 4, 5]
    assert rows[0]["loss_mask"] == [1, 0, 1]
    split = rows_for_rollout(episodes, grade, _Tokenizer(), step_wise=True)
    assert len(split) == 2
    assert split[0]["response_ids"] == [3]
    assert split[1]["response_ids"] == [5]
    exhausted = Grade(reward=2.0, reward_detail=dict(reward=2.0), exhausted=True)
    assert rows_for_rollout(episodes, exhausted, _Tokenizer(), step_wise=False)[0]["loss_mask"] == [0, 0, 0]


def test_re_rendered_history_that_changed_ids_is_not_trained():
    # The second call re-rendered the first turn as 9 instead of the sampled 3.
    episodes = [
        Episode([1, 2], [3], None, {}, False),
        Episode([1, 2, 9, 4], [5], None, {}, False),
    ]
    grade = Grade(reward=1.0, reward_detail=dict(reward=1.0), exhausted=False)
    rows = rows_for_rollout(episodes, grade, _Tokenizer(), step_wise=False)
    assert rows[0]["response_ids"] == [9, 4, 5]
    assert rows[0]["loss_mask"] == [0, 0, 1]


def test_a_rollout_with_no_episodes_is_one_masked_token():
    grade = Grade(reward=1.0, reward_detail=dict(reward=1.0), exhausted=False)
    rows = rows_for_rollout([], grade, _Tokenizer(), step_wise=False)
    assert rows[0]["loss_mask"] == [0]
    assert rows[0]["stop_reason"] == "error"


def test_trajectory_keeps_adapter_fields():
    payload = trajectory_payload(
        "inst",
        Grade(reward=2.0, reward_detail=dict(reward=2.0), exhausted=False),
        dict(messages=[dict(content="hi", prompt_token_ids=[1])], locations=[dict(file="a.py")]),
    )
    assert payload["locations"] == [dict(file="a.py")]
    assert payload["messages"] == [dict(content="hi")]
    assert payload["total_reward"] == 2.0
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
