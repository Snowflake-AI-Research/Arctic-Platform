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
"""Unit tests for the pure translation layer."""

from __future__ import annotations

import pytest

from arctic_platform.openai_compat import translation
from arctic_platform.openai_compat.errors import OpenAIError
from arctic_platform.openai_compat.schemas import ChatMessage


class TestContentNormalization:
    """Content arrives in three legal shapes; only one is a plain string."""

    def test_plain_string_passes_through(self):
        assert translation.flatten_content("hi", where="m") == "hi"

    def test_content_parts_are_flattened_not_json_dumped(self):
        # The regression this guards: json.dumps'ing the array puts
        # '[{"type": "text", ...}]' into the prompt and the model answers the
        # wrong question, with a 200 and no warning.
        parts = [{"type": "text", "text": "what is "}, {"type": "text", "text": "2+2?"}]
        assert translation.flatten_content(parts, where="m") == "what is 2+2?"

    def test_null_content_is_empty_not_the_string_null(self):
        # An assistant turn carrying only tool calls has content=null by spec.
        # json.dumps(None) == "null", which would be injected verbatim.
        assert translation.flatten_content(None, where="m") == ""

    @pytest.mark.parametrize("kind", ["image_url", "input_audio", "file", "video_url"])
    def test_multimodal_parts_are_refused(self, kind):
        with pytest.raises(OpenAIError) as caught:
            translation.flatten_content([{"type": kind}], where="m")
        assert caught.value.status_code == 400
        assert kind in caught.value.message

    def test_unknown_part_type_is_refused(self):
        with pytest.raises(OpenAIError):
            translation.flatten_content([{"type": "quantum"}], where="m")


class TestTemplateMessages:
    def test_tool_turn_fields_survive(self):
        calls = [{"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]
        messages = [
            ChatMessage(role="assistant", content=None, tool_calls=calls),
            ChatMessage(role="tool", content="ok", tool_call_id="call_1"),
        ]
        rendered = translation.to_template_messages(messages)
        assert rendered[0]["tool_calls"] == calls
        assert rendered[1]["tool_call_id"] == "call_1"
        assert rendered[0]["content"] == ""


class TestMaxTokens:
    """OpenAI: omitted means 'until the model stops'. vLLM's default is 16."""

    def test_omitted_defaults_to_remaining_context(self):
        assert translation.resolve_max_tokens(None, prompt_tokens=100, max_model_len=4096) == 3996

    def test_explicit_value_is_honored(self):
        assert translation.resolve_max_tokens(64, prompt_tokens=100, max_model_len=4096) == 64

    def test_explicit_value_is_clamped_to_the_window(self):
        assert translation.resolve_max_tokens(99_999, prompt_tokens=96, max_model_len=4096) == 4000

    def test_prompt_longer_than_context_is_a_400(self):
        with pytest.raises(OpenAIError) as caught:
            translation.resolve_max_tokens(None, prompt_tokens=5000, max_model_len=4096)
        assert caught.value.status_code == 400
        assert caught.value.code == "context_length_exceeded"


class TestToolCallParsing:
    def test_single_call(self):
        calls = translation.parse_tool_calls('<tool_call>{"name": "bash", "arguments": {"cmd": "ls"}}</tool_call>')
        assert calls is not None and len(calls) == 1
        assert calls[0]["function"]["name"] == "bash"
        # OpenAI requires arguments to be a JSON *string*, not an object.
        assert calls[0]["function"]["arguments"] == '{"cmd": "ls"}'
        assert calls[0]["type"] == "function"

    def test_multiple_calls(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments":'
            " {}}</tool_call>"
        )
        calls = translation.parse_tool_calls(text)
        assert [c["function"]["name"] for c in calls] == ["a", "b"]

    def test_nested_function_envelope(self):
        calls = translation.parse_tool_calls('<tool_call>{"function": {"name": "f", "arguments": "{}"}}</tool_call>')
        assert calls[0]["function"]["name"] == "f"

    def test_plain_text_has_no_calls(self):
        assert translation.parse_tool_calls("just talking about tools") is None

    def test_malformed_markup_falls_back_to_text(self):
        assert translation.parse_tool_calls("<tool_call>not json</tool_call>") is None

    def test_surrounding_prose_is_kept_as_content(self):
        text = 'thinking...<tool_call>{"name": "f", "arguments": {}}</tool_call>'
        assert translation.strip_tool_markup(text) == "thinking..."

    def test_markup_only_leaves_null_content(self):
        assert translation.strip_tool_markup('<tool_call>{"name": "f", "arguments": {}}</tool_call>') is None


class TestResponseShaping:
    def _results(self, text="hi", finish="stop"):
        return [{"text": text, "token_ids": [1, 2, 3], "finish_reason": finish}]

    def test_usage_counts_every_choice(self, tokenizer):
        body = translation.chat_completion(
            self._results() * 3,
            model="m",
            prompt_token_ids=[1] * 10,
            tokenizer=tokenizer,
            want_logprobs=False,
            tools_offered=False,
        )
        assert body["usage"] == {"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19}

    def test_finish_reason_becomes_tool_calls_when_a_call_is_parsed(self, tokenizer):
        results = self._results('<tool_call>{"name": "f", "arguments": {}}</tool_call>')
        body = translation.chat_completion(
            results, model="m", prompt_token_ids=[], tokenizer=tokenizer, want_logprobs=False, tools_offered=True
        )
        assert body["choices"][0]["finish_reason"] == "tool_calls"
        assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "f"

    def test_tool_markup_is_left_alone_when_no_tools_were_offered(self, tokenizer):
        # Without `tools` in the request the model isn't making a tool call;
        # it's just producing text that happens to look like one.
        results = self._results('<tool_call>{"name": "f", "arguments": {}}</tool_call>')
        body = translation.chat_completion(
            results, model="m", prompt_token_ids=[], tokenizer=tokenizer, want_logprobs=False, tools_offered=False
        )
        assert body["choices"][0]["finish_reason"] == "stop"
        assert "tool_calls" not in body["choices"][0]["message"]

    def test_unknown_finish_reason_falls_back_to_stop(self, tokenizer):
        body = translation.chat_completion(
            self._results(finish="abort"),
            model="m",
            prompt_token_ids=[],
            tokenizer=tokenizer,
            want_logprobs=False,
            tools_offered=False,
        )
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_logprobs_carry_real_token_strings(self, tokenizer):
        ids = tokenizer.encode("hello world", add_special_tokens=False)
        built = translation.chat_logprobs(ids, [-0.5] * len(ids), tokenizer)
        assert built is not None
        assert len(built["content"]) == len(ids)
        # OpenAI's schema wants the token text and its bytes, not a placeholder.
        assert all(entry["token"] for entry in built["content"])
        assert built["content"][0]["bytes"] == list(built["content"][0]["token"].encode("utf-8"))

    def test_logprobs_accept_the_dict_wire_shape(self, tokenizer):
        positions = [{7: {"logprob": -1.5, "rank": 1}}]
        built = translation.chat_logprobs([7], positions, tokenizer)
        assert built["content"][0]["logprob"] == -1.5
