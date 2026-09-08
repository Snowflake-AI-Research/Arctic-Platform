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

from arctic_platform.openai_compat import translation as tr
from arctic_platform.openai_compat.translation import OpenAIError


class TestContentNormalization:
    @pytest.mark.parametrize(
        "content, expected",
        [
            ("hi", "hi"),
            # json.dumps'ing the array would put a JSON blob in the prompt.
            ([{"type": "text", "text": "what is "}, {"type": "text", "text": "2+2?"}], "what is 2+2?"),
            # A tool-only assistant turn is content=null; json.dumps(None) is "null".
            (None, ""),
        ],
    )
    def test_shapes_are_flattened_not_dumped(self, content, expected):
        assert tr.flatten_content(content, where="m") == expected

    @pytest.mark.parametrize("kind", ["image_url", "input_audio", "file", "quantum"])
    def test_non_text_parts_are_refused(self, kind):
        with pytest.raises(OpenAIError) as caught:
            tr.flatten_content([{"type": kind}], where="m")
        assert caught.value.status_code == 400 and kind in caught.value.message


class TestMaxTokens:
    """OpenAI: omitted means 'until the model stops'. vLLM's default is 16."""

    @pytest.mark.parametrize(
        "requested, prompt_tokens, expected",
        [(None, 100, 3996), (64, 100, 64), (99_999, 96, 4000)],
    )
    def test_resolution(self, requested, prompt_tokens, expected):
        assert tr.resolve_max_tokens(requested, prompt_tokens=prompt_tokens, max_model_len=4096) == expected

    def test_prompt_longer_than_context_is_a_400(self):
        with pytest.raises(OpenAIError) as caught:
            tr.resolve_max_tokens(None, prompt_tokens=5000, max_model_len=4096)
        assert caught.value.code == "context_length_exceeded"


class TestToolCallParsing:
    def test_single_call(self):
        calls = tr.parse_tool_calls('<tool_call>{"name": "bash", "arguments": {"cmd": "ls"}}</tool_call>')
        assert calls[0]["function"]["name"] == "bash"
        # OpenAI types arguments as a JSON string, not an object.
        assert calls[0]["function"]["arguments"] == '{"cmd": "ls"}'

    def test_multiple_calls(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments":'
            " {}}</tool_call>"
        )
        assert [c["function"]["name"] for c in tr.parse_tool_calls(text)] == ["a", "b"]

    def test_nested_function_envelope(self):
        calls = tr.parse_tool_calls('<tool_call>{"function": {"name": "f", "arguments": "{}"}}</tool_call>')
        assert calls[0]["function"]["name"] == "f"

    @pytest.mark.parametrize("text", ["just talking about tools", "<tool_call>not json</tool_call>"])
    def test_non_calls_stay_text(self, text):
        assert tr.parse_tool_calls(text) is None


class TestResponseShaping:
    def _chat(self, results, tokenizer, **kwargs):
        options = {
            "model": "m",
            "prompt_token_ids": [1] * 10,
            "tokenizer": tokenizer,
            "want_logprobs": False,
            "tools_offered": False,
        }
        return tr.chat_completion(results, **{**options, **kwargs})

    def test_usage_counts_every_choice(self, tokenizer):
        results = [{"text": "hi", "token_ids": [1, 2, 3], "finish_reason": "stop"}] * 3
        assert self._chat(results, tokenizer)["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 9,
            "total_tokens": 19,
        }

    def test_tool_call_sets_finish_reason_and_clears_content(self, tokenizer):
        results = [{"text": 'sure<tool_call>{"name": "f", "arguments": {}}</tool_call>', "finish_reason": "stop"}]
        choice = self._chat(results, tokenizer, tools_offered=True)["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "f"
        assert choice["message"]["content"] == "sure"

    def test_tool_markup_is_left_alone_when_no_tools_were_offered(self, tokenizer):
        # Without `tools` the model isn't calling anything; it's producing text
        # that happens to look like a call.
        results = [{"text": '<tool_call>{"name": "f", "arguments": {}}</tool_call>', "finish_reason": "stop"}]
        choice = self._chat(results, tokenizer)["choices"][0]
        assert choice["finish_reason"] == "stop" and "tool_calls" not in choice["message"]

    def test_unknown_finish_reason_falls_back_to_stop(self, tokenizer):
        results = [{"text": "x", "finish_reason": "abort"}]
        assert self._chat(results, tokenizer)["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.parametrize("raw", [[-0.5, -0.5], [{7: {"logprob": -0.5}}, {8: {"logprob": -0.5}}]])
    def test_logprobs_carry_decoded_tokens_for_both_wire_shapes(self, tokenizer, raw):
        results = [{"text": "hi", "token_ids": [7, 8], "logprobs": raw, "finish_reason": "stop"}]
        content = self._chat(results, tokenizer, want_logprobs=True)["choices"][0]["logprobs"]["content"]
        # OpenAI's schema wants the token text and bytes, not a placeholder.
        assert [e["logprob"] for e in content] == [-0.5, -0.5]
        assert all(e["token"] and e["bytes"] == list(e["token"].encode()) for e in content)
