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
"""Where we diverge from the OpenAI spec, and how we diverge, over real HTTP.

Every test here pins behaviour a client can observe. The recurring theme is
that a parameter we cannot honor must produce an error, never a 200 whose body
quietly ignored it.
"""

from __future__ import annotations

import pytest
from openai_harness import MAX_MODEL_LEN
from openai_harness import MODEL
from openai_harness import RecordingBackend
from openai_harness import http_client
from openai_harness import make_app

CHAT = "/v1/chat/completions"
MESSAGES = [{"role": "user", "content": "hi"}]


def chat(http, **overrides):
    body = {"model": MODEL, "messages": MESSAGES}
    body.update(overrides)
    return http.post(CHAT, json=body)


class TestErrorEnvelope:
    """Clients parse failures out of {"error": {...}}, not {"detail": ...}."""

    def test_errors_use_openai_shape(self, http):
        response = chat(http, temperature="warm")
        assert response.status_code == 400
        error = response.json()["error"]
        assert set(error) == {"message", "type", "param", "code"}
        assert error["type"] == "invalid_request_error"
        assert error["message"]

    def test_malformed_json_is_a_clean_400(self, http):
        response = http.post(CHAT, content=b"{not json", headers={"content-type": "application/json"})
        assert response.status_code == 400
        assert "error" in response.json()

    def test_backend_failure_becomes_a_502_not_a_traceback(self, tokenizer):
        backend = RecordingBackend(raises=RuntimeError("job exploded"))
        with http_client(make_app(backend, tokenizer)) as http:
            response = chat(http)
        assert response.status_code == 502
        assert "job exploded" in response.json()["error"]["message"]


class TestRefusedParameters:
    """Accepting a parameter and ignoring it returns a wrong answer with a 200."""

    @pytest.mark.parametrize(
        "param, value",
        [
            ("response_format", {"type": "json_object"}),
            ("logit_bias", {"123": 5}),
            ("functions", [{"name": "f"}]),
            ("function_call", "auto"),
            ("audio", {"voice": "alloy"}),
            ("modalities", ["text", "audio"]),
            ("prediction", {"type": "content", "content": "x"}),
            ("reasoning_effort", "high"),
            ("best_of", 4),
        ],
    )
    def test_unsupported_parameter_is_named_in_a_400(self, http, param, value):
        response = chat(http, **{param: value})
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["param"] == param
        # The message has to say *why*, or the caller can't act on it.
        assert len(error["message"]) > len(param) + 40

    def test_streaming_is_refused_explicitly(self, http):
        response = chat(http, stream=True)
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "stream"

    def test_stream_false_is_fine(self, http):
        assert chat(http, stream=False).status_code == 200

    def test_forcing_a_specific_tool_is_refused(self, http):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        response = chat(http, tools=tools, tool_choice={"type": "function", "function": {"name": "f"}})
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "tool_choice"

    def test_unknown_parameter_fails_loudly(self, http):
        # The whole point of extra="forbid": a field in none of the three
        # buckets must not be silently dropped.
        response = chat(http, definitely_not_a_real_openai_field=1)
        assert response.status_code == 400

    @pytest.mark.parametrize("param, value", [("user", "u1"), ("store", False), ("metadata", {"k": "v"})])
    def test_inert_parameters_are_accepted(self, http, param, value):
        # These cannot change the sampled text, so rejecting them would be
        # gratuitous incompatibility.
        assert chat(http, **{param: value}).status_code == 200


class TestSamplingParameters:
    def test_omitted_max_tokens_becomes_remaining_context(self, http, backend):
        # Regression: forwarding nothing lets vLLM's SamplingParams default of
        # 16 apply, truncating every reply from a client that omits max_tokens.
        assert chat(http).status_code == 200
        assert backend.last_params["max_tokens"] > 1000
        assert backend.last_params["max_tokens"] <= MAX_MODEL_LEN

    def test_explicit_max_tokens_is_forwarded(self, http, backend):
        chat(http, max_tokens=7)
        assert backend.last_params["max_tokens"] == 7

    def test_max_completion_tokens_wins_over_max_tokens(self, http, backend):
        chat(http, max_tokens=7, max_completion_tokens=9)
        assert backend.last_params["max_tokens"] == 9

    def test_sampling_knobs_reach_the_sampler(self, http, backend):
        chat(http, temperature=0.3, top_p=0.9, seed=17, stop=["</s>"], frequency_penalty=0.5)
        params = backend.last_params
        assert params["temperature"] == 0.3
        assert params["top_p"] == 0.9
        assert params["seed"] == 17
        assert params["stop"] == ["</s>"]
        assert params["frequency_penalty"] == 0.5

    def test_unset_knobs_are_not_forwarded(self, http, backend):
        # Engine defaults should stand for anything the caller didn't set.
        chat(http)
        assert "temperature" not in backend.last_params
        assert "seed" not in backend.last_params

    def test_oversized_prompt_is_a_context_length_error(self, http):
        response = chat(http, messages=[{"role": "user", "content": "word " * 20_000}])
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "context_length_exceeded"

    def test_n_is_issued_as_repeated_prompts(self, http, backend):
        # The sampling worker only surfaces the first sub-output, so n must be
        # fanned out prompt-side or the caller silently gets one choice.
        body = chat(http, n=3).json()
        assert len(body["choices"]) == 3
        assert [c["index"] for c in body["choices"]] == [0, 1, 2]
        assert len(backend.calls[-1]["prompts"]) == 3
        assert backend.last_params["n"] == 1


class TestChatTemplate:
    def test_content_parts_do_not_leak_json_into_the_prompt(self, http, backend):
        chat(http, messages=[{"role": "user", "content": [{"type": "text", "text": "what is 2+2?"}]}])
        prompt = backend.last_prompt
        assert "what is 2+2?" in prompt
        assert '"type"' not in prompt

    def test_tools_are_rendered_into_the_prompt(self, http, backend):
        # Against a real Qwen chat template: if tools aren't passed through,
        # the model is never told they exist and can never call them.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_bash_command",
                    "description": "Run a shell command",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                },
            }
        ]
        assert chat(http, tools=tools).status_code == 200
        assert "run_bash_command" in backend.last_prompt

    def test_tool_choice_none_withholds_the_tools(self, http, backend):
        tools = [{"type": "function", "function": {"name": "secret_tool", "parameters": {}}}]
        chat(http, tools=tools, tool_choice="none")
        assert "secret_tool" not in backend.last_prompt

    def test_tool_results_are_rendered(self, http, backend):
        chat(
            http,
            messages=[
                {"role": "user", "content": "run ls"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "sh", "arguments": '{"cmd":"ls"}'}}
                    ],
                },
                {"role": "tool", "content": "file_a.txt", "tool_call_id": "c1"},
            ],
        )
        prompt = backend.last_prompt
        assert "file_a.txt" in prompt
        # 'null' would be the json.dumps of the assistant turn's None content.
        assert "null" not in prompt

    def test_thinking_is_not_forced_off(self, http, backend):
        # Hardcoding enable_thinking=False makes the same model behave
        # differently here than at any other endpoint serving it.
        chat(http)
        default_prompt = backend.last_prompt
        chat(http, chat_template_kwargs={"enable_thinking": False})
        assert backend.last_prompt != default_prompt


class TestResponseShape:
    def test_requested_model_is_echoed(self, http):
        body = chat(http, model="my-alias").json()
        assert body["model"] == "my-alias"

    def test_token_ids_are_exposed_for_rl_harnesses(self, http):
        body = chat(http).json()
        assert body["prompt_token_ids"]
        assert body["choices"][0]["token_ids"]

    def test_models_endpoint_lists_the_served_model(self, http):
        body = http.get("/v1/models").json()
        assert body["object"] == "list"
        assert [m["id"] for m in body["data"]] == [MODEL]

    def test_retrieve_model(self, http):
        assert http.get(f"/v1/models/{MODEL}").json()["id"] == MODEL

    def test_retrieve_unknown_model_is_404(self, http):
        response = http.get("/v1/models/not-served")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "model_not_found"


class TestAuth:
    def test_missing_key_is_401_when_configured(self, backend, tokenizer):
        with http_client(make_app(backend, tokenizer, api_key="sekret")) as http:
            assert chat(http).status_code == 401
            http.headers["authorization"] = "Bearer wrong"
            assert chat(http).status_code == 401
            http.headers["authorization"] = "Bearer sekret"
            assert chat(http).status_code == 200

    def test_no_key_configured_means_open(self, http):
        assert chat(http).status_code == 200

    def test_binding_off_box_without_a_key_is_refused(self):
        from arctic_platform.openai_compat.server import check_bind

        check_bind("127.0.0.1", None)
        check_bind("0.0.0.0", "key")
        with pytest.raises(ValueError, match="Refusing to bind"):
            check_bind("0.0.0.0", None)


class TestRateLimits:
    def test_backend_429_is_relayed_with_retry_after(self, tokenizer):
        class Response:
            status_code = 429
            headers = {"Retry-After": "12"}

        class Failure(Exception):
            response = Response()

        backend = RecordingBackend(raises=Failure("at capacity"))
        with http_client(make_app(backend, tokenizer)) as http:
            response = chat(http)
        # A 429 relayed as a 500 ends the caller's run; relayed as a 429 with
        # Retry-After, the stock client retry policy absorbs it.
        assert response.status_code == 429
        assert response.headers["retry-after"] == "12"
        assert response.json()["error"]["type"] == "rate_limit_error"


class TestLegacyCompletions:
    def test_string_prompt(self, http, backend):
        response = http.post("/v1/completions", json={"model": MODEL, "prompt": "2 + 2 ="})
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "text_completion"
        assert body["choices"][0]["text"]

    def test_prompt_array_numbers_choices_across_prompts(self, http):
        response = http.post("/v1/completions", json={"model": MODEL, "prompt": ["a", "b"], "n": 2})
        choices = response.json()["choices"]
        assert [c["index"] for c in choices] == [0, 1, 2, 3]

    def test_token_id_prompt_is_passed_through(self, http, backend):
        http.post("/v1/completions", json={"model": MODEL, "prompt": [1, 2, 3]})
        assert backend.calls[-1]["prompts"][0] == [1, 2, 3]

    def test_echo_is_refused(self, http):
        response = http.post("/v1/completions", json={"model": MODEL, "prompt": "x", "echo": True})
        assert response.status_code == 400
