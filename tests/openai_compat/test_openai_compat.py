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
"""The wire contract, over real HTTP.

Two levels. The ``http`` tests assert on the *call the stub received*, because
most of what can go wrong (a dropped sampling param, tools missing from the
prompt, max_tokens=16) is invisible in the response. The ``sdk`` tests drive the
real ``openai`` SDK, which parses every response into OpenAI's own Pydantic
models -- so a wrong-shaped response raises inside the client, making these a
conformance check against OpenAI's schema rather than against our idea of it.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any

import pytest
from conftest import MAX_MODEL_LEN
from conftest import MODEL
from conftest import BlockingClient
from conftest import StubClient
from conftest import http_client
from conftest import make_app
from conftest import serve

openai = pytest.importorskip("openai")

MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash_command",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        },
    }
]


def chat(http, **overrides):
    return http.post("/v1/chat/completions", json={"model": MODEL, "messages": MESSAGES, **overrides})


class TestErrorEnvelope:
    """Clients parse failures out of {"error": {...}}, not {"detail": ...}."""

    def test_shape(self, http):
        response = chat(http, temperature="warm")
        assert response.status_code == 400
        error = response.json()["error"]
        assert set(error) == {"message", "type", "param", "code"}
        assert error["type"] == "invalid_request_error" and error["message"]

    def test_malformed_json(self, http):
        response = http.post(
            "/v1/chat/completions", content=b"{not json", headers={"content-type": "application/json"}
        )
        assert response.status_code == 400 and "error" in response.json()

    def test_backend_failure_is_a_502_not_a_traceback(self, tokenizer):
        with http_client(make_app(StubClient(raises=RuntimeError("job exploded")), tokenizer)) as http:
            response = chat(http)
        assert response.status_code == 502 and "job exploded" in response.json()["error"]["message"]

    def test_capacity_error_relays_as_429_with_retry_after(self, tokenizer):
        # A 429 relayed as a 500 ends the caller's run; as a 429 with
        # Retry-After, stock client retry policy absorbs it.
        class Failure(Exception):
            response = type("R", (), {"status_code": 429, "headers": {"Retry-After": "12"}})()

        with http_client(make_app(StubClient(raises=Failure("at capacity")), tokenizer)) as http:
            response = chat(http)
        assert response.status_code == 429
        assert response.headers["retry-after"] == "12"
        assert response.json()["error"]["type"] == "rate_limit_error"


class TestParameterPolicy:
    """A parameter we can't honor must error, never return a 200 that ignored it."""

    @pytest.mark.parametrize(
        "param, value",
        [
            ("response_format", {"type": "json_object"}),
            ("logit_bias", {"123": 5}),
            ("functions", [{"name": "f"}]),
            ("function_call", "auto"),
            ("audio", {"voice": "alloy"}),
            ("modalities", ["text"]),
            ("prediction", {"type": "content", "content": "x"}),
            ("reasoning_effort", "high"),
            ("best_of", 4),
        ],
    )
    def test_unsupported_parameter_is_named_in_a_400(self, http, param, value):
        response = chat(http, **{param: value})
        assert response.status_code == 400
        error = response.json()["error"]
        # The message must say why, or the caller can't act on it.
        assert error["param"] == param and len(error["message"]) > len(param) + 40

    def test_streaming_is_refused(self, http):
        assert chat(http, stream=True).json()["error"]["param"] == "stream"
        assert chat(http, stream=False).status_code == 200

    def test_forcing_a_specific_tool_is_refused(self, http):
        response = chat(http, tools=TOOLS, tool_choice={"type": "function", "function": {"name": "f"}})
        assert response.json()["error"]["param"] == "tool_choice"

    def test_unknown_parameter_fails_loudly(self, http):
        assert chat(http, definitely_not_a_real_openai_field=1).status_code == 400

    @pytest.mark.parametrize("param, value", [("user", "u1"), ("store", False), ("metadata", {"k": "v"})])
    def test_inert_parameters_are_accepted(self, http, param, value):
        assert chat(http, **{param: value}).status_code == 200


class TestSamplingParameters:
    def test_omitted_max_tokens_becomes_remaining_context(self, http, stub):
        # Forwarding nothing lets vLLM's default of 16 truncate every reply.
        assert chat(http).status_code == 200
        assert 1000 < stub.params["max_tokens"] <= MAX_MODEL_LEN

    def test_explicit_and_max_completion_tokens(self, http, stub):
        chat(http, max_tokens=7)
        assert stub.params["max_tokens"] == 7
        chat(http, max_tokens=7, max_completion_tokens=9)
        assert stub.params["max_tokens"] == 9

    def test_knobs_reach_the_sampler_and_unset_ones_do_not(self, http, stub):
        chat(http, temperature=0.3, top_p=0.9, seed=17, stop=["</s>"], frequency_penalty=0.5)
        assert stub.params["temperature"] == 0.3 and stub.params["top_p"] == 0.9
        assert stub.params["seed"] == 17 and stub.params["stop"] == ["</s>"]
        assert stub.params["frequency_penalty"] == 0.5
        chat(http)
        assert "temperature" not in stub.params and "seed" not in stub.params

    def test_oversized_prompt_is_a_context_length_error(self, http):
        response = chat(http, messages=[{"role": "user", "content": "word " * 20_000}])
        assert response.json()["error"]["code"] == "context_length_exceeded"

    def test_n_is_issued_as_repeated_prompts(self, http, stub):
        # The worker only surfaces the first sub-output, so n must be fanned out
        # prompt-side or the caller silently gets one choice.
        body = chat(http, n=3).json()
        assert [c["index"] for c in body["choices"]] == [0, 1, 2]
        assert len(stub.calls[-1]["prompts"]) == 3 and stub.params["n"] == 1


class TestChatTemplate:
    """Against a real Qwen template."""

    def test_content_parts_do_not_leak_json_into_the_prompt(self, http, stub):
        chat(http, messages=[{"role": "user", "content": [{"type": "text", "text": "what is 2+2?"}]}])
        assert "what is 2+2?" in stub.prompt and '"type"' not in stub.prompt

    def test_tools_are_rendered_and_tool_choice_none_withholds_them(self, http, stub):
        assert chat(http, tools=TOOLS).status_code == 200
        assert "run_bash_command" in stub.prompt
        chat(http, tools=TOOLS, tool_choice="none")
        assert "run_bash_command" not in stub.prompt

    def test_tool_results_render_without_leaking_null(self, http, stub):
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
        # "null" would be the json.dumps of the assistant turn's None content.
        assert "file_a.txt" in stub.prompt and "null" not in stub.prompt

    def test_thinking_is_not_forced_off(self, http, stub):
        # Hardcoding enable_thinking=False makes the same model behave
        # differently here than at any other endpoint serving it.
        chat(http)
        default = stub.prompt
        chat(http, chat_template_kwargs={"enable_thinking": False})
        assert stub.prompt != default


class TestAuth:
    def test_key_is_enforced_when_configured(self, stub, tokenizer):
        with http_client(make_app(stub, tokenizer, api_key="sekret")) as http:
            assert chat(http).status_code == 401
            http.headers["authorization"] = "Bearer wrong"
            assert chat(http).status_code == 401
            http.headers["authorization"] = "Bearer sekret"
            assert chat(http).status_code == 200

    def test_binding_off_box_without_a_key_is_refused(self):
        from arctic_platform.openai_compat.server import check_bind

        check_bind("127.0.0.1", None)
        check_bind("0.0.0.0", "key")
        with pytest.raises(ValueError, match="Refusing to bind"):
            check_bind("0.0.0.0", None)


class TestLegacyCompletions:
    def test_prompt_shapes(self, http, stub):
        assert http.post("/v1/completions", json={"model": MODEL, "prompt": "2 + 2 ="}).status_code == 200
        choices = http.post("/v1/completions", json={"model": MODEL, "prompt": ["a", "b"], "n": 2}).json()["choices"]
        assert [c["index"] for c in choices] == [0, 1, 2, 3]
        http.post("/v1/completions", json={"model": MODEL, "prompt": [1, 2, 3]})
        assert stub.calls[-1]["prompts"][0] == [1, 2, 3]

    def test_echo_is_refused(self, http):
        assert http.post("/v1/completions", json={"model": MODEL, "prompt": "x", "echo": True}).status_code == 400


class TestOpenAISdk:
    """Every assertion here is really the SDK validating our JSON on the way in."""

    def test_chat_completion(self, sdk):
        completion = sdk.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=32)
        assert isinstance(completion, openai.types.chat.ChatCompletion)
        assert completion.choices[0].message.content == "sampled"
        assert completion.choices[0].finish_reason == "stop"
        assert completion.usage.total_tokens == completion.usage.prompt_tokens + completion.usage.completion_tokens
        assert completion.id.startswith("chatcmpl-") and completion.created > 0
        # Echoed, because clients match it against what they asked for.
        assert completion.model == MODEL

    def test_n_and_logprobs(self, sdk):
        completion = sdk.chat.completions.create(
            model=MODEL, messages=MESSAGES, n=3, max_tokens=8, logprobs=True, top_logprobs=1
        )
        assert [c.index for c in completion.choices] == [0, 1, 2]
        content = completion.choices[0].logprobs.content
        assert content and all(e.token and isinstance(e.logprob, float) for e in content)

    def test_legacy_completion_and_models(self, sdk):
        completion = sdk.completions.create(model=MODEL, prompt="2 + 2 =", max_tokens=8)
        assert isinstance(completion, openai.types.Completion) and completion.choices[0].text
        assert [m.id for m in sdk.models.list().data] == [MODEL]
        assert sdk.models.retrieve(MODEL).id == MODEL

    def test_token_ids_extension_is_exposed(self, sdk):
        # vLLM's OpenAI-server extension; RL harnesses build rollouts from these.
        completion = sdk.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=8)
        raw = completion.model_dump()
        assert raw["prompt_token_ids"] and raw["choices"][0]["token_ids"]

    def test_tool_call_round_trips(self, tokenizer):
        emitted = '<tool_call>{"name": "run_bash_command", "arguments": {"cmd": "ls -la"}}</tool_call>'
        with serve(make_app(StubClient(text=emitted), tokenizer)) as sdk:
            completion = sdk.chat.completions.create(model=MODEL, messages=MESSAGES, tools=TOOLS, max_tokens=32)
        choice = completion.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.content is None
        call = choice.message.tool_calls[0]
        assert call.type == "function" and call.function.name == "run_bash_command"
        assert call.function.arguments == '{"cmd": "ls -la"}'

    @pytest.mark.parametrize(
        "kwargs, exc, needle",
        [
            ({"response_format": {"type": "json_object"}}, "BadRequestError", "response_format"),
            ({"stream": True}, "BadRequestError", "non-streaming"),
            ({"messages": [{"role": "user", "content": "word " * 20_000}]}, "BadRequestError", "context length"),
        ],
    )
    def test_errors_surface_as_typed_sdk_exceptions(self, sdk, kwargs, exc, needle):
        # With FastAPI's default {"detail": ...} these messages would arrive as
        # "Error code: 400" and nothing else.
        with pytest.raises(getattr(openai, exc)) as caught:
            sdk.chat.completions.create(**{"model": MODEL, "messages": MESSAGES, **kwargs})
        assert needle in str(caught.value)

    def test_unknown_model_raises_not_found(self, sdk):
        with pytest.raises(openai.NotFoundError):
            sdk.models.retrieve("some-other-model")

    def test_bad_key_raises_authentication_error(self, stub, tokenizer):
        app = make_app(stub, tokenizer, api_key="right-key")
        with serve(app, api_key="right-key") as sdk:
            assert sdk.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=4)
            wrong = openai.OpenAI(base_url=str(sdk.base_url), api_key="wrong-key", max_retries=0)
            with pytest.raises(openai.AuthenticationError):
                wrong.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=4)

    def test_capacity_error_raises_rate_limit_error(self, tokenizer):
        class Failure(Exception):
            response = type("R", (), {"status_code": 429, "headers": {}})()

        with serve(make_app(StubClient(raises=Failure("at capacity")), tokenizer)) as sdk:
            with pytest.raises(openai.RateLimitError):
                sdk.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=4)


class TestConcurrency:
    def _hammer(self, sdk: Any, count: int) -> list[Any]:
        def one():
            return sdk.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=4)

        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            return [f.result() for f in [pool.submit(one) for _ in range(count)]]

    def test_a_blocking_client_does_not_serialize_callers(self, tokenizer):
        # Awaiting a blocking generate on the event loop turns an N-way parallel
        # eval into an N-times-longer serial one.
        delay_s, count = 0.4, 8
        blocking = BlockingClient(delay_s=delay_s)
        with serve(make_app(blocking, tokenizer, max_concurrency=count)) as sdk:
            started = time.monotonic()
            results = self._hammer(sdk, count)
            elapsed = time.monotonic() - started

        assert len(results) == count and blocking.calls == count
        serial = delay_s * count
        # Generous bound: the claim is "overlapped", not a latency SLA.
        assert elapsed < serial / 2, f"{count} concurrent requests took {elapsed:.2f}s (serial is {serial}s)"

    def test_concurrency_ceiling_is_enforced(self, tokenizer):
        stub = StubClient(delay_s=0.2)
        with serve(make_app(stub, tokenizer, max_concurrency=2)) as sdk:
            self._hammer(sdk, 6)
        assert stub.max_in_flight <= 2
