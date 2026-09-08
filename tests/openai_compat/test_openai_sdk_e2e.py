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
"""End to end against the real ``openai`` SDK over a real socket.

This is the test that actually says "compatible". The SDK parses every
response into its own generated Pydantic models, so a response that is the
wrong shape raises inside the client rather than quietly returning a dict --
which makes the suite a conformance check against OpenAI's schema rather than
against our idea of it. Errors are asserted as the SDK's exception classes for
the same reason.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import time
from typing import Any

import pytest
from openai_harness import MODEL
from openai_harness import BlockingClient
from openai_harness import RecordingBackend
from openai_harness import make_app

openai = pytest.importorskip("openai")

MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash_command",
            "description": "Run a shell command and return stdout.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }
]


@contextlib.contextmanager
def serving(app: Any, *, api_key: str | None = None):
    """Run ``app`` under uvicorn on a real port and yield an OpenAI client."""
    from arctic_platform.openai_compat.server import OpenAIGateway

    gateway = OpenAIGateway(app=app, host="127.0.0.1", api_key=api_key)
    gateway.start()
    try:
        yield openai.OpenAI(base_url=gateway.base_url, api_key=api_key or "not-checked", max_retries=0)
    finally:
        gateway.stop()


@pytest.fixture
def client(app: Any):
    with serving(app) as sdk_client:
        yield sdk_client


class TestHappyPath:
    def test_chat_completion_parses_into_the_sdk_model(self, client):
        completion = client.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=32)
        # Constructing this object is the assertion: the SDK validates the
        # whole payload against OpenAI's schema on the way in.
        assert isinstance(completion, openai.types.chat.ChatCompletion)
        assert completion.object == "chat.completion"
        assert completion.model == MODEL
        assert completion.choices[0].message.role == "assistant"
        assert completion.choices[0].message.content == "hello from the sampler"
        assert completion.choices[0].finish_reason == "stop"
        assert completion.usage.total_tokens == (completion.usage.prompt_tokens + completion.usage.completion_tokens)
        assert completion.id.startswith("chatcmpl-")
        assert completion.created > 0

    def test_n_returns_n_choices(self, client):
        completion = client.chat.completions.create(model=MODEL, messages=MESSAGES, n=3, max_tokens=8)
        assert [choice.index for choice in completion.choices] == [0, 1, 2]

    def test_logprobs_parse(self, client):
        completion = client.chat.completions.create(
            model=MODEL, messages=MESSAGES, logprobs=True, top_logprobs=1, max_tokens=8
        )
        content = completion.choices[0].logprobs.content
        assert content and all(entry.token for entry in content)
        assert all(isinstance(entry.logprob, float) for entry in content)

    def test_legacy_completions_parse(self, client):
        completion = client.completions.create(model=MODEL, prompt="2 + 2 =", max_tokens=8)
        assert isinstance(completion, openai.types.Completion)
        assert completion.object == "text_completion"
        assert completion.choices[0].text

    def test_models_list_and_retrieve(self, client):
        listed = client.models.list()
        assert [model.id for model in listed.data] == [MODEL]
        assert client.models.retrieve(MODEL).id == MODEL

    def test_sampling_knobs_round_trip(self, client, backend):
        client.chat.completions.create(
            model=MODEL, messages=MESSAGES, temperature=0.2, top_p=0.8, seed=11, stop=["END"], max_tokens=5
        )
        params = backend.last_params
        assert params["temperature"] == 0.2 and params["top_p"] == 0.8
        assert params["seed"] == 11 and params["stop"] == ["END"] and params["max_tokens"] == 5


class TestToolCalling:
    """A tool call has to survive both directions: into the prompt, and back."""

    def test_tools_reach_the_prompt(self, client, backend):
        client.chat.completions.create(model=MODEL, messages=MESSAGES, tools=TOOLS, max_tokens=16)
        assert "run_bash_command" in backend.last_prompt

    def test_tool_call_is_returned_in_openai_shape(self, tokenizer):
        emitted = '<tool_call>{"name": "run_bash_command", "arguments": {"cmd": "ls -la"}}</tool_call>'
        backend = RecordingBackend(text=emitted)
        with serving(make_app(backend, tokenizer)) as client:
            completion = client.chat.completions.create(model=MODEL, messages=MESSAGES, tools=TOOLS, max_tokens=32)

        choice = completion.choices[0]
        assert choice.finish_reason == "tool_calls"
        call = choice.message.tool_calls[0]
        assert call.type == "function"
        assert call.function.name == "run_bash_command"
        # The SDK types `arguments` as a JSON string; callers json.loads it.
        assert call.function.arguments == '{"cmd": "ls -la"}'
        assert choice.message.content is None

    def test_multi_turn_tool_transcript_is_accepted(self, client, backend):
        client.chat.completions.create(
            model=MODEL,
            max_tokens=16,
            tools=TOOLS,
            messages=[
                {"role": "user", "content": "list the files"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run_bash_command", "arguments": '{"cmd": "ls"}'},
                        }
                    ],
                },
                {"role": "tool", "content": "a.txt\nb.txt", "tool_call_id": "call_1"},
            ],
        )
        assert "a.txt" in backend.last_prompt


class TestErrorsSurfaceAsSdkExceptions:
    """The envelope only matters if the SDK turns it into a usable exception."""

    def test_unsupported_parameter_raises_bad_request_with_a_readable_message(self, client):
        with pytest.raises(openai.BadRequestError) as caught:
            client.chat.completions.create(model=MODEL, messages=MESSAGES, response_format={"type": "json_object"})
        # With FastAPI's default {"detail": ...} body this message would be
        # "Error code: 400" and nothing else.
        assert "response_format" in str(caught.value)
        assert "not supported" in str(caught.value)

    def test_streaming_request_raises_bad_request(self, client):
        with pytest.raises(openai.BadRequestError) as caught:
            client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
        assert "non-streaming" in str(caught.value)

    def test_unknown_model_raises_not_found(self, client):
        with pytest.raises(openai.NotFoundError):
            client.models.retrieve("some-other-model")

    def test_bad_key_raises_authentication_error(self, backend, tokenizer):
        app = make_app(backend, tokenizer, api_key="right-key")
        with serving(app, api_key="right-key") as client:
            assert client.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=4)
            wrong = openai.OpenAI(base_url=str(client.base_url), api_key="wrong-key", max_retries=0)
            with pytest.raises(openai.AuthenticationError):
                wrong.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=4)

    def test_capacity_error_raises_rate_limit_error(self, tokenizer):
        class Response:
            status_code = 429
            headers = {"Retry-After": "3"}

        class Failure(Exception):
            response = Response()

        backend = RecordingBackend(raises=Failure("account at capacity"))
        with serving(make_app(backend, tokenizer)) as client:
            with pytest.raises(openai.RateLimitError):
                client.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=4)

    def test_context_length_error_is_actionable(self, client):
        with pytest.raises(openai.BadRequestError) as caught:
            client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "word " * 20_000}])
        assert "context length" in str(caught.value)


class TestConcurrency:
    """A blocking client must not serialize concurrent callers.

    Any eval harness issues requests in parallel. Awaiting a blocking
    ``client.generate`` on the server's event loop stalls every other request
    behind it, turning an N-way parallel run into an N-times-longer serial one.
    """

    def test_blocking_client_requests_overlap(self, tokenizer):
        from arctic_platform.openai_compat.server import app_for_client

        delay_s, requests = 0.4, 8
        blocking = BlockingClient(delay_s=delay_s)
        app = app_for_client(
            blocking,
            tokenizer=tokenizer,
            model_name=MODEL,
            max_model_len=4096,
            max_concurrency=requests,
        )

        with serving(app) as client:

            def one() -> Any:
                return client.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=4)

            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=requests) as pool:
                results = [future.result() for future in [pool.submit(one) for _ in range(requests)]]
            elapsed = time.monotonic() - started

        assert len(results) == requests
        assert blocking.calls == requests
        serial = delay_s * requests
        # Generous bound: the point is "overlapped", not a latency SLA. Serial
        # execution would take >= 3.2s here; overlapped takes ~0.4s.
        assert elapsed < serial / 2, f"{requests} concurrent requests took {elapsed:.2f}s (serial would be {serial}s)"

    def test_concurrency_ceiling_is_enforced(self, tokenizer):
        backend = RecordingBackend(delay_s=0.2)
        from arctic_platform.openai_compat.backend import ArcticClientBackend

        class AsyncClient:
            async def generate(self, prompts, sampling_params):
                return await backend.generate(prompts, sampling_params)

        from arctic_platform.openai_compat.server import build_app

        app = build_app(
            backend=ArcticClientBackend(AsyncClient(), max_concurrency=2),
            tokenizer=tokenizer,
            model_name=MODEL,
            max_model_len=4096,
        )
        with serving(app) as client:

            def one() -> Any:
                return client.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=4)

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                [future.result() for future in [pool.submit(one) for _ in range(6)]]

        # Six callers, a ceiling of two: the job never sees more than two at once.
        assert backend.max_in_flight <= 2
