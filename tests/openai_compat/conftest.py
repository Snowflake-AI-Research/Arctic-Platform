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
"""Fixtures. The tokenizer is the real Qwen3 one: a stub proves the routes emit
the right JSON, but only a real chat template proves tools reach the prompt."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from typing import Any

import pytest

MODEL = "Qwen/Qwen3-0.6B"
MAX_MODEL_LEN = 4096


class StubClient:
    """Records calls, replays canned results.

    Recording the *request* matters as much as the response: a dropped sampling
    param, missing tools, or max_tokens=16 are invisible in the reply.
    """

    def __init__(
        self,
        text: str = "sampled",
        *,
        finish_reason: str = "stop",
        delay_s: float = 0.0,
        raises: BaseException | None = None,
        logprobs: Any = None,
    ) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.delay_s = delay_s
        self.raises = raises
        self.logprobs = [-0.1, -0.2, -0.3, -0.4] if logprobs is None else logprobs
        self.calls: list[dict[str, Any]] = []
        self.max_in_flight = 0
        self._in_flight = 0

    @property
    def params(self) -> dict[str, Any]:
        return self.calls[-1]["params"]

    @property
    def prompt(self) -> Any:
        return self.calls[-1]["prompts"][0]

    async def generate(self, prompts: list[Any], sampling_params: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls.append({"prompts": list(prompts), "params": dict(sampling_params)})
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            if self.raises is not None:
                raise self.raises
            return [
                {
                    "text": self.text,
                    "token_ids": [1000, 1001, 1002, 1003],
                    "logprobs": self.logprobs,
                    "finish_reason": self.finish_reason,
                }
                for _ in prompts
            ]
        finally:
            self._in_flight -= 1


class BlockingClient:
    """A client whose generate blocks the calling thread, like ArcticClient."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.calls = 0
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def generate(self, prompts: list[Any], sampling_params: dict[str, Any]) -> list[dict[str, Any]]:
        with self._lock:
            self.calls += 1
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            time.sleep(self.delay_s)
            return [{"text": "ok", "token_ids": [1], "finish_reason": "stop"} for _ in prompts]
        finally:
            with self._lock:
                self._in_flight -= 1


def make_app(client: Any, tokenizer: Any, *, api_key: str | None = None, max_concurrency: int = 32) -> Any:
    from arctic_platform.openai_compat import build_app

    return build_app(
        client=client,
        tokenizer=tokenizer,
        model_name=MODEL,
        max_model_len=MAX_MODEL_LEN,
        api_key=api_key,
        max_concurrency=max_concurrency,
    )


def http_client(app: Any):
    # raise_server_exceptions=False so the app's own 500 handler runs, as it
    # would behind uvicorn.
    from fastapi.testclient import TestClient

    return TestClient(app, raise_server_exceptions=False)


@contextlib.contextmanager
def serve(app: Any, *, api_key: str | None = None):
    """Run the app on a real socket, yield a real openai SDK client."""
    import openai
    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        yield openai.OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=api_key or "unchecked", max_retries=0)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture(scope="session")
def tokenizer() -> Any:
    transformers = pytest.importorskip("transformers")
    # Prefer the cache so the suite stays offline and fast, but fall back to a
    # download: on a cold CI runner the cache-only path would skip every test
    # that needs a chat template, which is most of the interesting ones.
    for kwargs in ({"local_files_only": True}, {}):
        try:
            return transformers.AutoTokenizer.from_pretrained(MODEL, **kwargs)
        except Exception:  # noqa: BLE001
            continue
    pytest.skip(f"{MODEL} tokenizer is unavailable offline and could not be downloaded")


@pytest.fixture
def stub() -> StubClient:
    return StubClient()


@pytest.fixture
def app(stub: StubClient, tokenizer: Any) -> Any:
    return make_app(stub, tokenizer)


@pytest.fixture
def http(app: Any):
    with http_client(app) as client:
        yield client


@pytest.fixture
def sdk(app: Any):
    with serve(app) as client:
        yield client
