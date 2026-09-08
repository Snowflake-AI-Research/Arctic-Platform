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
"""Shared pieces for the OpenAI-compat tests: a recording backend and app builders."""

from __future__ import annotations

import asyncio
from typing import Any

MODEL = "Qwen/Qwen3-0.6B"
MAX_MODEL_LEN = 4096


class RecordingBackend:
    """A ``SamplingBackend`` that records its calls and replays canned results.

    Recording the *request* matters as much as the response: most of what can
    go wrong here (a dropped sampling param, tools missing from the prompt, a
    max_tokens default of 16) is invisible in the reply and obvious in the call.
    """

    def __init__(
        self,
        text: str = "hello from the sampler",
        *,
        finish_reason: str = "stop",
        delay_s: float = 0.0,
        raises: BaseException | None = None,
    ) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.delay_s = delay_s
        self.raises = raises
        self.calls: list[dict[str, Any]] = []
        self.max_in_flight = 0
        self._in_flight = 0

    @property
    def last_params(self) -> dict[str, Any]:
        return self.calls[-1]["sampling_params"]

    @property
    def last_prompt(self) -> Any:
        return self.calls[-1]["prompts"][0]

    async def generate(self, prompts: list[str | list[int]], sampling_params: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls.append({"prompts": list(prompts), "sampling_params": dict(sampling_params)})
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
                    "token_ids": [1000 + i for i in range(4)],
                    "logprobs": [-0.1, -0.2, -0.3, -0.4],
                    "finish_reason": self.finish_reason,
                }
                for _ in prompts
            ]
        finally:
            self._in_flight -= 1


class BlockingClient:
    """A client whose ``generate`` blocks the calling thread, like ``ArcticClient``."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.calls = 0

    def generate(self, prompts: list[Any], sampling_params: dict[str, Any]) -> list[dict[str, Any]]:
        import time

        self.calls += 1
        time.sleep(self.delay_s)
        return [{"text": "ok", "token_ids": [1], "finish_reason": "stop"} for _ in prompts]


def make_app(backend: Any, tokenizer: Any, *, api_key: str | None = None) -> Any:
    from arctic_platform.openai_compat.server import build_app

    return build_app(
        backend=backend,
        tokenizer=tokenizer,
        model_name=MODEL,
        max_model_len=MAX_MODEL_LEN,
        api_key=api_key,
    )


def http_client(app: Any):
    """A raw HTTP client. ``raise_server_exceptions=False`` so the app's own
    500 handler runs exactly as it would behind uvicorn."""
    from fastapi.testclient import TestClient

    return TestClient(app, raise_server_exceptions=False)
