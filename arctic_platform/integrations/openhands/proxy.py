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
"""OpenAI chat server in front of Cortex ``generate``.

Adapted from ``src/cortex/sampling_proxy.py`` in
https://github.com/18jeffreyma/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2

OpenHands speaks chat completions. Cortex sampling takes token ids. This
server applies the chat template, calls ``generate``, parses tool calls, and
returns ``prompt_token_ids`` and ``token_ids`` so the trainer can stitch the
sampled tokens. Each trajectory gets ``{base}/r/{routing_key}/v1`` so every
turn of that trajectory hits the same sampler replica.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

from aiohttp import web

from arctic_platform.integrations.openhands.tool_parser import parse_tool_calls

GenerateFn = Callable[[list[int], dict[str, Any], str | None], Awaitable[dict[str, Any]]]


def _text_content(content: Any) -> Any:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten content parts and decode tool-call argument strings.

    Qwen's chat template runs ``arguments | tojson``. It needs a mapping, which
    is what vLLM produces before rendering.
    """
    normalized = []
    for message in messages:
        message = dict(message)
        if "content" in message:
            message["content"] = _text_content(message["content"])
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                call = json.loads(json.dumps(call))
                arguments = call.get("function", {}).get("arguments")
                if isinstance(arguments, str):
                    try:
                        call["function"]["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                calls.append(call)
            message["tool_calls"] = calls
        normalized.append(message)
    return normalized


def _error(status: int, message: str) -> web.Response:
    return web.json_response(
        dict(error=dict(message=message, type="BadRequestError", code=status)),
        status=status,
    )


class ChatCompletionsProxy:
    """One local server. Rollouts are distinguished by the routing-key path."""

    def __init__(
        self,
        generate_fn: GenerateFn,
        tokenizer: Any,
        model_name: str,
        max_model_len: int,
        default_max_tokens: int,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._generate_fn = generate_fn
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._max_model_len = max_model_len
        self._default_max_tokens = default_max_tokens
        self._host = host
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None
        self.base_url: str | None = None

    def url_for(self, routing_key: str) -> str:
        if self.base_url is None:
            raise RuntimeError("proxy is not running")
        return f"{self.base_url}/r/{routing_key}/v1"

    def start(self) -> str:
        started = threading.Event()
        failure: list[BaseException] = []

        def run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._serve())
            except BaseException as exc:
                failure.append(exc)
                started.set()
                return
            started.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, name="openhands-sampling-proxy", daemon=True)
        self._thread.start()
        started.wait()
        if len(failure) > 0:
            raise failure[0]
        if self.base_url is None:
            raise RuntimeError("proxy failed to bind")
        return self.base_url

    async def _serve(self) -> None:
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_post("/v1/chat/completions", self._chat_completions)
        app.router.add_post("/r/{routing_key}/v1/chat/completions", self._chat_completions)
        app.router.add_get("/health", self._health)
        app.router.add_get("/v1/models", self._models)
        app.router.add_get("/r/{routing_key}/v1/models", self._models)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        port = self._runner.addresses[0][1]
        self.base_url = f"http://{self._host}:{port}"

    def stop(self) -> None:
        if self._loop is None or self._thread is None:
            return

        async def shutdown() -> None:
            if self._runner is not None:
                await self._runner.cleanup()

        asyncio.run_coroutine_threadsafe(shutdown(), self._loop).result(timeout=30)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=30)
        self._loop = None

    async def _health(self, request: web.Request) -> web.Response:
        return web.json_response(dict(status="ok"))

    async def _models(self, request: web.Request) -> web.Response:
        return web.json_response(
            dict(object="list", data=[dict(id=self._model_name, object="model", owned_by="cortex")])
        )

    def render_prompt(self, body: dict[str, Any]) -> list[int]:
        template_kwargs = dict(body.get("chat_template_kwargs") or {})
        add_generation_prompt = template_kwargs.pop("add_generation_prompt", True)
        text = self._tokenizer.apply_chat_template(
            normalize_messages(body["messages"]),
            tools=body.get("tools") or None,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **template_kwargs,
        )
        return self._tokenizer.encode(text, add_special_tokens=False)

    def sampling_params(self, body: dict[str, Any], prompt_len: int) -> dict[str, Any]:
        requested = body.get("max_completion_tokens") or body.get("max_tokens") or self._default_max_tokens
        params: dict[str, Any] = dict(max_tokens=min(int(requested), self._max_model_len - prompt_len))
        for key in ("temperature", "top_p", "top_k", "stop", "seed"):
            if body.get(key) is not None:
                params[key] = body[key]
        return params

    async def _chat_completions(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error(400, "request body is not valid JSON")
        if body.get("stream"):
            return _error(400, "streaming is not supported")
        if body.get("n", 1) != 1:
            return _error(400, "only n=1 is supported")

        prompt_ids = await asyncio.to_thread(self.render_prompt, body)
        if len(prompt_ids) >= self._max_model_len:
            return _error(
                400,
                f"This model's maximum context length is {self._max_model_len} tokens. "
                f"However, your request has {len(prompt_ids)} input tokens.",
            )
        try:
            result = await self._generate_fn(
                prompt_ids,
                self.sampling_params(body, len(prompt_ids)),
                request.match_info.get("routing_key"),
            )
        except Exception as exc:
            return _error(500, f"Cortex generate failed: {exc}")

        response_ids = [int(token) for token in result.get("token_ids") or []]
        text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
        tools_offered = bool(body.get("tools")) and body.get("tool_choice") != "none"
        content, tool_calls = parse_tool_calls(text) if tools_offered else (text, [])
        finish_reason = result.get("finish_reason") or "stop"
        if len(tool_calls) > 0 and finish_reason == "stop":
            finish_reason = "tool_calls"
        return web.json_response(
            dict(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                object="chat.completion",
                created=int(time.time()),
                model=body.get("model", self._model_name),
                choices=[
                    dict(
                        index=0,
                        message=dict(role="assistant", content=content, tool_calls=tool_calls),
                        logprobs=None,
                        finish_reason=finish_reason,
                        token_ids=response_ids,
                    )
                ],
                usage=dict(
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=len(response_ids),
                    total_tokens=len(prompt_ids) + len(response_ids),
                ),
                prompt_token_ids=prompt_ids,
            )
        )
