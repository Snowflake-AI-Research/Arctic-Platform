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

Any harness that already speaks chat completions points its client at
``url_for(rollout_id)`` with the key from ``issue``. Each call is stored as an
episode. A request that continues the same messages is sent to ``generate``
with the previous sampled token ids still in the prompt. OpenHands, OpenCode,
and Codex are clients of this server. None of them are imported here.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import threading
import time
import uuid
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

from aiohttp import web

from arctic_platform.integrations.harness.episodes import Session
from arctic_platform.integrations.harness.episodes import normalize_messages
from arctic_platform.integrations.harness.episodes import plan_prompt
from arctic_platform.integrations.harness.episodes import remember
from arctic_platform.integrations.harness.tool_parser import parse_tool_calls as _parse_tool_calls

GenerateFn = Callable[[list[int], dict[str, Any], str | None], Awaitable[dict[str, Any]]]


def _bearer_ok(request: web.Request, api_key: str | None) -> bool:
    if api_key is None:
        return True
    return request.headers.get("Authorization", "") == f"Bearer {api_key}"


def _numeric_logprobs(raw: Any, length: int) -> list[float] | None:
    """Sampler logprobs, when they line up with the response tokens."""
    if not isinstance(raw, list) or len(raw) != length:
        return None
    values = []
    for item in raw:
        if isinstance(item, (int, float)):
            values.append(float(item))
        elif isinstance(item, dict) and isinstance(item.get("logprob"), (int, float)):
            values.append(float(item["logprob"]))
        else:
            return None
    return values


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
        parse_tool_calls: Callable[[str], tuple[str | None, list[dict[str, Any]]]] | None = None,
    ) -> None:
        self._generate_fn = generate_fn
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._max_model_len = max_model_len
        self._default_max_tokens = default_max_tokens
        self._parse_tool_calls = parse_tool_calls if parse_tool_calls is not None else _parse_tool_calls
        self._host = host
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None
        self.base_url: str | None = None
        self._mu = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._key_locks: dict[str, asyncio.Lock] = {}

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

        self._thread = threading.Thread(target=run, name="harness-chat-proxy", daemon=True)
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

    def issue(self, routing_key: str) -> str:
        """Mint the bearer token the harness sends as ``api_key`` for this rollout."""
        token = secrets.token_urlsafe(16)
        with self._mu:
            self._session(routing_key).api_key = token
        return token

    def episodes(self, routing_key: str) -> list:
        """Episodes recorded for one rollout. Empty if the rollout never called the model."""
        with self._mu:
            session = self._sessions.get(routing_key)
            if session is None:
                return []
            return list(session.episodes)

    def drop(self, routing_key: str) -> None:
        """Forget a finished rollout. The next request with this key starts a new session."""
        with self._mu:
            self._sessions.pop(routing_key, None)

    def _session(self, routing_key: str) -> Session:
        session = self._sessions.get(routing_key)
        if session is None:
            session = Session()
            self._sessions[routing_key] = session
        return session

    def _key_lock(self, routing_key: str) -> asyncio.Lock:
        lock = self._key_locks.get(routing_key)
        if lock is None:
            lock = asyncio.Lock()
            self._key_locks[routing_key] = lock
        return lock

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
        if "messages" not in body:
            return _error(400, "messages is required")

        routing_key = request.match_info.get("routing_key") or "_default"
        async with self._key_lock(routing_key):
            with self._mu:
                session = self._session(routing_key)
                if not _bearer_ok(request, session.api_key):
                    return _error(401, "invalid api key")
                snapshot = Session(
                    episodes=list(session.episodes),
                    messages=list(session.messages),
                    raw_messages=list(session.raw_messages),
                    assistant=session.assistant,
                )
            prompt_ids, extended = await asyncio.to_thread(plan_prompt, snapshot, self._tokenizer, body)
            if len(prompt_ids) >= self._max_model_len:
                return _error(
                    400,
                    f"This model's maximum context length is {self._max_model_len} tokens. "
                    f"However, your request has {len(prompt_ids)} input tokens.",
                )
            params = self.sampling_params(body, len(prompt_ids))
            try:
                result = await self._generate_fn(prompt_ids, params, routing_key)
            except Exception as exc:
                return _error(500, f"Cortex generate failed: {exc}")

            response_ids = [int(token) for token in result.get("token_ids") or []]
            text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
            tools_offered = bool(body.get("tools")) and body.get("tool_choice") != "none"
            content, tool_calls = self._parse_tool_calls(text) if tools_offered else (text, [])
            assistant_message = dict(role="assistant", content=content, tool_calls=tool_calls)
            with self._mu:
                remember(
                    self._session(routing_key),
                    body,
                    assistant_message,
                    prompt_ids,
                    response_ids,
                    params,
                    extended,
                    _numeric_logprobs(result.get("logprobs"), len(response_ids)),
                )
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
                        message=assistant_message,
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
