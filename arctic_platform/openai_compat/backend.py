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
"""The seam between the OpenAI routes and whatever actually samples.

The router only needs :class:`SamplingBackend`, so the wire contract can be
tested against a stub and the same routes serve any client that can turn
prompts into completions.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any
from typing import Protocol

from arctic_platform.openai_compat.errors import RATE_LIMIT_ERROR
from arctic_platform.openai_compat.errors import SERVER_ERROR
from arctic_platform.openai_compat.errors import OpenAIError

# Concurrent OpenAI requests all land on one sampling job. Without a ceiling a
# harness running dozens of trials at once buries the job in in-flight work; the
# requests queue here instead, where they cost a coroutine rather than a slot.
DEFAULT_MAX_CONCURRENCY = 32


class SamplingBackend(Protocol):
    """Turn prompts into completions. One call, many prompts."""

    async def generate(
        self, prompts: list[str | list[int]], sampling_params: dict[str, Any]
    ) -> list[dict[str, Any]]: ...


def _status_of(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return int(status) if isinstance(status, int) else None


def _retry_after(exc: BaseException) -> str | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        return None
    return str(value) if value is not None else None


def as_openai_error(exc: BaseException) -> OpenAIError:
    """Translate a backend failure into the envelope clients retry on.

    Cortex answers with 429 when the account is at capacity -- including for a
    while after a job is cancelled, since capacity accounting lags. Surfacing
    that as an OpenAI 429 with ``Retry-After`` lets the stock client retry
    policy absorb it; surfacing it as a 500 ends the caller's run.
    """
    status = _status_of(exc)
    if status == 429:
        headers = {}
        retry_after = _retry_after(exc)
        if retry_after is not None:
            headers["Retry-After"] = retry_after
        return OpenAIError(
            429,
            f"The sampling job is at capacity: {exc}",
            err_type=RATE_LIMIT_ERROR,
            code="rate_limit_exceeded",
            headers=headers,
        )
    if status is not None and 400 <= status < 500:
        return OpenAIError(status, f"The sampling job rejected the request: {exc}")
    if isinstance(exc, asyncio.TimeoutError):
        return OpenAIError(504, "The sampling job did not respond in time.", err_type=SERVER_ERROR)
    return OpenAIError(502, f"The sampling job failed: {type(exc).__name__}: {exc}", err_type=SERVER_ERROR)


class ArcticClientBackend:
    """A :class:`SamplingBackend` over ``ArcticClient`` / ``AsyncArcticClient``.

    The blocking client is driven through ``asyncio.to_thread`` rather than
    called inline: awaiting it directly on the server's event loop would stall
    every other in-flight request behind whichever one is talking to the job,
    silently serializing a concurrent workload down to one request at a time.
    """

    def __init__(self, client: Any, *, max_concurrency: int = DEFAULT_MAX_CONCURRENCY) -> None:
        self._client = client
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def generate(self, prompts: list[str | list[int]], sampling_params: dict[str, Any]) -> list[dict[str, Any]]:
        async with self._semaphore:
            try:
                result = self._client.generate(list(prompts), dict(sampling_params))
                if inspect.isawaitable(result):
                    return list(await result)
                return list(await asyncio.to_thread(lambda: result))
            except OpenAIError:
                raise
            except Exception as exc:  # noqa: BLE001 -- re-raised in OpenAI's envelope
                raise as_openai_error(exc) from exc


class BlockingArcticClientBackend(ArcticClientBackend):
    """``ArcticClientBackend`` for a client whose ``generate`` blocks.

    Kept separate so the call itself -- not just the result -- runs off the
    event loop thread.
    """

    async def generate(self, prompts: list[str | list[int]], sampling_params: dict[str, Any]) -> list[dict[str, Any]]:
        async with self._semaphore:
            try:
                return list(await asyncio.to_thread(self._client.generate, list(prompts), dict(sampling_params)))
            except OpenAIError:
                raise
            except Exception as exc:  # noqa: BLE001 -- re-raised in OpenAI's envelope
                raise as_openai_error(exc) from exc


def backend_for(client: Any, *, max_concurrency: int = DEFAULT_MAX_CONCURRENCY) -> SamplingBackend:
    """Pick the adapter that matches ``client.generate``'s call style."""
    if inspect.iscoroutinefunction(getattr(client, "generate", None)):
        return ArcticClientBackend(client, max_concurrency=max_concurrency)
    return BlockingArcticClientBackend(client, max_concurrency=max_concurrency)
