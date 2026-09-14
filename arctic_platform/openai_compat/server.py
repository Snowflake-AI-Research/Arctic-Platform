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
"""The ``/v1`` routes and the app that serves them. Non-streaming."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

from arctic_platform._dependency_groups import require_any_dep_group
from arctic_platform.openai_compat import translation as tr
from arctic_platform.openai_compat.translation import OpenAIError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["openai-compat"])

_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


class _State:
    def __init__(
        self,
        client: Any,
        tokenizer: Any,
        model_name: str,
        max_model_len: int,
        api_key: str | None,
        max_concurrency: int,
    ) -> None:
        self.client = client
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_model_len = int(max_model_len)
        self.api_key = api_key
        # Concurrent callers all land on one sampling job; the rest queue here,
        # where a request costs a coroutine rather than a slot.
        self.semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self.created = int(time.time())


def _state(request: Request) -> _State:
    state = getattr(request.app.state, "openai_compat", None)
    if state is None:
        raise OpenAIError(503, "The endpoint is still starting up.", err_type="server_error")
    if state.api_key is not None:
        scheme, _, token = (request.headers.get("authorization") or "").partition(" ")
        if scheme.lower() != "bearer" or token.strip() != state.api_key:
            raise OpenAIError(
                401, "Incorrect API key provided.", err_type="authentication_error", code="invalid_api_key"
            )
    return state


def _as_openai_error(exc: BaseException) -> OpenAIError:
    """Map a backend failure onto the status clients act on.

    Cortex answers 429 when the account is at capacity. Relayed as a 429 with
    Retry-After, stock client retry policy absorbs it; relayed as a 500 it ends
    the caller's run.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status == 429:
        retry_after = (getattr(response, "headers", None) or {}).get("Retry-After")
        return OpenAIError(
            429,
            f"The sampling job is at capacity: {exc}",
            err_type="rate_limit_error",
            code="rate_limit_exceeded",
            headers={"Retry-After": str(retry_after)} if retry_after else {},
        )
    if isinstance(status, int) and 400 <= status < 500:
        return OpenAIError(status, f"The sampling job rejected the request: {exc}")
    return OpenAIError(502, f"The sampling job failed: {type(exc).__name__}: {exc}", err_type="server_error")


async def _generate(state: _State, prompt: str | list[int], params: dict[str, Any], n: int) -> list[dict]:
    """Sample ``n`` completions for one prompt.

    Issued as ``n`` copies rather than SamplingParams(n=n): the sampling worker
    only surfaces the first sub-output, so asking the engine for n returns one.
    The copies share a prefix, so the extra prompts cost roughly their KV.
    """
    prompts: list[Any] = [prompt] * max(1, n)
    async with state.semaphore:
        try:
            generate = state.client.generate
            if inspect.iscoroutinefunction(generate):
                results = await generate(prompts, params)
            else:
                # Awaiting a blocking client on the event loop would stall every
                # other in-flight request behind it.
                results = await asyncio.to_thread(generate, prompts, params)
        except OpenAIError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _as_openai_error(exc) from exc

    if len(results) < len(prompts):
        raise OpenAIError(
            502,
            f"The sampling job returned {len(results)} result(s) for {len(prompts)} prompt(s).",
            err_type="server_error",
        )
    return list(results[: len(prompts)])


def _encode(tokenizer: Any, text: str) -> list[int]:
    # add_special_tokens=False: the chat template already placed the specials,
    # and a second BOS shifts every position.
    return [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]


async def _body(request: Request) -> Any:
    try:
        return await request.json()
    except Exception as exc:  # noqa: BLE001
        raise OpenAIError(400, f"Request body is not valid JSON: {exc}") from exc


def _card(state: _State) -> dict[str, Any]:
    return {"id": state.model_name, "object": "model", "created": state.created, "owned_by": "arctic-platform"}


@router.get("/models")
async def list_models(request: Request) -> dict[str, Any]:
    return {"object": "list", "data": [_card(_state(request))]}


@router.get("/models/{model_id:path}")
async def retrieve_model(model_id: str, request: Request) -> dict[str, Any]:
    state = _state(request)
    if model_id != state.model_name:
        raise OpenAIError(
            404,
            f"The model {model_id!r} does not exist. This endpoint serves {state.model_name!r}.",
            err_type="not_found_error",
            param="model",
            code="model_not_found",
        )
    return _card(state)


@router.post("/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    state = _state(request)
    req = tr.parse_request(await _body(request), tr.ChatCompletionRequest)

    prompt = tr.render_chat_prompt(state.tokenizer, req)
    prompt_token_ids = _encode(state.tokenizer, prompt)
    max_tokens = tr.resolve_max_tokens(
        req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens,
        prompt_tokens=len(prompt_token_ids),
        max_model_len=state.max_model_len,
    )
    want_logprobs = bool(req.logprobs)
    params = tr.sampling_params(
        req, max_tokens=max_tokens, logprobs_topk=(req.top_logprobs or 1) if want_logprobs else None
    )

    return tr.chat_completion(
        await _generate(state, prompt, params, req.n),
        # Echo what the caller asked for; clients match it against what they sent.
        model=req.model,
        prompt_token_ids=prompt_token_ids,
        tokenizer=state.tokenizer,
        want_logprobs=want_logprobs,
        tools_offered=bool(req.tools) and req.tool_choice != "none",
    )


@router.post("/completions")
async def completions(request: Request) -> dict[str, Any]:
    state = _state(request)
    req = tr.parse_request(await _body(request), tr.CompletionRequest)

    raw = req.prompt
    if isinstance(raw, str):
        prompts: list[Any] = [raw]
    elif isinstance(raw, list) and raw and all(isinstance(p, int) for p in raw):
        prompts = [list(raw)]
    else:
        prompts = [p if isinstance(p, str) else list(p) for p in raw]

    prompt_tokens, per_prompt = 0, []
    for prompt in prompts:
        token_ids = prompt if isinstance(prompt, list) else _encode(state.tokenizer, prompt)
        prompt_tokens += len(token_ids)
        params = tr.sampling_params(
            req,
            max_tokens=tr.resolve_max_tokens(
                req.max_tokens, prompt_tokens=len(token_ids), max_model_len=state.max_model_len
            ),
            logprobs_topk=req.logprobs,
        )
        per_prompt.append(await _generate(state, prompt, params, req.n))

    return tr.text_completion(
        per_prompt,
        model=req.model,
        prompt_tokens=prompt_tokens,
        tokenizer=state.tokenizer,
        want_logprobs=req.logprobs is not None,
    )


def build_app(
    *,
    client: Any,
    tokenizer: Any,
    model_name: str,
    max_model_len: int,
    api_key: str | None = None,
    max_concurrency: int = 32,
) -> FastAPI:
    """An app serving ``/v1``.

    ``client`` is anything with ``generate(prompts, sampling_params)``, sync or
    async -- an ``ArcticClient``, an ``AsyncArcticClient``, or a stub.
    """
    require_any_dep_group("openai")
    app = FastAPI(title="arctic-platform OpenAI-compatible endpoint")
    app.state.openai_compat = _State(client, tokenizer, model_name, max_model_len, api_key, max_concurrency)

    async def on_openai_error(_r: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, OpenAIError)
        return JSONResponse(status_code=exc.status_code, content=exc.body(), headers=exc.headers)

    async def on_unhandled(_r: Request, exc: Exception) -> JSONResponse:
        # Otherwise FastAPI renders a body no OpenAI client can parse, turning a
        # backend hiccup into "connection error" at the caller.
        err = OpenAIError(500, f"{type(exc).__name__}: {exc}", err_type="server_error")
        return JSONResponse(status_code=500, content=err.body())

    app.add_exception_handler(OpenAIError, on_openai_error)
    app.add_exception_handler(Exception, on_unhandled)
    app.include_router(router)
    return app


def check_bind(host: str, api_key: str | None) -> None:
    """Refuse to expose an unauthenticated endpoint off-box.

    Binding beyond loopback is what lets a container reach the gateway, and also
    what lets anything else on the network spend the job's GPUs.
    """
    if host not in _LOOPBACK and api_key is None:
        raise ValueError(
            f"Refusing to bind {host} without an API key: the endpoint would accept unauthenticated requests from"
            " anywhere that can route to this host. Pass --api-key, or bind 127.0.0.1."
        )


def main(argv: list[str] | None = None) -> None:
    """Attach to an existing sampling job and serve ``/v1`` until interrupted."""
    parser = argparse.ArgumentParser(prog="python -m arctic_platform.openai_compat")
    parser.add_argument("--config", required=True, type=Path, help="ArcticClientConfig JSON/YAML.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None, help="Required when --host is not loopback.")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--max-concurrency", type=int, default=32)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    check_bind(args.host, args.api_key)

    import uvicorn
    from transformers import AutoTokenizer

    from arctic_platform.client.base import ArcticClient
    from arctic_platform.client.config import ArcticClientConfig

    text = args.config.read_text()
    if args.config.suffix in (".yaml", ".yml"):
        import yaml

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)

    config = ArcticClientConfig.model_validate(raw)
    if config.sampling_job_id is None:
        raise SystemExit("--config must set sampling_job_id: this serves an endpoint, it does not create one.")

    app = build_app(
        client=ArcticClient(config),
        tokenizer=AutoTokenizer.from_pretrained(config.model_name),
        model_name=args.served_model_name or config.model_name,
        max_model_len=config.max_seq_len,
        api_key=args.api_key,
        max_concurrency=args.max_concurrency,
    )
    logger.info("Serving %s at http://%s:%s/v1", config.model_name, args.host, args.port)
    # Deliberately no client.shutdown() on exit: on Cortex that cancels the
    # parent job, taking the endpoint and its GPUs down with the gateway.
    logger.info("The sampling job stays up when this process exits.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
