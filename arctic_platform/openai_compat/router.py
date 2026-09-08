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
"""The ``/v1`` routes: models, chat completions, completions. Non-streaming."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi import Request

from arctic_platform.openai_compat import translation
from arctic_platform.openai_compat.backend import as_openai_error
from arctic_platform.openai_compat.errors import AUTHENTICATION_ERROR
from arctic_platform.openai_compat.errors import NOT_FOUND_ERROR
from arctic_platform.openai_compat.errors import SERVER_ERROR
from arctic_platform.openai_compat.errors import OpenAIError
from arctic_platform.openai_compat.schemas import ChatCompletionRequest
from arctic_platform.openai_compat.schemas import CompletionRequest
from arctic_platform.openai_compat.schemas import parse_request

router = APIRouter(prefix="/v1", tags=["openai-compat"])


class GatewayState:
    """What the routes need, resolved once at startup."""

    def __init__(
        self,
        *,
        backend: Any,
        tokenizer: Any,
        model_name: str,
        max_model_len: int,
        api_key: str | None = None,
    ) -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_model_len = int(max_model_len)
        self.api_key = api_key
        self.created = translation.now()


def _state(request: Request) -> GatewayState:
    state = getattr(request.app.state, "openai_compat", None)
    if state is None:
        raise OpenAIError(503, "The endpoint is still starting up.", err_type=SERVER_ERROR)
    return state


def _authorize(request: Request, state: GatewayState) -> None:
    """Check the bearer token, when one is configured.

    Configuring a key is what makes it safe to bind anywhere but loopback, and
    it lets callers keep their existing habit of exporting ``OPENAI_API_KEY``.
    """
    if state.api_key is None:
        return
    header = request.headers.get("authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or token.strip() != state.api_key:
        raise OpenAIError(
            401,
            "Incorrect API key provided.",
            err_type=AUTHENTICATION_ERROR,
            code="invalid_api_key",
        )


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Token ids for an already-rendered prompt.

    ``add_special_tokens=False`` because the chat template has already placed
    whatever specials the model expects; letting the tokenizer add another BOS
    shifts every position and quietly degrades output.
    """
    try:
        return [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]
    except TypeError:
        return [int(t) for t in tokenizer.encode(text)]


async def _generate(state: GatewayState, prompt: str | list[int], params: dict[str, Any], n: int) -> list[dict]:
    """Sample ``n`` completions for one prompt.

    Issued as ``n`` copies of the prompt rather than ``SamplingParams(n=n)``:
    the sampling worker only surfaces the first sub-output, so asking the
    engine for ``n`` returns one. The copies share a prefix, so the engine's
    prefix cache makes the extra prompts cost roughly their KV.
    """
    prompts: list[str | list[int]] = [prompt] * max(1, n)
    try:
        results = await state.backend.generate(prompts, params)
    except OpenAIError:
        raise
    except Exception as exc:  # noqa: BLE001
        # Translated here rather than only in the client adapter so the status
        # a caller sees (429 with Retry-After, 502, 504) is a property of the
        # endpoint, not of which backend happens to be plugged into it.
        raise as_openai_error(exc) from exc
    if len(results) < len(prompts):
        raise OpenAIError(
            502,
            f"The sampling job returned {len(results)} result(s) for {len(prompts)} prompt(s).",
            err_type=SERVER_ERROR,
        )
    return list(results[: len(prompts)])


@router.get("/models")
async def list_models(request: Request) -> dict[str, Any]:
    state = _state(request)
    _authorize(request, state)
    return {"object": "list", "data": [_model_card(state)]}


@router.get("/models/{model_id:path}")
async def retrieve_model(model_id: str, request: Request) -> dict[str, Any]:
    state = _state(request)
    _authorize(request, state)
    if model_id != state.model_name:
        raise OpenAIError(
            404,
            f"The model {model_id!r} does not exist. This endpoint serves {state.model_name!r}.",
            err_type=NOT_FOUND_ERROR,
            param="model",
            code="model_not_found",
        )
    return _model_card(state)


def _model_card(state: GatewayState) -> dict[str, Any]:
    return {
        "id": state.model_name,
        "object": "model",
        "created": state.created,
        "owned_by": "arctic-platform",
    }


@router.post("/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    state = _state(request)
    _authorize(request, state)
    req: ChatCompletionRequest = parse_request(await _json(request), ChatCompletionRequest)

    prompt = translation.render_chat_prompt(
        state.tokenizer,
        req.messages,
        tools=req.tools,
        tool_choice=req.tool_choice,
        chat_template_kwargs=req.chat_template_kwargs,
    )
    prompt_token_ids = _encode(state.tokenizer, prompt)
    max_tokens = translation.resolve_max_tokens(
        req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens,
        prompt_tokens=len(prompt_token_ids),
        max_model_len=state.max_model_len,
    )
    want_logprobs = bool(req.logprobs)
    params = translation.sampling_params(
        req,
        max_tokens=max_tokens,
        logprobs_topk=(req.top_logprobs or 1) if want_logprobs else None,
    )

    results = await _generate(state, prompt, params, req.n)
    return translation.chat_completion(
        results,
        # Echo what the caller asked for. Clients (and cost/telemetry layers
        # downstream of them) match this against the model they requested.
        model=req.model,
        prompt_token_ids=prompt_token_ids,
        tokenizer=state.tokenizer,
        want_logprobs=want_logprobs,
        tools_offered=bool(req.tools) and req.tool_choice != "none",
    )


@router.post("/completions")
async def completions(request: Request) -> dict[str, Any]:
    state = _state(request)
    _authorize(request, state)
    req: CompletionRequest = parse_request(await _json(request), CompletionRequest)

    prompts = _normalize_prompts(req.prompt)
    prompt_tokens = 0
    per_prompt: list[list[dict[str, Any]]] = []
    for prompt in prompts:
        token_ids = prompt if isinstance(prompt, list) else _encode(state.tokenizer, prompt)
        prompt_tokens += len(token_ids)
        max_tokens = translation.resolve_max_tokens(
            req.max_tokens, prompt_tokens=len(token_ids), max_model_len=state.max_model_len
        )
        params = translation.sampling_params(req, max_tokens=max_tokens, logprobs_topk=req.logprobs)
        per_prompt.append(await _generate(state, prompt, params, req.n))

    return translation.text_completion(
        per_prompt,
        model=req.model,
        prompt_tokens=prompt_tokens,
        tokenizer=state.tokenizer,
        want_logprobs=req.logprobs is not None,
    )


def _normalize_prompts(prompt: Any) -> list[str | list[int]]:
    """OpenAI's ``prompt`` is a string, a token-id list, or an array of either."""
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list) and prompt and all(isinstance(p, int) for p in prompt):
        return [list(prompt)]
    if isinstance(prompt, list):
        return [p if isinstance(p, str) else list(p) for p in prompt]
    raise OpenAIError(400, "'prompt' must be a string, a token-id array, or an array of either.", param="prompt")


async def _json(request: Request) -> Any:
    try:
        return await request.json()
    except Exception as exc:  # noqa: BLE001 -- malformed JSON is the caller's problem, said plainly
        raise OpenAIError(400, f"Request body is not valid JSON: {exc}") from exc
