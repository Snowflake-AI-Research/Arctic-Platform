"""Token-capturing gateway: the interception seam for black-box agents.

A black-box agent never reports token ids, so the driver cannot reconstruct
RL-grade data from its text output: re-tokenizing a rendered transcript drifts at
turn boundaries and silently corrupts credit assignment. The only place the exact
ids exist is the request path itself, which is why prime-rl puts an interception
server here.

This subclasses the PR's DriverOpenAIGateway and adds one HTTP middleware that
records ``prompt_token_ids`` and per-choice ``token_ids`` (the vLLM extensions
arctic_platform.openai_compat already emits) into a per-rollout buffer, keyed by
the bearer token the agent was launched with. The agent stays untouched.

Sampler logprobs are captured alongside the ids. Cortex's grpo loss reads
``old_log_probs_shifted`` when the driver supplies it and otherwise falls back
to ``logprobs.detach()``, i.e. pi_old == pi_new — which is only correct while a
batch is used once. Capturing them here is what lets a batch be replayed across
policy versions the way prime-rl's ``max_off_policy_steps`` does.

The agent is never asked to opt in: a black-box harness has no reason to set
``logprobs=true``, so the middleware sets it on the way in.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from typing import Any

from arctic_platform.integrations.harbor.openai_gateway import DriverOpenAIGateway


class _RawLoggingPool:
    """Tees the sampler's raw text to a file, ahead of any parsing.

    A harness only ever reports its verdict ("format_invalid"), and the router
    rewrites the model's text into content / reasoning_content / tool_calls
    before anyone sees it. When those two disagree, the raw completion is the
    only artifact that says which side is wrong, and by then it is gone.
    """

    def __init__(self, inner: Any, path: str) -> None:
        self._inner = inner
        self._path = path
        self._lock = threading.Lock()
        self._config = inner._config

    async def generate(self, prompts: list[Any], sampling_params: dict[str, Any]) -> Any:
        results = await self._inner.generate(prompts, sampling_params)
        try:
            with self._lock, open(self._path, "a") as fh:
                for r in results:
                    fh.write(json.dumps({
                        "finish_reason": r.get("finish_reason"),
                        "n_tokens": len(r.get("token_ids") or []),
                        "text": r.get("text"),
                    }) + "\n")
        except OSError:
            pass
        return results


def _length_stop_payload() -> dict[str, Any]:
    """Minimal well-formed completion whose only job is to end the rollout."""
    return {
        "id": "chatcmpl-turn-budget",
        "object": "chat.completion",
        "created": 0,
        "model": "turn-budget",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": None},
            "finish_reason": "length",
            "token_ids": [],
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "prompt_token_ids": [],
    }


def _completion_logprobs(choice: dict[str, Any]) -> list[float] | None:
    """Pull the flat per-token logprob list out of an OpenAI choice."""
    block = choice.get("logprobs")
    if not isinstance(block, dict):
        return None
    content = block.get("content")
    if not isinstance(content, list) or not content:
        return None
    out: list[float] = []
    for entry in content:
        if not isinstance(entry, dict) or not isinstance(entry.get("logprob"), (int, float)):
            return None
        out.append(float(entry["logprob"]))
    return out


async def _normalize_sampling(
    request: Any, default_max_tokens: int, enable_thinking: bool = True
) -> None:
    """Fill in the sampling fields a black-box agent has no reason to send.

    Two of them, for different reasons:

    ``logprobs`` — needed for off-policy replay, and no harness asks for it.

    ``max_tokens`` — vLLM's SamplingParams default is 16, so an agent that
    omits it gets a 16-token completion and ``finish_reason="length"``. Under
    mini-swe-agent-plus's truncation grading that is a terminal invalid on turn
    one, which reads as a hard zero-reward wall rather than a missing default.

    ``enable_thinking`` — the router defaults it off to save tokens, which
    pre-closes Qwen3's ``<think>`` block. The harness requires a reasoning
    block on every turn, so with thinking off every rollout is a format
    violation no matter what the policy does.

    Mutates the request in place rather than returning a new one. Starlette's
    BaseHTTPMiddleware re-invokes the downstream app with the *original* scope
    and receive channel, ignoring whatever object is handed to ``call_next``, so
    a replacement request is silently discarded and the handler still parses the
    unmodified body. What does carry through is the body cache: once ``body()``
    has been awaited, ``_CachedRequest.wrapped_receive`` replays ``_body`` to the
    app, so overwriting it here is what the handler actually sees.
    """
    body = await request.body()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return
    payload["logprobs"] = True
    if payload.get("max_tokens") is None and payload.get("max_completion_tokens") is None:
        payload["max_tokens"] = default_max_tokens
    if enable_thinking:
        ct = payload.get("chat_template_kwargs")
        payload["chat_template_kwargs"] = {
            **(ct if isinstance(ct, dict) else {}),
            "enable_thinking": True,
        }
    new_body = json.dumps(payload).encode()

    request._body = new_body
    request.scope["headers"] = [
        (k, v) for k, v in request.scope.get("headers", []) if k.lower() != b"content-length"
    ] + [(b"content-length", str(len(new_body)).encode())]


class Turn:
    __slots__ = ("prompt_token_ids", "completion_token_ids", "logprobs")

    def __init__(
        self,
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        logprobs: list[float] | None = None,
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.completion_token_ids = completion_token_ids
        self.logprobs = logprobs

    def __repr__(self) -> str:
        lp = "none" if self.logprobs is None else str(len(self.logprobs))
        return (
            f"Turn(prompt={len(self.prompt_token_ids)}, "
            f"completion={len(self.completion_token_ids)}, logprobs={lp})"
        )


class CapturingGateway(DriverOpenAIGateway):
    """DriverOpenAIGateway that records token ids per rollout id."""

    def __init__(
        self,
        max_turns: int | None = None,
        default_max_tokens: int = 4096,
        raw_log: str | None = None,
        enable_thinking: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._default_max_tokens = default_max_tokens
        self._raw_log = raw_log
        self._enable_thinking = enable_thinking
        self._turns: dict[str, list[Turn]] = defaultdict(list)
        self._lock = threading.Lock()
        # mini-swe-agent-plus loops on ``while True`` and relies on its
        # orchestrator for a turn budget; running it standalone, the gateway is
        # the only component that sees every turn of a rollout, so the cap
        # lives here. Without it a wedged agent burns the whole rollout
        # timeout instead of terminating as a truncation.
        self._max_turns = max_turns

    def _over_budget(self, rollout_id: str) -> bool:
        if self._max_turns is None:
            return False
        with self._lock:
            return len(self._turns.get(rollout_id, ())) >= self._max_turns

    def turns(self, rollout_id: str) -> list[Turn]:
        with self._lock:
            return list(self._turns.get(rollout_id, ()))

    def reset(self, rollout_id: str) -> None:
        with self._lock:
            self._turns.pop(rollout_id, None)

    def _record(self, rollout_id: str, payload: dict[str, Any]) -> None:
        prompt_ids = payload.get("prompt_token_ids") or []
        choices = payload.get("choices") or []
        if not prompt_ids or not choices:
            return
        completion_ids = choices[0].get("token_ids") or []
        if not completion_ids:
            return
        logprobs = _completion_logprobs(choices[0])
        # A length mismatch means the two came from different views of the
        # completion; pairing them anyway would misalign the importance ratio
        # token-by-token, so drop the weaker signal and stay on-policy.
        if logprobs is not None and len(logprobs) != len(completion_ids):
            logprobs = None
        with self._lock:
            self._turns[rollout_id].append(
                Turn(list(prompt_ids), list(completion_ids), logprobs)
            )

    def _build_app(self) -> Any:
        from fastapi import Request, Response

        app = super()._build_app()
        if self._raw_log is not None:
            app.state.sampling_pool = _RawLoggingPool(
                app.state.sampling_pool, self._raw_log
            )

        @app.middleware("http")
        async def capture_tokens(request: Request, call_next):  # type: ignore[no-untyped-def]
            if request.url.path.endswith("/chat/completions"):
                rid = (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
                if rid and self._over_budget(rid):
                    # ``finish_reason="length"`` is the signal the harness
                    # already handles: with truncation grading on it records a
                    # response_length stop and exits, which is the same
                    # terminal class the reference max_turns cap produces.
                    return Response(
                        content=json.dumps(_length_stop_payload()),
                        status_code=200,
                        media_type="application/json",
                    )
                await _normalize_sampling(
                    request, self._default_max_tokens, self._enable_thinking
                )
            response = await call_next(request)
            if not request.url.path.endswith("/chat/completions"):
                return response

            body = b"".join([chunk async for chunk in response.body_iterator])
            rollout_id = (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
            if rollout_id and response.status_code == 200:
                try:
                    self._record(rollout_id, json.loads(body))
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass

            headers = {
                k: v for k, v in response.headers.items() if k.lower() != "content-length"
            }
            return Response(
                content=body,
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type,
            )

        return app
