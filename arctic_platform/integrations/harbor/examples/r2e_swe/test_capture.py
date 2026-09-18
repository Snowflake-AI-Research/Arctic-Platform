"""The interception seam, pinned.

Every failure mode here is silent in production: a dropped max_tokens looks
like a model that always truncates, a dropped logprob list looks like an
on-policy run, and a misaligned one looks like a converging run that is
actually optimising the wrong ratio. None of them raise.

    /data-fast/ap-venv/bin/python -m pytest poc/test_capture.py -q
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from capture import CapturingGateway
from capture import _completion_logprobs
from capture import _normalize_sampling


def _request(payload: Any):
    """A real Starlette ``_CachedRequest``, because the body-cache replay path
    is the thing under test and a hand-rolled stub would not have it."""
    from starlette.middleware.base import _CachedRequest

    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"content-length", str(len(body)).encode())],
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return _CachedRequest(scope, receive)


def _norm(payload: Any, default_max_tokens: int = 4096) -> dict[str, Any]:
    """Normalize, then read the body back the way the downstream app does."""
    req = _request(payload)
    asyncio.run(_normalize_sampling(req, default_max_tokens))
    return json.loads(asyncio.run(req.wrapped_receive())["body"])


def _gateway(max_turns: int | None = None) -> CapturingGateway:
    """Bypass __init__: the parent needs a live Cortex client we don't want."""
    gw = CapturingGateway.__new__(CapturingGateway)
    import threading
    from collections import defaultdict

    gw._turns = defaultdict(list)
    gw._lock = threading.Lock()
    gw._max_turns = max_turns
    gw._default_max_tokens = 4096
    return gw


class TestSamplingDefaults:
    def test_missing_max_tokens_is_filled(self):
        """vLLM would otherwise default to 16 and truncate turn one."""
        assert _norm({"model": "m", "messages": []})["max_tokens"] == 4096

    def test_explicit_max_tokens_is_respected(self):
        assert _norm({"model": "m", "max_tokens": 128})["max_tokens"] == 128

    def test_max_completion_tokens_counts_as_set(self):
        """The renamed field means the same thing; filling both would conflict."""
        out = _norm({"model": "m", "max_completion_tokens": 256})
        assert "max_tokens" not in out

    def test_logprobs_is_forced_on(self):
        assert _norm({"model": "m"})["logprobs"] is True

    def test_thinking_is_enabled(self):
        """Off by default in the router, and the harness rejects every turn
        that carries no reasoning block."""
        out = _norm({"model": "m"})
        assert out["chat_template_kwargs"]["enable_thinking"] is True

    def test_thinking_injection_preserves_other_template_kwargs(self):
        req = _request({"model": "m", "chat_template_kwargs": {"custom": 1}})
        asyncio.run(_normalize_sampling(req, 4096))
        ct = json.loads(asyncio.run(req.wrapped_receive())["body"])["chat_template_kwargs"]
        assert ct == {"custom": 1, "enable_thinking": True}

    def test_thinking_can_be_left_alone(self):
        req = _request({"model": "m"})
        asyncio.run(_normalize_sampling(req, 4096, False))
        body = json.loads(asyncio.run(req.wrapped_receive())["body"])
        assert "chat_template_kwargs" not in body

    def test_edit_survives_the_middleware_receive_replay(self):
        """The regression that cost a live run.

        Returning a new Request from the middleware looks correct and does
        nothing: ``call_next`` re-invokes the app with the original receive
        channel. Only the body cache is replayed, so assert through
        ``wrapped_receive`` — the exact path the handler reads — rather than
        through ``body()``, which would pass either way.
        """
        req = _request({"model": "m", "messages": []})
        asyncio.run(_normalize_sampling(req, 4096))
        replayed = json.loads(asyncio.run(req.wrapped_receive())["body"])
        assert replayed["max_tokens"] == 4096
        assert replayed["logprobs"] is True

    def test_content_length_matches_the_rewritten_body(self):
        """A stale content-length truncates the body the handler parses."""
        req = _request({"model": "m", "messages": []})
        asyncio.run(_normalize_sampling(req, 4096))
        declared = dict(req.scope["headers"])[b"content-length"]
        assert int(declared) == len(asyncio.run(req.wrapped_receive())["body"])

    def test_non_json_body_passes_through_untouched(self):
        req = _request(b"not json")
        asyncio.run(_normalize_sampling(req, 4096))
        assert asyncio.run(req.wrapped_receive())["body"] == b"not json"


class TestLogprobExtraction:
    def test_flat_list_is_extracted_in_order(self):
        choice = {"logprobs": {"content": [{"logprob": -0.5}, {"logprob": -1.5}]}}
        assert _completion_logprobs(choice) == [-0.5, -1.5]

    def test_absent_block_is_none(self):
        assert _completion_logprobs({}) is None

    def test_malformed_entry_rejects_the_whole_list(self):
        """A partial list would be padded with zeros, i.e. pi_old = 1."""
        choice = {"logprobs": {"content": [{"logprob": -0.5}, {"no_logprob": 1}]}}
        assert _completion_logprobs(choice) is None


class TestRecording:
    def test_ids_and_logprobs_are_paired(self):
        gw = _gateway()
        gw._record("r1", {
            "prompt_token_ids": [1, 2],
            "choices": [{"token_ids": [7, 8], "logprobs": {"content": [
                {"logprob": -0.1}, {"logprob": -0.2}]}}],
        })
        (turn,) = gw.turns("r1")
        assert turn.prompt_token_ids == [1, 2]
        assert turn.completion_token_ids == [7, 8]
        assert turn.logprobs == [-0.1, -0.2]

    def test_length_mismatch_drops_logprobs_but_keeps_ids(self):
        """Misaligned logprobs are worse than none: they corrupt the ratio."""
        gw = _gateway()
        gw._record("r1", {
            "prompt_token_ids": [1],
            "choices": [{"token_ids": [7, 8, 9],
                         "logprobs": {"content": [{"logprob": -0.1}]}}],
        })
        (turn,) = gw.turns("r1")
        assert turn.completion_token_ids == [7, 8, 9]
        assert turn.logprobs is None

    def test_turns_accumulate_per_rollout_and_reset_independently(self):
        gw = _gateway()
        for rid in ("a", "a", "b"):
            gw._record(rid, {"prompt_token_ids": [1], "choices": [{"token_ids": [2]}]})
        assert (len(gw.turns("a")), len(gw.turns("b"))) == (2, 1)
        gw.reset("a")
        assert (len(gw.turns("a")), len(gw.turns("b"))) == (0, 1)

    def test_empty_completion_is_not_recorded(self):
        """The turn-budget stop reply carries no ids and must not become a turn."""
        gw = _gateway()
        gw._record("r1", {"prompt_token_ids": [], "choices": [{"token_ids": []}]})
        assert gw.turns("r1") == []


class TestTurnBudget:
    def test_no_budget_never_trips(self):
        gw = _gateway(max_turns=None)
        for _ in range(5):
            gw._record("r1", {"prompt_token_ids": [1], "choices": [{"token_ids": [2]}]})
        assert gw._over_budget("r1") is False

    def test_budget_trips_exactly_at_the_cap(self):
        gw = _gateway(max_turns=2)
        assert gw._over_budget("r1") is False
        gw._record("r1", {"prompt_token_ids": [1], "choices": [{"token_ids": [2]}]})
        assert gw._over_budget("r1") is False
        gw._record("r1", {"prompt_token_ids": [1], "choices": [{"token_ids": [2]}]})
        assert gw._over_budget("r1") is True

    def test_budget_is_per_rollout(self):
        gw = _gateway(max_turns=1)
        gw._record("a", {"prompt_token_ids": [1], "choices": [{"token_ids": [2]}]})
        assert gw._over_budget("a") is True
        assert gw._over_budget("b") is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
