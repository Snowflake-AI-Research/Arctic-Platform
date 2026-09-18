"""Prove the gateway hands the sampler the right params, without a GPU.

The previous live run burned ~5 minutes of Cortex provisioning to discover that
the middleware's edits never reached the router. This checks the same path in a
second against a stub client that simply records what it was asked to sample,
so an injection bug shows up here instead of as a wall of zero rewards.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture import CapturingGateway  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
REPLY = "<think>Looking.</think>\n<tool_call>\n<function=execute_bash>\n<parameter=command>\nls /testbed\n</parameter>\n</function>\n</tool_call>"


class StubClient:
    """Records sampling_params; replies with one well-formed tool call."""

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def generate(self, prompts, sampling_params):
        self.seen.append(dict(sampling_params))
        ids = [101, 102, 103]
        return [{
            "text": REPLY,
            "token_ids": ids,
            "finish_reason": "stop",
            # vLLM's shape: one dict per position, keyed by token id.
            "logprobs": [
                {t: {"logprob": -0.1 * (i + 1), "rank": 1, "decoded_token": "x"}}
                for i, t in enumerate(ids)
            ],
        } for _ in prompts]


TOOLS = [{
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": "Run a bash command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]


def main() -> int:
    from transformers import AutoTokenizer

    client = StubClient()
    gw = CapturingGateway(
        client=client,
        tokenizer=AutoTokenizer.from_pretrained(MODEL),
        model_name=MODEL,
        host="127.0.0.1",
        max_turns=2,
        default_max_tokens=4096,
    )
    base = gw.start()
    failures: list[str] = []

    def post(rollout_id: str) -> dict:
        # Exactly the body mini-swe-agent-plus sends: no max_tokens, no logprobs.
        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Fix the bug in /testbed."}],
            "tools": TOOLS,
            "parallel_tool_calls": False,
            "temperature": 1.0,
        }).encode()
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {rollout_id}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    try:
        first = post("r1")

        if not client.seen:
            failures.append("sampler was never called")
        else:
            sp = client.seen[-1]
            if sp.get("max_tokens") != 4096:
                failures.append(f"max_tokens not injected: {sp.get('max_tokens')!r}")
            # 0 is the wanted value — sampled-token log-probs, no alternatives —
            # so this must be a None check, not a truthiness check.
            if sp.get("logprobs") is None:
                failures.append("logprobs not requested")

        choice = first["choices"][0]
        if choice.get("finish_reason") != "tool_calls":
            failures.append(f"finish_reason={choice.get('finish_reason')!r}, want tool_calls")
        calls = choice["message"].get("tool_calls") or []
        if len(calls) != 1 or calls[0]["function"]["name"] != "execute_bash":
            failures.append(f"tool_calls not parsed: {calls!r}")
        elif json.loads(calls[0]["function"]["arguments"]).get("command") != "ls /testbed":
            failures.append(f"bad arguments: {calls[0]['function']['arguments']!r}")
        if not choice["message"].get("reasoning_content"):
            failures.append("reasoning_content missing (harness requires reasoning)")

        turns = gw.turns("r1")
        if len(turns) != 1:
            failures.append(f"captured {len(turns)} turns, want 1")
        elif turns[0].logprobs != [-0.1, -0.2, -0.30000000000000004]:
            failures.append(f"logprobs not captured: {turns[0].logprobs!r}")
        elif turns[0].completion_token_ids != [101, 102, 103]:
            failures.append(f"token ids not captured: {turns[0].completion_token_ids!r}")
        elif not turns[0].prompt_token_ids:
            failures.append("prompt token ids empty")

        # Turn budget: cap is 2, so the third call must come back as a
        # truncation without reaching the sampler.
        post("r1")
        before = len(client.seen)
        third = post("r1")
        if third["choices"][0].get("finish_reason") != "length":
            failures.append(f"turn budget did not trip: {third['choices'][0]}")
        if len(client.seen) != before:
            failures.append("over-budget request still hit the sampler")
    finally:
        gw.stop()

    if failures:
        print("[check] FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[check] PASS: max_tokens injected, logprobs requested and captured, "
          "tool calls parsed, reasoning split, turn budget enforced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
