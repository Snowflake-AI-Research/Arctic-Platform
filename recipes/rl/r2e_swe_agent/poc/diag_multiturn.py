#!/usr/bin/env python3
"""Replay a multi-turn conversation against the live gateway, as the harness does.

Single-turn probes against the gateway return a well-formed ``tool_calls``
array, yet real rollouts die at turn ~38 with ``missing_tool_call``. The
difference between the two is conversation length and the replay of prior
assistant turns, so this reproduces exactly that: same request shape the
harness sends, same feeding back of tool results, growing context.

Reports the first turn whose response carries no ``tool_calls``, and dumps
enough of that response to tell a parse failure from an empty generation.

  python3 diag_multiturn.py <base_url> [turns]
"""

import json
import sys
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Run a bash command in the repository.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]

# Padding that makes each replayed tool result resemble real command output,
# so context grows at roughly the rate a real rollout sees rather than
# staying trivially short.
FILLER = "\n".join(f"/testbed/module_{i}/file_{i}.py" for i in range(120))


def post(base_url: str, messages: list[dict], api_key: str) -> dict:
    body = json.dumps(
        {
            "model": "Qwen/Qwen3.5-4B",
            "messages": messages,
            "tools": TOOLS,
            "parallel_tool_calls": False,
            "temperature": 1.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://10.42.0.1:19914/v1"
    turns = int(sys.argv[2]) if len(sys.argv) > 2 else 45

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                "Fix the failing tests in /testbed. Investigate the repository "
                "step by step using execute_bash."
            ),
        }
    ]
    api_key = "diag-multiturn"

    for turn in range(1, turns + 1):
        resp = post(base_url, messages, api_key)
        choice = resp["choices"][0]
        msg = choice["message"]
        calls = msg.get("tool_calls") or []
        n_prompt = resp.get("usage", {}).get("prompt_tokens", -1)

        if not calls:
            print(f"\n*** turn {turn}: NO tool_calls  (prompt_tokens={n_prompt}) ***")
            print("finish_reason  :", choice.get("finish_reason"))
            print("content        :", repr((msg.get("content") or "")[:600]))
            print("reasoning tail :", repr((msg.get("reasoning_content") or "")[-600:]))
            return 1

        print(f"turn {turn:3d}  ok  calls={len(calls)}  prompt_tokens={n_prompt}")
        messages.append(
            {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": calls,
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": calls[0]["id"],
                "content": f"<returncode>0</returncode>\n<output>\n{FILLER}\n</output>",
            }
        )

    print(f"\nall {turns} turns returned tool_calls; did not reproduce")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
