"""Replay captured sampler text through the gateway's parse path.

The harness validates the JSON message the gateway returns, not the raw text
the sampler produced, so a well-formed completion can still be scored as a
format violation if the parse leaves anything behind. Running the real
functions over the real text is the only way to tell those two apart.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from arctic_platform.openai_compat import _parse_tool_calls, _split_reasoning


def message_for(text: str, think_open: bool, tools: bool) -> dict:
    """Mirror the assembly in ``chat_completions`` for one choice."""
    body, reasoning = _split_reasoning(text, think_open=think_open)
    if tools and reasoning and not body:
        reasoning, calls = _parse_tool_calls(reasoning)
    else:
        body, calls = _parse_tool_calls(body) if tools else (body, [])
    msg: dict = {"content": body or None}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = calls
    return msg


def main() -> None:
    path = Path(sys.argv[1])
    think_open = "--no-think-open" not in sys.argv
    rows = [json.loads(line) for line in path.open() if line.strip()]

    counts: Counter[str] = Counter()
    examples: list[str] = []
    for row in rows:
        text = row.get("text") or row.get("content") or ""
        msg = message_for(text, think_open=think_open, tools=True)

        # These are exactly the harness's two format gates.
        nonempty = msg["content"] not in (None, "", [])
        missing = not msg.get("tool_calls")
        verdict = (
            "nonempty_content" if nonempty else
            "missing_tool_call" if missing else
            "ok"
        )
        counts[verdict] += 1
        if verdict == "nonempty_content" and len(examples) < 3:
            examples.append(repr(msg["content"])[:400])

    print(f"think_open={think_open} turns={len(rows)}")
    for name, n in counts.most_common():
        print(f"{n:5d}  {name}")
    for i, ex in enumerate(examples):
        print(f"\n--- leftover content {i} ---\n{ex}")


if __name__ == "__main__":
    main()
