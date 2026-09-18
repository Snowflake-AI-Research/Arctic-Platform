"""Categorize raw completions against the harness's format rules.

The harness only reports a verdict per rollout, so a run that dies on turn 14
looks identical to one that dies on turn 1. This replays every captured
completion through the same parser the gateway uses and counts which rule each
one would break, which is what says whether the remaining zeros are a parser
gap or the policy being bad at the task.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

sys.path.insert(0, "/modeling-code/karthik/abstract-remote-exps/ap-harbor")

from arctic_platform.openai_compat import _parse_tool_calls  # noqa: E402
from arctic_platform.openai_compat import _split_reasoning  # noqa: E402

path = sys.argv[1]
rows = [json.loads(line) for line in open(path)]

verdicts: Counter[str] = Counter()
examples: dict[str, str] = {}

for r in rows:
    text = r.get("text") or ""
    if r.get("finish_reason") == "length":
        verdicts["truncated_response_length"] += 1
        examples.setdefault("truncated_response_length", text[-600:])
        continue

    content, reasoning = _split_reasoning(text, think_open=True)
    if reasoning and not content:
        reasoning, calls = _parse_tool_calls(reasoning)
    else:
        content, calls = _parse_tool_calls(content)

    problems = []
    if not reasoning:
        problems.append("missing_reasoning_block")
    if content.strip():
        problems.append("nonempty_content")
    if len(calls) == 0:
        problems.append("no_tool_call")
    elif len(calls) > 1:
        problems.append("too_many_tool_calls")

    key = ",".join(problems) if problems else "OK"
    verdicts[key] += 1
    if key != "OK":
        examples.setdefault(key, text[:1200])

total = len(rows)
print(f"{total} completions in {path}\n")
for key, n in verdicts.most_common():
    print(f"{n:5d}  {100 * n / total:5.1f}%  {key}")

for key, sample in examples.items():
    print(f"\n{'=' * 70}\nEXAMPLE: {key}\n{'=' * 70}\n{sample}")
