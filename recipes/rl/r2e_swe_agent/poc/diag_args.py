"""Find tool-call arguments the gateway hands over with the wrong JSON type.

Qwen3.5 emits parameters as untyped XML text, so the parser has to decide what
each value *is*. Guessing with ``json.loads`` silently turns ``0`` into an int
and ``true`` into a bool, and the harness's schema check requires plain strings
for every argument the SWE tools take. A turn lost this way is a well-formed
tool call scored as a protocol violation.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from arctic_platform.openai_compat import _parse_tool_calls, _split_reasoning

# Arguments the harness validates with isinstance(x, str).
STRING_ARGS = {"command", "path", "old_str", "new_str"}


def main() -> None:
    rows = [json.loads(l) for l in Path(sys.argv[1]).open() if l.strip()]

    bad: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    argnames: Counter[str] = Counter()
    examples: list[str] = []
    total_calls = 0

    for row in rows:
        text = row.get("text") or row.get("content") or ""
        body, reasoning = _split_reasoning(text, think_open=True)
        if reasoning and not body:
            _, calls = _parse_tool_calls(reasoning)
        else:
            _, calls = _parse_tool_calls(body)

        for call in calls:
            total_calls += 1
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            tools[name] += 1
            for key in args:
                argnames[f"{name}.{key}"] += 1
            for key, value in args.items():
                if key in STRING_ARGS and not isinstance(value, str):
                    bad[f"{name}.{key} -> {type(value).__name__}"] += 1
                    if len(examples) < 6:
                        examples.append(f"{name}.{key} = {value!r}")

    print(f"tool calls: {total_calls}   mistyped args: {sum(bad.values())}")
    for name, n in bad.most_common():
        print(f"{n:4d}  {name}")
    for ex in examples:
        print(f"  e.g. {ex}")

    print("\n--- tool names ---")
    for name, n in tools.most_common():
        print(f"{n:5d}  {name}")
    print("\n--- argument names ---")
    for name, n in argnames.most_common():
        print(f"{n:5d}  {name}")


if __name__ == "__main__":
    main()
