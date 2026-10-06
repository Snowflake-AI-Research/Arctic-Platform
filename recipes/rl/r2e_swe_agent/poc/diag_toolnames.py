#!/usr/bin/env python3
"""Does `unknown_tool` come from the model or from our parser?

`unknown_tool` is the single largest cause of thrown-away solves: across the
first three steps it ended 47 rollouts and cost 16 trajectories that had
already fixed the bug. A naive scan of the raw completions finds only 25
invented names in 14,000 tool calls, which is nowhere near enough to explain
that, so one of the two numbers is measuring the wrong thing.

The way to tell them apart is to stop grepping and run the real parser. This
replays every raw completion through the exact `_parse_tool_calls` the gateway
uses and reports the names it emits. A name the model never wrote, or a call
the model wrote and the parser dropped, is our bug; a genuinely invented name
is the model's.

  python3 diag_toolnames.py <run_dir>
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ap-harbor"))

from arctic_platform.openai_compat import _parse_tool_calls  # noqa: E402

ALLOWED = {"execute_bash", "edit_via_str_replace"}

TOOLS = [
    {"type": "function", "function": {
        "name": "execute_bash",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    }},
    {"type": "function", "function": {
        "name": "edit_via_str_replace",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
        }},
    }},
]

# What the model literally typed, independent of whether the parser accepted it.
RAW_FN = re.compile(r"<function=([^>\s]{1,60})")
RAW_WRAP = re.compile(r"<tool_call>")


def main() -> int:
    run = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    path = run / "raw_completions.jsonl"

    parsed_names: collections.Counter[str] = collections.Counter()
    written_names: collections.Counter[str] = collections.Counter()
    n = wrote_call = parser_found = unwrapped = 0

    for line in path.open():
        try:
            text = json.loads(line).get("text") or ""
        except json.JSONDecodeError:
            continue
        n += 1

        written = RAW_FN.findall(text)
        written_names.update(written)
        if written:
            wrote_call += 1

        _, calls = _parse_tool_calls(text, TOOLS)
        if calls:
            parser_found += 1
            parsed_names.update(c["function"]["name"] for c in calls)
        elif written and not RAW_WRAP.search(text):
            # The model wrote a call but omitted the <tool_call> wrapper the
            # parser keys on, so the call never becomes a call.
            unwrapped += 1

    print(f"{n} completions")
    print(f"  wrote something that looks like a call : {wrote_call}")
    print(f"  parser produced a call                 : {parser_found}")
    print(f"  wrote a call but had no <tool_call>    : {unwrapped}")

    def table(title: str, c: collections.Counter[str]) -> None:
        total = sum(c.values())
        bad = sum(v for k, v in c.items() if k not in ALLOWED)
        print(f"\n{title}: {total} calls, {bad} not a real tool ({bad / max(total, 1) * 100:.2f}%)")
        for name, count in c.most_common(18):
            flag = "" if name in ALLOWED else "   <-- not a real tool"
            print(f"  {count:7d}  {name!r}{flag}")

    table("as the model wrote them", written_names)
    table("as the parser reports them", parsed_names)

    only_parser = {k: v for k, v in parsed_names.items() if k not in written_names}
    if only_parser:
        print(f"\nnames the parser invented that the model never wrote: {only_parser}")
    else:
        print("\nno name appears from the parser that the model did not write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
