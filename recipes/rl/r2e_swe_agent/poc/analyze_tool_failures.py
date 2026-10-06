"""Classify the tool-call names a run emitted against the harness's real schema.

``unknown_tool`` is the single largest killer of trajectories, but the label
hides three very different causes, and only one of them is the model being
wrong about which tools exist:

  syntax     the name is a valid tool wearing a malformed envelope, e.g. a
             doubled ``function=`` prefix or a leading space. A server-side
             tool-call parser would absorb these; ours does not, so they are
             the defensible thing to normalise.
  collapse   the model wrote the shell command where the function name goes,
             dropping the <parameter=command> wrapper. Recoverable in
             principle but it is a real modelling error, not a parse artifact.
  invented   a tool that simply does not exist.

The split matters because it bounds how much a gateway fix can buy.

  python3 analyze_tool_failures.py runs/<run>/raw_completions.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

VALID = {"execute_bash", "edit_via_str_replace"}
NAME_RE = re.compile(r"<function=([^>]*)>")
# A function name is a bare identifier. Anything with a space, slash, quote or
# newline in it is a shell command that lost its <parameter=command> wrapper.
COMMANDISH = re.compile(r"[\s/'\"|&]")


def classify(name: str) -> tuple[str, str | None]:
    """Return (category, repaired_name)."""
    stripped = name.strip()
    if stripped in VALID:
        return "syntax", stripped
    while stripped.startswith("function="):
        stripped = stripped[len("function=") :]
        if stripped in VALID:
            return "syntax", stripped
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name.strip()).lower()
    if snake in VALID:
        return "syntax", snake
    if COMMANDISH.search(name):
        return "collapse", None
    return "invented", None


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    counts: Counter[str] = Counter()
    names: Counter[str] = Counter()
    total_calls = 0
    with open(sys.argv[1]) as fh:
        for line in fh:
            text = json.loads(line).get("text", "")
            for raw in NAME_RE.findall(text):
                total_calls += 1
                if raw in VALID:
                    continue
                cat, _ = classify(raw)
                counts[cat] += 1
                names[f"[{cat}] {raw[:46]}"] += 1

    bad = sum(counts.values())
    print(f"tool calls: {total_calls}   invalid: {bad} ({bad / max(total_calls,1):.2%})\n")
    for cat in ("syntax", "collapse", "invented"):
        print(f"  {cat:<10}{counts[cat]:>5}")
    print("\ntop offenders:")
    for k, v in names.most_common(15):
        print(f"  {v:>4}  {k}")
    print(
        f"\nrecoverable by envelope normalisation alone: {counts['syntax']} "
        f"({counts['syntax'] / max(bad,1):.0%} of invalid calls)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
