#!/usr/bin/env python3
"""Which tool names does the model invent, and are those traces terminal-invalid?

`unknown_tool` was the largest format failure in the 3-step R2E run. The harness
records the offending name, so the transcripts can say whether the model is
reaching for a submit/finish tool -- which our instance prompt asks for by name
while the harness only exposes execute_bash and edit_via_str_replace.

  python3 diag_unknown_tool.py [transcript_dir]
"""

from __future__ import annotations

import collections
import glob
import json
import re
import sys

ALLOWED = {"execute_bash", "edit_via_str_replace"}


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "/data-fast/poc/r2e-grpo/transcripts"
    paths = sorted(glob.glob(f"{root}/*"))
    print(f"[diag] {len(paths)} transcripts under {root}")

    names: collections.Counter[str] = collections.Counter()
    terminal: collections.Counter[str] = collections.Counter()
    tool_calls: collections.Counter[str] = collections.Counter()
    files_with = 0

    for path in paths:
        try:
            text = open(path, errors="ignore").read()
        except OSError:
            continue
        if "unknown_tool" in text:
            files_with += 1

        # Every tool name the model emitted, whether or not it was rejected.
        for name in re.findall(r'"name"\s*:\s*"([^"]{1,60})"', text):
            tool_calls[name] += 1

        # The harness detail field is f"name={name!r}", adjacent to the code.
        for window in re.findall(r".{0,200}unknown_tool.{0,300}", text, re.S):
            for name in re.findall(r"name=['\"]([^'\"]{1,60})['\"]", window):
                names[name] += 1
            for val in re.findall(r'"terminal_invalid"\s*:\s*(true|false)', window):
                terminal[val] += 1

    print(f"[diag] transcripts containing unknown_tool: {files_with}")

    print("\n[diag] names reported by the unknown_tool detector:")
    if names:
        for name, count in names.most_common(20):
            print(f"  {count:4d}  {name!r}")
    else:
        print("  (detector detail not captured in transcript text)")

    print("\n[diag] all tool names the model emitted:")
    for name, count in tool_calls.most_common(20):
        flag = "" if name in ALLOWED else "   <-- NOT ALLOWED"
        print(f"  {count:6d}  {name!r}{flag}")

    print(f"\n[diag] terminal_invalid values near unknown_tool: {dict(terminal) or 'none found'}")

    # A transcript may be JSON; if so, report its top-level shape once to make
    # further digging cheap rather than guessing at the format.
    if paths:
        try:
            obj = json.load(open(paths[0], errors="ignore"))
            keys = list(obj)[:15] if isinstance(obj, dict) else f"list[{len(obj)}]"
            print(f"[diag] transcript[0] is JSON, top level: {keys}")
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            print("[diag] transcript[0] is not plain JSON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
