"""Print raw sampler completions from a run's raw_completions.jsonl."""

from __future__ import annotations

import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/data-fast/poc/r2e-raw/raw_completions.jsonl"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 2

rows = [json.loads(line) for line in open(path)]
print(f"{len(rows)} completions in {path}")
for i, r in enumerate(rows[:limit]):
    print(f"\n--- [{i}] finish_reason={r['finish_reason']} n_tokens={r['n_tokens']}")
    text = r["text"] or ""
    print(f"    has </think>: {'</think>' in text}")
    print(f"    has <tool_call>: {'<tool_call>' in text}")
    print(f"    has <function=: {'<function=' in text}")
    print("    ---- text ----")
    print(text[:2500])
