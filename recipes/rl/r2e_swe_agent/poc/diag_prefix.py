"""Locate where a turn's prompt stops matching its predecessor, and decode it.

A multi-turn agent should send turn N+1 as turn N's prompt plus turn N's
completion plus the new tool result. When it does not, two things break at
once: the trainer is fed a context the sampler never saw, and the sampler's
prefix cache misses from the divergence point onward, so every turn re-prefills
the whole conversation instead of just the new tokens.

Pointing at the first differing token and printing the text on both sides says
which of the two it is -- a re-rendering artifact in our own gateway, or the
harness legitimately rewriting history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """Index of the first mismatch, or None when ``a`` is a prefix of ``b``."""
    if len(a) > len(b):
        return min(len(b), next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(b)))
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("turn_ids", help="a s<step>-<rollout>.json from the run's turn_ids dir")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--window", type=int, default=60, help="tokens of context to decode")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    turns = json.loads(Path(args.turn_ids).read_text())
    print(f"{len(turns)} turns")

    for i in range(len(turns) - 1):
        # What turn i+1's prompt should have been, if history is append-only.
        expected = turns[i]["prompt"] + turns[i]["completion"]
        actual = turns[i + 1]["prompt"]
        d = first_divergence(expected, actual)
        if d is None:
            continue

        # Re-tokenizing a concatenation does not have to reproduce the token
        # boundaries the sampler produced incrementally, so ids can differ
        # where the text does not. Only a text-level difference means history
        # was actually rewritten.
        same_text = tok.decode(actual).startswith(tok.decode(expected))
        print(f"\ntext-level append-only holds: {same_text}"
              f"  ({'tokenization boundary artifact' if same_text else 'REAL content divergence'})")

        print(f"\nfirst divergence: turn {i} -> {i + 1} at token {d}")
        print(f"  prompt len {len(turns[i]['prompt'])}, completion len {len(turns[i]['completion'])}")
        print(f"  expected {len(expected)} tokens, actual prompt {len(actual)} tokens")
        lo = max(0, d - args.window)
        print(f"\n--- common context before divergence ---\n{tok.decode(expected[lo:d])!r}")
        print(f"\n--- expected (turn {i} completion continues) ---\n{tok.decode(expected[d:d + args.window])!r}")
        print(f"\n--- actual   (what turn {i + 1} was sent) ---\n{tok.decode(actual[d:d + args.window])!r}")
        return

    print("append-only holds for every turn")


if __name__ == "__main__":
    main()
