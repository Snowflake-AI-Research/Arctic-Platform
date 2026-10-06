#!/usr/bin/env python3
"""Separate genuine content drift from tokenizer boundary noise.

After the parser fix, packing still rejects most trajectories, but the decoded
windows around the divergence look identical -- the ids differ because
tokenizing a long context greedily merges differently than sampling did one
token at a time. Those two cases call for opposite responses. Real drift means
the trajectory cannot be reconstructed and must stay per-turn. Boundary noise
means the *text* is append-only and only the segmentation moved, so the
trajectory can be packed by locating each turn's text and deriving the mask
from character offsets.

Reports, per trajectory, whether the concatenated turns reproduce the final
context as text, and how many tokens differ when they do.

  python3 diag_retok.py <turn_ids_dir> [limit]
"""

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pack import pack_trajectory, pack_trajectory_exact  # noqa: E402


def main() -> int:
    d = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)

    files = sorted(glob.glob(os.path.join(d, "*.json")))[:limit]
    packable = 0
    text_ok = 0
    text_bad = 0
    tok_deltas = []

    for f in files:
        turns = [(t["prompt"], t["completion"]) for t in json.load(open(f))]
        packed, err = pack_trajectory(turns)
        if packed is not None:
            packable += 1
            continue

        k = err.turn
        prompt, completion = turns[k]
        final = list(turns[-1][0]) + list(turns[-1][1])
        start = len(prompt)
        got = final[start : start + len(completion)]

        # Same question the packer asks, but in text rather than ids.
        if tok.decode(got) == tok.decode(completion):
            text_ok += 1
            diff = sum(1 for a, b in zip(got, completion) if a != b)
            tok_deltas.append(diff / max(1, len(completion)))
        else:
            text_bad += 1
            if text_bad <= 3:
                a = tok.decode(completion)
                b = tok.decode(got)
                i = next(
                    (j for j in range(min(len(a), len(b))) if a[j] != b[j]),
                    min(len(a), len(b)),
                )
                print("=" * 78)
                print(f"{Path(f).name}  turn {k}/{len(turns)}  drift at char {i}")
                print(f"  sampled: {a[max(0, i - 120):i + 200]!r}")
                print(f"  final  : {b[max(0, i - 120):i + 200]!r}")
                print()

    n = len(files)
    print(f"trajectories            : {n}")
    print(f"  packable as-is        : {packable}")
    print(f"  fail, but TEXT matches: {text_ok}   <- boundary noise, recoverable")
    print(f"  fail, TEXT differs    : {text_bad}   <- genuine drift, not recoverable")
    exact_ok = 0
    exact_err: dict[str, int] = {}
    saved_tokens = 0
    per_turn_tokens = 0
    for f in files:
        turns = [(t["prompt"], t["completion"]) for t in json.load(open(f))]
        per_turn_tokens += sum(len(p) + len(c) for p, c in turns)
        packed, err = pack_trajectory_exact(turns, tok)
        if packed is not None:
            exact_ok += 1
            saved_tokens += len(packed.input_ids)
        else:
            exact_err[err.kind] = exact_err.get(err.kind, 0) + 1
            saved_tokens += sum(len(p) + len(c) for p, c in turns)

    print(f"\ntext-space packer        : {exact_ok}/{n}   errors={exact_err or '{}'}")
    if per_turn_tokens:
        print(
            f"  tokens per step        : {per_turn_tokens:,} -> {saved_tokens:,}"
            f"  ({per_turn_tokens / max(1, saved_tokens):.1f}x less)"
        )

    if tok_deltas:
        tok_deltas.sort()
        mid = tok_deltas[len(tok_deltas) // 2]
        print(f"\n  among recoverable, share of differing ids at the failing turn:")
        print(f"    min={tok_deltas[0]:.1%}  median={mid:.1%}  max={tok_deltas[-1]:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
