#!/usr/bin/env python3
"""Explain why a trajectory fails the append-only check, in text rather than ids.

``pack_trajectory`` reports the token offset where a turn stops matching the
final context, which is enough to reject the trajectory but not enough to fix
it. The interesting question is what the tokens *say*: a divergence that is one
token of whitespace is a tokenizer boundary artifact and can be absorbed, while
a divergence that rewrites tool-call arguments means the replay path is not the
inverse of the parse path and the trajectory genuinely cannot be reconstructed.

  python3 diag_pack.py <turn_ids_dir> [limit]
"""

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pack import pack_trajectory  # noqa: E402


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)


def main() -> int:
    d = sys.argv[1] if len(sys.argv) > 1 else "turn_ids"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tok = load_tokenizer()

    files = sorted(glob.glob(os.path.join(d, "*.json")))
    print(f"{len(files)} trajectories\n")

    ok = 0
    shown = 0
    kinds: dict[str, int] = {}
    for f in files:
        turns = [(t["prompt"], t["completion"]) for t in json.load(open(f))]
        packed, err = pack_trajectory(turns)
        if packed is not None:
            ok += 1
            continue
        kinds[err.kind] = kinds.get(err.kind, 0) + 1
        if shown >= limit:
            continue
        shown += 1

        k = err.turn
        prompt, completion = turns[k]
        final = list(turns[-1][0]) + list(turns[-1][1])
        start = len(prompt)

        print("=" * 78)
        print(f"{os.path.basename(f)}   turn {k}/{len(turns)}   kind={err.kind}")
        print(f"  {err.detail}")

        if err.kind == "completion":
            got = final[start : start + len(completion)]
            # Find the first differing index, then show a window either side so
            # the mismatch is visible in context rather than as a bare offset.
            i = next(
                (j for j in range(min(len(got), len(completion))) if got[j] != completion[j]),
                min(len(got), len(completion)),
            )
            lo, hi = max(0, i - 40), i + 40
            print(f"\n  --- sampled completion [{lo}:{hi}] ---")
            print("  " + repr(tok.decode(completion[lo:hi])))
            print(f"\n  --- final context     [{lo}:{hi}] ---")
            print("  " + repr(tok.decode(got[lo:hi])))
            print(f"\n  first differing token #{i}: "
                  f"sampled={completion[i] if i < len(completion) else None!r} "
                  f"({tok.decode([completion[i]])!r}) vs "
                  f"final={got[i] if i < len(got) else None!r} "
                  f"({tok.decode([got[i]]) if i < len(got) else None!r})")
        elif err.kind == "prefix":
            i = next(
                (j for j in range(min(start, len(final))) if final[j] != prompt[j]),
                0,
            )
            lo, hi = max(0, i - 40), i + 40
            print(f"\n  --- turn prompt   [{lo}:{hi}] ---")
            print("  " + repr(tok.decode(prompt[lo:hi])))
            print(f"\n  --- final context [{lo}:{hi}] ---")
            print("  " + repr(tok.decode(final[lo:hi])))
        print()

    print("=" * 78)
    print(f"packable: {ok}/{len(files)}")
    print("failure kinds:", kinds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
