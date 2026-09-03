"""Compare the client-side-loss arm against its sign-flipped control across seeds.

The claim under test is not "reward goes up" -- a single rising curve is consistent
with a rollout sampler that improves for reasons unrelated to our gradient. The
claim is that reward goes up *because of the sign we put in the payload*, so every
seed is run twice with only that sign changed and the arms are compared pairwise.
"""

from __future__ import annotations

import json
import os
import statistics as st

SEEDS = (7, 42, 123)
ARMS = ("normal", "control")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "seedmatrix")
WINDOW = 30


def series(path: str, key: str) -> list[float]:
    hist = json.load(open(path))["log_history"]
    return [float(r[key]) for r in hist if key in r]


def slope(ys: list[float]) -> float:
    """Least-squares slope of y on step index, in reward units per 100 steps."""
    n = len(ys)
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return 100.0 * num / den if den else 0.0


def welch(a: list[float], b: list[float]) -> float:
    """Welch t on the two window means; returns t, not a p-value."""
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    return (st.mean(a) - st.mean(b)) / ((va + vb) ** 0.5) if va + vb else 0.0


def main() -> int:
    table: dict[tuple[int, str], dict] = {}
    print(f"{'seed':>5} {'arm':>8} {'steps':>6} {'first30':>8} {'last30':>8} "
          f"{'delta':>8} {'t':>6} {'slope/100':>10} {'ratio':>7} {'kl':>9}")
    print("-" * 86)
    for seed in SEEDS:
        for arm in ARMS:
            path = os.path.join(OUT, f"s{seed}_{arm}.json")
            if not os.path.exists(path):
                print(f"{seed:>5} {arm:>8}   MISSING")
                continue
            r = series(path, "reward")
            first, last = r[:WINDOW], r[-WINDOW:]
            delta = st.mean(last) - st.mean(first)
            ratio = series(path, "ratio")
            kl = series(path, "kl")
            table[(seed, arm)] = {
                "reward": r, "delta": delta, "slope": slope(r),
                "ratio": st.mean(ratio) if ratio else float("nan"),
                "kl": st.mean(kl) if kl else float("nan"),
            }
            print(f"{seed:>5} {arm:>8} {len(r):>6} {st.mean(first):>8.4f} {st.mean(last):>8.4f} "
                  f"{delta:>+8.4f} {welch(last, first):>6.2f} {table[(seed, arm)]['slope']:>+10.4f} "
                  f"{table[(seed, arm)]['ratio']:>7.4f} {table[(seed, arm)]['kl']:>9.2e}")

    print("\n" + "=" * 86)
    print("PAIRED: same seed, same data order, only the advantage sign differs")
    print("=" * 86)
    pairs = [s for s in SEEDS if (s, "normal") in table and (s, "control") in table]
    if not pairs:
        print("no complete pairs")
        return 1

    print(f"{'seed':>5} {'normal delta':>13} {'control delta':>14} {'separation':>12}")
    seps = []
    for s in pairs:
        dn, dc = table[(s, "normal")]["delta"], table[(s, "control")]["delta"]
        seps.append(dn - dc)
        print(f"{s:>5} {dn:>+13.4f} {dc:>+14.4f} {dn - dc:>+12.4f}")

    n_deltas = [table[(s, "normal")]["delta"] for s in pairs]
    c_deltas = [table[(s, "control")]["delta"] for s in pairs]
    n_slopes = [table[(s, "normal")]["slope"] for s in pairs]
    c_slopes = [table[(s, "control")]["slope"] for s in pairs]

    def summarize(name, vals):
        m = st.mean(vals)
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        sem = sd / len(vals) ** 0.5 if len(vals) > 1 else 0.0
        print(f"  {name:<26} {m:>+8.4f}  (sd {sd:.4f}, sem {sem:.4f}, n={len(vals)})")

    print(f"\nacross {len(pairs)} seeds:")
    summarize("normal reward delta", n_deltas)
    summarize("control reward delta", c_deltas)
    summarize("normal slope /100 steps", n_slopes)
    summarize("control slope /100 steps", c_slopes)
    summarize("paired separation", seps)

    print("\nVERDICT")
    all_pos = all(d > 0 for d in n_deltas)
    all_sep = all(s > 0 for s in seps)
    ratios_ok = all(abs(table[(s, "normal")]["ratio"] - 1.0) < 0.05 for s in pairs)
    print(f"  normal improves on every seed ..... {all_pos}  {[round(d, 3) for d in n_deltas]}")
    print(f"  normal beats control on every seed  {all_sep}  {[round(s, 3) for s in seps]}")
    print(f"  importance ratio ~1 (no drift) .... {ratios_ok}  "
          f"{[round(table[(s, 'normal')]['ratio'], 4) for s in pairs]}")
    if all_pos and all_sep:
        print("\n  PASS: reward improvement tracks the advantage sign on every seed.")
        return 0
    print("\n  INCONCLUSIVE: see the per-seed rows above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
