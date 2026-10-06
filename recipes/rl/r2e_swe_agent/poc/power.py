#!/usr/bin/env python3
"""How much improvement a run of this shape can actually detect.

``verdict.py`` answers "is this run learning". This answers the question that
has to come first: "if it were learning, would we be able to tell". A 15-step
run at 60 rollouts per step has a reward mean whose sampling noise is around
five points, so a real but modest improvement is invisible to any test. Finding
that out after thirty hours of GPU time is the expensive way to find it out.

The simulation draws each step's reward as a binomial mean at the run's actual
group size, with the success probability walking linearly from ``start`` to
``start + move``, and asks how often ``verdict.py``'s own permutation test
calls it. Reusing that exact test matters: the answer is about the decision
rule we will really apply, not a t-test standing in for it.

  python3 power.py --steps 15 --n 60 --start 0.21
  python3 power.py --steps 15 --n 60 --start 0.21 --move 0.15
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verdict import trend_p_value  # noqa: E402


def simulate(
    steps: int,
    n: int,
    start: float,
    move: float,
    reps: int,
    alpha: float,
    perm_trials: int,
    seed: int,
    noise_sd: float | None = None,
) -> float:
    """Share of simulated runs the permutation test calls significant.

    Noise defaults to binomial at ``n`` draws, which is the right model for a
    reward mean. Pass ``noise_sd`` to use a measured scatter instead: not every
    series is binomial-noisy. The terminal-invalid rate in particular is far
    steadier step to step than its base rate implies, because it is driven by
    systematic policy behaviour rather than per-rollout chance, and modelling it
    as binomial would badly understate what the run can detect.
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(reps):
        ys = []
        for s in range(steps):
            p = start + move * (s / max(steps - 1, 1))
            p = min(max(p, 0.0), 1.0)
            if noise_sd is None:
                ys.append(sum(rng.random() < p for _ in range(n)) / n)
            else:
                ys.append(p + rng.gauss(0.0, noise_sd))
        if trend_p_value(ys, trials=perm_trials, seed=rng.randrange(1 << 30)) < alpha:
            hits += 1
    return hits / reps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--n", type=int, default=60, help="rollouts per step")
    ap.add_argument("--start", type=float, default=0.21, help="reward at step 0")
    ap.add_argument(
        "--move",
        type=float,
        default=None,
        help="total reward gain over the run; omit to sweep",
    )
    ap.add_argument(
        "--noise-sd",
        type=float,
        default=None,
        help="measured step-to-step scatter; overrides the binomial model",
    )
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--perm-trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.move is not None:
        power = simulate(
            args.steps, args.n, args.start, args.move,
            args.reps, args.alpha, args.perm_trials, args.seed, args.noise_sd,
        )
        print(
            f"{args.steps} steps, n={args.n}, {args.start:.2f} -> "
            f"{args.start + args.move:.2f}: power {power:.0%}"
        )
        return 0

    model = (
        f"measured scatter sd={args.noise_sd:.4f}"
        if args.noise_sd is not None
        else f"binomial noise at n={args.n}"
    )
    print(
        f"{model}, start={args.start:.2f}, alpha={args.alpha}, "
        f"{args.reps} simulated runs per cell"
    )
    print("power = share of runs where the trend test fires\n")

    moves = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    step_grid = [s for s in (10, 15, 20, 30, 50) if s >= 3]

    print(f"{'total gain':>11} " + "".join(f"{s:>7}st" for s in step_grid))
    for mv in moves:
        cells = []
        for st in step_grid:
            p = simulate(
                st, args.n, args.start, mv,
                args.reps, args.alpha, args.perm_trials, args.seed, args.noise_sd,
            )
            cells.append(f"{p:>8.0%}")
        print(f"{args.start:.2f}->{args.start + mv:.2f} " + "".join(cells))

    print(
        "\nA run is worth launching when the gain you actually expect lands in a "
        "cell at or above ~80%. Below that the run is likely to end in "
        "'not distinguishable from noise' even if the training is working."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
