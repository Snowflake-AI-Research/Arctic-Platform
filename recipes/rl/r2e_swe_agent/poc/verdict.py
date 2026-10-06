#!/usr/bin/env python3
"""Decide whether a run is learning, without eyeballing a curve.

Reward here is a mean over sixty Bernoulli trials, so step-to-step swings of
several points are routine and two adjacent steps say almost nothing. The
question worth asking is whether there is a trend across all the steps, and
whether it is larger than what reshuffling the same numbers would produce.

Two series are reported because they answer different questions. ``reward`` is
what the optimizer actually maximizes, and includes the zeroing of
protocol-violating trajectories. ``pass_rate`` is the share that solved the
instance before that zeroing, so it tracks capability alone. A run can improve
on one and not the other, and which one moves says what the model learned: to
fix bugs, or to stop breaking the harness contract.

  python3 verdict.py <run_dir> [--min-steps N]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def per_turn_invalid(terminal_rate: float, mean_turns: float) -> float:
    """Back out the per-turn protocol error rate from a trajectory-level rate.

    A trajectory is zeroed if any single turn violates the contract, so the
    trajectory-level rate is ``1 - (1 - r)**T``. Reporting that rate alone
    mixes two things that move independently: how careful the policy is on a
    given turn, and how many turns it spends. Inverting to ``r`` isolates the
    part training can act on, and makes runs with different trajectory lengths
    comparable at all.
    """
    if mean_turns <= 0:
        return float("nan")
    clean = max(1.0 - terminal_rate, 1e-9)
    return 1.0 - math.exp(math.log(clean) / mean_turns)


def slope(ys: list[float]) -> float:
    """Least-squares slope against step index, in reward units per step."""
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def trend_p_value(ys: list[float], trials: int = 20000, seed: int = 0) -> float:
    """How often shuffling the steps produces a slope this large.

    A permutation test rather than a t-test: with a dozen points and bounded,
    non-normal values, the null distribution is worth constructing directly
    instead of assumed.
    """
    if len(ys) < 3:
        return 1.0
    observed = abs(slope(ys))
    rng = random.Random(seed)
    shuffled = list(ys)
    hits = 0
    for _ in range(trials):
        rng.shuffle(shuffled)
        if abs(slope(shuffled)) >= observed:
            hits += 1
    return (hits + 1) / (trials + 1)


def residual_sd(ys: list[float]) -> float:
    """Step-to-step scatter about the fitted trend, not about the mean.

    Using the raw standard deviation would count a real trend as noise and
    overstate how much movement it takes to see one.
    """
    n = len(ys)
    if n < 3:
        return 0.0
    s = slope(ys)
    mx = (n - 1) / 2
    my = sum(ys) / n
    resid = [y - (my + s * (x - mx)) for x, y in enumerate(ys)]
    return (sum(r * r for r in resid) / max(n - 2, 1)) ** 0.5


def min_detectable_move(noise_sd: float, steps: int) -> float:
    """Total change over the run that this test would catch ~80% of the time.

    Normal-theory approximation to the permutation test, with the constant
    calibrated against ``power.py``'s simulation of the real decision rule
    (matched to within a few percent across 10-50 steps and both noise
    regimes seen in this run). Reported so that "no trend" can be read
    correctly: it means either the run is flat, or the gain was smaller than
    this, and those are very different conclusions.
    """
    if steps < 3 or noise_sd <= 0:
        return float("inf")
    sxx = steps * (steps * steps - 1) / 12.0
    return 3.15 * noise_sd * (steps - 1) / (sxx ** 0.5)


def summarize(name: str, ys: list[float]) -> str:
    if len(ys) < 3:
        return f"{name:<10} {len(ys)} step(s): too few to call a trend"
    s = slope(ys)
    p = trend_p_value(ys)
    first, last = ys[0], ys[-1]
    verdict = "rising" if s > 0 else "falling"
    sig = "significant" if p < 0.05 else "not distinguishable from noise"
    line = (
        f"{name:<10} {first:.3f} -> {last:.3f}   "
        f"slope {s:+.4f}/step   p={p:.3f}   {verdict}, {sig}"
    )
    if p >= 0.05:
        mdm = min_detectable_move(residual_sd(ys), len(ys))
        line += f"\n{'':<10} would have needed a move of {mdm:+.3f} over {len(ys)} steps to show"
    return line


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--min-steps", type=int, default=5)
    args = ap.parse_args()

    path = Path(args.run_dir) / "history.json"
    hist = json.loads(path.read_text())
    hist.sort(key=lambda h: h["step"])

    turns = [h.get("mean_turns") for h in hist]
    have_turns = all(isinstance(t, (int, float)) and t for t in turns)

    head = f"{'step':>4}  {'reward':>7}  {'pass':>7}  {'solved':>7}  {'zeroed':>7}"
    if have_turns:
        head += f"  {'turns':>7}  {'err/turn':>9}"
    print(head)
    for h in hist:
        n = h.get("n") or 1
        row = (
            f"{h['step']:>4}  {h['mean_reward']:>7.3f}  {h['pass_rate']:>7.3f}"
            f"  {h['solved']:>7}  {h['zeroed'] / n:>7.2f}"
        )
        if have_turns:
            r = per_turn_invalid(h["zeroed"] / n, h["mean_turns"])
            row += f"  {h['mean_turns']:>7.1f}  {r * 100:>8.2f}%"
        print(row)

    print()
    if len(hist) < args.min_steps:
        print(f"only {len(hist)} steps; need {args.min_steps} before calling anything")
        return 0

    print(summarize("reward", [h["mean_reward"] for h in hist]))
    print(summarize("pass_rate", [h["pass_rate"] for h in hist]))
    print(summarize("terminal", [h["zeroed"] / (h.get("n") or 1) for h in hist]))
    if have_turns:
        print(summarize("turns", [float(h["mean_turns"]) for h in hist]))
        print(summarize("err/turn", [
            per_turn_invalid(h["zeroed"] / (h.get("n") or 1), h["mean_turns"])
            for h in hist
        ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
