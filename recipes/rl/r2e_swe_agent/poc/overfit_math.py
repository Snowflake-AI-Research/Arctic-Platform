#!/usr/bin/env python3
"""Can a fixed batch of 12 tasks be overfit stably, and in how many steps?

Two constraints have to be satisfied at once, and they pull the learning rate
in opposite directions:

  progress   the weights must travel far enough to memorise the batch, which
             for a fixed lr is a question of step count
  stability  the gradient is a noisy estimate, and SGD only settles into a ball
             whose radius grows with lr and with the noise

The second constraint is the one a small batch tightens. Fewer prompts means a
noisier gradient means a *smaller* admissible lr -- exactly the opposite of what
compressing a long run into few steps demands. This module computes both sides
from the rates the control run actually measured, so the question of whether the
two constraints leave a usable window is arithmetic rather than opinion.
"""

from __future__ import annotations

# Post-zeroing solve rate per task, averaged over the three control steps.
# These are what GRPO sees, which is the only thing that produces gradient.
CONTROL_RATES = {
    "aiohttp@fa628a21": 0.133,
    "aiohttp@fecb85a9": 0.000,
    "coveragepy@5c3d0946": 0.200,
    "coveragepy@c4fc3833": 0.267,
    "datalad@2df9d5fb": 0.067,
    "numpy@a5322429": 0.133,
    "orange3@78213643": 0.600,
    "pandas@5f5350b8": 0.000,
    "pillow@bfaa0a1f": 0.267,
    "pyramid@a43abd25": 0.600,
    "scrapy@b51b52ff": 0.067,
    "tornado@37081d79": 0.133,
}

GROUP = 5


def informative_prob(p: float, group: int = GROUP) -> float:
    """Chance a group of ``group`` draws at rate ``p`` yields any advantage.

    GRPO centres rewards within a group, so a group that is unanimous -- all
    solved or all failed -- has every advantage exactly zero and contributes no
    gradient at all. A task is therefore not learned at a rate set by its solve
    probability but at a rate set by how often its group happens to disagree
    with itself, which is zero at both ends and maximal at p = 0.5.
    """
    return 1.0 - p ** group - (1.0 - p) ** group


def signal_mass(p: float, group: int = GROUP) -> float:
    """Expected total |advantage| a group contributes, E[sum_i |A_i|].

    With k of ``group`` solved, the advantages are (1 - k/G) on each success and
    -k/G on each failure, so the mass is 2k(G-k)/G. Averaging k over the
    binomial gives 2(G-1)p(1-p): the same inverted parabola, peaking at p = 0.5.
    """
    return 2.0 * (group - 1) * p * (1.0 - p)


def batch_signal(rates: dict[str, float], group: int = GROUP) -> dict[str, float]:
    """Per-step gradient budget of a fixed batch, and its effective width."""
    ps = list(rates.values())
    live = [p for p in ps if informative_prob(p, group) > 0]
    masses = [signal_mass(p, group) for p in ps]
    return {
        "tasks": len(ps),
        "reachable": len(live),
        "informative_groups": sum(informative_prob(p, group) for p in ps),
        "signal_mass": sum(masses),
        "ceiling": len(live) / len(ps),
    }


def mis_signed_fraction(pass_rate: float, train_reward: float) -> float:
    """Share of the negative-advantage mass sitting on correct solutions.

    A trajectory that fixed the bug but broke protocol is scored zero, so inside
    its group it lands below the mean and the update pushes its log-probability
    down. The malformed turn deserves that; the several dozen correct edits in
    front of it do not, and a sequence-level advantage cannot tell them apart.
    """
    zeroed_solves = max(pass_rate - train_reward, 0.0)
    apparent_failures = max(1.0 - train_reward, 1e-9)
    return zeroed_solves / apparent_failures


def max_stable_lr(reference_lr: float, our_groups: float, ref_groups: float) -> float:
    """Largest lr with the same gradient-noise floor as the reference run.

    SGD does not converge to the optimum but to a ball whose radius goes like
    lr * sigma^2. Averaging over n independent groups puts sigma^2 ~ 1/n, so
    holding the ball fixed while shrinking the batch means shrinking lr in
    proportion. This is a scaling statement, not a bound: it says how the
    admissible lr moves, given that the reference's own lr sits somewhere safe.
    """
    return reference_lr * our_groups / ref_groups


def lr_to_compress(reference_lr: float, ref_steps: int, our_steps: int) -> float:
    """lr needed to cover the reference's parameter travel in fewer steps.

    Under Adam the per-step displacement is set by lr almost independently of
    the gradient magnitude, so coherent travel over a run is about lr * steps.
    Matching a 500-step run in 15 therefore needs 33x the lr.
    """
    return reference_lr * ref_steps / our_steps


def steps_for_updates(p: float, updates: int, group: int = GROUP) -> float:
    """Optimizer steps before a task at rate ``p`` receives ``updates`` of them."""
    rate = informative_prob(p, group)
    return float("inf") if rate <= 0 else updates / rate


def wall_clock_hours(steps: int, collect_s: float = 7669.0, mops: int = 1) -> float:
    """Hours to reach ``steps`` optimizer updates at a measured collection cost.

    Collection dominates by two orders of magnitude, so off-policy replay is not
    a small efficiency gain: it is the difference between a run that fits in a
    day and one that does not fit in a week. ``mops`` optimizer steps share a
    single collection, and the clipping that makes that safe is the same
    clipping that is inert at ``mops = 1``.
    """
    return steps / mops * collect_s / 3600.0


def main() -> None:
    b = batch_signal(CONTROL_RATES)
    print(f"batch: {b['tasks']} tasks, {b['reachable']} reachable "
          f"(ceiling {b['ceiling']:.0%} mean reward)")
    print(f"informative groups per step: {b['informative_groups']:.1f} "
          f"of {b['tasks']}  (logged: 7, 6, 6)")
    print(f"signal mass per step: {b['signal_mass']:.1f}")

    ref = {f"t{i}": 0.21 for i in range(32)}
    rb = batch_signal(ref)
    print(f"\nreference batch 160 / group 5 = 32 prompts, zero_advantage filter enforced")
    print(f"informative groups per step: 32.0 (filtered, so all of them)")
    print(f"signal mass per step: {rb['signal_mass']:.1f}")

    ours, theirs = b["informative_groups"], 32.0
    print(f"\ngradient SNR ratio (sqrt of group ratio): "
          f"{(ours / theirs) ** 0.5:.2f}x -- we are {(theirs / ours) ** 0.5:.1f}x noisier")

    print(f"\nmis-signed gradient: "
          f"{mis_signed_fraction(0.494, 0.206):.0%} of negative-advantage mass "
          f"sits on trajectories that solved the task")

    print("\nsteps until a task has had N informative updates:")
    print(f"  {'rate p':>8} {'inform/step':>12} {'N=10':>8} {'N=30':>8}")
    for p in (0.00, 0.067, 0.133, 0.20, 0.267, 0.60):
        r = informative_prob(p)
        s10, s30 = steps_for_updates(p, 10), steps_for_updates(p, 30)
        f10 = "never" if s10 == float("inf") else f"{s10:.0f}"
        f30 = "never" if s30 == float("inf") else f"{s30:.0f}"
        print(f"  {p:>8.3f} {r:>12.2f} {f10:>8} {f30:>8}")

    print("\ncan lr buy back step count?")
    stable = max_stable_lr(1e-6, ours, theirs)
    print(f"  noise-matched lr for a 12-task batch: {stable:.1e} "
          f"(below the reference's 1e-6, not above)")
    for n in (15, 30, 60, 120):
        need = lr_to_compress(1e-6, 500, n)
        print(f"  {n:>4} steps would need lr {need:.1e} -> "
              f"{need / stable:>5.0f}x the noise-matched value")

    print("\nwall clock to a real overfit (~50 informative updates on the weak tasks):")
    for mops in (1, 4, 8):
        for n in (50, 100):
            print(f"  mops={mops}  {n:>3} optimizer steps: "
                  f"{wall_clock_hours(n, mops=mops):>6.1f} h "
                  f"({wall_clock_hours(n, mops=mops) / 24:.1f} d)")


if __name__ == "__main__":
    main()
