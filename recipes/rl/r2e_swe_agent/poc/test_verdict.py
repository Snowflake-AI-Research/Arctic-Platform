"""The convergence call, pinned.

The risk this guards against is declaring victory on noise: sixty Bernoulli
trials per step wander by several points on their own, and a rising pair of
steps is the most common way to fool yourself in an RL run.
"""

import random

from power import simulate
from verdict import (
    min_detectable_move,
    per_turn_invalid,
    residual_sd,
    slope,
    summarize,
    trend_p_value,
)


def test_flat_series_has_no_slope():
    assert slope([0.2] * 10) == 0.0


def test_rising_series_has_positive_slope():
    assert slope([0.1, 0.2, 0.3, 0.4]) > 0


def test_falling_series_has_negative_slope():
    assert slope([0.4, 0.3, 0.2, 0.1]) < 0


def test_slope_is_per_step_units():
    """A clean ramp of 0.05 per step must report 0.05, not a correlation."""
    assert abs(slope([0.0, 0.05, 0.10, 0.15, 0.20]) - 0.05) < 1e-9


def test_a_clear_trend_is_significant():
    ys = [0.10, 0.14, 0.19, 0.23, 0.28, 0.33, 0.37, 0.42]
    assert trend_p_value(ys) < 0.05


def test_pure_noise_is_not_significant():
    rng = random.Random(7)
    ys = [rng.uniform(0.15, 0.35) for _ in range(12)]
    assert trend_p_value(ys) > 0.05


def test_noise_is_not_called_a_trend_across_many_seeds():
    """The false-positive rate must actually sit near the 5% it claims."""
    false_positives = 0
    for seed in range(40):
        rng = random.Random(seed)
        ys = [rng.uniform(0.15, 0.35) for _ in range(10)]
        if trend_p_value(ys, trials=2000, seed=seed) < 0.05:
            false_positives += 1
    assert false_positives <= 6, f"{false_positives}/40 flagged as trending"


def test_a_single_lucky_last_step_does_not_carry_a_trend():
    """Twelve flat steps and one spike is not convergence."""
    ys = [0.20] * 12 + [0.45]
    assert trend_p_value(ys) > 0.05


def test_too_few_steps_refuses_to_call_it():
    assert trend_p_value([0.1, 0.5]) == 1.0


def test_verdict_is_reproducible():
    ys = [0.1, 0.3, 0.2, 0.4, 0.3, 0.5]
    assert trend_p_value(ys, seed=3) == trend_p_value(ys, seed=3)


def test_residual_sd_of_a_perfect_line_is_zero():
    assert residual_sd([0.1, 0.2, 0.3, 0.4, 0.5]) < 1e-12


def test_residual_sd_measures_scatter_not_trend():
    """A steep clean ramp must not be reported as noisy."""
    steep = [0.0, 0.2, 0.4, 0.6, 0.8]
    jittery = [0.30, 0.26, 0.33, 0.28, 0.31]
    assert residual_sd(steep) < residual_sd(jittery)


def test_detectable_move_scales_with_noise():
    assert min_detectable_move(0.02, 15) > min_detectable_move(0.01, 15)


def test_more_steps_detect_smaller_moves():
    assert min_detectable_move(0.05, 30) < min_detectable_move(0.05, 15)


def test_detectable_move_is_calibrated_against_the_real_test():
    """The approximation must track the permutation test it stands in for.

    This is the claim the number rests on: at the reported move, simulating
    the actual decision rule should fire close to 80% of the time. Tolerance
    is wide because the simulation itself is sampled.
    """
    sd, steps = 0.0096, 15
    move = min_detectable_move(sd, steps)
    power = simulate(
        steps=steps, n=60, start=0.72, move=move,
        reps=120, alpha=0.05, perm_trials=1200, seed=11, noise_sd=sd,
    )
    assert 0.6 <= power <= 0.95, f"power {power:.0%} at the stated move"


def test_insignificant_verdict_reports_what_it_would_have_taken():
    """"No trend" is ambiguous unless the detection floor comes with it."""
    rng = random.Random(3)
    flat = [0.2 + rng.gauss(0, 0.05) for _ in range(12)]
    out = summarize("reward", flat)
    assert "not distinguishable from noise" in out
    assert "would have needed a move of" in out


def test_significant_verdict_omits_the_floor():
    out = summarize("reward", [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    assert "significant" in out
    assert "would have needed" not in out


def test_per_turn_rate_inverts_the_compounding():
    """Round-trip: compounding a recovered per-turn rate rebuilds the input."""
    for terminal, turns in ((0.72, 62.0), (0.15, 53.0), (0.5, 10.0)):
        r = per_turn_invalid(terminal, turns)
        assert abs((1 - (1 - r) ** turns) - terminal) < 1e-9


def test_longer_trajectories_mean_a_lower_per_turn_rate():
    """The same trajectory-level rate is less alarming over more turns."""
    assert per_turn_invalid(0.72, 100.0) < per_turn_invalid(0.72, 50.0)


def test_flat_per_turn_rate_under_growing_trajectories():
    """The case this metric exists for.

    Unchanged per-turn behaviour with lengthening trajectories must show a
    rising terminal rate but a flat per-turn rate, so the two series
    disagree and the disagreement is the finding.
    """
    r = 0.019
    turns = [62.0, 68.0, 71.0, 80.0]
    terminal = [1 - (1 - r) ** t for t in turns]
    assert terminal[0] < terminal[-1]
    recovered = [per_turn_invalid(t, n) for t, n in zip(terminal, turns)]
    assert max(recovered) - min(recovered) < 1e-9


def test_zero_turns_does_not_explode():
    import math as _m
    assert _m.isnan(per_turn_invalid(0.5, 0.0))
