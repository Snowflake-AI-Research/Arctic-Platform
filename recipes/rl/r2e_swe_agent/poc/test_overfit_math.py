"""The feasibility arithmetic for a fixed-batch overfit, pinned.

The claim these guard is that a 12-task batch cannot be overfit by raising the
learning rate, only by taking more steps. That rests on two curves -- how often
a group disagrees with itself, and how gradient noise scales with batch width --
so both are checked against closed forms and against the rates the control run
logged.
"""

import pytest

from overfit_math import (
    CONTROL_RATES,
    batch_signal,
    informative_prob,
    lr_to_compress,
    max_stable_lr,
    mis_signed_fraction,
    signal_mass,
    steps_for_updates,
    wall_clock_hours,
)


def test_unanimous_groups_carry_no_gradient():
    assert informative_prob(0.0) == 0.0
    assert informative_prob(1.0) == 0.0


def test_informative_probability_peaks_at_even_odds():
    peak = informative_prob(0.5)
    assert peak > informative_prob(0.2)
    assert peak > informative_prob(0.8)


def test_signal_mass_matches_the_binomial_expectation():
    # E[sum_i |A_i|] = 2(G-1)p(1-p); check against a direct enumeration.
    from math import comb

    group, p = 5, 0.3
    direct = sum(
        comb(group, k) * p**k * (1 - p) ** (group - k) * 2 * k * (group - k) / group
        for k in range(group + 1)
    )
    assert signal_mass(p, group) == pytest.approx(direct)


def test_signal_mass_vanishes_at_both_ends():
    assert signal_mass(0.0) == 0.0
    assert signal_mass(1.0) == 0.0


def test_model_reproduces_the_logged_informative_group_count():
    # The driver logged 7, 6, 6 informative groups on the three control steps.
    b = batch_signal(CONTROL_RATES)
    assert 6.0 <= b["informative_groups"] <= 7.0


def test_never_solved_tasks_are_excluded_from_the_ceiling():
    b = batch_signal(CONTROL_RATES)
    assert b["tasks"] == 12
    assert b["reachable"] == 10
    assert b["ceiling"] == pytest.approx(10 / 12)


def test_a_task_never_solved_never_receives_an_update():
    assert steps_for_updates(0.0, 10) == float("inf")


def test_weaker_tasks_need_more_steps_for_the_same_updates():
    assert steps_for_updates(0.067, 10) > steps_for_updates(0.6, 10)


def test_mis_signed_fraction_is_zero_when_nothing_is_zeroed():
    assert mis_signed_fraction(0.4, 0.4) == 0.0


def test_mis_signed_fraction_grows_with_the_zeroing_gap():
    assert mis_signed_fraction(0.6, 0.2) > mis_signed_fraction(0.3, 0.2)


def test_shrinking_the_batch_lowers_the_noise_matched_lr():
    # The direction is the load-bearing claim: fewer prompts means a noisier
    # gradient, so the admissible lr falls rather than rises.
    assert max_stable_lr(1e-6, 6.2, 32.0) < 1e-6


def test_compressing_a_run_raises_the_required_lr():
    assert lr_to_compress(1e-6, 500, 15) > lr_to_compress(1e-6, 500, 120)


def test_the_two_constraints_do_not_overlap_for_a_12_task_batch():
    stable = max_stable_lr(1e-6, 6.2, 32.0)
    for steps in (15, 30, 60, 120):
        assert lr_to_compress(1e-6, 500, steps) > stable


def test_off_policy_replay_divides_wall_clock():
    assert wall_clock_hours(100, mops=8) == pytest.approx(
        wall_clock_hours(100, mops=1) / 8
    )


def test_wall_clock_uses_the_measured_collection_cost():
    # ~7669s per collection, so a single on-policy step is a bit over two hours.
    assert 2.0 < wall_clock_hours(1) < 2.2
