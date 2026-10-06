"""The picker decides how much of each step produces a gradient at all.

A group whose rollouts all score the same has identically zero advantages, so
a step spent on instances that are uniformly unsolved buys nothing but wall
clock. These tests pin the two behaviours that were silently wasting half of
every step: exploration crowding out known-good instances, and instances being
sampled once and then abandoned before reaching a verdict.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curriculum import Curriculum, Record  # noqa: E402


def _instances(n: int) -> list[dict]:
    return [{"instance_id": f"i{k}"} for k in range(n)]


def _curriculum(**records: Record) -> Curriculum:
    c = Curriculum(path=Path("/tmp/does-not-matter.json"), min_attempts=24)
    c.records = dict(records)
    return c


def test_most_of_a_step_goes_to_instances_with_spread() -> None:
    c = _curriculum(**{f"i{k}": Record(attempts=8, solved=4) for k in range(8)})
    picked = c.pick(_instances(1000), 8, random.Random(0), explore_frac=0.25)

    have_spread = sum(1 for r in picked if r["instance_id"] in c.records)
    assert have_spread == 6, "expected 6 exploit + 2 explore at explore_frac=0.25"


def test_no_instance_is_picked_twice() -> None:
    """The exploit slice and the top-up both read the spread bucket."""
    c = _curriculum(**{f"i{k}": Record(attempts=8, solved=4) for k in range(3)})
    # Only 3 spread and 0 unseen beyond them, so the top-up has to reach back
    # into spread -- the case where a naive re-slice repeats an instance.
    picked = c.pick(_instances(3), 3, random.Random(0), explore_frac=0.25)

    ids = [r["instance_id"] for r in picked]
    assert len(ids) == len(set(ids))


def test_half_measured_instances_are_revisited_before_new_ones() -> None:
    """``min_attempts`` is patience, and patience needs re-sampling.

    An instance seen once and never again can never reach a verdict, so it is
    never evicted and the pool fills with dead weight. Pending must outrank
    unseen for the eviction rule to mean anything.
    """
    c = _curriculum(**{f"i{k}": Record(attempts=8, solved=0) for k in range(4)})
    picked = c.pick(_instances(1000), 4, random.Random(0), explore_frac=1.0)

    assert {r["instance_id"] for r in picked} == {"i0", "i1", "i2", "i3"}


def test_a_verdict_reached_means_eviction() -> None:
    c = _curriculum()
    # Three failed groups of eight is the configured patience.
    for _ in range(3):
        c.observe("i0", [0.0] * 8)

    assert c.records["i0"].evicted == "too_hard"
    assert not c.is_live("i0")
    assert c.pick(_instances(1), 1, random.Random(0)) == []


def test_exploration_still_reaches_unseen_instances() -> None:
    """Exploitation must not freeze the pool."""
    c = _curriculum(**{f"i{k}": Record(attempts=8, solved=4) for k in range(8)})
    picked = c.pick(_instances(1000), 8, random.Random(0), explore_frac=0.25)

    assert any(r["instance_id"] not in c.records for r in picked)
