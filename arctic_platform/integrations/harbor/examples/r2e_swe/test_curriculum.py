"""Difficulty filtering decides what GRPO ever gets to learn from.

Every bug in here is expensive and quiet. Evicting too eagerly throws away the
handful of instances that produce any gradient at all; evicting too slowly
spends whole steps on groups whose advantages are identically zero. Neither
shows up as an error — both look like a flat reward curve.

    /data-fast/ap-venv/bin/python -m pytest test_curriculum.py -q
"""

from __future__ import annotations

import json
import random

import pytest

from curriculum import Curriculum


def _c(tmp_path, min_attempts: int = 4) -> Curriculum:
    return Curriculum.load(tmp_path / "c.json", min_attempts=min_attempts)


def _instances(n: int) -> list[dict]:
    return [{"instance_id": f"i{k}"} for k in range(n)]


class TestEviction:
    def test_no_verdict_before_min_attempts(self):
        """One unlucky group is not evidence. The reference hard_window is 3 groups."""
        c = Curriculum(path=None, min_attempts=8)  # type: ignore[arg-type]
        c.observe("i0", [0.0] * 4)
        assert c.is_live("i0")
        c.observe("i0", [0.0] * 4)
        assert not c.is_live("i0")

    def test_all_failures_evicts_as_too_hard(self, tmp_path):
        c = _c(tmp_path)
        c.observe("i0", [0.0, 0.0, 0.0, 0.0])
        assert c.records["i0"].evicted == "too_hard"

    def test_all_successes_evicts_as_too_easy(self, tmp_path):
        c = _c(tmp_path)
        c.observe("i0", [1.0, 1.0, 1.0, 1.0])
        assert c.records["i0"].evicted == "too_easy"

    def test_any_spread_stays_live(self, tmp_path):
        c = _c(tmp_path)
        c.observe("i0", [1.0, 0.0, 0.0, 0.0])
        assert c.records["i0"].evicted is None
        assert c.is_live("i0")

    def test_eviction_is_reversible_when_the_policy_improves(self, tmp_path):
        """An instance written off under a weak policy must be able to return.

        Permanent eviction would shrink the pool monotonically and stall
        training on the tasks the policy has just become able to attempt.
        """
        c = _c(tmp_path)
        c.observe("i0", [0.0] * 4)
        assert not c.is_live("i0")
        c.observe("i0", [1.0, 0.0, 0.0, 0.0])
        assert c.is_live("i0")

    def test_unmeasured_instances_are_live(self, tmp_path):
        assert _c(tmp_path).is_live("never-seen")


class TestPicking:
    def test_evicted_instances_are_never_picked(self, tmp_path):
        c = _c(tmp_path)
        for k in range(8):
            c.observe(f"i{k}", [0.0] * 4)
        c.observe("i3", [1.0, 0.0, 0.0, 0.0])
        picked = c.pick(_instances(8), 4, random.Random(0))
        ids = {p["instance_id"] for p in picked}
        assert ids <= {"i3"}, f"picked evicted instances: {ids - {'i3'}}"

    def test_known_spread_is_preferred_but_not_exclusive(self, tmp_path):
        """Drawing only from known-spread instances would freeze the pool and
        overfit a handful of repos, so fresh ones must keep coming in."""
        c = _c(tmp_path)
        c.observe("i0", [1.0, 0.0, 0.0, 0.0])
        c.observe("i1", [1.0, 1.0, 0.0, 0.0])
        picked = c.pick(_instances(20), 4, random.Random(0))
        ids = [p["instance_id"] for p in picked]
        assert len(ids) == 4
        assert len(set(ids)) == 4, "must not repeat an instance within a step"
        assert {"i0", "i1"} & set(ids), "known-spread instances were not preferred"
        assert any(i not in {"i0", "i1"} for i in ids), "no fresh instances mixed in"

    def test_returns_a_full_step_even_when_fresh_runs_out(self, tmp_path):
        """A short step wastes the group slots it did not fill."""
        c = _c(tmp_path)
        for k in range(3):
            c.observe(f"i{k}", [1.0, 0.0, 0.0, 0.0])
        picked = c.pick(_instances(3), 3, random.Random(0))
        assert len(picked) == 3

    def test_cannot_return_more_than_asked(self, tmp_path):
        c = _c(tmp_path)
        assert len(c.pick(_instances(50), 4, random.Random(0))) == 4

    def test_empty_pool_returns_empty(self, tmp_path):
        c = _c(tmp_path)
        c.observe("i0", [0.0] * 4)
        assert c.pick([{"instance_id": "i0"}], 4, random.Random(0)) == []


class TestPersistence:
    def test_tally_survives_a_restart(self, tmp_path):
        """Cortex jobs idle out in ~35 minutes; restarting the driver must not
        discard the pool those rollouts paid for."""
        c = _c(tmp_path)
        c.observe("i0", [1.0, 0.0, 0.0, 0.0])
        c.observe("i1", [0.0] * 4)
        c.save()

        again = _c(tmp_path)
        assert again.records["i0"].solved == 1
        assert again.records["i0"].attempts == 4
        assert again.records["i1"].evicted == "too_hard"

    def test_save_is_atomic(self, tmp_path):
        """Written via a temp file and renamed, so a crash mid-save cannot
        leave truncated JSON that fails to load on the next run."""
        c = _c(tmp_path)
        c.observe("i0", [1.0, 0.0])
        c.save()
        c.observe("i1", [1.0, 0.0])
        c.save()
        assert json.loads((tmp_path / "c.json").read_text()).keys() == {"i0", "i1"}
        assert not (tmp_path / "c.tmp").exists()

    def test_rate_reports_the_solve_fraction(self, tmp_path):
        c = _c(tmp_path)
        c.observe("i0", [1.0, 1.0, 0.0, 0.0])
        assert c.records["i0"].rate == pytest.approx(0.5)

    def test_rate_of_an_unattempted_record_is_zero_not_a_divide_by_zero(self):
        from curriculum import Record

        assert Record().rate == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
