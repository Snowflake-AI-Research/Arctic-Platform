"""Fixed-batch overfit mode.

On a batch that never changes, a correct loop must drive reward up — nothing
prevents it from memorising. So a flat curve under --overfit indicts the
plumbing (advantages, loss mask, token alignment, weight sync) rather than the
task mix, which is what makes it worth wiring as its own mode.
"""

from __future__ import annotations

import pytest

from r2e_driver import _fixed_set


class Args:
    def __init__(self, overfit: int = 0, task_ids: str | None = None) -> None:
        self.overfit = overfit
        self.task_ids = task_ids


@pytest.fixture
def instances() -> list[dict]:
    return [
        {"instance_id": f"repo@{i:04x}", "problem_statement": f"issue {i}"}
        for i in range(50)
    ]


def test_normal_mode_returns_none(instances) -> None:
    """Without either flag the curriculum stays in charge."""
    assert _fixed_set(instances, Args()) is None


def test_overfit_draw_is_seeded_prefix(instances) -> None:
    """load_instances shuffles under --seed, so a prefix is a reproducible draw."""
    got = _fixed_set(instances, Args(overfit=16))
    assert [g["instance_id"] for g in got] == [i["instance_id"] for i in instances[:16]]


def test_overfit_draw_is_stable_across_calls(instances) -> None:
    a = _fixed_set(instances, Args(overfit=16))
    b = _fixed_set(instances, Args(overfit=16))
    assert [x["instance_id"] for x in a] == [x["instance_id"] for x in b]


def test_task_ids_preserve_given_order(instances) -> None:
    """Explicit ids let us pin instances already known to produce spread."""
    got = _fixed_set(instances, Args(task_ids="repo@0005, repo@0001,repo@0003"))
    assert [g["instance_id"] for g in got] == ["repo@0005", "repo@0001", "repo@0003"]


def test_task_ids_beat_overfit(instances) -> None:
    got = _fixed_set(instances, Args(overfit=16, task_ids="repo@0002"))
    assert [g["instance_id"] for g in got] == ["repo@0002"]


def test_unknown_task_id_fails_loudly(instances) -> None:
    """Silently dropping a typo'd id would quietly change the experiment."""
    with pytest.raises(SystemExit, match="not in dataset"):
        _fixed_set(instances, Args(task_ids="repo@0001,typo@9999"))


def test_overfit_larger_than_dataset_fails(instances) -> None:
    with pytest.raises(SystemExit, match="exceeds"):
        _fixed_set(instances, Args(overfit=len(instances) + 1))


def test_records_are_the_dataset_rows_not_copies(instances) -> None:
    """The driver passes these straight to the sandbox, so they must carry the
    full record (docker_image, expected_output_json), not a trimmed view."""
    got = _fixed_set(instances, Args(overfit=3))
    assert all(g is instances[i] for i, g in enumerate(got))


@pytest.mark.parametrize("n", [1, 4, 16, 50])
def test_draw_sizes(instances, n: int) -> None:
    assert len(_fixed_set(instances, Args(overfit=n))) == n
