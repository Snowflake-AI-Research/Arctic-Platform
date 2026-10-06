"""Choose the fixed instance set for an overfit convergence run.

Selection is not "pick the easiest". A GRPO group contributes gradient only
when its rewards disagree, so a task solved every time is as useless as one
never solved: P(informative) = 1 - p^G - (1-p)^G, maximised at p = 0.5. The
job here is to fill the set with tasks whose post-gate solve rate sits in a
usable band, using measured rates where we have them and repo diversity as
the only available proxy where we do not.

  python3 select_overfit_tasks.py --n 16 --group 8
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = (
    "/data/fshu/important/swe_data/r2e_family/"
    "R2E-Gym-Subset_validgold_unique_baseline/train.jsonl"
)
DEFAULT_REFERENCE = REPO / "runs" / "r2e160-20260923-155841" / "reference.toml"

# Post-gate solve rates measured over 35 rollouts each in
# runs/r2e160-mops8-20260928-172754. These are the only instances in the pool
# whose difficulty we actually know.
MEASURED: dict[str, float] = {
    "pyramid@a43abd25": 0.69,
    "coveragepy@c4fc3833": 0.43,
    "pillow@bfaa0a1f": 0.37,
    "coveragepy@5c3d0946": 0.34,
    "scrapy@b51b52ff": 0.31,
    "aiohttp@fa628a21": 0.20,
    "orange3@78213643": 0.20,
    "numpy@a5322429": 0.11,
    "tornado@37081d79": 0.09,
    "datalad@2df9d5fb": 0.03,
}

# Measured at zero over 15 rollouts in the earlier 12-task probe. Excluded:
# a task that never varies contributes exactly nothing to the gradient.
KNOWN_DEAD = {"aiohttp@fecb85a9", "pandas@5f5350b8"}

# Below this, a group is unanimous often enough that the task mostly burns
# collection time. datalad at 0.03 yields P(informative) = 0.22 even at G=8.
MIN_USABLE_RATE = 0.05

# A repo whose measured instances essentially never survive the gate. Its other
# instances share a test harness and problem style, so they are the worst place
# to spend an unmeasured slot.
MIN_REPO_RATE = 0.05


def informative(p: float, group: int) -> float:
    return 1.0 - p**group - (1.0 - p) ** group


def reference_hashes(path: Path) -> set[str]:
    """The 160 commit hashes from the reference taskset filter_fn."""
    text = path.read_text()
    return set(re.findall(r'"([a-f0-9]{40})"', text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.dataset)]
    hashes = reference_hashes(Path(args.reference))
    pool = [r for r in rows if r.get("commit_hash") in hashes]
    print(f"dataset {len(rows)} instances -> {len(pool)} in the reference subset")

    keep = [t for t, p in MEASURED.items() if p >= MIN_USABLE_RATE]
    keep.sort(key=lambda t: -informative(MEASURED[t], args.group))
    dropped = [t for t, p in MEASURED.items() if p < MIN_USABLE_RATE]

    by_id = {r["instance_id"]: r for r in pool}
    for t in keep:
        if t not in by_id:
            raise SystemExit(f"measured task {t} is not in the reference subset")

    # Every repo in the subset is one we have already measured, so there is no
    # unseen repo to diversify into. Repo-level tractability is then the only
    # signal available for an unmeasured instance: instances of a repo we have
    # never solved are far likelier to be unreachable than instances of one we
    # solve a third of the time. Rank eligible repos by how close their measured
    # mean sits to 0.5, where informativeness peaks, and fill round-robin so no
    # single repo's quirks dominate the set.
    repo_rate: dict[str, list[float]] = {}
    for task, p in MEASURED.items():
        repo_rate.setdefault(task.split("@")[0], []).append(p)
    repo_mean = {r: sum(v) / len(v) for r, v in repo_rate.items()}
    eligible = sorted(
        (r for r, m in repo_mean.items() if m >= MIN_REPO_RATE),
        key=lambda r: abs(repo_mean[r] - 0.5),
    )
    excluded = sorted(r for r, m in repo_mean.items() if m < MIN_REPO_RATE)
    print(f"eligible repos {eligible}; excluded as untractable {excluded}")

    candidates: dict[str, list[str]] = {}
    for r in pool:
        iid = r["instance_id"]
        repo = iid.split("@")[0]
        if repo in eligible and iid not in MEASURED and iid not in KNOWN_DEAD:
            candidates.setdefault(repo, []).append(iid)
    rng = random.Random(args.seed)
    for v in candidates.values():
        rng.shuffle(v)

    need = args.n - len(keep)
    picked: list[str] = []
    while len(picked) < need:
        progressed = False
        for repo in eligible:
            if len(picked) == need:
                break
            if candidates.get(repo):
                picked.append(candidates[repo].pop())
                progressed = True
        if not progressed:
            raise SystemExit("ran out of candidate instances in eligible repos")
    chosen = keep + picked

    print(f"\nkeeping {len(keep)} measured, adding {need} unmeasured, dropping {dropped}")
    print(f"\n{'instance_id':<28}{'p_kept':>9}{'P(inform)':>12}")
    total = 0.0
    for t in chosen:
        p = MEASURED.get(t)
        if p is None:
            print(f"{t:<28}{'?':>9}{'?':>12}")
        else:
            total += informative(p, args.group)
            print(f"{t:<28}{p:>9.2f}{informative(p, args.group):>12.3f}")

    print(f"\nexpected informative groups from the {len(keep)} measured: {total:.2f}")
    print(f"sequences/step if all {args.n} inform: {args.n * args.group}")
    print("\n--task-ids " + ",".join(chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
