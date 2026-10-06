#!/usr/bin/env python3
"""Per-task pass@k and the turn budget a cap would actually need.

Two questions have to be answered before a turn cap can be chosen, and the run
artifacts answer them through different keys, which is why this is a script and
not a one-liner:

- **Can the agent solve the task at all?** ``run.log``'s per-rollout lines carry
  the task name and are in the same order as ``history.json``'s arrays, so the
  task joins to ``earned_rewards`` by position. ``run.log``'s own ``reward=`` is
  the *post-gating* value, so it under-counts solves; ``earned_rewards`` is the
  one that answers pass@k. (``run.log`` also duplicates every line through the
  log mirror, so records are deduped by index.)
- **How many turns does an outcome take?** ``turn_ids/`` and ``transcripts/``
  are both keyed by ``(task, per-task index)``, which is a *different* key from
  the global completion index above. So turns join to the stop condition, but
  not directly to reward. The stop condition is the usable proxy: a trajectory
  that ends on a format or repetition gate spent its turns failing, and one that
  ends cleanly spent them working.

Usage:
  python3 analyze_pass_and_turns.py runs/overfit16d-20260929-145849 [--group 8]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROLLOUT_LINE = re.compile(
    r"\[(\d+)/(\d+)\]\s+(\S+)\s+reward=([0-9.]+)\s+stop=(\S+)\s+t=([0-9.]+)s"
)
# transcripts/<prefix>-<task>-<idx>.log and turn_ids/<prefix>-<task>-<idx>.json,
# where <task> itself contains a '@'. Anchor on the trailing -<idx>.
KEYED_NAME = re.compile(r"^(?:s\d+-)+(?P<task>.+?)-(?P<idx>\d+)$")


def rollout_records(run: Path) -> list[dict]:
    """One record per rollout: task, gated reward, earned reward, stop, seconds."""
    hist = json.loads((run / "history.json").read_text())[0]
    earned = hist["earned_rewards"]
    gated = hist["rewards"]

    by_index: dict[int, tuple] = {}
    for line in (run / "run.log").read_text(errors="replace").splitlines():
        m = ROLLOUT_LINE.search(line)
        if m:
            by_index.setdefault(
                int(m.group(1)),
                (m.group(3), float(m.group(4)), m.group(5), float(m.group(6))),
            )

    out = []
    for i in sorted(by_index):
        task, rline, stop, secs = by_index[i]
        if i - 1 >= len(gated):
            continue
        # Positional join is only trustworthy if run.log's own reward column
        # reproduces history's gated array; bail loudly rather than silently
        # mis-attributing a solve to the wrong task.
        out.append(
            {
                "task": task,
                "gated": gated[i - 1],
                "earned": earned[i - 1],
                "stop": stop,
                "secs": secs,
                "line_reward": rline,
            }
        )
    mismatch = sum(1 for r in out if abs(r["line_reward"] - r["gated"]) > 1e-9)
    if mismatch:
        raise SystemExit(
            f"run.log/history ordering disagrees on {mismatch}/{len(out)} rollouts; "
            "the task->reward join is unsafe"
        )
    return out


def turns_by_key(run: Path) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for f in (run / "turn_ids").glob("*.json"):
        m = KEYED_NAME.match(f.stem)
        if not m:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(d, list) and d:
            out[(m.group("task"), m.group("idx"))] = len(d)
    return out


def stop_by_key(run: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for f in (run / "transcripts").glob("*.log"):
        m = KEYED_NAME.match(f.stem)
        if not m:
            continue
        text = f.read_text(errors="replace")
        marker = "__MINI_SWE_AGENT_PLUS_STOP__:"
        if marker not in text:
            continue
        blob = text.split(marker, 1)[1].splitlines()[0]
        try:
            rec = json.loads(blob)
        except Exception:
            continue
        out[(m.group("task"), m.group("idx"))] = rec.get("stop_condition", "?")
    return out


def cumulative_lengths(run: Path) -> dict[tuple[str, str], list[int]]:
    """Packed token length after each turn, per rollout."""
    out: dict[tuple[str, str], list[int]] = {}
    for f in (run / "turn_ids").glob("*.json"):
        m = KEYED_NAME.match(f.stem)
        if not m:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(d, list) or not d:
            continue
        out[(m.group("task"), m.group("idx"))] = [
            len(t.get("prompt", [])) + len(t.get("completion", [])) for t in d
        ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=151936)
    args = ap.parse_args()

    recs = rollout_records(args.run)
    per_task: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        per_task[r["task"]].append(r)

    print(f"rollouts={len(recs)}  tasks={len(per_task)}  group={args.group}\n")
    print(f"{'task':<26} {'pass@k':>7} {'solved':>7} {'gated':>6} {'lost':>5}")
    print("-" * 60)
    rows = []
    for task, rs in per_task.items():
        earned = sum(1 for r in rs if r["earned"] > 0)
        kept = sum(1 for r in rs if r["gated"] > 0)
        rows.append((earned / len(rs), task, earned, len(rs), kept))
    for frac, task, earned, n, kept in sorted(rows, reverse=True):
        print(
            f"{task:<26} {frac:>7.2f} {earned:>3}/{n:<3} {kept:>6} {earned - kept:>5}"
        )

    solvable = [t for f, t, e, n, k in rows if e > 0]
    print(
        f"\nsolvable at pass@{args.group}: {len(solvable)}/{len(per_task)} tasks  "
        f"| overall earned {sum(r['earned'] > 0 for r in recs)}/{len(recs)} "
        f"= {sum(r['earned'] > 0 for r in recs) / len(recs):.3f}"
    )
    print(
        f"kept after gating: {sum(r['gated'] > 0 for r in recs)}/{len(recs)} "
        f"= {sum(r['gated'] > 0 for r in recs) / len(recs):.3f}  "
        f"(gating discards {sum(r['earned'] > 0 for r in recs) - sum(r['gated'] > 0 for r in recs)})"
    )

    # ── turns by outcome ────────────────────────────────────────────────
    turns = turns_by_key(args.run)
    stops = stop_by_key(args.run)
    joined = [(k, turns[k], stops.get(k, "?")) for k in turns if k in stops]
    print(f"\nturns joined to a stop condition: {len(joined)} rollouts")
    by_stop: dict[str, list[int]] = defaultdict(list)
    for _k, n, s in joined:
        by_stop[s].append(n)
    print(f"\n{'stop condition':<22} {'n':>4} {'median':>7} {'p90':>5} {'max':>5}")
    print("-" * 48)
    for s, ns in sorted(by_stop.items(), key=lambda kv: -len(kv[1])):
        ns.sort()
        print(
            f"{s:<22} {len(ns):>4} {st.median(ns):>7.0f} "
            f"{ns[int(0.9 * (len(ns) - 1))]:>5} {ns[-1]:>5}"
        )

    # ── what a cap costs ────────────────────────────────────────────────
    cum = cumulative_lengths(args.run)
    print(f"\n{'cap':>5} {'max tok':>9} {'median tok':>11} {'GiB(max)':>9} {'calls':>8} {'reach cap':>10}")
    print("-" * 60)
    total_rollouts = len(cum)
    for cap in (20, 30, 40, 50, 60, 80, 100):
        lens, calls, hit = [], 0, 0
        for _k, series in cum.items():
            k = min(cap, len(series))
            calls += k
            lens.append(series[k - 1])
            if len(series) > cap:
                hit += 1
        lens.sort()
        mx = lens[-1]
        scaled = round(calls * (len(recs) / max(total_rollouts, 1)))
        print(
            f"{cap:>5} {mx:>9,} {lens[len(lens) // 2]:>11,} "
            f"{mx * args.vocab * 4 / 2**30:>9.1f} {scaled:>8,} "
            f"{hit / total_rollouts:>9.0%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
