#!/usr/bin/env python3
"""Tabulate rollout outcomes across driver run logs.

The 74% terminal-invalid figure is an aggregate, and aggregates hide the thing
we actually need: *which* detector fired. A trace killed by `unknown_tool` on
turn 1 and a trace killed by `truncation` on turn 48 are different bugs with
different owners -- one is our prompt/parser, the other is the model genuinely
running long. This splits them so we stop treating the 74% as one problem.

  python3 diag_stops.py <run.log> [run.log ...]
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

# "    numpy@0e65b716   reward=0.0 turns=18 stop=format_invalid/unknown_tool t=394.3s"
ROLLOUT = re.compile(
    r"^\s+(?P<inst>\S+@\S+)\s+reward=(?P<reward>[\d.]+)\s+turns=(?P<turns>\d+)\s+"
    r"stop=(?P<stop>.*?)\s+t=(?P<t>[\d.]+)s"
)
STEP = re.compile(r"=====\s*step\s+(\d+)\s*=====")


def parse(path: Path) -> list[dict]:
    rows, step = [], -1
    for line in path.read_text(errors="replace").splitlines():
        m = STEP.search(line)
        if m:
            step = int(m.group(1))
            continue
        m = ROLLOUT.match(line)
        if not m:
            continue
        stop = m.group("stop").strip()
        # "format_invalid/nonempty_content, missing_tool_call" -> class + detectors
        cls, _, detail = stop.partition("/")
        rows.append({
            "log": path.name,
            "step": step,
            "inst": m.group("inst"),
            "reward": float(m.group("reward")),
            "turns": int(m.group("turns")),
            "cls": cls or "ok",
            "detail": detail.strip(),
            "secs": float(m.group("t")),
        })
    return rows


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "    -"


def main(paths: list[str]) -> None:
    rows: list[dict] = []
    for p in paths:
        rows += parse(Path(p))
    if not rows:
        print("no rollout lines matched")
        return

    n = len(rows)
    solved = sum(1 for r in rows if r["reward"] > 0)
    print(f"{n} rollouts across {len({r['log'] for r in rows})} log(s), "
          f"{len({(r['log'], r['step']) for r in rows})} step(s)")
    print(f"earned reward (pass rate): {solved}/{n} = {pct(solved, n)}\n")

    print("outcome class:")
    by_cls = collections.Counter(r["cls"] for r in rows)
    for cls, c in by_cls.most_common():
        s = sum(1 for r in rows if r["cls"] == cls and r["reward"] > 0)
        print(f"  {cls:16s} {c:4d}  {pct(c, n)}   solved={s}")

    print("\ndetector detail (what actually fired):")
    det = collections.Counter()
    for r in rows:
        if not r["detail"]:
            continue
        # a trace can trip several detectors at once
        for d in (x.strip() for x in r["detail"].split(",")):
            det[d] += 1
    for d, c in det.most_common():
        print(f"  {d:55s} {c:4d}  {pct(c, n)}")

    print("\nturns at which each class ends (median / min / max):")
    for cls in by_cls:
        t = sorted(r["turns"] for r in rows if r["cls"] == cls)
        print(f"  {cls:16s} med={t[len(t)//2]:4d}  min={t[0]:4d}  max={t[-1]:4d}")

    # A failure on turn 1 cannot be the model losing the thread; it means the
    # very first exchange was already unacceptable, which points at us.
    early = [r for r in rows if r["cls"] != "None" and r["turns"] <= 2]
    print(f"\nfailures within 2 turns (systematic, not model drift): "
          f"{len(early)}  {pct(len(early), n)}")
    for r in early[:10]:
        print(f"  {r['inst']:28s} turns={r['turns']:3d} {r['cls']}/{r['detail']}")

    print("\nterminal-invalid share (everything except a clean finish):")
    bad = sum(1 for r in rows if r["cls"] != "None")
    print(f"  {bad}/{n} = {pct(bad, n)}")
    lost = sum(1 for r in rows if r["cls"] != "None" and r["reward"] > 0)
    print(f"  of which had earned reward > 0 and get zeroed: {lost}")


if __name__ == "__main__":
    main(sys.argv[1:])
