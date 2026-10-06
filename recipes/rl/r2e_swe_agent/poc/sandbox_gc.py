#!/usr/bin/env python3
"""Delete sandbox pods left behind by runs that were killed.

A driver deletes its own sandboxes when a rollout ends, but SIGKILL skips
that, so every abandoned run leaves its whole fleet running. They are not
idle: each holds a checked-out repository and its dependencies resident on a
node the live run is also using, so orphans from yesterday quietly tax
today's throughput.

Age is the only safe discriminator available, hence a minimum age rather than
a name match: pods younger than the cutoff may belong to a run in flight.
Default is a dry run; pass --delete to act.

  python3 sandbox_gc.py --older-than 5h [--delete]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sandbox import NAMESPACE, _kubectl  # noqa: E402


def _parse_age(text: str) -> float:
    """Hours from a duration like '90m', '5h', '2h30m'."""
    total, num = 0.0, ""
    units = {"s": 1 / 3600, "m": 1 / 60, "h": 1.0, "d": 24.0}
    for ch in text:
        if ch.isdigit():
            num += ch
        elif ch in units and num:
            total += int(num) * units[ch]
            num = ""
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--older-than", default="5h")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--prefix", default="r2e-")
    args = ap.parse_args()

    cutoff = _parse_age(args.older_than)
    now = datetime.now(timezone.utc)

    r = _kubectl("get", "pods", "-n", NAMESPACE, "-o", "json")
    if r.returncode != 0:
        print(r.stderr[:400])
        return 1

    stale = []
    for item in json.loads(r.stdout).get("items", []):
        name = item["metadata"]["name"]
        if not name.startswith(args.prefix):
            continue
        created = datetime.fromisoformat(
            item["metadata"]["creationTimestamp"].replace("Z", "+00:00")
        )
        hours = (now - created).total_seconds() / 3600
        if hours >= cutoff:
            stale.append((name, hours))

    stale.sort(key=lambda kv: -kv[1])
    print(f"{len(stale)} pods older than {args.older_than}")
    for name, hours in stale[:5]:
        print(f"  {name}  {hours:.1f}h")
    if len(stale) > 5:
        print(f"  ... and {len(stale) - 5} more")

    if not args.delete:
        print("\ndry run; pass --delete to remove them")
        return 0

    # The Sandbox custom resource owns the pod, so deleting the pod alone
    # would just have the controller recreate it.
    for name, _ in stale:
        _kubectl("delete", "sandbox", name, "-n", NAMESPACE, "--wait=false")
    print(f"\ndeleted {len(stale)} sandboxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
