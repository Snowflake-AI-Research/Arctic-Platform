"""Delete Sandbox CRs left behind by finished rollouts.

The controller recreates a sandbox's pod for as long as its CR exists, so
deleting pods frees nothing -- a previous run accumulated 100 CRs up to 16h old
and ate the node's allocatable memory. Rollouts are bounded (tens of minutes at
the concurrency we run), so a CR older than ``--max-age`` belongs to a rollout
that is gone.

Run alongside a driver:  python3 reap_sandboxes.py --loop 300
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from datetime import timezone

KUBECTL = ["/data-fast/k3s/bin/k3s", "kubectl"]
RESOURCE = "sandboxes.agents.x-k8s.io"


def _kubectl(*args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*KUBECTL, *args], capture_output=True, text=True, timeout=timeout
    )


def sweep(namespace: str, max_age: float, dry_run: bool) -> tuple[int, int]:
    got = _kubectl("get", RESOURCE, "-n", namespace, "-o", "json")
    if got.returncode != 0:
        return (0, 0)
    items = json.loads(got.stdout or "{}").get("items", []) or []
    now = datetime.now(timezone.utc)
    stale = []
    for it in items:
        ts = (it.get("metadata") or {}).get("creationTimestamp")
        name = (it.get("metadata") or {}).get("name")
        if not ts or not name:
            continue
        created = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if (now - created).total_seconds() > max_age:
            stale.append(name)
    if stale and not dry_run:
        # Batch the deletes; one call per CR is slow once there are hundreds.
        for i in range(0, len(stale), 20):
            _kubectl(
                "delete", RESOURCE, "-n", namespace, "--wait=false", *stale[i : i + 20]
            )
    return (len(items), len(stale))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="default")
    ap.add_argument(
        "--max-age",
        type=float,
        default=7200.0,
        help="seconds; a CR older than this cannot belong to a live rollout",
    )
    ap.add_argument(
        "--loop",
        type=float,
        default=0.0,
        help="sweep every N seconds; 0 sweeps once and exits",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    while True:
        total, reaped = sweep(args.namespace, args.max_age, args.dry_run)
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(
            f"[reaper {stamp}] sandboxes={total} stale={reaped}"
            f"{' (dry-run)' if args.dry_run else ' deleted'}",
            flush=True,
        )
        if not args.loop:
            return 0
        time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
