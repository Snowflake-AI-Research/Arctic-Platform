#!/usr/bin/env python3
"""Stop one run from destroying another's work.

Two full overfit runs were lost to cancellation, and the launch path makes that
easy to do by accident: every launch script force-deletes *all* ``r2e-`` pods
and reaps every sandbox before it starts, so a second launch kills the first
run's in-flight rollouts even though it never meant to touch them. Several
agents or shells share this checkout, so "just don't do that" is not a control.

This records which run owns the cluster right now and lets the launch and
cancel paths refuse to stomp on it.

  python3 run_lock.py acquire <run_dir>   claim; exit 1 if someone else holds it
  python3 run_lock.py check <job_id>      exit 1 if that job belongs to the holder
  python3 run_lock.py show                print the holder, if any
  python3 run_lock.py release             drop the claim

``FORCE=1`` overrides acquire, for when you really do mean to replace a run.

Liveness is the run log's modification time, not a pid: the driver runs inside
another pod, so its pid means nothing here, and a stale lock from a crashed run
must not block the next launch forever.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "ACTIVE_RUN.json"
# A step is ~0.8-2.9h of collection, but the driver writes a rollout line every
# few minutes throughout, so silence this long means it is gone, not working.
STALE_SECONDS = 1800


def _holder() -> dict | None:
    """The current claim, or None if there is none or it has gone stale."""
    if not LOCK.exists():
        return None
    try:
        rec = json.loads(LOCK.read_text())
    except (OSError, ValueError):
        return None
    log = Path(rec.get("run_dir", "")) / "run.log"
    if not log.exists():
        return None
    age = time.time() - log.stat().st_mtime
    rec["log_age_s"] = int(age)
    return None if age > STALE_SECONDS else rec


def _job_ids(run_dir: str) -> list[str]:
    """Cortex job ids the driver recorded for a run, if it got that far."""
    f = Path(run_dir) / "cortex_job_ids.json"
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text())
    except (OSError, ValueError):
        return []
    if isinstance(data, dict):
        # Values may be plain ids or "<id>:training:0" style sub-job names.
        return [str(v).split(":")[0] for v in data.values() if v]
    return [str(v).split(":")[0] for v in data] if isinstance(data, list) else []


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    held = _holder()

    if cmd == "show":
        print(json.dumps(held, indent=2) if held else "no active run")
        return 0

    if cmd == "release":
        LOCK.unlink(missing_ok=True)
        print("released")
        return 0

    if cmd == "acquire":
        if len(sys.argv) < 3:
            print(__doc__)
            return 2
        run_dir = sys.argv[2]
        if held and held.get("run_dir") != run_dir:
            if os.environ.get("FORCE") != "1":
                print(
                    f"REFUSED: {held['run_dir']} is active "
                    f"(log touched {held['log_age_s']}s ago, jobs "
                    f"{','.join(_job_ids(held['run_dir'])) or 'not yet recorded'}).\n"
                    f"Launching would delete its pods and sandboxes. "
                    f"Re-run with FORCE=1 to replace it deliberately.",
                    file=sys.stderr,
                )
                return 1
            print(f"FORCE=1: replacing active run {held['run_dir']}")
        LOCK.write_text(json.dumps({
            "run_dir": run_dir,
            "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "in use -- see poc/run_lock.py before cancelling or relaunching",
        }, indent=2))
        print(f"acquired for {run_dir}")
        return 0

    if cmd == "check":
        if len(sys.argv) < 3:
            print(__doc__)
            return 2
        target = sys.argv[2].split(":")[0]
        if held and target in _job_ids(held["run_dir"]):
            print(
                f"REFUSED: job {target} belongs to the active run "
                f"{held['run_dir']} (log touched {held['log_age_s']}s ago).",
                file=sys.stderr,
            )
            return 1
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
