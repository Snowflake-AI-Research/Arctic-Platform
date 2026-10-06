"""Release a Cortex job's GPUs.

A job keeps its allocation until it is explicitly cancelled: killing the local
driver only stops the client, so the sub-jobs stay RUNNING and keep billing.
Cancel the parent id, not the sub-jobs -- ``{job_id}:cancel`` tears down both
the training and sampling sides.

  python3 job_cancel.py <parent_job_id> [--force]

``--force`` is required to cancel the job belonging to the currently active
run: two overfit runs were already destroyed by a cancel issued from another
shell, and the job id alone gives no hint that hours of collection are riding
on it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import requests

from job_status import _headers, _prefix

_LOCK_TOOL = Path(__file__).resolve().parent / "run_lock.py"


def _protected(job_id: str) -> bool:
    """True if this job belongs to the run that currently holds the lock."""
    try:
        return subprocess.run(
            [sys.executable, str(_LOCK_TOOL), "check", job_id],
            capture_output=True, text=True, timeout=30,
        ).returncode != 0
    except Exception:  # noqa: BLE001 - a broken guard must not block a cancel
        return False


def main() -> int:
    args = sys.argv[1:]
    force = "--force" in args
    ids = [a for a in args if not a.startswith("-")]
    if not ids:
        print(__doc__)
        return 2
    for job_id in ids:
        parent = job_id.split(":")[0]
        if not force and _protected(parent):
            print(f"{parent}: REFUSED (active run; pass --force if deliberate)")
            continue
        url = f"{_prefix()}/{parent}:cancel"
        try:
            r = requests.post(url, headers=_headers(), timeout=60)
            print(f"{parent}: HTTP {r.status_code} {r.text[:200]}")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"{parent}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
