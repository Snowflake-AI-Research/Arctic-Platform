#!/usr/bin/env python3
"""Is a Cortex job still holding its GPUs, and what did the server accept?

``cortex_job_info.py`` answers what a job was configured as. This answers the
two questions that cost money or change conclusions: whether the sub-jobs are
still live, and what the training side actually took for the settings we reason
about (gradient clipping, optimizer, batch size). A job that responds to GET is
not necessarily a job that is running, so the status field is the thing to read
before assuming GPUs were released.

  python3 job_status.py <job_id> [<job_id> ...]
"""

from __future__ import annotations

import json
import os
import sys

import requests

_STATUS_KEYS = ("status", "state", "job_status", "execution_status", "lifecycle_state")


def _prefix() -> str:
    base = os.environ["ARCTIC_CORTEX_HOST"].rstrip("/")
    if not base.startswith("http"):
        base = f"https://{base}"
    db = os.environ["ARCTIC_CORTEX_DATABASE"]
    schema = os.environ["ARCTIC_CORTEX_SCHEMA"]
    endpoint = os.environ.get("ARCTIC_CORTEX_ENDPOINT", "cortex-training")
    return f"{base}/api/v2/databases/{db}/schemas/{schema}/{endpoint}"


def _headers() -> dict[str, str]:
    pat = os.environ.get("CORTEX_PAT") or os.environ["ARCTIC_CORTEX_PAT"]
    return {
        "Authorization": f"Bearer {pat}",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    }


def _statuses(obj: dict) -> str:
    found = {k: obj[k] for k in _STATUS_KEYS if obj.get(k) is not None}
    return json.dumps(found) if found else "(no status field)"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for raw in sys.argv[1:]:
        job_id = raw.split(":")[0]
        r = requests.get(f"{_prefix()}/{job_id}", headers=_headers(), timeout=60)
        print(f"\n=== {job_id}  HTTP {r.status_code}")
        if r.status_code != 200:
            print(r.text[:300])
            continue
        body = r.json()
        print(f"  job: {_statuses(body)}")
        # Any key we did not anticipate is worth seeing once rather than
        # guessing at: the schema is the server's, not ours.
        print(f"  top-level keys: {sorted(body)}")
        # The only field that says *why* a FAILED job failed. Without it a
        # startup failure is indistinguishable from a capacity rejection, and
        # the client's exception only ever reports the terminal state.
        if body.get("reason"):
            print(f"  reason: {body['reason']}")
        # What the job actually holds, which is the only number that matters
        # when asking what a run costs. Our own flags say what we requested;
        # this says what the server gave us.
        if body.get("hardware"):
            print(f"  hardware: {json.dumps(body['hardware'])}")
        for sub in body.get("sub_jobs") or []:
            print(f"  {sub.get('job_type')} {sub.get('sub_job_id')}")
            print(f"    {_statuses(sub)}")
            if sub.get("reason"):
                print(f"    reason: {sub['reason']}")
            if sub.get("hardware"):
                print(f"    hardware: {json.dumps(sub['hardware'])}")
            cfg = sub.get("training_config") or {}
            if cfg:
                for k in sorted(cfg):
                    print(f"    {k} = {cfg[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
