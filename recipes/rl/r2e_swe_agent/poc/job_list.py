#!/usr/bin/env python3
"""Which jobs exist in our schema, and which still hold GPUs?

``job_status.py`` needs a job id, which is useless when the id was never
printed -- a driver killed between job creation and its first log line leaves a
job holding GPUs with nothing on disk pointing at it. This lists the schema so
an orphan can be found and released.

Every job in a shared schema reports ``submitted_by=ADMIN``, so ownership has to
be read off the comment field. Nothing here mutates: cancelling is
``job_cancel.py``'s job, deliberately, because a wrong guess here would take
down someone else's run.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ap-harbor"))

ALIVE = {"PENDING", "RUNNING", "STARTING", "QUEUED", "CREATED"}


def main() -> int:
    from arctic_platform.client.config import CortexConfig
    from arctic_platform.client.transports.cortex import CortexTransport

    cx = CortexConfig(
        base_url=os.environ.get("ARCTIC_CORTEX_BASE_URL"),
        host=os.environ["ARCTIC_CORTEX_HOST"],
        pat_env_var="CORTEX_PAT",
        database=os.environ["ARCTIC_CORTEX_DATABASE"],
        endpoint=os.environ.get("ARCTIC_CORTEX_ENDPOINT", "cortex-training"),
        **{"schema": os.environ["ARCTIC_CORTEX_SCHEMA"]},
    )

    from arctic_platform.client.config import ArcticRLClientConfig

    # Bypass __init__ so listing never creates a job, but hand-set the few
    # attributes _build_session and _send read.
    t = CortexTransport.__new__(CortexTransport)
    t.config = ArcticRLClientConfig(
        model_name="unused", training_gpus=0, sampling_gpus=0, backend_config=cx
    )
    t.cortex = cx
    t.job_id = None
    t.pool_maxsize = 8
    t.poll_interval = 0.5
    t.max_retries = 2
    t.request_timeout = 60.0
    t.session = t._build_session()

    listing = t._send("GET", t._prefix)
    jobs = listing if isinstance(listing, list) else listing.get("jobs", listing)
    if isinstance(jobs, dict):
        jobs = [jobs]

    only_alive = "--all" not in sys.argv
    for j in jobs:
        state = str(j.get("status") or j.get("state") or "?").upper()
        if only_alive and state not in ALIVE:
            continue
        gpus = sum(
            int(s.get("n_gpus") or 0)
            for s in (j.get("sub_jobs") or j.get("sub_job_configs") or [])
        )
        print(
            # Full id, not a prefix: the API rejects a truncated id with 404,
            # so a short form here would be unusable with job_status/job_cancel.
            f"{j.get('job_id')}  {state:<10} gpus={gpus:<3} "
            f"{j.get('created_at') or ''}  {(j.get('comment') or '')[:70]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
