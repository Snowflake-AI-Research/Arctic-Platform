# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Show, and optionally release, Cortex jobs holding GPUs on this account.

Why this exists: Cortex caps an account at 8 GPUs, and this recipe asks for
exactly 8 (4 training + 4 sampling). A driver that exits cleanly releases them.
One that dies first -- SIGKILL, an OOM, a dropped SSH session -- does not, and
the job keeps its GPUs. The next launch then fails with

    429 ... per-account GPU cap reached: 8 GPUs in use plus 8 requested
    exceeds the cap of 8

which never mentions that the GPUs are held by your own dead run, so there is
no obvious way out. This is the way out.

    python cortex_jobs.py            # what is holding GPUs right now
    python cortex_jobs.py --cancel   # release all of it
    python cortex_jobs.py --all      # include finished jobs

Reads the same ARCTIC_CORTEX_* environment the recipe does.
"""

from __future__ import annotations

import sys

from arctic_platform.client import ArcticClientConfig
from arctic_platform.client.config import CortexConfig
from arctic_platform.client.transports.cortex import CortexTransport

# A job in any of these has already given its GPUs back.
_TERMINAL = {"failed", "done", "cancelled", "canceled", "succeeded", "terminated"}


def main() -> int:
    show_all = "--all" in sys.argv
    do_cancel = "--cancel" in sys.argv

    try:
        # model_name is required by the config but unused for job queries: this
        # never provisions anything, it only reads and cancels.
        config = ArcticClientConfig(model_name="unused", backend=CortexConfig.from_env())
        transport = CortexTransport(config)
    except Exception as exc:
        print(f"could not reach Cortex from the environment: {exc}", file=sys.stderr)
        print("Set ARCTIC_CORTEX_HOST and the PAT variable it names. See README.md.", file=sys.stderr)
        return 2

    jobs = transport.list_jobs()
    live = []
    for job in jobs:
        job_id = job.get("job_id") or job.get("id") or job.get("name")
        status = str(job.get("status") or job.get("state") or "?").lower()
        if status not in _TERMINAL:
            live.append(job_id)
            print(f"{job_id}  {status}  <-- holding GPUs")
        elif show_all:
            print(f"{job_id}  {status}")

    if not live:
        print(f"nothing holding GPUs ({len(jobs)} job(s) total, all finished).")
        return 0

    print(f"\n{len(live)} job(s) holding GPUs out of {len(jobs)} total.")
    if not do_cancel:
        print("Re-run with --cancel to release them.")
        return 0

    failed = 0
    for job_id in live:
        try:
            transport.cancel_job(job_id)
            print(f"released {job_id}")
        except Exception as exc:  # a job can finish between listing and cancelling
            failed += 1
            print(f"could not release {job_id}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
