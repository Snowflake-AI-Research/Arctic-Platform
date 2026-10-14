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

"""Cancel the jobs an onboarding run left behind.

    python -m arctic_platform.inference.semi_persistence.onboarding.cleanup config=CONN.json ledger=onboarding_reports/jobs-TIME.jsonl \\
        [dry_run=true]

``onboard`` appends every job to its ledger when it submits it and again when it
cancels it. A run that ended normally cancelled everything; one that was killed,
or whose machine slept or lost its connection, may not have. This cancels every
job the ledger shows submitted but not cancelled and still running or starting.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from arctic_platform.inference.semi_persistence.onboarding.common import as_bool
from arctic_platform.inference.semi_persistence.onboarding.common import parse_kv_args
from arctic_platform.inference.semi_persistence.onboarding.jobs import TERMINAL
from arctic_platform.inference.semi_persistence.onboarding.jobs import normalize_status
from arctic_platform.inference.semi_persistence.onboarding.client import make_client

logger = logging.getLogger(__name__)


def open_jobs(ledger: Path) -> dict[str, str]:
    """Job id -> "model/variant" for every job submitted and never cancelled."""
    jobs: dict[str, str] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if "cancelled_at" in entry or entry.get("cancelled"):
            jobs.pop(entry["job_id"], None)
        else:
            jobs[entry["job_id"]] = f"{entry.get('model')}/{entry.get('variant')}"
    return jobs


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_kv_args(argv)
    if "config" not in args or "ledger" not in args:
        raise SystemExit("config=CONNECTION_FILE and ledger=LEDGER.jsonl are required")
    dry_run = as_bool(args.get("dry_run", "false"))
    client = make_client(args["config"])
    pending = open_jobs(Path(args["ledger"]).expanduser())
    logger.info("%d job(s) in the ledger were never cancelled", len(pending))
    for job_id, what in pending.items():
        status = normalize_status(client.get_job(job_id).get("status"))
        if status in TERMINAL:
            logger.info("%s (%s) already %s", job_id, what, status)
            continue
        if dry_run:
            logger.info("would cancel %s (%s), now %s", job_id, what, status)
            continue
        client.cancel_job(job_id)
        logger.info("cancelled %s (%s), was %s", job_id, what, status)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
