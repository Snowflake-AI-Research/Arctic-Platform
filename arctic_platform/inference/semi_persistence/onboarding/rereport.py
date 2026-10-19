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

"""Rebuild onboarding reports offline from their JSON and saved logs.

    python -m arctic_platform.inference.semi_persistence.onboarding.rereport report=REPORT.json[,REPORT.json...] \\
        [log_files_on=POD_LOG_DIR] [log_files_off=POD_LOG_DIR] [out_dir=DIR]

Re-parses the logs the run saved (``<report>-logs/``) plus any GPU pod logs
given, re-scores the perf checks from the saved measurements, then rewrites
the Markdown and JSON in place (or into ``out_dir``). No jobs are needed, so
it works after the jobs are gone. With several reports,
``log_files_on`` may hold one subdirectory per semi_p job named by the first
eight characters of its job id; each report picks its own.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from arctic_platform.inference.semi_persistence.onboarding import book
from arctic_platform.inference.semi_persistence.onboarding import perf
from arctic_platform.inference.semi_persistence.onboarding import report
from arctic_platform.inference.semi_persistence.onboarding import startup
from arctic_platform.inference.semi_persistence.onboarding.common import parse_kv_args

logger = logging.getLogger(__name__)


def _saved_api_logs(log_dir: Path, side: str) -> dict[str, list[str]]:
    if not log_dir.is_dir():
        return {}
    return {p.name[len(side) + 1:]: p.read_text(encoding="utf-8", errors="replace").splitlines()
            for p in sorted(log_dir.glob(f"{side}-*"))}


def _pod_logs(paths: list[str], job_id: str | None) -> dict[str, list[str]]:
    """Pod logs for one job: a ``<job8>/`` subdirectory when one exists, else ``paths`` as given."""
    chosen = []
    for raw in paths:
        p = Path(raw).expanduser()
        if not p.is_dir():
            chosen.append(str(p))
            continue
        per_job = [c for c in p.iterdir() if c.is_dir()]
        if not per_job:
            chosen.append(str(p))
        elif job_id and (p / job_id[:8]).is_dir():
            chosen.append(str(p / job_id[:8]))
        # A per-job layout without this job's directory contributes nothing.
    return startup.read_files(chosen)


def rebuild(path: Path, on_paths: list[str], off_paths: list[str], out_dir: str | None) -> Path:
    result = json.loads(path.read_text(encoding="utf-8"))
    result.pop("verdict", None)
    result.pop("reasons", None)
    st = result.setdefault("startup", {})
    log_dir = Path(st.get("log_dir") or path.with_name(f"{path.stem}-logs"))
    on_api, off_api = _saved_api_logs(log_dir, "semi_p"), _saved_api_logs(log_dir, "plain")
    on_pods = _pod_logs(on_paths, result.get("on_job_id"))
    off_pods = _pod_logs(off_paths, result.get("off_job_id"))
    if on_api or on_pods:
        st["semi_p"] = startup.restore_report(on_api, on_pods)
        st["generate_counts"] = startup.generate_counts(on_pods)
        st["on_sources"] = sorted(on_api) + sorted(on_pods)
    if off_api or off_pods:
        st["vllm_off"] = startup.parse_vllm([x for logs in (off_api, off_pods) for v in logs.values() for x in v])
        st["off_sources"] = sorted(off_api) + sorted(off_pods)
    pf = result.get("perf") or {}
    if pf.get("rows") and pf.get("latency"):
        replicas = int(result["model"].get("replicas", 1)) * int(result["model"].get("copies", 1))
        pf.update(perf.score(pf["rows"], pf["latency"], replicas, pf.get("thresholds") or {}))
    md = report.write(result, out_dir or str(path.parent))
    logger.info("%s: %s (%d semi_p pod log(s))", md, report.verdict(result)[0], len(on_pods))
    return md


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_kv_args(argv)
    reports = [Path(p).expanduser() for p in (args.get("report") or "").split(",") if p]
    if not reports:
        raise SystemExit("report=REPORT.json[,...] is required")
    on_paths = [p for p in args.get("log_files_on", "").split(",") if p]
    off_paths = [p for p in args.get("log_files_off", "").split(",") if p]
    for path in reports:
        rebuild(path, on_paths, off_paths, args.get("out_dir"))
    for directory in {args.get("out_dir") or str(p.parent) for p in reports}:
        logger.info("onboarding book: %s", book.write(directory))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
