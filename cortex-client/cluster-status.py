#!/usr/bin/env python3
"""Live summary of running Neutrino jobs and GPU usage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


LIST_COMMAND_PREFIX = ["dss-neutrino", "list", "--status"]
WATCH_STATUSES = ("running", "placing")
GPU_CONFIG_KEYS = (
    "training_config",
    "inference_config",
    "sampling_config",
)


def _normalize_status(status: Any) -> str:
    return str(status or "").lower().removeprefix("job_state_")


def _gpu_count(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _format_count(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _sub_job_gpu_count(sub_job: dict[str, Any]) -> float:
    direct_count = _gpu_count(sub_job.get("n_gpus"))
    if direct_count is not None:
        return direct_count

    for key in GPU_CONFIG_KEYS:
        config = sub_job.get(key)
        if isinstance(config, dict):
            count = _gpu_count(config.get("n_gpus"))
            if count is not None:
                return count

    return 0.0


def _job_gpu_count(job: dict[str, Any]) -> float:
    sub_jobs = job.get("sub_jobs")
    if isinstance(sub_jobs, list):
        return sum(
            _sub_job_gpu_count(sub_job)
            for sub_job in sub_jobs
            if isinstance(sub_job, dict)
        )

    direct_count = _gpu_count(job.get("n_gpus"))
    return direct_count or 0.0


def _sub_job_summary(job: dict[str, Any]) -> str:
    sub_jobs = job.get("sub_jobs")
    if not isinstance(sub_jobs, list) or not sub_jobs:
        return "-"

    parts = []
    for sub_job in sub_jobs:
        if not isinstance(sub_job, dict):
            continue
        job_type = str(sub_job.get("job_type") or "sub-job").lower()
        count = _sub_job_gpu_count(sub_job)
        parts.append(f"{job_type}:{_format_count(count)}")
    return ", ".join(parts) if parts else "-"


def _load_jobs(raw_output: str) -> list[dict[str, Any]]:
    parsed = json.loads(raw_output)
    if isinstance(parsed, dict):
        jobs = parsed.get("jobs", [])
    elif isinstance(parsed, list):
        jobs = parsed
    else:
        raise ValueError("expected dss-neutrino list output to be a JSON object or list")

    if not isinstance(jobs, list):
        raise ValueError("expected dss-neutrino list output to contain a jobs list")

    return [job for job in jobs if isinstance(job, dict)]


def _run_list_command(status: str) -> str:
    command = [*LIST_COMMAND_PREFIX, status]
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout

    stderr = result.stderr.strip()
    detail = f": {stderr}" if stderr else ""
    raise RuntimeError(
        f"{' '.join(command)} failed with exit code {result.returncode}{detail}"
    )


def _read_jobs(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        jobs = []
        for status in WATCH_STATUSES:
            jobs.extend(_load_jobs(_run_list_command(status)))
        return jobs

    return _load_jobs(Path(path).read_text(encoding="utf-8"))


def _status_jobs(jobs: list[dict[str, Any]], status: str) -> list[dict[str, Any]]:
    return [job for job in jobs if _normalize_status(job.get("status")) == status]


def _job_rows(jobs: list[dict[str, Any]]) -> tuple[list[tuple[str, str, str]], float]:
    rows = []
    total_gpus = 0.0

    for job in jobs:
        job_id = str(job.get("job_id") or "-")
        count = _job_gpu_count(job)
        total_gpus += count
        rows.append((job_id, _format_count(count), _sub_job_summary(job)))

    return rows, total_gpus


def _append_status_section(
    lines: list[str],
    title: str,
    rows: list[tuple[str, str, str]],
) -> None:
    lines.append(title.upper())
    if not rows:
        lines.append("No jobs.")
        return

    job_width = max(36, min(64, max(len(row[0]) for row in rows)))
    lines.append(f"{'JOB ID':<{job_width}}  {'GPUS':>6}  SUB-JOBS")
    lines.append(f"{'-' * job_width}  {'-' * 6}  {'-' * 32}")
    for job_id, gpus, sub_jobs in rows:
        lines.append(f"{job_id:<{job_width}}  {gpus:>6}  {sub_jobs}")


def _render(jobs: list[dict[str, Any]], interval: float) -> str:
    rows_by_status = {}
    gpus_by_status = {}
    for status in WATCH_STATUSES:
        rows, total_gpus = _job_rows(_status_jobs(jobs, status))
        rows_by_status[status] = rows
        gpus_by_status[status] = total_gpus

    combined_gpus = sum(gpus_by_status.values())
    lines = [
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Refresh: every {_format_count(interval)}s",
        (
            f"Running jobs: {len(rows_by_status['running'])} | "
            f"GPUs: {_format_count(gpus_by_status['running'])}"
        ),
        (
            f"Placing jobs: {len(rows_by_status['placing'])} | "
            f"GPUs: {_format_count(gpus_by_status['placing'])}"
        ),
        f"Combined GPUs: {_format_count(combined_gpus)}",
        "",
    ]

    for index, status in enumerate(WATCH_STATUSES):
        if index:
            lines.append("")
        _append_status_section(lines, status, rows_by_status[status])

    return "\n".join(lines)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a live GPU usage summary for running Neutrino jobs.",
    )
    parser.add_argument(
        "-n",
        "--interval",
        type=float,
        default=5.0,
        help="Refresh interval in seconds. Defaults to 5.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one summary and exit.",
    )
    parser.add_argument(
        "--input",
        help="Read saved dss-neutrino list JSON instead of running the command.",
    )
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be greater than 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    while True:
        try:
            jobs = _read_jobs(args.input)
            output = _render(jobs, args.interval)
        except Exception as exc:
            output = f"Error: {exc}"

        if not args.once:
            print("\033[2J\033[H", end="")
        print(output, flush=True)

        if args.once:
            return 0

        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print()
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
