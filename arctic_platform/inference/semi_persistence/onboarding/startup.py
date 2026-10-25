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

"""Start-up phases from job logs: the semi_p restore, and vLLM's cold start.

For each engine the semi_p server logs ``semi_p: resolved model_dir``, then
either ``cache miss`` (a cold start that dumps) or ``materialized`` and
``reading weights in place`` (a restore from the published image), then
``restoring image``, ``restore complete`` and ``front end attached``, after
which the engine serves; key-to-ready runs to that last line, or to ``restore
complete`` in logs from before the AsyncLLM front end. vLLM's own cold start
logs the weight load, torch.compile, profiling, graph capture and the
engine-init total.

Two line formats carry these markers. The sub-job log tags each engine's lines
with its worker process, ``(InferenceWorker pid=P, ip=A) YYYY-MM-DD HH:MM:SS,mmm``;
an engine is identified by that worker plus the ``replica<K>`` or ``node<K>``
directory of its model_dir. Logs saved from the GPU pod instead tag lines
``HH:MM:SS.fff [r<K> <process>]`` and also carry each restore step as
``<<< <step> OK (<seconds>s)``, which the sub-job log does not.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from arctic_platform.inference.semi_persistence.onboarding.common import log_text

logger = logging.getLogger(__name__)

TAG = re.compile(r"(\d\d):(\d\d):(\d\d(?:\.\d+)?) \[(r\d+) (\w+)\]")
WORKER = re.compile(r"\((\w+) pid=(\d+), ip=([0-9a-fA-F.:]+)\)")
STAMP = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})")
SUBDIR = re.compile(r"model_dir=\S*?/((?:replica|node)\d+)\b")
NOT_RESTORE_STEPS = {"teardown"}
MARK = re.compile(r"semi_p: (?:multi-node )?(resolved model_dir|cache miss|materialized|"
                  r"reading weights in place|dump complete|restoring image|restore of|restore complete|"
                  r"front end attached)")
STEP = re.compile(r"<<< (\w+) (OK|FAILED) \(([0-9.]+)s\)")
# The child's step that hands the restored core to the AsyncLLM front end, after
# ``restore complete`` and before ``front end attached``.
ATTACH_STEPS = {"serve_core"}
DUMP_SECONDS = re.compile(r"dump complete .*? in ([0-9.]+)s")
ATTACH_SECONDS = re.compile(r"front end attached in ([0-9.]+)s")
VLLM_PHASES = {
    "load_weights_s": re.compile(r"Loading weights took ([0-9.]+) seconds"),
    "model_loading_s": re.compile(r"Model loading took [0-9.]+ GiB(?: memory)? and ([0-9.]+) seconds"),
    "torch_compile_s": re.compile(r"torch\.compile took ([0-9.]+) s in total"),
    "profiling_s": re.compile(r"Initial profiling/warmup run took ([0-9.]+) s"),
    "graph_capture_s": re.compile(r"Graph capturing finished in ([0-9.]+) secs"),
    "engine_init_s": re.compile(r"init engine \(profile, create kv cache, warmup model\) took ([0-9.]+) s"),
}


def read_sub_job_log(client: Any, job_id: str, max_pages: int = 200) -> list[str]:
    """The sub-job log so far. Pages until one comes back empty: a running job's
    log never reports ``eof``, so ``stream_logs(follow=False)`` would not return."""
    lines: list[str] = []
    cursor = None
    for _ in range(max_pages):
        page = client.tail_logs(job_id, cursor=cursor, max_lines=1000, sub_job_type="sampling")
        entries = page.get("entries") or []
        for entry in entries:
            lines.extend(log_text(entry).splitlines())
        cursor = page.get("next_cursor", cursor)
        if not entries or page.get("eof"):
            break
    return lines


def collect_logs(client: Any, job_id: str) -> dict[str, list[str]]:
    """The logs the API serves for ``job_id``: the sub-job log and the archived
    execution logs, keyed by a source name (also the saved file name)."""
    logs: dict[str, list[str]] = {}
    try:
        live = read_sub_job_log(client, job_id)
        if live:
            logs["sub-job.log"] = live
    except Exception as exc:  # noqa: BLE001 - a missing source is reported, not fatal
        logger.warning("sub-job log for %s unavailable: %s", job_id, exc)
    try:
        for f in client.fetch_execution_logs(job_id):
            logs[f"execution-{Path(f.get('filename') or 'log').name}"] = (f.get("content") or "").splitlines()
    except Exception as exc:  # noqa: BLE001
        logger.warning("execution logs for %s unavailable: %s", job_id, exc)
    return logs


def read_files(paths: list[str]) -> dict[str, list[str]]:
    """Saved log files, e.g. GPU pod logs, keyed by file name. A directory
    contributes every ``*.log`` under it."""
    out: dict[str, list[str]] = {}
    for raw in paths:
        path = Path(raw).expanduser()
        files = sorted(path.rglob("*.log")) if path.is_dir() else [path]
        for f in files:
            out[f.name] = f.read_text(encoding="utf-8", errors="replace").splitlines()
    return out


def save_logs(logs: dict[str, list[str]], directory: Path, prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, lines in logs.items():
        (directory / f"{prefix}-{name}").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pod_label(name: str) -> str:
    """A short label for a saved per-pod log: its name's last dash-separated part."""
    stem = name[:-4] if name.endswith(".log") else name
    return stem.rsplit("-", 1)[-1]


def restore_report(api_logs: dict[str, list[str]], pod_logs: dict[str, list[str]]) -> dict:
    """The semi_p restore summary. Saved per-pod logs carry every restore step,
    so when any are given they are parsed one pod at a time (each pod numbers its
    replicas from r0) and replace the API view; otherwise the API logs are used."""
    if pod_logs:
        merged: dict[str, dict] = {}
        for name, lines in sorted(pod_logs.items()):
            merged.update(_collect(lines, prefix=f"{_pod_label(name)} "))
        if merged:
            out = _summarize(merged)
            out["steps_source"] = "pod logs"
            return out
    out = _summarize(_collect([line for lines in api_logs.values() for line in lines]))
    out["steps_source"] = "sub-job log (no per-step breakdown)"
    return out


def _secs(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + float(s)


def _engine_of(line: str, workers: dict[str, str]) -> tuple[str, float, str] | None:
    """``(engine id, timestamp, process)`` for a tagged line, else None."""
    tag = TAG.search(line)
    if tag:
        return tag.group(4), _secs(*tag.group(1, 2, 3)), tag.group(5)
    worker = WORKER.search(line)
    stamp = STAMP.search(line)
    if not (worker and stamp):
        return None
    ts = datetime.strptime(stamp.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(stamp.group(2)) / 1000
    key = f"{worker.group(3)} pid {worker.group(2)}"
    sub = SUBDIR.search(line)
    if sub and "resolved model_dir" in line:
        workers[key] = f"{worker.group(3)} {sub.group(1)}"
    return workers.get(key, key), ts, "engine"


def parse_semi_p(lines: list[str]) -> dict:
    """Per-engine restore markers, and step durations when the log has them."""
    return _summarize(_collect(lines))


def _collect(lines: list[str], prefix: str = "") -> dict[str, dict]:
    """Raw markers and steps per engine, keyed ``prefix + engine id``."""
    replicas: dict[str, dict] = {}
    workers: dict[str, str] = {}
    for line in lines:
        if "semi_p:" not in line and "<<< " not in line:
            continue
        tagged = _engine_of(line, workers)
        if not tagged:
            continue
        rep, ts, proc = tagged
        r = replicas.setdefault(prefix + rep, {"marks": {}, "steps": [], "dump_s": None, "attach_s": None})
        mark = MARK.search(line)
        if mark:
            name = "restoring image" if mark.group(1) == "restore of" else mark.group(1)
            r["marks"].setdefault(name, ts)
            if name == "dump complete":
                d = DUMP_SECONDS.search(line)
                r["dump_s"] = float(d.group(1)) if d else None
            if name == "front end attached":
                a = ATTACH_SECONDS.search(line)
                r["attach_s"] = float(a.group(1)) if a else None
            continue
        step = STEP.search(line)
        # Steps count once a restore has begun; a pod holding only a later
        # node-partition logs no engine markers, so its steps always count.
        restoring = "restoring image" in r["marks"] and "restore complete" not in r["marks"]
        attaching = (step is not None and step.group(1) in ATTACH_STEPS
                     and "restore complete" in r["marks"] and "front end attached" not in r["marks"])
        if step and proc in ("worker", "child") and (restoring or attaching or not r["marks"]):
            if step.group(1) in NOT_RESTORE_STEPS:
                continue
            entry = (step.group(1), float(step.group(3)), step.group(2) == "OK")
            if not r["steps"] or r["steps"][-1][0] != entry[0]:
                r["steps"].append(entry)
    return replicas


def _summarize(replicas: dict[str, dict]) -> dict:
    out: dict[str, Any] = {"replicas": {}, "markers_visible": False, "cache_miss": False, "restored": False}
    for rep in sorted(replicas, key=_natural):
        mk = replicas[rep]["marks"]
        if "resolved model_dir" not in mk:
            if replicas[rep]["steps"]:
                # A node-partition pod's agent: steps only, restored with its leader.
                out["replicas"][rep] = {"partition_agent": True,
                                        "steps": {n: s for n, s, _ in replicas[rep]["steps"]},
                                        "failed_steps": [n for n, _, ok in replicas[rep]["steps"] if not ok]}
            continue
        out["markers_visible"] = True
        t0 = mk["resolved model_dir"]
        info: dict[str, Any] = {
            "cache_miss": "cache miss" in mk,
            "from_published_image": "materialized" in mk,
            "reading_weights_in_place": "reading weights in place" in mk,
            "restored": "restore complete" in mk,
        }
        if "materialized" in mk:
            info["materialize_s"] = round(mk["materialized"] - t0, 1)
        if replicas[rep]["dump_s"] is not None:
            info["cold_start_and_dump_s"] = replicas[rep]["dump_s"]
        if "restore complete" in mk:
            start = mk.get("restoring image", mk.get("materialized", t0))
            info["restore_s"] = round(mk["restore complete"] - start, 1)
            ready = mk.get("front end attached", mk["restore complete"])
            info["key_to_ready_s"] = round(ready - t0, 1)
        if "front end attached" in mk:
            info["attach_s"] = replicas[rep]["attach_s"]
        info["steps"] = {name: secs for name, secs, _ in replicas[rep]["steps"]}
        info["failed_steps"] = [name for name, _, ok in replicas[rep]["steps"] if not ok]
        out["replicas"][rep] = info
    reps = [r for r in out["replicas"].values() if not r.get("partition_agent")]
    out["cache_miss"] = any(r["cache_miss"] for r in reps)
    out["restored"] = bool(reps) and all(r["restored"] and r["from_published_image"] for r in reps)
    if reps and all("key_to_ready_s" in r for r in reps):
        out["key_to_ready_s"] = max(r["key_to_ready_s"] for r in reps)
    steps: dict[str, float] = {}
    for r in out["replicas"].values():
        for name, secs in r["steps"].items():
            steps[name] = max(steps.get(name, 0.0), secs)
    out["slowest_replica_steps"] = steps
    return out


def _natural(key: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", key)]


def parse_vllm(lines: list[str]) -> dict:
    """vLLM cold-start phases; with several replicas, the slowest one of each."""
    out: dict[str, float] = {}
    for line in lines:
        for name, rx in VLLM_PHASES.items():
            m = rx.search(line)
            if m:
                out[name] = max(out.get(name, 0.0), float(m.group(1)))
    return out


def generate_counts(pod_logs: dict[str, list[str]]) -> dict[str, int]:
    """Completed generate requests per semi_p replica, from saved pod logs'
    ``<<< generate`` lines (the sub-job log does not carry them)."""
    counts: dict[str, int] = {}
    for name, lines in sorted(pod_logs.items()):
        for line in lines:
            if "<<< generate" not in line or " OK " not in line:
                continue
            tag = TAG.search(line)
            if tag and tag.group(5) == "child":
                key = f"{_pod_label(name)} {tag.group(4)}"
                counts[key] = counts.get(key, 0) + 1
    return counts
