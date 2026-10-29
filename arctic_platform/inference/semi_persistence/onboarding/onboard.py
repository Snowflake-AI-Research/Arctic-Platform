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

"""Onboard models for semi-persistence: semi_p checked against plain vLLM.

    python -m arctic_platform.inference.semi_persistence.onboarding.onboard config=CONN.json models=qwen36-35b-a3b-tp1

Per model, up to four jobs on the same backend image:

  on_hit       semi_p restored from the published image (the reference)
  off_default  plain vLLM as shipped
  on_miss      semi_p forced to cold-start and dump (published images off)
  off_matched  plain vLLM given semi_p's collective setup (TP >= 2, one pod)

Before submitting a model's jobs it waits until the node cache has verified the
published skeleton and weights on every node (see verify.py), since on_hit
cannot restore from a node that has not.

on_hit and off_default run copies x the published layout and are compared for
throughput and latency; that runs first, while on_miss may still be dumping.
Then every check sends the same requests to every job: on_miss and off_matched
must match on_hit token for token, and off_default's differences are reported
with where they start. Start-up phases come from the logs; all jobs are then
cancelled and a report written. With several models, they run concurrently
within ``max_gpus``, largest first.

Arguments (``key=value``):
  config           connection file (required)
  models           comma-separated model names under onboarding/models/, or paths
  debug_image_tag  backend image for every job, instead of each model's entry in
                   onboarding/models/backends.json
  variants         subset of on_hit,off_default,on_miss,off_matched (default all;
                   on_hit and off_default are required)
  copies           the perf pair runs copies x the published n_gpus (default: 2
                   for a whole-node config, 1 otherwise, or the model file's)
  miss_image_cache local image-cache root for on_miss, unique per run, so a copy
                   left on the node by an earlier job cannot turn it into a hit
  checks           comma-separated subset of correctness,perf (default both)
  require_restore  stop when on_hit cold-starts (default true)
  max_gpus         GPU budget for concurrent models (default: free GPUs now
                   minus reserve_gpus)
  max_nodes        node budget; each job counts as whole nodes (default: none)
  reserve_gpus     submit only while this many GPUs stay free for others
                   (default 0)
  gpus_per_node    for the node count (default 8)
  out_dir          report directory (default onboarding_reports)
  keep_jobs        leave the jobs running (default false)
  job_on_hit / job_off_default / job_on_miss / job_off_matched
                   attach to running jobs instead (one model only; kept)
  log_files_on / log_files_off   saved GPU pod logs of on_hit / off_default (files
                   or directories); they add the per-step restore breakdown
  timeout_s        how long to wait for the jobs to run (default 3600)
  publish_dir      where publish wrote <image_tag>/summary.json, which names each
                   model's skeleton and weights (default <out_dir>/publish)
  verify_timeout_s how long to wait for them to be verified on every node
                   (default 3600)
  pre_cancel_hook  shell command run for each job just before it is cancelled,
                   with {job_id}, {variant} and {miss_image_cache} (the on_miss
                   job's cache directory, else empty) filled in; e.g. to delete
                   what on_miss left on the node's disk

Every job id is appended to <out_dir>/jobs-<time>.jsonl as it is submitted and
cancelled. SIGINT, SIGTERM and SIGHUP cancel every job still held before
exiting; if the process dies harder than that, cancel the leftovers with
python -m arctic_platform.inference.semi_persistence.onboarding.cleanup ledger=<out_dir>/jobs-<time>.jsonl.

Every log read is saved under <out_dir>/<model>-<time>-logs/. To rebuild a
report later with pod logs: python -m arctic_platform.inference.semi_persistence.onboarding.rereport report=X.json
log_files_on=DIR.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from arctic_platform.inference.semi_persistence.onboarding import book
from arctic_platform.inference.semi_persistence.onboarding import correctness
from arctic_platform.inference.semi_persistence.onboarding import perf
from arctic_platform.inference.semi_persistence.onboarding import report
from arctic_platform.inference.semi_persistence.onboarding import startup
from arctic_platform.inference.semi_persistence.onboarding import verify
from arctic_platform.inference.semi_persistence.onboarding.common import Pair
from arctic_platform.inference.semi_persistence.onboarding.common import as_bool
from arctic_platform.inference.semi_persistence.onboarding.common import parse_kv_args
from arctic_platform.inference.semi_persistence.onboarding.jobs import GPUS_PER_NODE
from arctic_platform.inference.semi_persistence.onboarding.jobs import MATCH_APPLIED_MARKER
from arctic_platform.inference.semi_persistence.onboarding.jobs import PERF_VARIANTS
from arctic_platform.inference.semi_persistence.onboarding.jobs import VARIANTS
from arctic_platform.inference.semi_persistence.onboarding.jobs import JobTracker
from arctic_platform.inference.semi_persistence.onboarding.jobs import ModelSpec
from arctic_platform.inference.semi_persistence.onboarding.jobs import build_variant_bodies
from arctic_platform.inference.semi_persistence.onboarding.jobs import load_model
from arctic_platform.inference.semi_persistence.onboarding.jobs import plan_variants
from arctic_platform.inference.semi_persistence.onboarding.jobs import resolve_image_tag
from arctic_platform.inference.semi_persistence.onboarding.jobs import copies_for
from arctic_platform.inference.semi_persistence.onboarding.jobs import timeline
from arctic_platform.inference.semi_persistence.onboarding.jobs import wait_running
from arctic_platform.inference.semi_persistence.onboarding.client import enable_debug_options
from arctic_platform.inference.semi_persistence.onboarding.client import make_client

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def analyze_startup(on_api: dict, off_api: dict, on_pods: dict, off_pods: dict) -> dict:
    """Start-up section of a report from the API logs and any saved pod logs."""
    off_lines = [line for logs in (off_api, off_pods) for lines in logs.values() for line in lines]
    return {
        "semi_p": startup.restore_report(on_api, on_pods),
        "vllm_off": startup.parse_vllm(off_lines),
        "generate_counts": startup.generate_counts(on_pods),
        "on_sources": sorted(on_api) + sorted(on_pods),
        "off_sources": sorted(off_api) + sorted(off_pods),
    }


class Runner:
    def __init__(self, args: dict[str, str]):
        if "config" not in args:
            raise SystemExit("config=CONNECTION_FILE is required")
        self.config = args["config"]
        self.image_tag_override = args.get("debug_image_tag") or None
        self.variants = [v for v in (args.get("variants") or ",".join(VARIANTS)).split(",") if v]
        self.checks = set((args.get("checks") or "correctness,perf").split(","))
        self.require_restore = as_bool(args.get("require_restore", "true"))
        self.keep_jobs = as_bool(args.get("keep_jobs", "false"))
        self.out_dir = args.get("out_dir", "onboarding_reports")
        self.timeout_s = float(args.get("timeout_s", "3600"))
        self.publish_dir = args.get("publish_dir") or str(Path(self.out_dir) / "publish")
        self.verify_timeout_s = float(args.get("verify_timeout_s", "3600"))
        self.copies = int(args["copies"]) if args.get("copies") else None
        self.miss_image_cache = args.get("miss_image_cache") or None
        self.max_gpus = int(args["max_gpus"]) if args.get("max_gpus") else None
        self.max_nodes = int(args["max_nodes"]) if args.get("max_nodes") else None
        self.reserve_gpus = int(args.get("reserve_gpus", "0"))
        self.gpus_per_node = int(args.get("gpus_per_node", str(GPUS_PER_NODE)))
        self.pre_cancel_hook = args.get("pre_cancel_hook") or None
        Path(self.out_dir).expanduser().mkdir(parents=True, exist_ok=True)
        self.ledger = Path(self.out_dir).expanduser() / f"jobs-{utc_now().replace(':', '').replace('-', '')}.jsonl"
        self.live: dict[str, str] = {}
        self.live_lock = threading.Lock()
        self.attach = {v: args.get(f"job_{v}") for v in VARIANTS if args.get(f"job_{v}")}
        if args.get("on_job_id"):
            self.attach["on_hit"] = args["on_job_id"]
        if args.get("off_job_id"):
            self.attach["off_default"] = args["off_job_id"]
        self.log_files = {side: [p for p in args.get(f"log_files_{side}", "").split(",") if p]
                          for side in ("on", "off")}
        enable_debug_options()
        self.client = self.fast_client()

    def fast_client(self) -> Any:
        """A client that polls requests every 20 ms without backoff, so client
        latency is not dominated by the poll interval."""
        client = make_client(self.config, poll_interval=0.02)
        client.poll_backoff_multiplier = 1.0
        return client

    def footprint(self, spec: ModelSpec) -> tuple[int, int]:
        """``(GPUs, nodes)`` all of one model's jobs hold together. A job counts as
        whole nodes, since a node it shares is not one another job can use whole."""
        variants, _ = plan_variants(spec, self.variants)
        copies = copies_for(spec, self.copies)
        gpus = [spec.n_gpus * (copies if v in PERF_VARIANTS else 1) for v in variants]
        return sum(gpus), sum(-(-g // self.gpus_per_node) for g in gpus)

    def wait_for_capacity(self, gpus: int, name: str) -> None:
        """Block until ``gpus`` are free with ``reserve_gpus`` still left over."""
        last = None
        while True:
            free = int(self.client.get_capacity().get("available_gpus", 0))
            if free - gpus >= self.reserve_gpus:
                return
            if free != last:
                logger.info("%s: waiting for %d GPUs plus %d in reserve (%d free)", name, gpus,
                            self.reserve_gpus, free)
                last = free
            time.sleep(30)

    def submitted(self, job_id: str, spec: ModelSpec, variant: str) -> None:
        with self.live_lock:
            self.live[job_id] = f"{spec.name}/{variant}"
            with self.ledger.open("a") as f:
                f.write(json.dumps({"job_id": job_id, "model": spec.name, "variant": variant,
                                    "submitted_at": utc_now()}) + "\n")

    def cancelled(self, job_id: str) -> None:
        with self.live_lock:
            self.live.pop(job_id, None)
            with self.ledger.open("a") as f:
                f.write(json.dumps({"job_id": job_id, "cancelled_at": utc_now()}) + "\n")

    def abort(self, signum: int, _frame: Any) -> None:
        """On SIGINT/SIGTERM/SIGHUP: cancel every job still held, then exit."""
        with self.live_lock:
            held = dict(self.live)
        logger.error("signal %d: cancelling %d running job(s)", signum, len(held))
        for job_id, what in held.items():
            try:
                self.client.cancel_job(job_id)
                self.cancelled(job_id)
                logger.info("cancelled %s (%s)", job_id, what)
            except Exception:  # noqa: BLE001
                logger.exception("failed to cancel %s -- run arctic_platform.inference.semi_persistence.onboarding.cleanup ledger=%s",
                                 job_id, self.ledger)
        os._exit(128 + signum)

    def run_one(self, spec: ModelSpec, capacity_checked: bool = False) -> dict:
        copies = copies_for(spec, self.copies)
        variants, skipped = plan_variants(spec, list(self.attach) if self.attach else self.variants)
        image_tag = resolve_image_tag(spec, self.image_tag_override)
        result: dict[str, Any] = {
            "model": {"name": spec.name, "model_name": spec.sub_job["model_name"], "n_gpus": spec.n_gpus,
                      "tensor_parallel_size": spec.tensor_parallel_size, "replicas": spec.replicas,
                      "copies": copies, "test_n_gpus": spec.n_gpus * copies,
                      "test_replicas": spec.replicas * copies,
                      "max_seq_len": spec.max_seq_len, "file": spec.path},
            "image_tag": image_tag, "started_at": utc_now(),
            "variants": variants, "variants_skipped": skipped,
        }
        client = self.fast_client()
        attached = bool(self.attach)
        trackers: dict[str, JobTracker] = {}
        try:
            miss_cache = (f"{self.miss_image_cache.rstrip('/')}/{report.stem(result)}"
                          if self.miss_image_cache else None)
            bodies = build_variant_bodies(spec, image_tag, variants, copies, miss_cache)
            if not attached and "on_hit" in variants:
                summary = Path(self.publish_dir).expanduser() / image_tag / "summary.json"
                result["published_image"] = verify.wait_verified(
                    verify.published_dirs(summary, spec.name), self.verify_timeout_s, spec.name)
                checked = not result["published_image"].startswith("not checked")
                logger.log(logging.INFO if checked else logging.WARNING, "%s: published image %s",
                           spec.name, result["published_image"])
            if not (attached or capacity_checked):
                self.wait_for_capacity(self.footprint(spec)[0], spec.name)
            for v in variants:
                if attached:
                    trackers[v] = JobTracker(v, self.attach[v], time.time())
                    continue
                submitted = time.time()
                trackers[v] = JobTracker(v, client.create_job_from_body(bodies[v])["job_id"], submitted)
                self.submitted(trackers[v].job_id, spec, v)
                logger.info("submitted %s job %s", v, trackers[v].job_id)
            result["jobs"] = {v: t.job_id for v, t in trackers.items()}
            result["on_job_id"], result["off_job_id"] = trackers["on_hit"].job_id, trackers["off_default"].job_id

            # The perf jobs first: on_miss may still be cold-starting and dumping.
            wait_running(client, [trackers[v] for v in variants if v in PERF_VARIANTS], self.timeout_s)
            restore = startup.restore_report(startup.collect_logs(client, trackers["on_hit"].job_id),
                                             startup.read_files(self.log_files["on"]))
            result["startup"] = {"semi_p": restore}
            if restore["cache_miss"] and self.require_restore:
                raise RuntimeError("semi_p job cold-started (cache miss); publish this config's image first, "
                                   "or pass require_restore=false")
            if "perf" in self.checks:
                pair = Pair(on=trackers["on_hit"].job_id, off=trackers["off_default"].job_id,
                            replicas=spec.replicas * copies, max_seq_len=spec.max_seq_len)
                matched = trackers["off_matched"].job_id if "off_matched" in trackers else None
                result["perf"] = perf.run(self.fast_client, pair, spec.perf, spec.thresholds, matched)

            if "correctness" in self.checks:
                rest = [trackers[v] for v in variants if v not in PERF_VARIANTS]
                wait_running(client, rest, self.timeout_s)
                gated = self.gated_variants(client, trackers, result)
                replicas = {v: spec.replicas * (copies if v in PERF_VARIANTS else 1) for v in variants}
                result["correctness"] = correctness.run(client, {v: t.job_id for v, t in trackers.items()},
                                                        "on_hit", gated, replicas, spec.max_seq_len,
                                                        spec.correctness)
        except Exception as exc:  # noqa: BLE001 - recorded in the report
            logger.exception("%s: %s", spec.name, exc)
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if trackers:
                self.finish(client, spec, trackers, result, attached)
            result["finished_at"] = utc_now()
            result["report"] = str(report.write(result, self.out_dir))
            logger.info("%s: %s, report %s", spec.name, report.verdict(result)[0], result["report"])
        return result

    def gated_variants(self, client: Any, trackers: dict[str, JobTracker], result: dict) -> set[str]:
        """Which jobs must agree with on_hit exactly. on_miss must really have
        cold-started; off_matched only counts once the backend applied the match."""
        gated = set()
        notes = result.setdefault("variant_notes", {})
        if "on_miss" in trackers:
            miss = startup.restore_report(startup.collect_logs(client, trackers["on_miss"].job_id), {})
            if miss["cache_miss"]:
                notes["on_miss"] = "cold-started and dumped (forced miss)"
            else:
                notes["on_miss"] = ("did not cold-start: it found a local image (set miss_image_cache to force a "
                                    "miss); still compared as a second restore")
            gated.add("on_miss")
        if "off_matched" in trackers:
            lines = [x for v in startup.collect_logs(client, trackers["off_matched"].job_id).values() for x in v]
            if any(MATCH_APPLIED_MARKER in line for line in lines):
                notes["off_matched"] = "backend applied semi_p's collective setup (worker class, workspace, copy path)"
                gated.add("off_matched")
            else:
                notes["off_matched"] = ("backend did not report SEMIP_MATCH_COLLECTIVES (older image): only the "
                                        "environment and pass_config were matched, so it is reported, not gated")
        if "off_default" in trackers:
            notes["off_default"] = "stock vLLM collectives; differences at TP >= 2 are expected and reported"
        return gated

    def finish(self, client: Any, spec: ModelSpec, trackers: dict[str, JobTracker], result: dict,
               attached: bool) -> None:
        st = result.setdefault("startup", {})
        if not attached:
            st["timelines"] = {v: timeline(t) for v, t in trackers.items()}
            st["on_timeline"], st["off_timeline"] = st["timelines"]["on_hit"], st["timelines"]["off_default"]
        try:
            logs = {v: startup.collect_logs(client, t.job_id) for v, t in trackers.items()}
            log_dir = Path(self.out_dir).expanduser() / f"{report.stem(result)}-logs"
            for v, logs_ in logs.items():
                startup.save_logs(logs_, log_dir, v)
            st.update(analyze_startup(logs["on_hit"], logs["off_default"],
                                      startup.read_files(self.log_files["on"]),
                                      startup.read_files(self.log_files["off"])))
            if "on_miss" in logs:
                st["semi_p_miss"] = startup.restore_report(logs["on_miss"], {})
            if "off_matched" in logs:
                st["vllm_matched"] = startup.parse_vllm([x for v in logs["off_matched"].values() for x in v])
            st["log_dir"] = str(log_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read start-up logs: %s", exc)
        if self.keep_jobs or attached:
            return
        for v, t in trackers.items():
            self.run_pre_cancel_hook(v, t.job_id, result)
            try:
                client.cancel_job(t.job_id)
                self.cancelled(t.job_id)
                logger.info("cancelled %s job %s", v, t.job_id)
            except Exception:  # noqa: BLE001
                logger.exception("failed to cancel %s -- cancel it by hand", t.job_id)

    def run_pre_cancel_hook(self, variant: str, job_id: str, result: dict) -> None:
        """Run ``pre_cancel_hook`` for one job, while its pods still exist (for
        example to delete what on_miss dumped to local disk). A failure is
        logged and the job is cancelled anyway."""
        if not self.pre_cancel_hook:
            return
        miss_cache = (f"{self.miss_image_cache.rstrip('/')}/{report.stem(result)}"
                      if self.miss_image_cache and variant == "on_miss" else "")
        command = self.pre_cancel_hook.format(job_id=job_id, variant=variant, miss_image_cache=miss_cache)
        try:
            done = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=900)
            logger.info("pre-cancel hook for %s %s: exit %d %s", variant, job_id, done.returncode,
                        (done.stdout + done.stderr).strip()[-300:])
        except Exception:  # noqa: BLE001
            logger.exception("pre-cancel hook for %s %s failed", variant, job_id)

    def run_many(self, specs: list[ModelSpec]) -> list[dict]:
        max_gpus = self.max_gpus
        if max_gpus is None:
            max_gpus = int(self.client.get_capacity().get("available_gpus", 0)) - self.reserve_gpus
        max_nodes = self.max_nodes or max_gpus // self.gpus_per_node + len(specs)
        queue = sorted(specs, key=lambda s: self.footprint(s)[0], reverse=True)
        too_big = [f"{s.name} ({self.footprint(s)[0]} GPUs, {self.footprint(s)[1]} nodes)" for s in queue
                   if self.footprint(s)[0] > max_gpus or self.footprint(s)[1] > max_nodes]
        if too_big:
            raise SystemExit(f"max_gpus={max_gpus} / max_nodes={max_nodes} cannot fit: {', '.join(too_big)}")
        logger.info("budget: %d GPUs, %d nodes, %d GPUs left free for others", max_gpus, max_nodes, self.reserve_gpus)
        results: list[dict] = []
        running: dict[threading.Thread, tuple[int, int]] = {}
        lock = threading.Lock()

        def work(spec: ModelSpec) -> None:
            r = self.run_one(spec, capacity_checked=True)
            with lock:
                results.append(r)

        while queue or running:
            for t in [t for t in running if not t.is_alive()]:
                running.pop(t)
            gpus_used = sum(g for g, _ in running.values())
            nodes_used = sum(n for _, n in running.values())
            launched = False
            for spec in list(queue):
                gpus, nodes = self.footprint(spec)
                if gpus_used + gpus > max_gpus or nodes_used + nodes > max_nodes:
                    continue
                free = int(self.client.get_capacity().get("available_gpus", 0))
                if free - gpus < self.reserve_gpus:
                    logger.info("waiting for %d GPUs for %s plus %d in reserve (%d free)", gpus, spec.name,
                                self.reserve_gpus, free)
                    break
                queue.remove(spec)
                t = threading.Thread(target=work, args=(spec,), name=spec.name, daemon=False)
                running[t] = (gpus, nodes)
                gpus_used += gpus
                nodes_used += nodes
                t.start()
                launched = True
                time.sleep(5)  # let the submission show up in capacity before the next check
            if not launched:
                time.sleep(30)
        return results


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(levelname)s %(message)s")
    for noisy in ("httpx", "urllib3", "botocore", "boto3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_kv_args(argv)
    runner = Runner(args)
    specs = [load_model(m) for m in (args.get("models") or "").split(",") if m]
    if not specs:
        raise SystemExit("models=NAME[,NAME...] is required")
    if runner.attach and len(specs) != 1:
        raise SystemExit("attaching to running jobs works for one model only")
    for spec in specs:
        try:
            logger.info("%s: backend image %s", spec.name, resolve_image_tag(spec, runner.image_tag_override))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    started = utc_now()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, runner.abort)
    logger.info("job ledger: %s", runner.ledger)
    results = [runner.run_one(specs[0])] if len(specs) == 1 else runner.run_many(specs)
    summary = report.write_summary(results, runner.out_dir, started)
    for r in results:
        logger.info("%-28s %s", r["model"]["name"], report.verdict(r)[0])
    logger.info("summary: %s", summary)
    logger.info("onboarding book: %s", book.write(runner.out_dir))
    return 0 if all(report.verdict(r)[0] != "FAIL" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
