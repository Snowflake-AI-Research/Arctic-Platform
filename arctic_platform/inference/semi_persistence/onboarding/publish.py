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

"""Dump and publish semi_p images for a backend image, one job per model config.

    python -m arctic_platform.inference.semi_persistence.onboarding.publish \\
        config=CONN.json image_tag=TAG models=a,b,c [max_gpus=40] [reserve_gpus=8] \\
        [out_dir=onboarding_reports/publish] [timeout_s=7200]

For each model file under onboarding/models/, at its published size:

  1. submit the sampling job with semi_p on, SEMIP_DUMP_PUBLISH=1, pinned to
     ``image_tag``;
  2. wait for ``running``; a job that restored from a published image needs
     nothing (the image is already published);
  3. otherwise the engine cold-started, dumped, restored and is publishing in
     the background: wait for its ``semi_p: published`` line in the sub-job log;
  4. cancel the job.

Jobs start as GPUs allow within ``max_gpus``, always leaving ``reserve_gpus``
free. Every job is appended to ``<out_dir>/<image_tag>/jobs.jsonl``; SIGINT,
SIGTERM and SIGHUP cancel every job still held. Results go to
``<out_dir>/<image_tag>/summary.json``. Afterwards, point each model's entry in
onboarding/models/backends.json at ``image_tag``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from arctic_platform.inference.semi_persistence.onboarding.client import enable_debug_options
from arctic_platform.inference.semi_persistence.onboarding.client import make_client
from arctic_platform.inference.semi_persistence.onboarding.common import parse_kv_args
from arctic_platform.inference.semi_persistence.onboarding.jobs import TERMINAL
from arctic_platform.inference.semi_persistence.onboarding.jobs import ModelSpec
from arctic_platform.inference.semi_persistence.onboarding.jobs import load_model
from arctic_platform.inference.semi_persistence.onboarding.jobs import normalize_status
from arctic_platform.inference.semi_persistence.onboarding.startup import parse_semi_p
from arctic_platform.inference.semi_persistence.onboarding.startup import read_sub_job_log

logger = logging.getLogger(__name__)

DUMP_PUBLISH_ENV = {"SEMIP_DUMP_PUBLISH": "1"}
PUBLISHED = re.compile(r"semi_p: published (\S+) in ([0-9.]+)s")
NOT_PUBLISHED = re.compile(r"semi_p: (?:publishing \S+ failed|publishing \S+ raised|"
                           r"dumped \S+ but not publishing it|not publishing \S+)")
SKELETON = re.compile(r"image-cache/([^/\s]+)/skeleton/([0-9a-f]{12}_[0-9a-f]{12}_[0-9a-f]{12})")
POLL_S = 30.0


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def publish_body(spec: ModelSpec, image_tag: str) -> dict:
    """The model's sub-job with semi_p on and a miss allowed to dump and publish."""
    sub = deepcopy(spec.sub_job)
    ic = sub["inference_config"]
    ic["semi_p"] = True
    vc = ic.setdefault("vllm_config", {})
    vc["extra_env"] = {**(vc.get("extra_env") or {}), **DUMP_PUBLISH_ENV}
    return {"sub_job_configs": [sub], "debug": {"job": {"image_tag": image_tag}}}


def publish_outcome(lines: list[str]) -> dict | None:
    """What the job's log says so far: ``None`` while it is still dumping or
    publishing, else ``{"status": ..., ...}``."""
    restore = parse_semi_p(lines)
    if restore["markers_visible"] and restore["restored"] and not restore["cache_miss"]:
        return {"status": "already published (restored from it); nothing to publish"}
    for line in lines:
        refused = NOT_PUBLISHED.search(line)
        if refused:
            return {"status": f"FAILED: {line[refused.start():][:400]}"}
    for i, line in enumerate(lines):
        done = PUBLISHED.search(line)
        if done:
            out: dict[str, Any] = {"status": "published", "key_dir": done.group(1),
                                   "publish_s": float(done.group(2))}
            skeleton = next((m for m in map(SKELETON.search, lines[i:]) if m), None)
            if skeleton:
                out.update(slug=skeleton.group(1), skeleton=skeleton.group(2),
                           weight_hash=skeleton.group(2)[-12:])
            return out
    return None


class Publisher:
    def __init__(self, args: dict[str, str]):
        for key in ("config", "image_tag", "models"):
            if not args.get(key):
                raise SystemExit(f"{key}= is required")
        self.image_tag = args["image_tag"]
        self.models = [m for m in args["models"].split(",") if m]
        self.max_gpus = int(args.get("max_gpus", "40"))
        self.reserve = int(args.get("reserve_gpus", "8"))
        self.timeout_s = float(args.get("timeout_s", "7200"))
        self.out = Path(args.get("out_dir", "onboarding_reports/publish")).expanduser() / self.image_tag
        self.out.mkdir(parents=True, exist_ok=True)
        enable_debug_options()
        self.client = make_client(args["config"])
        self.ledger = self.out / "jobs.jsonl"
        self.live: dict[str, str] = {}
        self.lock = threading.Lock()
        # A re-run of some models keeps the others' entries, which onboard reads.
        summary = self.out / "summary.json"
        self.results: dict[str, dict] = json.loads(summary.read_text()) if summary.exists() else {}

    def note(self, entry: dict) -> None:
        with self.lock, self.ledger.open("a") as f:
            f.write(json.dumps({**entry, "at": utc()}) + "\n")

    def cancel(self, job_id: str) -> None:
        try:
            self.client.cancel_job(job_id)
        except Exception:  # noqa: BLE001
            logger.exception("cancel %s failed -- cancel it by hand", job_id)
            return
        with self.lock:
            self.live.pop(job_id, None)
        self.note({"job_id": job_id, "cancelled": True})
        logger.info("cancelled %s", job_id)

    def abort(self, signum: int, _frame: Any) -> None:
        with self.lock:
            held = list(self.live)
        logger.error("signal %d: cancelling %d job(s)", signum, len(held))
        for job_id in held:
            self.cancel(job_id)
        os._exit(128 + signum)

    def wait_running(self, name: str, job_id: str, deadline: float) -> None:
        last = None
        while time.time() < deadline:
            job = self.client.get_job(job_id)
            status = normalize_status(job.get("status"))
            if status != last:
                logger.info("%s: %s", name, status)
                last = status
            if status == "running":
                return
            if status in TERMINAL:
                raise RuntimeError(f"job ended {status}: {job.get('reason')}")
            time.sleep(20)
        raise TimeoutError("not running in time")

    def wait_published(self, name: str, job_id: str, deadline: float) -> dict:
        while time.time() < deadline:
            lines = read_sub_job_log(self.client, job_id)
            outcome = publish_outcome(lines)
            if outcome:
                return outcome
            status = normalize_status(self.client.get_job(job_id).get("status"))
            if status in TERMINAL:
                raise RuntimeError(f"job ended {status} before publishing")
            time.sleep(POLL_S)
        raise TimeoutError("publish did not finish in time")

    def prepare(self, name: str) -> dict:
        spec = load_model(name)
        res: dict[str, Any] = {"model": name, "model_name": spec.sub_job["model_name"],
                               "n_gpus": spec.n_gpus, "started_at": utc()}
        job_id = self.client.create_job_from_body(publish_body(spec, self.image_tag))["job_id"]
        res["job_id"] = job_id
        with self.lock:
            self.live[job_id] = name
        self.note({"job_id": job_id, "model": name})
        logger.info("%s: submitted %s", name, job_id)
        deadline = time.time() + self.timeout_s
        try:
            self.wait_running(name, job_id, deadline)
            res["running_at"] = utc()
            res.update(self.wait_published(name, job_id, deadline))
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s: %s", name, exc)
            res["status"] = f"FAILED: {type(exc).__name__}: {exc}"
        finally:
            try:
                (self.out / f"{name}.log").write_text("\n".join(read_sub_job_log(self.client, job_id)) + "\n")
            except Exception:  # noqa: BLE001
                logger.warning("%s: could not save the sub-job log", name, exc_info=True)
            self.cancel(job_id)
            res["finished_at"] = utc()
            logger.info("%s: %s", name, res["status"])
        return res

    def run(self) -> None:
        queue = sorted(self.models, key=lambda m: load_model(m).n_gpus, reverse=True)
        running: dict[threading.Thread, int] = {}

        def work(name: str) -> None:
            r = self.prepare(name)
            with self.lock:
                self.results[name] = r
                (self.out / "summary.json").write_text(json.dumps(self.results, indent=1))

        while queue or running:
            for t in [t for t in running if not t.is_alive()]:
                running.pop(t)
            used = sum(running.values())
            started = False
            for name in list(queue):
                need = load_model(name).n_gpus
                if used + need > self.max_gpus:
                    continue
                free = int(self.client.get_capacity().get("available_gpus", 0))
                if free - need < self.reserve:
                    logger.info("waiting: %s needs %d GPUs plus %d in reserve, %d free",
                                name, need, self.reserve, free)
                    break
                queue.remove(name)
                t = threading.Thread(target=work, args=(name,), name=name)
                running[t] = need
                used += need
                t.start()
                started = True
                time.sleep(5)
            if not started:
                time.sleep(30)
        logger.info("all done: %s", {k: v["status"] for k, v in self.results.items()})


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(levelname)s %(message)s")
    for noisy in ("httpx", "urllib3", "botocore", "boto3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    publisher = Publisher(parse_kv_args(argv))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, publisher.abort)
    logger.info("image %s, %d model(s), budget %d GPUs, reserve %d; ledger %s", publisher.image_tag,
                len(publisher.models), publisher.max_gpus, publisher.reserve, publisher.ledger)
    publisher.run()
    return 0 if all(r["status"].startswith(("published", "already")) for r in publisher.results.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
