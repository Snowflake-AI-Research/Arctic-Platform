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

"""Model configs from the model catalog, the semi_p / plain job bodies built from them, and job status tracking."""

from __future__ import annotations

import functools
import json
import logging
import re
import subprocess
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

# Where catalog entries are read from when no local checkout is given.
CATALOG_REPO = "snowflakedb/cortex-training"
CATALOG_BRANCH = "main"
CATALOG_DIR = "model-catalog/models"

logger = logging.getLogger(__name__)

# At tensor_parallel_size >= 2 the semi_p engine moves vLLM's collectives onto
# paths a checkpoint can carry: no fused allreduce+RMSNorm, NVLS, symmetric-memory
# or FlashInfer allreduce, and no MoE shared-experts side stream. Each changes the
# reduction order, so a default plain job differs from semi_p in the last bits of
# the logits -- enough to flip near-tie greedy tokens. The "matched" baseline
# gives the plain job the same settings so the comparison is exact.
MATCHED_TP_ENV = {
    "NCCL_NVLS_ENABLE": "0",
    "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
    "VLLM_ALLREDUCE_USE_FLASHINFER": "0",
}
MATCHED_PASS_CONFIG = {"fuse_allreduce_rms": False}
# Asks the backend to apply the rest of semi_p's TP>1 setup to a plain engine:
# the semi_p worker class, which refuses the FlashInfer all-reduce workspace and
# forces CustomAllreduce's copy path during graph capture, and sleep mode. Those
# cannot be expressed as engine args or environment variables, and a backend
# that predates the flag ignores it (the report shows whether it took effect).
MATCH_FLAG_ENV = {"SEMIP_MATCH_COLLECTIVES": "1"}
MATCH_APPLIED_MARKER = "SEMIP_MATCH_COLLECTIVES=1: plain engine matched"
# An empty image source turns off published images, so a semi_p job misses; the
# flag lets the miss cold-start and dump. With the mirror off the engine never
# publishes, so a forced miss cannot add a second skeleton for a published key.
MISS_ENV = {"SEMIP_IMAGE_SOURCE": "", "SEMIP_DUMP_PUBLISH": "1"}
# The bucket a semi_p engine publishes to (QA6's when unset). The device-manager
# pod's environment does not carry the cluster's, so a job elsewhere sets it.
MODEL_BUCKET_ENV = "MODEL_BUCKET"

VARIANTS = ("on_hit", "off_default", "on_miss", "off_matched")
PERF_VARIANTS = ("on_hit", "off_default", "off_matched")
VARIANT_LABELS = {"on_hit": "semi_p, published image", "off_default": "plain, default",
                  "on_miss": "semi_p, forced miss", "off_matched": "plain, matched"}

TERMINAL = ("failed", "cancelled", "canceled", "terminated", "done")

# A published semi_p image is keyed by what one pod holds, so a job with twice
# the GPUs of a whole-node config restores the same image on two pods. Configs
# smaller than a node would put the extra replicas in the same pod (a different
# shape, so a cold start), and multi-node configs already span pods.
GPUS_PER_NODE = 8


@dataclass
class ModelSpec:
    name: str
    sub_job: dict
    thresholds: dict = field(default_factory=dict)
    perf: dict = field(default_factory=dict)
    correctness: dict = field(default_factory=dict)
    copies: int | None = None
    path: str = ""

    @property
    def inference_config(self) -> dict:
        return self.sub_job["inference_config"]

    @property
    def vllm_config(self) -> dict:
        return self.inference_config.get("vllm_config") or {}

    @property
    def n_gpus(self) -> int:
        return int(self.inference_config["n_gpus"])

    @property
    def tensor_parallel_size(self) -> int:
        return int(self.vllm_config.get("tensor_parallel_size") or 1)

    @property
    def replicas(self) -> int:
        per_replica = self.tensor_parallel_size * int(self.vllm_config.get("pipeline_parallel_size") or 1)
        return max(1, self.n_gpus // per_replica)

    @property
    def max_seq_len(self) -> int:
        return int(self.inference_config["max_seq_len"])


def catalog_name(rel: str) -> str:
    """A model's name from its catalog path: glm-5.3/standard/h200-inference-full -> glm-5.3-standard-full."""
    parts = rel.removesuffix(".json").strip("/").split("/")
    parts[-1] = parts[-1].removeprefix("h200-inference-")
    return "-".join(parts)


@functools.lru_cache(maxsize=None)
def _fetch_catalog_entry(rel: str) -> str:
    """A catalog entry from CATALOG_REPO at CATALOG_BRANCH, through the gh CLI and its login."""
    api = f"repos/{CATALOG_REPO}/contents/{CATALOG_DIR}/{rel}?ref={CATALOG_BRANCH}"
    try:
        out = subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github.raw", api],
                             capture_output=True, text=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", "") or exc
        raise ValueError(f"could not read {CATALOG_REPO}:{CATALOG_DIR}/{rel} at {CATALOG_BRANCH} ({detail}); "
                         "pass catalog=PATH to a local checkout of the catalog's models/ directory") from exc
    return out.stdout


def load_model(ref: str, catalog: str | None = None) -> ModelSpec:
    """The job config ``ref`` names.

    ``ref`` is a path to a JSON file, or a catalog entry's path under the model
    catalog's ``models/`` directory (``glm-5.3/standard/h200-inference-full``),
    read from ``catalog``, a local checkout of that directory, or else from
    CATALOG_REPO at CATALOG_BRANCH. A catalog entry holds its one sampling
    sub-job under ``sub_job_configs``; a model file holds it under ``sub_job``
    and may add ``name``, ``thresholds``, ``perf``, ``correctness`` and
    ``copies``.
    """
    path = Path(ref).expanduser()
    if path.exists():
        rel, source, text = "/".join(path.parts[-3:]), str(path), path.read_text(encoding="utf-8")
    else:
        rel = ref if ref.endswith(".json") else f"{ref}.json"
        if catalog:
            local = Path(catalog).expanduser() / rel
            source, text = str(local), local.read_text(encoding="utf-8")
        else:
            source, text = f"{CATALOG_REPO}@{CATALOG_BRANCH}:{CATALOG_DIR}/{rel}", _fetch_catalog_entry(rel)
    raw = json.loads(text)
    if "sub_job_configs" in raw:
        subs = raw["sub_job_configs"]
        if len(subs) != 1:
            raise ValueError(f"{source}: expected one sub-job in sub_job_configs, found {len(subs)}")
        raw = {"name": catalog_name(rel), "sub_job": subs[0]}
    return _model_file(raw, source, Path(source).stem)


def _model_file(raw: dict, source: str, default_name: str) -> ModelSpec:
    """A model file's ``ModelSpec``, its sub-job checked."""
    sub_job = raw["sub_job"]
    if sub_job.get("job_type") != "sampling":
        raise ValueError(f"{source}: sub_job.job_type must be 'sampling'")
    if "semi_p" in sub_job.get("inference_config", {}):
        raise ValueError(f"{source}: leave inference_config.semi_p out; the suite sets it per job")
    return ModelSpec(
        name=raw.get("name", default_name),
        sub_job=sub_job,
        thresholds=raw.get("thresholds", {}),
        perf=raw.get("perf", {}),
        correctness=raw.get("correctness", {}),
        copies=raw.get("copies"),
        path=source,
    )


JOB_BODY_KEYS = ("hardware", "sub_job_configs")


def seq_label(max_seq_len: int) -> str:
    """``max_seq_len`` for a label: 32768 -> 32k, 720896 -> 704k, else the number."""
    return f"{max_seq_len // 1024}k" if max_seq_len % 1024 == 0 else str(max_seq_len)


def job_label(sub_job: dict) -> str:
    """A semi_p job's label from its sub-job: model basename, context, full or
    lora (``peft_config`` set), e.g. ``qwen3-0.6b-32k-full``."""
    ic = sub_job["inference_config"]
    base = re.sub(r"[^a-z0-9._-]+", "-", str(sub_job["model_name"]).rstrip("/").rsplit("/", 1)[-1].lower()).strip("-")
    return f"{base or 'model'}-{seq_label(int(ic['max_seq_len']))}-{'lora' if ic.get('peft_config') else 'full'}"


def _job_body_line(raw: Any, source: str) -> dict:
    """The sampling sub-job of one ``.jsonl`` line, a job body as submitted for a
    semi_p job, with ``semi_p`` taken out (the suite sets it per job)."""
    if not isinstance(raw, dict) or not isinstance(raw.get("sub_job_configs"), list):
        raise ValueError(f"{source}: expected a semi_p job body, "
                         '{"hardware": ..., "sub_job_configs": [one sampling sub-job]}')
    extra = sorted(set(raw) - set(JOB_BODY_KEYS))
    if extra:
        raise ValueError(f"{source}: unexpected field(s) {', '.join(extra)}; a line holds only "
                         f"{' and '.join(JOB_BODY_KEYS)}")
    subs = raw["sub_job_configs"]
    if len(subs) != 1:
        raise ValueError(f"{source}: expected one sub-job in sub_job_configs, found {len(subs)}")
    sub = deepcopy(subs[0])
    if not isinstance(sub, dict) or sub.get("job_type") != "sampling":
        raise ValueError(f"{source}: sub_job_configs[0].job_type must be 'sampling'")
    ic = sub.get("inference_config")
    if not isinstance(ic, dict):
        raise ValueError(f"{source}: sub_job_configs[0] has no inference_config")
    if ic.get("semi_p") is not True:
        found = "missing" if "semi_p" not in ic else f"{ic['semi_p']!r}"
        raise ValueError(f"{source}: inference_config.semi_p must be true (the line is the semi_p job "
                         f"as submitted), found {found}")
    del ic["semi_p"]
    return sub


def load_models(ref: str, catalog: str | None = None) -> list[ModelSpec]:
    """The job configs ``ref`` names: for a ``.jsonl`` file, one per non-blank
    line, each a semi_p job body exactly as a user submits it (a catalog entry's
    ``{"hardware", "sub_job_configs": [one sampling sub-job]}`` with
    ``inference_config.semi_p: true``), labelled by ``job_label`` with -2, -3, ...
    appended in file order to a label already taken; else
    ``[load_model(ref, catalog)]``."""
    path = Path(ref).expanduser()
    if not (ref.endswith(".jsonl") and path.is_file()):
        return [load_model(ref, catalog)]
    specs: dict[str, ModelSpec] = {}
    bodies: dict[str, int] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        source = f"{path}:{n}"
        try:
            raw = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{source}: not JSON ({exc})") from exc
        sub = _job_body_line(raw, source)
        canonical = json.dumps(raw, sort_keys=True)
        if canonical in bodies:
            raise ValueError(f"{source}: the same job as line {bodies[canonical]}")
        bodies[canonical] = n
        try:
            label = job_label(sub)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{source}: cannot label the job (needs model_name and "
                             f"inference_config.max_seq_len): {exc!r}") from exc
        name, k = label, 1
        while name in specs:
            k += 1
            name = f"{label}-{k}"
        specs[name] = _model_file({"name": name, "sub_job": sub}, source, name)
    if not specs:
        raise ValueError(f"{path}: no job configs")
    return list(specs.values())


def copies_for(spec: ModelSpec, override: int | None = None, gpus_per_node: int = GPUS_PER_NODE) -> int:
    """How many copies of the published layout the test jobs run: the override,
    else the model file's ``copies``, else 2 for a whole-node config and 1 otherwise."""
    if override is not None:
        return int(override)
    if spec.copies is not None:
        return int(spec.copies)
    return 2 if spec.n_gpus == gpus_per_node else 1


def pod_spanning(spec: ModelSpec, gpus_per_node: int = GPUS_PER_NODE) -> bool:
    """One engine spans pods (its TP x PP group is larger than a node)."""
    return spec.n_gpus // spec.replicas > gpus_per_node


def plan_variants(spec: ModelSpec, wanted: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
    """The jobs to run and, for any dropped, why. Any set that includes the
    reference ``on_hit`` runs. ``off_matched`` equals ``off_default`` below TP 2,
    and cannot be matched for a pod-spanning engine."""
    wanted = list(wanted or VARIANTS)
    unknown = [v for v in wanted if v not in VARIANTS]
    if unknown or "on_hit" not in wanted:
        raise ValueError(f"variants must include on_hit and come from {VARIANTS}, got {wanted}")
    skipped = {}
    if "off_matched" in wanted:
        if spec.tensor_parallel_size < 2:
            skipped["off_matched"] = "TP=1: semi_p changes no collective setting, so it equals off_default"
            if "off_default" not in wanted:
                skipped["off_matched"] += " (add off_default for a plain baseline)"
        elif pod_spanning(spec):
            skipped["off_matched"] = "pod-spanning engine: its NCCL network pinning cannot be matched on plain workers"
    return [v for v in VARIANTS if v in wanted and v not in skipped], skipped


def plain_baseline(variants: Any) -> str | None:
    """The plain job semi_p's perf and start-up are compared with: off_default,
    else off_matched, else None (no plain job in the run)."""
    return next((v for v in ("off_default", "off_matched") if v in variants), None)


def build_variant_bodies(spec: ModelSpec, image_tag: str | None, variants: list[str], copies: int = 1,
                         miss_image_cache: str | None = None,
                         model_bucket: str | None = None) -> dict[str, dict]:
    """A create-job body per variant. The perf jobs (on_hit, off_default and
    off_matched, which splits the perf gap into settings and semi_p) run
    ``copies`` x the published layout; on_miss, correctness only, runs it as
    published. Every body is the model file's sub-job plus only what its
    variant needs, and ``model_bucket``, if given, as every job's MODEL_BUCKET."""
    bodies = {}
    for variant in variants:
        sub = deepcopy(spec.sub_job)
        ic = sub["inference_config"]
        ic["semi_p"] = variant.startswith("on_")
        ic["n_gpus"] = spec.n_gpus * (copies if variant in PERF_VARIANTS else 1)
        vc = ic.setdefault("vllm_config", {})
        if variant == "on_miss":
            extra = dict(MISS_ENV)
            if miss_image_cache:
                extra["SEMIP_IMAGE_CACHE"] = miss_image_cache
            vc["extra_env"] = {**(vc.get("extra_env") or {}), **extra}
        elif variant == "off_matched":
            vc["extra_env"] = {**(vc.get("extra_env") or {}), **MATCHED_TP_ENV, **MATCH_FLAG_ENV}
            engine = vc.setdefault("extra_engine_kwargs", {})
            compilation = engine.setdefault("compilation_config", {})
            compilation["pass_config"] = {**(compilation.get("pass_config") or {}), **MATCHED_PASS_CONFIG}
        if model_bucket:
            vc["extra_env"] = {**(vc.get("extra_env") or {}), MODEL_BUCKET_ENV: model_bucket}
        body: dict[str, Any] = {"sub_job_configs": [sub]}
        if image_tag:
            body["debug"] = {"job": {"image_tag": image_tag}}
        bodies[variant] = body
    return bodies


def normalize_status(status: Any) -> str:
    text = str(status or "").strip().lower()
    return text[len("job_state_"):] if text.startswith("job_state_") else text


@dataclass
class JobTracker:
    """A job's id, the wall-clock time each status was first seen, and when the suite cancelled it."""

    label: str
    job_id: str
    submitted_at: float
    transitions: dict = field(default_factory=dict)
    reason: str = ""
    cancelled_at: float | None = None

    @property
    def status(self) -> str:
        return next(reversed(self.transitions), "") if self.transitions else ""

    def poll(self, client: Any) -> str:
        job = client.get_job(self.job_id)
        status = normalize_status(job.get("status"))
        self.transitions.setdefault(status, time.time())
        self.reason = job.get("reason") or self.reason
        return status


def wait_until_running(client: Any, tracker: JobTracker, deadline: float, interval_s: float = 5.0,
                       stop: threading.Event | None = None) -> bool:
    """Poll one tracker until it is running (True) or ``stop`` is set (False);
    raise if it ends or is not running by ``deadline`` (a ``time.time()``)."""
    stop = stop or threading.Event()
    while True:
        status = tracker.poll(client)
        if status == "running":
            logger.info("%s job %s is running", tracker.label, tracker.job_id)
            return True
        if status in TERMINAL:
            raise RuntimeError(f"{tracker.label} job {tracker.job_id} ended {status}: {tracker.reason}")
        if time.time() > deadline:
            raise TimeoutError(f"{tracker.label} job {tracker.job_id} not running "
                               f"{time.time() - tracker.submitted_at:.0f}s after submission ({status})")
        if stop.wait(interval_s):
            return False


def timeline(tracker: JobTracker) -> dict:
    """Seconds from submission to each status, the placement/start-up split,
    and how long the job ran before the suite cancelled it."""
    out = {status: round(t - tracker.submitted_at, 1) for status, t in tracker.transitions.items()}
    running = tracker.transitions.get("running")
    initializing = tracker.transitions.get("initializing")
    if running is not None:
        out["submit_to_running_s"] = round(running - tracker.submitted_at, 1)
        if initializing is not None:
            out["initializing_to_running_s"] = round(running - initializing, 1)
        if tracker.cancelled_at is not None:
            out["running_to_cancelled_s"] = round(tracker.cancelled_at - running, 1)
    return out
