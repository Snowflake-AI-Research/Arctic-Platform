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

"""Model files, the semi_p / plain job bodies built from them, and job status tracking."""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

# BACKENDS_FILE: the backend image each model's semi_p images are published on. A
# published image only restores on the backend it was dumped on, and the cluster
# default backend may not run semi_p at all (it would then serve plain without
# saying so).
MODELS_DIR = Path(__file__).resolve().parent / "models"
BACKENDS_FILE = MODELS_DIR / "backends.json"

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


def load_model(ref: str) -> ModelSpec:
    """Load ``models/<ref>.json``, or ``ref`` itself when it is a path."""
    path = Path(ref).expanduser()
    if not path.exists():
        path = MODELS_DIR / (ref if ref.endswith(".json") else f"{ref}.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    sub_job = raw["sub_job"]
    if sub_job.get("job_type") != "sampling":
        raise ValueError(f"{path}: sub_job.job_type must be 'sampling'")
    if "semi_p" in sub_job.get("inference_config", {}):
        raise ValueError(f"{path}: leave inference_config.semi_p out; the suite sets it per job")
    return ModelSpec(
        name=raw.get("name", path.stem),
        sub_job=sub_job,
        thresholds=raw.get("thresholds", {}),
        perf=raw.get("perf", {}),
        correctness=raw.get("correctness", {}),
        copies=raw.get("copies"),
        path=str(path),
    )


def resolve_image_tag(spec: ModelSpec, override: str | None = None, backends_file: Path | str = BACKENDS_FILE) -> str:
    """The backend image for ``spec``: ``override`` when given, else the model's
    entry in ``backends_file``. Raises when the model has no entry."""
    if override:
        return override
    path = Path(backends_file).expanduser()
    backends = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    tag = (backends.get(spec.name) or {}).get("image_tag")
    if not tag:
        raise ValueError(f"{spec.name}: no image_tag in {path}; add one or pass debug_image_tag=")
    return tag


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
    """The jobs to run and, for any dropped, why. ``off_matched`` equals
    ``off_default`` below TP 2, and cannot be matched for a pod-spanning engine."""
    wanted = list(wanted or VARIANTS)
    unknown = [v for v in wanted if v not in VARIANTS]
    if unknown or "on_hit" not in wanted or "off_default" not in wanted:
        raise ValueError(f"variants must include on_hit and off_default and come from {VARIANTS}, got {wanted}")
    skipped = {}
    if "off_matched" in wanted:
        if spec.tensor_parallel_size < 2:
            skipped["off_matched"] = "TP=1: semi_p changes no collective setting, so it equals off_default"
        elif pod_spanning(spec):
            skipped["off_matched"] = "pod-spanning engine: its NCCL network pinning cannot be matched on plain workers"
    return [v for v in VARIANTS if v in wanted and v not in skipped], skipped


def build_variant_bodies(spec: ModelSpec, image_tag: str | None, variants: list[str], copies: int = 1,
                         miss_image_cache: str | None = None) -> dict[str, dict]:
    """A create-job body per variant. The perf jobs (on_hit, off_default and
    off_matched, which splits the perf gap into settings and semi_p) run
    ``copies`` x the published layout; on_miss, correctness only, runs it as
    published. Every body is the model file's sub-job plus only what its
    variant needs."""
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
    """A job's id and the wall-clock time each status was first seen."""

    label: str
    job_id: str
    submitted_at: float
    transitions: dict = field(default_factory=dict)
    reason: str = ""

    @property
    def status(self) -> str:
        return next(reversed(self.transitions), "") if self.transitions else ""

    def poll(self, client: Any) -> str:
        job = client.get_job(self.job_id)
        status = normalize_status(job.get("status"))
        self.transitions.setdefault(status, time.time())
        self.reason = job.get("reason") or self.reason
        return status


def wait_running(client: Any, trackers: list[JobTracker], timeout_s: float, interval_s: float = 5.0) -> None:
    """Poll every tracker until all are running; raise if one fails or time runs out."""
    deadline = time.time() + timeout_s
    pending = list(trackers)
    while pending:
        for tracker in list(pending):
            status = tracker.poll(client)
            if status == "running":
                logger.info("%s job %s is running", tracker.label, tracker.job_id)
                pending.remove(tracker)
            elif status in TERMINAL:
                raise RuntimeError(f"{tracker.label} job {tracker.job_id} ended {status}: {tracker.reason}")
        if pending:
            if time.time() > deadline:
                names = ", ".join(f"{t.label}={t.status}" for t in pending)
                raise TimeoutError(f"jobs not running after {timeout_s:.0f}s ({names})")
            time.sleep(interval_s)


def timeline(tracker: JobTracker) -> dict:
    """Seconds from submission to each status, and the placement/start-up split."""
    out = {status: round(t - tracker.submitted_at, 1) for status, t in tracker.transitions.items()}
    running = tracker.transitions.get("running")
    initializing = tracker.transitions.get("initializing")
    if running is not None:
        out["submit_to_running_s"] = round(running - tracker.submitted_at, 1)
        if initializing is not None:
            out["initializing_to_running_s"] = round(running - initializing, 1)
    return out
