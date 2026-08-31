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
"""The Cortex control plane: what you do *about* a job rather than *inside* one.

`ArcticClient` owns a session it created. This owns nothing: every call takes a
job id, so it can inspect, wait on, or cancel a job that belongs to someone else.

`attach` is the bridge between the two. It reads a live job's sub-jobs back into
an `ArcticClientConfig` that reconnects to them, so data-plane tooling runs on the
real client instead of re-deriving job tokens and payload shapes.
"""

from __future__ import annotations

import builtins
import os
import time
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from arctic_platform.client.config import ArcticClientConfig
from arctic_platform.client.config import CortexConfig
from arctic_platform.client.transports.cortex import CortexSession
from arctic_platform.client.transports.cortex import _is_connect_error
from arctic_platform.client.transports.cortex import _next_delay
from arctic_platform.client.transports.cortex import _poll_progress
from arctic_platform.client.transports.cortex import _short

# Cortex sub-job job_type -> the client's role name.
_ROLE = {"training": "training", "sampling": "sampling", "log_probability": "log_prob"}

_TERMINAL = ("failed", "done", "cancelled", "canceled")

# A CreateJob `debug` block (e.g. pinning a job to a specific backend image) is an
# internal capability, gated client-side as well as on the account parameter.
DEBUG_OPTIONS_ENV = "CORTEX_ENABLE_DEBUG_OPTIONS"


def _debug_options_enabled() -> bool:
    return os.environ.get(DEBUG_OPTIONS_ENV, "").strip().lower() in ("1", "true", "yes", "on")


class Capacity(BaseModel):
    """The caller account's GPU reservation.

    Defaults matter: the server emits proto3 JSON, which omits zero/false fields, so
    an unreserved account's response is literally ``{}``.
    """

    model_config = ConfigDict(extra="ignore")

    has_reservation: bool = Field(False, description="Whether the account has a configured GPU reservation.")
    reserved_gpus: int = Field(0, description="Total GPUs reserved for the account.")
    in_use_gpus: int = Field(0, description="GPUs consumed by the account's running + placing jobs.")
    available_gpus: int = Field(0, description="reserved_gpus - in_use_gpus, floored at 0.")


class CortexJobs:
    """Job control plane for one Cortex connection."""

    def __init__(
        self,
        config: CortexConfig,
        *,
        request_timeout: float = 1800.0,
        poll_timeout: float = 1800.0,
        poll_interval: float = 0.5,
    ) -> None:
        self.config = config
        self.session = CortexSession(config, request_timeout=request_timeout)
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval

    @property
    def prefix(self) -> str:
        return self.session.prefix

    # ── jobs ─────────────────────────────────────────────────────────────────
    def submit(self, body: dict) -> dict:
        """Create a job from a raw SnowAPI CreateJob body."""
        sub_job_configs = body.get("sub_job_configs")
        if not isinstance(sub_job_configs, list) or not sub_job_configs:
            raise ValueError("CreateJob body needs a non-empty sub_job_configs list")
        if body.get("debug") and not _debug_options_enabled():
            raise ValueError(
                f"CreateJob debug options are internal-only; set {DEBUG_OPTIONS_ENV}=1 to send a `debug` block"
            )
        # Only retry when the request provably never landed, so a retry can't create
        # a second job.
        return self.session.send("POST", self.prefix, retry_on=_is_connect_error, json=body)

    def get(self, job_id: str) -> dict:
        return self.session.send("GET", f"{self.prefix}/{job_id}")

    def list(self, status: str | None = None) -> builtins.list[dict]:
        params = {"status": status} if status else None
        return self.session.send("GET", self.prefix, params=params).get("jobs", [])

    def cancel(self, job_id: str) -> None:
        self.session.send("POST", f"{self.prefix}/{job_id}:cancel")  # GS colon-action syntax

    def wait(self, job_id: str) -> dict:
        """Poll until the job is running. Raises on a terminal state or timeout."""
        deadline = time.monotonic() + self.poll_timeout
        delay = self.poll_interval
        while time.monotonic() < deadline:
            job = self.get(job_id)
            state = _short(_unwrap(job).get("status"))
            if state == "running":
                return job
            if state in _TERMINAL:
                reason = _unwrap(job).get("reason", "")
                raise RuntimeError(f"job {job_id} reached terminal state '{state}': {reason}")
            time.sleep(delay)
            delay = _next_delay(delay)
        raise TimeoutError(f"job {job_id} did not become running within {self.poll_timeout}s")

    def capacity(self) -> Capacity:
        return Capacity(**self.session.send("GET", f"{self.prefix}/capacity"))

    def checkpoints(self, job_id: str) -> builtins.list[dict]:
        return self.session.send("GET", f"{self.prefix}/{job_id}/checkpoints").get("checkpoints", [])

    def experiment_run(self, job_id: str) -> dict:
        """``{experiment_name, experiment_run_name}`` -- the handle for the job's log stage."""
        return self.session.send("GET", f"{self.prefix}/{job_id}/experiment-run")

    # ── checkpoint load ──────────────────────────────────────────────────────
    # Cortex's /load takes a checkpoint *id* from the job's checkpoint store, which is
    # a different operation from the client's `load_checkpoint` (an on-prem resume by
    # path, absent from the Cortex wire). It lives here rather than in the op
    # vocabulary because no other backend has it.
    def load(
        self,
        job_id: str,
        checkpoint_id: str,
        *,
        source_job_id: str | None = None,
        target_sub_job_id: str | None = None,
    ) -> str:
        """Load a checkpoint into a running job. Returns a request id to poll."""
        body: dict[str, Any] = {"checkpoint_id": checkpoint_id}
        if source_job_id is not None:
            body["source_job_id"] = source_job_id
        if target_sub_job_id is not None:
            body["target_sub_job_id"] = target_sub_job_id
        return str(self.session.send("POST", f"{self.prefix}/{job_id}/load", json=body)["request_id"])

    def poll_request(self, job_id: str, request_id: str) -> dict:
        """Poll an async request to completion, draining any chunked result."""
        deadline = time.monotonic() + self.poll_timeout
        delay = self.poll_interval
        chunks: list[bytes] = []
        cursor: str | None = None
        while time.monotonic() < deadline:
            url = f"{self.prefix}/{job_id}/requests/{request_id}"
            params = {"cursor": cursor} if cursor else None
            action, value = _poll_progress(self.session.send("GET", url, params=params), chunks, request_id)
            if action == "done":
                return value
            if action == "drain":
                cursor = value  # more chunks queued; re-poll without backing off
                continue
            time.sleep(delay)
            delay = _next_delay(delay)
        raise TimeoutError(f"cortex request {request_id} did not complete within {self.poll_timeout}s")

    # ── bridge to the data plane ─────────────────────────────────────────────
    def attach(self, job_id: str) -> ArcticClientConfig:
        """A config that reconnects `ArcticClient` to this job's existing sub-jobs.

        GPU counts come back from the server because the client uses them to decide
        which roles are live; the created-job fields (dtype, peft, ds_config) are not
        reconstructed, since nothing is created on a reconnect.

        The caller owns the resulting client but *not* the job: never `shutdown()` it,
        or the CLI would cancel a job it merely looked at.
        """
        job = _unwrap(self.get(job_id))
        sub_jobs = job.get("sub_jobs") or []
        if not sub_jobs:
            raise ValueError(f"job {job_id} reports no sub_jobs to attach to")

        settings: dict[str, Any] = {}
        model_name = ""
        max_seq_len = 0
        for sub in sub_jobs:
            role = _ROLE.get(_short(sub.get("job_type"), "job_type_"))
            if role is None:
                continue
            sub_config = sub.get("training_config") or sub.get("inference_config") or {}
            # A sub-job that exists is a live role; n_gpus only sizes it, so fall back
            # to 1 rather than 0, which would read as "role disabled".
            settings[f"{role}_gpus"] = int(sub_config.get("n_gpus") or 1)
            settings[f"{role}_job_id"] = str(sub["sub_job_id"])
            model_name = model_name or str(sub.get("model_name") or "")
            max_seq_len = max(max_seq_len, int(sub_config.get("max_seq_len") or 0))

        if not settings:
            raise ValueError(f"job {job_id} has no sub-job of a known type: {[s.get('job_type') for s in sub_jobs]}")

        if max_seq_len:
            settings["max_seq_len"] = max_seq_len
        return ArcticClientConfig(model_name=model_name or "unknown", backend=self.config, **settings)


def _unwrap(job: dict) -> dict:
    """GS answers either the job or ``{"job": {...}}``."""
    return job.get("job", job) if isinstance(job, dict) else job
