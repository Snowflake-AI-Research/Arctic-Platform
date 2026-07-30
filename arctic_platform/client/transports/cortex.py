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
"""Cortex (cortex-training) transport over the SnowAPI GS REST surface.

The transport *owns* the Cortex wire mechanics: it builds the SnowAPI session,
submits each op over HTTP, and polls the async request to completion. There is
no separately-held client object — `call` is an op->handler table, the only
place Cortex specifics live. Unlisted ops (fwd-no-grad, log-probs, save-weights)
raise NotImplementedError.

The types and codec needed to talk to Cortex live in the client package:
`JobType`/`SubJobConfig`/`TrainingConfig`/`InferenceConfig` (the CreateJob wire
schema) below, and the DSSST1 safetensors codec in `arctic_platform.client.wire`.
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Callable

import requests

from arctic_platform.client import wire
from arctic_platform.client.config import ArcticRLClientConfig
from arctic_platform.client.transport import JOB_TYPES
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transport import Request
from arctic_platform.client.transport import Transport

# Ask the server for a chunked DSSST1 (safetensors) response for tensor-bearing ops.
_CHUNKED_DSSST1 = {"response_options": {"format": "dssst1", "delivery": "chunked"}}


# ─── CreateJob wire schema ────────────────────────────────────────────────


class JobType(str, Enum):
    TRAINING = "training"
    SAMPLING = "sampling"
    LOG_PROBABILITY = "log_probability"


@dataclass
class TrainingConfig:
    optimizer: dict
    max_seq_len: int
    train_batch_size: int
    n_gpus: int
    gradient_clipping: float | None = None
    load_optimizer_states: bool | None = None
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.optimizer, dict) or not self.optimizer:
            raise ValueError("training.optimizer is required and must be a non-empty dict")
        if self.max_seq_len <= 0:
            raise ValueError("training.max_seq_len must be > 0")
        if self.train_batch_size <= 0:
            raise ValueError("training.train_batch_size must be > 0")
        if self.n_gpus <= 0:
            raise ValueError("training.n_gpus must be > 0")

    def to_wire(self) -> dict:
        out: dict = {
            "optimizer": self.optimizer,
            "max_seq_len": self.max_seq_len,
            "train_batch_size": self.train_batch_size,
            "n_gpus": self.n_gpus,
        }
        if self.gradient_clipping is not None:
            out["gradient_clipping"] = self.gradient_clipping
        if self.load_optimizer_states is not None:
            out["load_optimizer_states"] = self.load_optimizer_states
        for k, v in self.extra.items():
            out.setdefault(k, v)
        return out


@dataclass
class InferenceConfig:
    max_seq_len: int
    n_gpus: int
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.max_seq_len <= 0:
            raise ValueError("sampling.max_seq_len must be > 0")
        if self.n_gpus <= 0:
            raise ValueError("sampling.n_gpus must be > 0")

    def to_wire(self) -> dict:
        out: dict = {"max_seq_len": self.max_seq_len, "n_gpus": self.n_gpus}
        for k, v in self.extra.items():
            out.setdefault(k, v)
        return out


@dataclass
class SubJobConfig:
    job_type: JobType
    model_name: str
    training: TrainingConfig | None = None
    sampling: InferenceConfig | None = None
    dtype: str | None = None
    seed: int | None = None

    @classmethod
    def training_job(
        cls,
        model_name: str,
        *,
        optimizer: dict,
        max_seq_len: int,
        train_batch_size: int,
        n_gpus: int,
        extra_training: dict | None = None,
        dtype: str | None = None,
        seed: int | None = None,
    ) -> SubJobConfig:
        return cls(
            job_type=JobType.TRAINING,
            model_name=model_name,
            training=TrainingConfig(
                optimizer=optimizer,
                max_seq_len=max_seq_len,
                train_batch_size=train_batch_size,
                n_gpus=n_gpus,
                extra=dict(extra_training) if extra_training else {},
            ),
            dtype=dtype,
            seed=seed,
        )

    @classmethod
    def sampling_job(
        cls,
        model_name: str,
        *,
        max_seq_len: int,
        n_gpus: int,
        extra_sampling: dict | None = None,
        job_type: JobType = JobType.SAMPLING,
        dtype: str | None = None,
        seed: int | None = None,
    ) -> SubJobConfig:
        if job_type not in (JobType.SAMPLING, JobType.LOG_PROBABILITY):
            raise ValueError(f"sampling_job() only accepts SAMPLING or LOG_PROBABILITY, got {job_type!r}")
        return cls(
            job_type=job_type,
            model_name=model_name,
            sampling=InferenceConfig(
                max_seq_len=max_seq_len,
                n_gpus=n_gpus,
                extra=dict(extra_sampling) if extra_sampling else {},
            ),
            dtype=dtype,
            seed=seed,
        )

    def validate(self) -> None:
        if not self.model_name:
            raise ValueError("sub_job.model_name is required")
        if self.job_type == JobType.TRAINING:
            if self.training is None:
                raise ValueError("training sub-job requires a `training` block")
            self.training.validate()
        else:
            if self.sampling is None:
                raise ValueError(f"{self.job_type.value} sub-job requires a `sampling` block")
            self.sampling.validate()

    def to_wire(self) -> dict:
        out: dict = {"job_type": self.job_type.value, "model_name": self.model_name}
        if self.dtype is not None:
            out["dtype"] = self.dtype
        if self.seed is not None:
            out["seed"] = self.seed
        if self.training is not None:
            out["training_config"] = self.training.to_wire()
        if self.sampling is not None:
            out["inference_config"] = self.sampling.to_wire()
        return out


# ─── Result decoding ──────────────────────────────────────────────────────


def _load_torch():
    import torch

    return torch


def _restore_generate_result_lists(value: Any, torch_module: Any = None) -> Any:
    torch_module = _load_torch() if torch_module is None else torch_module
    if torch_module.is_tensor(value):
        return value.cpu().tolist()
    if isinstance(value, dict):
        return {key: _restore_generate_result_lists(item, torch_module) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_restore_generate_result_lists(item, torch_module) for item in value)
    return value


def _decode_result_payload(result: dict) -> dict | None:
    if not isinstance(result, dict) or result.get("wire_format") != wire.WIRE_FORMAT_VERSION:
        return None
    if result.get("encoding") != "base64":
        raise RuntimeError("DSSST1 result payload must use base64 encoding")
    raw = result.get("payload_b64")
    if not isinstance(raw, str) or not raw:
        raise RuntimeError("DSSST1 result payload missing payload_b64")
    decoded = wire.loads(base64.b64decode(raw))
    if not isinstance(decoded, dict):
        raise RuntimeError("DSSST1 result payload did not decode to a dict")
    return decoded


def _decode_result_chunk_event(event: dict) -> bytes | None:
    if not isinstance(event, dict) or event.get("type") != "result_chunk":
        return None
    raw = event.get("payload_b64")
    if not isinstance(raw, str) or not raw:
        raise RuntimeError("result_chunk event missing payload_b64")
    payload = base64.b64decode(raw)
    expected = event.get("payload_sha256")
    if expected is not None and hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError("result_chunk payload_sha256 mismatch")
    return payload


# ─── Transport ─────────────────────────────────────────────────────────────


class CortexTransport(Transport):
    _MAX_FWD_BWD_BYTES = 60 * 1024 * 1024
    _MAX_GENERATE_BYTES = 60 * 1024 * 1024

    poll_interval = 0.5
    poll_timeout = 1800.0
    poll_backoff_multiplier = 1.25
    poll_max_interval = 6.0

    def __init__(self, config: ArcticRLClientConfig) -> None:
        self.config = config
        self.jobs = JobHandles()
        self.job_id: str | None = None
        self.sub_jobs: dict[str, str] = {}
        self._generate_request_ids: set[str] = set()
        self._session = self._build_session()
        self._handlers: dict[str, Callable[[dict], dict]] = {
            "fwd-bwd": self._fwd_bwd,
            "fwd-no-grad": self._fwd_no_grad,
            "log-probs": self._log_probs,
            "step": self._step,
            "save-checkpoint": self._save_checkpoint,
            "generate": self._generate,
            "sync-weights": self._sync_weights,
            "reset-prefix-cache": self._reset_prefix_cache,
        }
        # Colocation lifecycle ops are no-ops on Cortex (sub-jobs live in
        # separate placements; there is no wake/sleep concept). We register
        # them so SkyRL's colocated code path stays call-shape identical
        # without branching on backend. See ArcticRLClient._colo.
        for op in (
            "wake-training",
            "sleep-training",
            "wake-inference",
            "sleep-inference",
            "wake-log-prob",
            "sleep-log-prob",
            "empty-training-cache",
            "weight-norm",
            "save-weights",
        ):
            self._handlers[op] = self._colo_noop

    def initialize(self) -> JobHandles:
        cfg = self.config
        if cfg.training_job_id is not None:  # reconnect
            self.job_id = str(cfg.training_job_id)
            self.jobs = JobHandles.from_config(cfg)
            self.sub_jobs = {jt: f"{self.job_id}:{jt}" for jt in ("training", "sampling", "log_probability")}
            return self.jobs
        self.job_id = self._create_job(self._build_sub_jobs())
        self._capture_sub_jobs(self._wait_for_job(self.job_id))
        for job_type in JOB_TYPES:
            if cfg.gpus_for(job_type) > 0:
                self.jobs.set(job_type, self.job_id)
        return self.jobs

    def call(self, request: Request) -> dict:
        handler = self._handlers.get(request.op)
        if handler is None:
            raise NotImplementedError(f"cortex has no {request.op}")
        return handler(request.body)

    def shutdown(self) -> None:
        if self.job_id is not None:
            try:
                self._send("POST", f"{self._prefix}/{self.job_id}:cancel")
            except Exception:
                pass

    # ── op handlers: canonical body -> SnowAPI call -> canonical dict ──────
    def _fwd_bwd(self, body: dict) -> dict:
        body = self._normalize_train_body(body)
        payload = wire.dumps(body, metadata=_CHUNKED_DSSST1)
        request_id = self._post_octet_request_chunks(
            path_suffix="forward-backward",
            operation="fwd-bwd",
            frame=payload,
            max_bytes=self._MAX_FWD_BWD_BYTES,
        )["request_id"]
        return self._shape_train_response(self._poll(request_id))

    def _fwd_no_grad(self, body: dict) -> dict:
        """Cortex-side forward-only pass; returns model_outputs (log-probs).

        SkyRL and verl both need this every training step. Symmetric to
        `_fwd_bwd`: same octet-chunked submit + poll, hitting
        ``/{job_id}/forward-no-grad``. Requires the Neutrino GS to expose the
        endpoint (see the tracking issue). Same envelope translation applies:
        callers that ship the verl-GRPO ``{batch, meta, processing}`` shape
        have it repackaged into ``{args, kwargs}`` for Cortex.
        """
        body = self._normalize_train_body(body)
        payload = wire.dumps(body, metadata=_CHUNKED_DSSST1)
        request_id = self._post_octet_request_chunks(
            path_suffix="forward-no-grad",
            operation="fwd-no-grad",
            frame=payload,
            max_bytes=self._MAX_FWD_BWD_BYTES,
        )["request_id"]
        return self._shape_train_response(self._poll(request_id))

    def _log_probs(self, body: dict) -> dict:
        """Cortex-side log-probs endpoint (JSON in / DSSST1-decoded out).

        Symmetric to ``_generate`` in framing but returns a log-probs dict
        rather than sampled sequences. Requires the Neutrino GS endpoint at
        ``/{job_id}/log-probs``.
        """
        payload: dict = {"prompts": body["prompts"]}
        if body.get("completions") is not None:
            payload["completions"] = body["completions"]
        if body.get("top_k") is not None:
            payload["top_k"] = body["top_k"]
        request_id = self._send(
            "POST",
            f"{self._prefix}/{self.job_id}/log-probs",
            json=payload,
        ).json()["request_id"]
        return self._poll(request_id)

    def _step(self, body: dict) -> dict:
        req_body = {} if body["learning_rate"] is None else {"learning_rate": body["learning_rate"]}
        request_id = self._send("POST", f"{self._prefix}/{self.job_id}/step", json=req_body).json()["request_id"]
        return self._shape_train_response(self._poll(request_id))

    def _save_checkpoint(self, body: dict) -> dict:
        request_id = self._send(
            "POST",
            f"{self._prefix}/{self.job_id}/save",
            json={"checkpoint_type": "resumable"},
        ).json()["request_id"]
        return self._poll(request_id)

    def _generate(self, body: dict) -> dict:
        payload: dict = {"prompts": body["prompts"]}
        if body["sampling_params"] is not None:
            payload["sampling_params"] = body["sampling_params"]
        if body["routing_key"] is not None:
            payload["routing_key"] = body["routing_key"]
        frame = wire.dumps(payload, metadata=_CHUNKED_DSSST1)
        request_id = self._post_octet_request_chunks(
            path_suffix="generate",
            operation="generate",
            frame=frame,
            max_bytes=self._MAX_GENERATE_BYTES,
        )["request_id"]
        self._generate_request_ids.add(request_id)
        return {"results": self._poll(request_id).get("results", [])}

    def _sync_weights(self, body: dict) -> dict:
        # cuda_ipc / low_memory flags are on-prem colocation hints; Cortex has
        # separate sub-jobs and does a server-driven pull regardless. Drop
        # them silently rather than raising so SkyRL's colocated call site
        # (`sync_weights(cuda_ipc=True)`) works unchanged.
        source = self.sub_jobs["training"]
        request_id = self._operation(
            "weight-sync",
            payload={"source_sub_job_id": source, "target_sub_job_ids": [self.sub_jobs["sampling"]]},
            sub_job_id=source,
            sub_job_type="training",
        )["request_id"]
        return self._poll(request_id)

    def _colo_noop(self, body: dict) -> dict:
        """No-op handler for colocation-lifecycle ops.

        SkyRL calls wake/sleep/empty_training_cache/weight_norm unconditionally
        under `colocate=True`. Cortex has no colocation lifecycle (training
        and sampling live in separate sub-jobs), so these are no-ops. Returning
        `{}` matches on-prem's post-hoc metrics-less responses closely enough
        that downstream `.get(...)` reads don't blow up.
        """
        return {}

    def _normalize_train_body(self, body: dict) -> dict:
        """Translate the verl-GRPO envelope into Cortex's RPC-style body.

        The unified frontend forwards ``batch`` verbatim, but SkyRL and verl
        build ``{batch: {input_ids, labels, ...}, meta: {...}, processing: {...}}``
        while Cortex expects ``{args: [], kwargs: {input_ids, labels, ...}}``.
        We detect the verl-GRPO shape (``"batch"`` key holding a dict) and
        repack it into Cortex's shape. Bodies already in Cortex shape pass
        through unchanged.

        Extra sibling keys (``meta``, ``processing``, ``router_replay``,
        ``context``, ``post_processors``, ``reference_model``, …) are copied
        alongside so the Neutrino trainer — whose proto is
        ``additionalProperties: true`` — can route them without a schema
        change. ``reference_model`` in particular is how verl toggles between
        the actor and reference forward-only pass.
        """
        if not isinstance(body, dict):
            return body
        batch = body.get("batch")
        if not isinstance(batch, dict):
            return body
        kwargs = dict(batch)
        out: dict = {"args": [], "kwargs": kwargs}
        for key in (
            "meta",
            "processing",
            "router_replay",
            "context",
            "post_processors",
            "reference_model",
        ):
            if body.get(key) is not None:
                out[key] = body[key]
        return out

    def _shape_train_response(self, result: dict) -> dict:
        """Coerce a Cortex train-op response into the shape callers expect.

        Cortex today returns loss-only for fwd_bwd; SkyRL reads
        ``result["grad_norm"]`` and verl reads ``result["avg_loss"]`` /
        ``result["post_process_outputs"]``. We surface the loss under
        ``avg_loss`` (and mirror it as ``loss`` if the server used a different
        key) so the integrations Just Work. Anything the server does return
        (``model_outputs``, extra scalars) passes through untouched.

        For fwd_no_grad, on-prem returns ``{"batch": {"logprobs": ...,
        "entropy": ...}, ...}`` while Cortex packages the same fields under
        ``model_outputs``. verl's adapter reads ``response["batch"]["log_probs"]``
        after renaming; we alias ``model_outputs`` -> ``batch`` here so the
        response schema is uniform across transports without touching the
        integration.
        """
        if not isinstance(result, dict):
            return result
        out = dict(result)
        if "avg_loss" not in out and "loss" in out:
            out["avg_loss"] = out["loss"]
        elif "loss" not in out and "avg_loss" in out:
            out["loss"] = out["avg_loss"]
        # Alias model_outputs -> batch so on-prem's response shape is the
        # canonical one, regardless of which server produced the result.
        if "batch" not in out and isinstance(out.get("model_outputs"), dict):
            out["batch"] = dict(out["model_outputs"])
        # Ensure the two dicts SkyRL/verl reach into always exist.
        out.setdefault("metrics", {})
        out.setdefault("post_process_outputs", {})
        # ``grad_norm`` is not returned by Cortex today. Surface a None so
        # ``.get("grad_norm")`` returns None rather than raising KeyError from
        # downstream code that does dict subscripting.
        out["metrics"].setdefault("grad_norm", None)
        return out

    def _reset_prefix_cache(self, body: dict) -> dict:
        result = self._operation(
            "reset-prefix-cache",
            payload={"drain": body["drain"], "timeout_s": body["timeout_s"], "retry_interval_s": 0.1},
            sub_job_type="sampling",
        )
        request_id = result.get("request_id") if isinstance(result, dict) else None
        return self._poll(request_id) if request_id else result

    # ── SnowAPI HTTP layer ─────────────────────────────────────────────────
    def _build_session(self) -> requests.Session:
        cfg = self.config
        session = requests.Session()
        if cfg.cortex_base_url is not None:
            self.base_url = cfg.cortex_base_url.rstrip("/")
            return session
        self.base_url = f"https://{cfg.cortex_host}"
        session.headers["Authorization"] = f"Bearer {os.environ[cfg.cortex_pat_env_var]}"
        session.headers["X-Snowflake-Authorization-Token-Type"] = "PROGRAMMATIC_ACCESS_TOKEN"
        return session

    @property
    def _prefix(self) -> str:
        cfg = self.config
        return (
            f"{self.base_url}/api/v2/databases/{cfg.cortex_database}/schemas/{cfg.cortex_schema}/{cfg.cortex_endpoint}"
        )

    def _send(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        resp = getattr(self._session, method.lower())(url, **kwargs)
        resp.raise_for_status()
        return resp

    def _create_job(self, sub_jobs: list[SubJobConfig]) -> str:
        if not sub_jobs:
            raise ValueError("create_job requires a non-empty sub_jobs list")
        for sj in sub_jobs:
            sj.validate()
        body = {"sub_job_configs": [sj.to_wire() for sj in sub_jobs]}
        return self._send("POST", self._prefix, json=body).json()["job_id"]

    def _wait_for_job(self, job_id: str) -> dict:
        deadline = time.monotonic() + self.poll_timeout
        delay = self.poll_interval
        while time.monotonic() < deadline:
            job = self._send("GET", f"{self._prefix}/{job_id}").json()
            status = str(job.get("status", "")).lower().removeprefix("job_state_")
            if status == "running":
                return job
            if status in ("failed", "done", "cancelled", "canceled"):
                raise RuntimeError(f"Job {job_id} reached terminal state '{status}': {job.get('reason', '')}")
            delay = self._sleep_with_backoff(delay, deadline)
        raise TimeoutError(f"Job {job_id} did not become running within {self.poll_timeout}s")

    def _post_octet_request_chunks(self, *, path_suffix: str, operation: str, frame: bytes, max_bytes: int) -> dict:
        chunks = wire.encode_byte_chunks(frame, kind="request", operation=operation, max_bytes=max_bytes)
        final_body: dict | None = None
        for idx, chunk in enumerate(chunks):
            body = self._send(
                "POST",
                f"{self._prefix}/{self.job_id}/{path_suffix}",
                data=chunk,
                headers={"Content-Type": "application/octet-stream"},
            ).json()
            if idx < len(chunks) - 1:
                if isinstance(body, dict) and body.get("request_id"):
                    raise RuntimeError(f"{operation} chunk {idx} unexpectedly returned request_id")
                continue
            final_body = body
        if final_body is None:
            raise RuntimeError(f"{operation} produced no request body")
        return final_body

    def _operation(
        self,
        operation_type: str,
        *,
        payload: dict,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        body: dict = {"operation_type": operation_type, "payload": payload}
        if sub_job_id is not None:
            body["sub_job_id"] = sub_job_id
        if sub_job_type is not None:
            body["sub_job_type"] = sub_job_type
        return self._send("POST", f"{self._prefix}/{self.job_id}/operation", json=body).json()

    def _poll(self, request_id: str) -> dict:
        deadline = time.monotonic() + self.poll_timeout
        delay = self.poll_interval
        result_chunks: list[bytes] = []
        cursor: str | None = None
        while time.monotonic() < deadline:
            status = self._get_request_status(request_id, cursor)
            state = str(status.get("status", "")).lower().removeprefix("request_state_")
            received_chunk = False
            for event in status.get("events") or []:
                chunk = _decode_result_chunk_event(event)
                if chunk is not None:
                    result_chunks.append(chunk)
                    received_chunk = True
            next_cursor = status.get("next_cursor")
            if isinstance(next_cursor, str) and next_cursor:
                cursor = next_cursor
                continue
            if state in ("completed", "done", "succeeded"):
                return self._finalize_result(request_id, status, result_chunks)
            if state in ("failed", "cancelled", "canceled"):
                self._generate_request_ids.discard(request_id)
                raise RuntimeError(f"Request {request_id} ended with state '{state}': {status.get('error', '')}")
            if received_chunk:
                continue
            delay = self._sleep_with_backoff(delay, deadline)
        self._generate_request_ids.discard(request_id)
        raise TimeoutError(f"Request {request_id} did not complete within {self.poll_timeout}s")

    def _get_request_status(self, request_id: str, cursor: str | None) -> dict:
        params = {"cursor": cursor} if cursor else None
        return self._send("GET", f"{self._prefix}/{self.job_id}/requests/{request_id}", params=params).json()

    def _finalize_result(self, request_id: str, status: dict, result_chunks: list[bytes]) -> dict:
        if result_chunks:
            result = wire.decode_result_chunks(result_chunks)
        else:
            result = status.get("result") or {}
            decoded = _decode_result_payload(result)
            if decoded is not None:
                result = decoded
        is_generate = request_id in self._generate_request_ids
        self._generate_request_ids.discard(request_id)
        if is_generate and isinstance(result, dict) and "results" in result:
            result = {**result, "results": _restore_generate_result_lists(result["results"])}
        return result

    def _sleep_with_backoff(self, delay: float, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return delay
        time.sleep(min(delay, remaining))
        return min(delay * self.poll_backoff_multiplier, self.poll_max_interval)

    # ── sub-job wiring ─────────────────────────────────────────────────────
    def _build_sub_jobs(self) -> list[SubJobConfig]:
        cfg = self.config
        tc = cfg.training_config or {}
        subs: list[SubJobConfig] = []
        if cfg.sampling_gpus > 0:
            subs.append(
                SubJobConfig.sampling_job(
                    model_name=cfg.model_name,
                    max_seq_len=cfg.max_seq_len,
                    n_gpus=cfg.sampling_gpus,
                    extra_sampling=cfg.vllm_config or {},
                    job_type=JobType.SAMPLING,
                    dtype=cfg.dtype,
                    seed=cfg.seed,
                )
            )
        if cfg.log_prob_gpus > 0:
            subs.append(
                SubJobConfig.sampling_job(
                    model_name=cfg.model_name,
                    max_seq_len=cfg.max_seq_len,
                    n_gpus=cfg.log_prob_gpus,
                    extra_sampling=cfg.vllm_config or {},
                    job_type=JobType.LOG_PROBABILITY,
                    dtype=cfg.dtype,
                    seed=cfg.seed,
                )
            )
        if cfg.training_gpus > 0:
            subs.append(
                SubJobConfig.training_job(
                    model_name=cfg.model_name,
                    optimizer=tc.get("optimizer", {"type": "adamw", "lr": 1e-5}),
                    max_seq_len=cfg.max_seq_len,
                    train_batch_size=tc.get("train_batch_size", 1),
                    n_gpus=cfg.training_gpus,
                    extra_training=tc,
                    dtype=cfg.dtype,
                    seed=cfg.seed,
                )
            )
        return subs

    def _capture_sub_jobs(self, job_info: dict) -> None:
        job = job_info.get("job", job_info)
        for sub in job.get("sub_jobs", []) or []:
            job_type = str(sub.get("job_type", "")).lower().removeprefix("job_type_")
            self.sub_jobs[job_type] = str(sub["sub_job_id"])
        self.sub_jobs.setdefault("training", f"{self.job_id}:training")
        self.sub_jobs.setdefault("sampling", f"{self.job_id}:sampling")
