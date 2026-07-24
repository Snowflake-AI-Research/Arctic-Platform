# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-prem transport base for the Arctic-Platform server.

HTTP (onprem/dss) and in-process Ray share one control plane (`OnPremTransport`:
job creation, ordering, payload building, `call`); they differ only in three
primitives (`_start`, `_rpc`, `_destroy`). `call` is uniform — post/dispatch the
op against its target job — because the server exposes a uniform
`op(job_id, body) -> dict` surface.

Concrete transports live alongside this base: `HttpTransport` in
`onprem_http.py` and `RayTransport` in `onprem_ray.py`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from arctic_platform.client.config import ArcticRLClientConfig
from arctic_platform.client.config import JobId
from arctic_platform.client.transport import JOB_CREATE_ORDER
from arctic_platform.client.transport import JOB_TYPES
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transport import Request
from arctic_platform.client.transport import Transport


class OnPremTransport(Transport):
    def __init__(self, config: ArcticRLClientConfig) -> None:
        self.config = config
        self.jobs = JobHandles()

    def initialize(self) -> JobHandles:
        reconnect = JobHandles.from_config(self.config)
        if reconnect.any_set:
            self.jobs = reconnect
            return self.jobs
        cfg = self.config
        for job_type in JOB_CREATE_ORDER:
            if cfg.gpus_for(job_type) > 0:
                self.jobs.set(job_type, self._start(self._init_payload(job_type)))
        self._wait_running()
        return self.jobs

    def call(self, request: Request) -> dict:
        # The client already resolved the job id onto the request; just deliver it.
        return self._rpc(request)

    def shutdown(self) -> None:
        for job_type in JOB_TYPES:
            job_id = getattr(self.jobs, job_type)
            if job_id is not None:
                self._destroy(job_id, job_type)

    def _init_payload(self, job_type: str) -> dict[str, Any]:
        cfg = self.config
        payload: dict[str, Any] = {"model_name": cfg.model_name, "job_type": job_type, "seed": cfg.seed}
        if job_type in ("training", "log_prob"):
            if cfg.ds_config:
                payload["ds_config"] = cfg.ds_config
            if job_type == "training":
                if cfg.training_config:
                    payload["training_config"] = cfg.training_config
                if cfg.checkpoint_path:
                    payload["checkpoint_path"] = cfg.checkpoint_path
        elif cfg.vllm_config:
            payload["vllm_config"] = cfg.vllm_config
        return payload

    # delivery primitives — the only things a concrete transport implements
    @abstractmethod
    def _start(self, payload: dict) -> JobId: ...
    @abstractmethod
    def _rpc(self, request: Request) -> dict: ...
    @abstractmethod
    def _destroy(self, job_id: JobId, job_type: str) -> None: ...
    @abstractmethod
    def _wait_running(self) -> None: ...
