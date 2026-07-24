# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""The pluggable seam: a transport is just "how do we deliver an op here".

The op *surface* (names, args, canonical body, which job each op targets,
response contract) is defined once in the client. A transport owns only job
identity + wire mechanics: given a `Request`, deliver it to its deployment and
return a canonical response dict. It never redefines the API.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from arctic_platform.client.config import ArcticRLClientConfig
from arctic_platform.client.config import JobId

JOB_TYPES = ("training", "sampling", "log_prob")
# Creation order shared by every backend: inference jobs first so training (the
# NCCL root) rendezvouses last.
JOB_CREATE_ORDER = ("sampling", "log_prob", "training")


@dataclass
class JobHandles:
    training: JobId | None = None
    sampling: JobId | None = None
    log_prob: JobId | None = None

    @classmethod
    def from_config(cls, config: ArcticRLClientConfig) -> JobHandles:
        return cls(config.training_job_id, config.sampling_job_id, config.log_prob_job_id)

    @property
    def any_set(self) -> bool:
        return any(getattr(self, jt) is not None for jt in JOB_TYPES)

    def set(self, job_type: str, job_id: JobId) -> None:
        setattr(self, job_type, job_id)

    def require(self, job_type: str) -> JobId:
        job_id = getattr(self, job_type)
        if job_id is None:
            raise ValueError(f"No {job_type} job initialized.")
        return job_id


@dataclass
class Request:
    """A canonical op call, built by the client — identical across backends."""

    op: str                              # canonical op name, e.g. "fwd-bwd"
    target: str                          # job type the op addresses
    body: dict[str, Any] = field(default_factory=dict)  # user data only (no job ids)


class Transport(ABC):
    jobs: JobHandles

    @abstractmethod
    def initialize(self) -> JobHandles:
        """Create jobs (or attach to reconnect ids) and return their handles."""

    @abstractmethod
    def call(self, request: Request) -> dict:
        """Deliver one op to this deployment; return a canonical response dict."""

    @abstractmethod
    def shutdown(self) -> None: ...
