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
"""The pluggable seam: a transport is just "how do we deliver an op here".

The op *surface* (names, args, canonical body, which job each op targets,
response contract) is defined once in the client. A transport owns job identity,
wire mechanics, and temporary normalization of deployment-specific responses.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from arctic_platform.client.config import ArcticClientConfig
from arctic_platform.client.config import JobId

JOB_TYPES = ("training", "sampling", "log_prob")
# Creation order shared by every backend: inference jobs first so training (the
# NCCL root) rendezvouses last.
JOB_CREATE_ORDER = ("sampling", "log_prob", "training")

# Live-engine operations that every initialized client transport must resolve.
# `unresolved_ops` checks on-prem server coverage before work starts.
OPS = frozenset(
    {
        "forward-backward",
        "forward",
        "step",
        "save",
        "load-checkpoint",  # SFT resume; not in Cortex surface
        "generate",
        "log-probs",
        # Control-plane ops share one canonical envelope (op_type + payload),
        # matching Cortex's /operation; the transport routes them to /operation.
        "operation",
        # Colocate sleep/wake (SFT generate / RL e2e); direct routes, not /operation.
        "sleep-inference",
        "wake-inference",
        "sleep-training",
        "wake-training",
    }
)

CLUSTER_OPS = frozenset(
    {
        "list-jobs",
        "get-job",
        "cancel-job",
        "capacity",
        "list-checkpoints",
        "export-checkpoint",
        "delete-checkpoint",
        "download-artifacts",
    }
)


def method_name(op: str) -> str:
    """Canonical op name -> the method name a transport target exposes."""
    return op.replace("-", "_")


def initialize_or_cleanup(transport: "Transport", client_config: ArcticClientConfig) -> "JobHandles":
    """``initialize(client_config)``, or clean up so a launched server is not orphaned.

    ``shutdown`` is idempotent; safe even when the transport already self-cleaned.
    """
    try:
        return transport.initialize(client_config)
    except Exception:
        try:
            transport.shutdown()
        except Exception:
            pass
        raise


def unresolved_ops(target: object) -> list[str]:
    """Ops in `OPS` that `target` exposes no matching method for."""
    return sorted(op for op in OPS if not hasattr(target, method_name(op)))


@dataclass
class JobHandles:
    training: JobId | None = None
    sampling: JobId | None = None
    log_prob: JobId | None = None

    @classmethod
    def from_config(cls, config: ArcticClientConfig) -> JobHandles:
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
    """A canonical backend operation, built without knowing its transport.

    A transport may fulfill one operation with one request, several requests, or
    a client-side workflow. ``job_id`` is absent for account-level operations.
    """

    op: str  # canonical op name, e.g. "forward-backward"
    job_id: JobId | None = None  # primary target id (client-resolved); None -> omit
    body: dict[str, Any] = field(default_factory=dict)
    binary: bool = False  # body carries tensors -> octet, else JSON


class Transport(ABC):
    jobs: JobHandles

    @abstractmethod
    def initialize(self, client_config: ArcticClientConfig) -> JobHandles:
        """Provision jobs from the stateful client config, or attach to its reconnect ids."""

    @abstractmethod
    def call(self, request: Request) -> Any:
        """Deliver one operation to this deployment."""

    @abstractmethod
    async def acall(self, request: Request) -> Any:
        """Async delivery of one operation to this deployment."""

    async def aclose(self) -> None:
        """Release async resources (e.g. an aiohttp session); no-op by default."""

    @abstractmethod
    def shutdown(self) -> None:
        """Tear down jobs / connections."""
