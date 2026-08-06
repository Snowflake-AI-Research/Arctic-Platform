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

# The canonical op vocabulary: every op the client can lower to a Request. A
# transport is expected to resolve all of these; `unresolved_ops` lets it check
# its own coverage up front instead of failing with an AttributeError mid-run.
OPS = frozenset(
    {
        "forward-backward",
        "forward",
        "step",
        "save",
        "generate",
        "log-probs",
        # Control-plane ops share one canonical envelope (op_type + payload),
        # matching Cortex's /operation; the transport routes them to /operation.
        "operation",
    }
)


def method_name(op: str) -> str:
    """Canonical op name -> the method name a transport target exposes."""
    return op.replace("-", "_")


def unresolved_ops(target: object) -> list[str]:
    """Ops in `OPS` that `target` exposes no matching method for."""
    return sorted(op for op in OPS if not hasattr(target, method_name(op)))


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
    """A canonical op call, built by the client.

    The client owns job identity (it holds `JobHandles` after `initialize`), so it
    resolves and sets `job_id` here directly; `body` carries everything else. This
    keeps transports as pure forwarders: on-prem hands `(job_id, body)` straight to
    the server, `binary` just picks the body codec (octet tensors vs JSON).
    """

    op: str  # canonical op name, e.g. "forward-backward"
    job_id: JobId | None = None  # primary target id (client-resolved); None -> omit
    body: dict[str, Any] = field(default_factory=dict)
    binary: bool = False  # body carries tensors -> octet, else JSON


class Transport(ABC):
    jobs: JobHandles

    @abstractmethod
    def initialize(self) -> JobHandles:
        """Create jobs (or attach to reconnect ids) and return their handles."""

    @abstractmethod
    def call(self, request: Request) -> dict:
        """Deliver one op to this deployment; return a canonical response dict."""

    @abstractmethod
    def shutdown(self) -> None:
        """Tear down jobs / connections."""
