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

"""The interface a check drives a Arctic Platform training job through, so one measurement runs on either transport.

A check that trains a loss trajectory, writes a checkpoint and resumes a second job from it needs three
operations and no knowledge of where the job runs: start a job, send one forward-backward followed by one
optimizer step, and save weights together with optimizer and scheduler state. Both implementations provide
exactly those -- ``dss_driver.GatewayTransport`` against a Ray gateway this process starts, and
``hosted.HostedTransport`` against the hosted control plane.

The checkpoint each mints is opaque to the caller, because the two transports name a checkpoint
differently: the gateway returns the DeepSpeed tag a directory was written under, while the hosted server
returns the durable ``cp_<uuid>`` resource its control plane catalogs. A check carries the value back to
the transport that issued it and never reads inside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ContextManager
from typing import Optional
from typing import Protocol


@dataclass(frozen=True)
class Checkpoint:
    """A saved checkpoint and the job that wrote it, in whatever spelling the issuing transport uses."""

    reference: str
    source_job: str


class TrainingJob(Protocol):
    """A running training job, reduced to the two operations a loss trajectory needs."""

    def fwd_bwd_step(self, body: bytes, *, learning_rate: float) -> float:
        """Run one request's forward-backward and then one optimizer step, and return the request's loss.

        ``body`` is a DSSST1 frame. The loss is the one the forward-backward reported, so it is measured
        before the step that follows it changes any parameter.
        """

    def save_resumable(self) -> Checkpoint:
        """Save model weights, optimizer moments and LR-scheduler position, and return what identifies them."""


class Transport(Protocol):
    """Somewhere jobs can be created, held open for as long as a check needs to create them."""

    def __enter__(self) -> "Transport": ...

    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

    def job(self, *, resume_from: Optional[Checkpoint] = None) -> ContextManager[TrainingJob]:
        """One job running the config this transport was built for, and destroyed when the block exits.

        With ``resume_from`` the job initializes its weights, optimizer moments and scheduler position from
        that checkpoint instead of from the base model.
        """
