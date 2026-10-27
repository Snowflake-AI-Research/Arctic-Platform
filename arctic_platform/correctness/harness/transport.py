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


"Training transport contracts and explicitly requested microbatch packing."

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import ContextManager
from typing import Optional
from typing import Protocol


class UnsafeCleanupError(RuntimeError):
    """Owned resources may remain; do not start another correctness workload."""

    def __init__(self, message, *, owned_parent=None, owned_children=None, shutdown_error=None):
        super().__init__(message)
        self.owned_parent = owned_parent
        self.owned_children = dict(owned_children or {})
        self.shutdown_error = shutdown_error
        self.workload_error = None

    def preserve_workload(self, error):
        self.workload_error = error
        self.args = (f"Original workload failed: {type(error).__name__}: {error}; {self}",)


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


def pack_microbatches(
    batch,
    microbatches: int,
    *,
    model_provider: str = "huggingface",
    loss_fn: str = "sft",
    processing_config=None,
) -> dict:
    """Pack an explicitly requested number of microbatches without changing the input config."""
    from .dss_driver import pack

    body = pack(batch, model_provider=model_provider)
    body["processing"] = {"loss_fn": loss_fn}
    if processing_config is not None:
        body["processing"]["config"] = deepcopy(processing_config)
    if microbatches <= 1:
        return body
    if batch.rows < microbatches:
        raise ValueError(f"cannot split {batch.rows} row(s) into {microbatches} microbatches")
    base_rows, extra_rows = divmod(batch.rows, microbatches)
    tensors = body["batch"]
    splits = []
    start = 0
    for index in range(microbatches):
        end = start + base_rows + (index < extra_rows)
        splits.append({key: value[start:end].contiguous() for key, value in tensors.items()})
        start = end
    body["batch"] = splits
    return body
