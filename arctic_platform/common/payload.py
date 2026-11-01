# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend-neutral payload for Arctic Platform processing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class APPayload:
    """The execution fields shared by on-prem and managed-service adapters."""

    batch_data: Any
    meta_data: dict
    processing: dict
    loss_object: Any
    args: tuple


@dataclass(frozen=True)
class ScheduledAPPayload:
    """One model call in a possibly padded, request-sized AP schedule."""

    payload: APPayload
    source_index: int
    is_padding: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_index, int)
            or isinstance(self.source_index, bool)
            or self.source_index < 0
        ):
            raise ValueError(
                f"source_index must be a non-negative integer, got {self.source_index!r}"
            )


@dataclass(frozen=True)
class APExecution:
    """Per-call outputs plus the objective reduction AP applied."""

    outputs: tuple[dict, ...]
    loss_reduction: Any


def execute_payload(
    engine: Any,
    payload: APPayload,
    *,
    loop_count: int,
    backward: bool,
    device: Any,
    return_tensors: bool = False,
    step_fn: Callable[[], None] | None = None,
    prepare_call: Callable[[dict, dict], dict] | None = None,
    before_backward: Callable[[int, dict], None] | None = None,
) -> dict:
    """Execute one AP payload without exposing backend wire concerns."""
    from arctic_platform.common.unpacked_gas import run_unpacked_gas

    return run_unpacked_gas(
        engine,
        loop_count=loop_count,
        batch_data=payload.batch_data,
        meta_data=payload.meta_data,
        processing=payload.processing,
        args=payload.args,
        loss_object=payload.loss_object,
        backward=backward,
        device=device,
        return_tensors=return_tensors,
        step_fn=step_fn,
        prepare_call=prepare_call,
        before_backward=before_backward,
    )


def execute_payload_schedule(
    engine: Any,
    calls: Sequence[ScheduledAPPayload],
    *,
    backward: bool,
    device: Any,
    return_tensors: bool = False,
    step_fn: Callable[[], None] | None = None,
    before_backward: Callable[[int, dict], None] | None = None,
) -> APExecution:
    """Execute a client-built call schedule with AP-owned loss scaling.

    Source indexes identify the unpadded calls used to resolve objective
    weights. Repeated padding calls run the same payload at zero loss scale.
    Managed engines step between calls and require configured GAS alignment;
    unmanaged engines accumulate every call and leave the only step to the
    service's final ``/step`` operation.
    """
    from arctic_platform.common.unpacked_gas import run_payload_schedule

    outputs, reduction = run_payload_schedule(
        engine,
        calls,
        backward=backward,
        device=device,
        return_tensors=return_tensors,
        step_fn=step_fn,
        before_backward=before_backward,
    )
    return APExecution(outputs=tuple(outputs), loss_reduction=reduction)
