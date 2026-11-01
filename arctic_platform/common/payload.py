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


def build_packed_schedule(
    windows: Sequence[dict],
    pack_metas: Sequence[Any],
    *,
    sp_size: int,
    sp_rank: int,
    request_call_count: int | None,
    processing: dict,
    loss_object: Any,
) -> list[ScheduledAPPayload]:
    """Turn one rank's packed windows into a padded call schedule.

    Each window is one ``APPayload``. ``batch_data`` keeps the ``[1, T]``
    window. ``meta_data`` is that window squeezed to ``[T]``, plus this rank's
    ``cu_seqlens`` and the window's ``packed_loss_scale_factor``. A shard with
    fewer windows than ``request_call_count`` repeats its last window. Those
    repeated slots set ``is_padding`` and share the source payload so a later
    edit of the real call is the call the padding slot runs.
    """
    from arctic_platform.common.packing.windows import window_cu_seqlens

    if len(windows) != len(pack_metas):
        raise ValueError(
            f"packed windows ({len(windows)}) must match pack metadata ({len(pack_metas)})"
        )
    if not windows:
        return []
    import torch

    call_count = max(len(windows), int(request_call_count or len(windows)))
    source_payloads: list[APPayload] = []
    for window, pack_meta in zip(windows, pack_metas, strict=True):
        meta = {
            key: value.squeeze(0)
            if torch.is_tensor(value) and value.ndim == 2 and value.shape[0] == 1
            else value
            for key, value in window.items()
        }
        meta["cu_seqlens"] = window_cu_seqlens(pack_meta, sp_size, sp_rank)
        meta["packed_loss_scale_factor"] = pack_meta.sequence_length
        source_payloads.append(
            APPayload(
                batch_data=dict(window),
                meta_data=meta,
                processing=processing,
                loss_object=loss_object,
                args=(),
            )
        )
    calls: list[ScheduledAPPayload] = []
    last = len(source_payloads) - 1
    for call_index in range(call_count):
        source_index = min(call_index, last)
        calls.append(
            ScheduledAPPayload(
                source_payloads[source_index],
                source_index=source_index,
                is_padding=call_index >= len(source_payloads),
            )
        )
    return calls


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
