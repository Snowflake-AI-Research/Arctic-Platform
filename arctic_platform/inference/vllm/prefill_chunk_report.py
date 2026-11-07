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

"""Opt-in: report how the scheduler split each request's prefill, for a trainer that reproduces it.

A GatedDeltaNet layer prefills each scheduled chunk as its own chunked-scan call, so a trainer that reproduces the
sampler's numbers needs every chunk's boundaries, and with prefix caching also where the request resumed from the
cache. With ``ARCTIC_REPORT_PREFILL_CHUNKS=1`` the scheduler records ``[start, end)`` of every prefill chunk it
schedules for a request and returns them, with whether the request was preempted, in the finished request's
``kv_transfer_params`` under :data:`KEY`. The first chunk's start is the cache-hit length.
"""

from __future__ import annotations

import os
from typing import Any

ENV_VAR = "ARCTIC_REPORT_PREFILL_CHUNKS"
KEY = "arctic_prefill_chunks"

_APPLIED = False


def reporting_requested() -> bool:
    return os.environ.get(ENV_VAR, "0") == "1"


def record_scheduled_chunk(request: Any, num_scheduled_tokens: int) -> None:
    """Record ``[start, start + n)`` if the scheduled tokens are (part of) the request's prefill."""
    start = int(request.num_computed_tokens)
    if start >= int(request.num_prompt_tokens) or num_scheduled_tokens <= 0:
        return
    chunks = getattr(request, "_arctic_prefill_chunks", None)
    if chunks is None:
        chunks = request._arctic_prefill_chunks = []
    if chunks and start < chunks[-1][1]:
        request._arctic_preempted = True
    chunks.append((start, start + int(num_scheduled_tokens)))


def chunk_report(request: Any) -> dict[str, Any] | None:
    chunks = getattr(request, "_arctic_prefill_chunks", None)
    if chunks is None:
        return None
    return {"chunks": [list(chunk) for chunk in chunks], "preempted": bool(getattr(request, "_arctic_preempted", False))}


def ensure_prefill_chunk_report() -> None:
    global _APPLIED
    if _APPLIED or not reporting_requested():
        return
    from vllm.v1.core.sched.scheduler import Scheduler

    original_update = Scheduler._update_after_schedule
    original_free = Scheduler._free_request

    def _update_after_schedule(self, scheduler_output):
        for req_id, num_scheduled in scheduler_output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            if request is not None:
                record_scheduled_chunk(request, num_scheduled)
        return original_update(self, scheduler_output)

    def _free_request(self, request, delay_free_blocks: bool = False):
        kv_params, ec_params = original_free(self, request, delay_free_blocks)
        report = chunk_report(request)
        if report is not None:
            kv_params = {**(kv_params or {}), KEY: report}
        return kv_params, ec_params

    Scheduler._update_after_schedule = _update_after_schedule
    Scheduler._free_request = _free_request
    _APPLIED = True
