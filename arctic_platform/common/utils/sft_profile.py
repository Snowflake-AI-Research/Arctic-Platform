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
"""Lightweight SFT step timing (opt-in via ``ARL_SFT_PROFILE=1``).

Records wall-clock ms for the buckets listed in PERFORMANCE.md:
``serialize``, ``rpc``, ``h2d``, ``fwd``, ``bwd``, ``step``.

Server-side buckets are attached to the ``fwd-bwd`` / ``step`` response under
``metrics["_profile_ms"]`` so a CPU client can print them. Client-side
``serialize`` / ``rpc`` are printed locally (they never leave the client).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator


def enabled() -> bool:
    return os.environ.get("ARL_SFT_PROFILE", "").strip().lower() in ("1", "true", "yes", "on")


class _BucketAccum:
    """Per-process accumulators (client OR one DeepSpeed worker rank)."""

    __slots__ = ("sums", "counts", "last")

    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.last: dict[str, float] = {}

    def add(self, name: str, ms: float) -> None:
        self.sums[name] += ms
        self.counts[name] += 1
        # Within a step, same bucket may fire per microbatch — sum into `last`
        # until ``take_last`` clears it (so gas>1 reports total fwd/bwd ms).
        self.last[name] = self.last.get(name, 0.0) + ms

    def last_snapshot(self) -> dict[str, float]:
        return dict(self.last)

    def mean_snapshot(self) -> dict[str, float]:
        return {k: self.sums[k] / self.counts[k] for k in self.sums if self.counts[k]}


_ACCUM = _BucketAccum()


@contextmanager
def timed(name: str) -> Iterator[None]:
    """Accumulate wall-clock ms under *name* when profiling is enabled."""
    if not enabled():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _ACCUM.add(name, (time.perf_counter() - t0) * 1000.0)


def take_last() -> dict[str, float]:
    """Return and clear the last-step bucket snapshot (empty if disabled)."""
    if not enabled():
        return {}
    snap = _ACCUM.last_snapshot()
    _ACCUM.last.clear()
    return snap


def merge_into_metrics(metrics: dict | None, extra: dict[str, float] | None = None) -> dict:
    """Attach ``_profile_ms`` (last step) onto a metrics dict for the wire response."""
    out = dict(metrics or {})
    if not enabled():
        return out
    prof = take_last()
    if extra:
        prof.update(extra)
    if prof:
        out["_profile_ms"] = {k: round(v, 3) for k, v in sorted(prof.items())}
    return out


def format_line(prefix: str, buckets: dict[str, float]) -> str:
    parts = " ".join(f"{k}={v:.2f}ms" for k, v in sorted(buckets.items()))
    return f"[ARL_SFT_PROFILE] {prefix}: {parts}"


def maybe_print(prefix: str, buckets: dict[str, float] | None = None) -> None:
    if not enabled():
        return
    data = buckets if buckets is not None else _ACCUM.last_snapshot()
    if data:
        print(format_line(prefix, data), flush=True)
