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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wall protocol: climb by 1.25x from a known length, then bisect to 8K tokens."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from arctic_platform.sft.eval.arms import AT_WALLS
from arctic_platform.sft.eval.arms import PROBE_STEPS
from arctic_platform.sft.eval.arms import WARMUP_STEPS


@dataclass
class ProbeResult:
    length: int
    passed: bool
    failure: str | None
    step_s: list[float]
    peak_gpu_mib: float | None
    host_rss_gib: float | None


def climb_then_bisect(
    start: int,
    probe: Callable[[int], ProbeResult],
    *,
    factor: float = 1.25,
    bisect_step: int = 8192,
    max_probes: int = 40,
) -> tuple[int | None, list[ProbeResult]]:
    """Return the longest passing length and the probe history.

    Starts at ``start``. Climbs by ``factor`` while probes pass. On the first
    failure, bisects between the last pass and that failure down to ``bisect_step``.
    """
    history: list[ProbeResult] = []
    length = int(start)
    last_pass: int | None = None
    probes = 0
    while probes < max_probes:
        result = probe(length)
        history.append(result)
        probes += 1
        if result.passed:
            last_pass = length
            length = max(length + 1, int(length * factor))
            continue
        if last_pass is None:
            # start itself failed: step down until a pass or length collapses
            if length <= bisect_step:
                return None, history
            length = max(bisect_step, length // 2)
            continue
        lo, hi = last_pass, length
        while hi - lo > bisect_step and probes < max_probes:
            mid = lo + ((hi - lo) // 2 // bisect_step) * bisect_step
            if mid <= lo:
                mid = lo + bisect_step
            if mid >= hi:
                break
            result = probe(mid)
            history.append(result)
            probes += 1
            if result.passed:
                lo = mid
                last_pass = mid
            else:
                hi = mid
        return last_pass, history
    return last_pass, history


def at_wall(model_name: str, arm: str) -> int:
    try:
        return AT_WALLS[model_name][arm]
    except KeyError as exc:
        raise KeyError(f"no AT wall for model={model_name!r} arm={arm!r}") from exc


def write_history(path: Path, model_name: str, arm: str, wall: int | None, history: list[ProbeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "arm": arm,
        "wall": wall,
        "warmup_steps": WARMUP_STEPS,
        "probe_steps": PROBE_STEPS,
        "history": [asdict(item) for item in history],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def classify_failure(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    lowered = text.lower()
    if "out of memory" in lowered or "cuda out of memory" in lowered:
        return "gpu_oom"
    if "killed" in lowered or "sigkill" in lowered:
        return "host_kill"
    if "cuda" in lowered and ("fault" in lowered or "illegal" in lowered):
        return "cuda_fault"
    return text.splitlines()[0][:200]


def host_rss_gib() -> float | None:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    kib = int(line.split()[1])
                    return kib / (1024 * 1024)
    except OSError:
        return None
    return None


def peak_gpu_mib() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None


def timed_steps(step_fn: Callable[[], None], *, warmup: int, steps: int) -> tuple[list[float], str | None]:
    try:
        for _ in range(warmup):
            step_fn()
        times: list[float] = []
        for _ in range(steps):
            start = time.perf_counter()
            step_fn()
            times.append(time.perf_counter() - start)
        return times, None
    except Exception as exc:  # noqa: BLE001 — probe must classify every failure
        return [], classify_failure(exc)
