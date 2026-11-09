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
"""Step-0 gradient parity across arms (plan section 4 layer 3).

Acceptance (plan): per-tensor norm difference at most 1%, cosine at least 0.999
on sampled elements, and no worse than the gc-vs-gc rerun.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


def named_grad_norms(parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, parameter in parameters:
        if parameter.grad is None:
            continue
        out[name] = float(parameter.grad.detach().float().norm().item())
    return out


def relative_norm_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def compare_grad_norms(
    reference: dict[str, float],
    candidate: dict[str, float],
) -> tuple[float, str | None]:
    """Return (worst relative norm diff, name of that tensor). Missing names are skipped."""
    worst = 0.0
    worst_name: str | None = None
    for name, ref in reference.items():
        if name not in candidate:
            continue
        diff = relative_norm_diff(ref, candidate[name])
        if diff >= worst:
            worst = diff
            worst_name = name
    return worst, worst_name


def cosine_on_concat(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    *,
    max_elements: int = 1_000_000,
) -> float:
    """Cosine similarity on a concat of matching grads, capped at ``max_elements``."""
    pieces_a: list[torch.Tensor] = []
    pieces_b: list[torch.Tensor] = []
    remaining = max_elements
    for name, ref in reference.items():
        if name not in candidate or remaining <= 0:
            continue
        a = ref.detach().float().reshape(-1)
        b = candidate[name].detach().float().reshape(-1)
        n = min(a.numel(), b.numel(), remaining)
        pieces_a.append(a[:n].cpu())
        pieces_b.append(b[:n].cpu())
        remaining -= n
    if not pieces_a:
        return float("nan")
    a = torch.cat(pieces_a)
    b = torch.cat(pieces_b)
    denom = float(a.norm().item() * b.norm().item())
    if denom == 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / denom)


def passes_plan_gates(
    worst_rel: float,
    cosine: float,
    *,
    gc_vs_gc_worst: float,
    max_rel: float = 0.01,
    min_cosine: float = 0.999,
) -> bool:
    """Plan gate: at most 1% norm diff, cosine >= 0.999, and no worse than gc-vs-gc."""
    if math.isnan(cosine) or cosine < min_cosine:
        return False
    if worst_rel > max_rel:
        return False
    if worst_rel > gc_vs_gc_worst:
        return False
    return True
