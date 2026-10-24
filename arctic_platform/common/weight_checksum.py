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

"""Logical-weight checksum carried by every training -> sampling weight-sync transport.

The trainer squares each full (un-sharded) tensor it hands to a transport, and the sampler squares each full tensor
it receives before vLLM shards, fuses or quantizes it. Both sides therefore measure the same logical weights, so the
totals are comparable at any tensor-parallel size and for quantized samplers, which a walk over the sampler's live
parameters is not.

The sampler imports this module, so it must not pull in DeepSpeed or Ray.
"""

from __future__ import annotations

import math

import torch

# Both sides square bit-identical tensors with ``tensor_l2_sq``, so every per-element square is identical. Only the
# order of the float64 additions can differ (GPU reduction trees, Python sums over chunks / params). Summing n
# non-negative float64 terms in any order is off by at most (n - 1) * 2**-53 relative. Inside one 2**20-element
# chunk that bound is about 1.2e-10, so 1e-9 leaves about one order of magnitude. The live-parameter norm's 1e-3
# would hide a dropped small tensor such as one RMSNorm weight.
WEIGHT_CHECKSUM_RTOL = 1e-9


def tensor_l2_sq(value: torch.Tensor, chunk_numel: int = 1 << 20) -> torch.Tensor:
    """Sum of squares of ``value`` as a float64 scalar on ``value``'s device.

    Squared in float64, one chunk at a time. A float32 square overflows to inf once an element exceeds about
    1e19, and ``math.isclose(inf, inf)`` is true, so that overflow would hide every other tensor.
    """
    total = torch.zeros((), dtype=torch.float64, device=value.device)
    # reshape (not view) so non-contiguous parameters still checksum.
    for chunk in value.detach().reshape(-1).split(chunk_numel):
        total.add_(chunk.double().square().sum())
    return total


def _checksums_agree(left: float, right: float, rtol: float) -> bool:
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    return math.isclose(left, right, rel_tol=rtol, abs_tol=0.0)


def compare_weight_checksums(
    sent: dict[str, float | None],
    received: dict[str, float | None],
    rtol: float = WEIGHT_CHECKSUM_RTOL,
) -> dict:
    """Whether every receiver got the bytes the trainer sent.

    ``sent`` maps each trainer rank that shipped a full model to its sum of squares, and ``received`` maps each
    sampler rank to the sum of squares of what it received. Every rank in ``sent`` ships the same logical model, so
    they must agree with one another as well. A missing value is a failed check, not a pass: ``checksum_match`` is
    True only when every value is present and within ``rtol``, and ``checksum_error`` names what failed otherwise.
    """
    reference = next(iter(sent.values()), None)
    problems = []
    unreported = [label for label, value in sent.items() if value is None]
    nonfinite_sent = [label for label, value in sent.items() if value is not None and not math.isfinite(value)]
    if not sent or unreported:
        problems.append(f"no sent checksum from {unreported or 'any training rank'}")
    elif nonfinite_sent:
        problems.append(f"non-finite sent checksum from {nonfinite_sent}")
    elif any(not _checksums_agree(value, reference, rtol) for value in sent.values()):
        problems.append(f"training ranks sent different checksums {sent}")
    unreceived = [label for label, value in received.items() if value is None]
    nonfinite_received = [
        label for label, value in received.items() if value is not None and not math.isfinite(value)
    ]
    if not received or unreceived:
        problems.append(f"no received checksum from {unreceived or 'any sampling rank'}")
    elif nonfinite_received:
        problems.append(f"non-finite received checksum from {nonfinite_received}")
    elif reference is not None and math.isfinite(reference):
        differing = {
            label: value
            for label, value in received.items()
            if value is not None and not _checksums_agree(value, reference, rtol)
        }
        if differing:
            problems.append(f"received {differing}, sent {reference}")
    result = {
        "sent_l2_sq": reference,
        "recv_l2_sq": dict(received),
        "checksum_rtol": rtol,
        "checksum_match": not problems,
    }
    if problems:
        result["checksum_error"] = "; ".join(problems)
    return result
