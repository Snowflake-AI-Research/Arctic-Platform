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

"""Align DeepSpeed/PrimeRL parameter names with the Hugging Face reference.

Container aliases are removed first. PrimeRL then needs one architecture conversion: Qwen3.5 MoE stores
routed gate/up projections as separate ``w1`` and ``w3`` tensors, while Hugging Face stores their
concatenation as ``gate_up_proj``. The L2 norm of that concatenated gradient is exactly
``hypot(norm(w1), norm(w3))``. Router and shared-expert paths are renamed without changing their values.

Checkpoint wrapping can expose more than one name for one tensor. Equal-norm aliases collapse to one
semantic parameter; unequal aliases remain unmatched so the correctness test cannot hide a real collision.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

_CONTAINER_SEGMENTS = frozenset(
    {
        "module",
        "_forward_module",
        "_orig_mod",
        "_checkpoint_wrapped_module",
        "model",
        "language_model",
        "text_model",
        "base_model",
        "transformer",
    }
)

_ROUTED_EXPERT = re.compile(r"^(?P<prefix>.*\.mlp\.experts)\.(?P<weight>w[123])$")
_SHARED_EXPERT = re.compile(r"^(?P<layer>(?:.*\.)?layers\.\d+)\.(?:mlp\.)?shared_expert\.(?P<weight>w[123])\.weight$")
_SHARED_NAMES = {
    "w1": "gate_proj.weight",
    "w2": "down_proj.weight",
    "w3": "up_proj.weight",
}


@dataclass
class _Entry:
    originals: List[str]
    value: float


def normalize(name: str) -> str:
    return ".".join(part for part in name.split(".") if part not in _CONTAINER_SEGMENTS)


def _insert(entries: Dict[str, _Entry], key: str, original: str, value: float) -> None:
    existing = entries.get(key)
    if existing is None:
        entries[key] = _Entry([original], float(value))
        return
    if existing.value == float(value):
        existing.originals.append(original)
        return
    # Do not let normalization silently overwrite two different gradients. The synthetic key cannot match
    # anything on the other side and therefore turns the collision into an explicit coverage failure.
    entries[f"{key}::alias::{original}"] = _Entry([original], float(value))


def _normalized(values: Dict[str, float]) -> Dict[str, _Entry]:
    entries: Dict[str, _Entry] = {}
    for original, value in values.items():
        _insert(entries, normalize(original), original, float(value))
    return entries


def _canonical_dss(values: Dict[str, float]) -> Dict[str, _Entry]:
    source = _normalized(values)
    canonical: Dict[str, _Entry] = {}
    routed: Dict[str, Dict[str, _Entry]] = {}

    for key, entry in source.items():
        # Text-only correctness batches never execute the VLM vision tower. DeepSpeed materializes zero
        # gradients for those unused trainable parameters, while plain PyTorch leaves ``grad`` absent.
        # Exclude only a proven zero; a nonzero vision gradient remains unmatched and fails coverage.
        if key.startswith("visual.") and entry.value == 0.0:
            continue
        match = _ROUTED_EXPERT.match(key)
        if match:
            weight = match.group("weight")
            prefix = match.group("prefix")
            if weight in {"w1", "w3"}:
                routed.setdefault(prefix, {})[weight] = entry
                continue
            _insert(canonical, f"{prefix}.down_proj", entry.originals[0], entry.value)
            canonical[f"{prefix}.down_proj"].originals.extend(entry.originals[1:])
            continue

        match = _SHARED_EXPERT.match(key)
        if match:
            mapped = f"{match.group('layer')}.mlp.shared_expert.{_SHARED_NAMES[match.group('weight')]}"
            _insert(canonical, mapped, entry.originals[0], entry.value)
            canonical[mapped].originals.extend(entry.originals[1:])
            continue

        if ".mlp.router.gate.weight" in key:
            mapped = key.replace(".mlp.router.gate.weight", ".mlp.gate.weight")
        elif ".shared_expert_gate.weight" in key and ".mlp.shared_expert_gate.weight" not in key:
            mapped = key.replace(".shared_expert_gate.weight", ".mlp.shared_expert_gate.weight")
        else:
            mapped = key
        _insert(canonical, mapped, entry.originals[0], entry.value)
        canonical[mapped].originals.extend(entry.originals[1:])

    for prefix, pieces in routed.items():
        if set(pieces) == {"w1", "w3"}:
            w1, w3 = pieces["w1"], pieces["w3"]
            key = f"{prefix}.gate_up_proj"
            canonical[key] = _Entry(w1.originals + w3.originals, math.hypot(w1.value, w3.value))
        else:
            for weight, entry in pieces.items():
                canonical[f"{prefix}.{weight}"] = entry
    return canonical


@dataclass(frozen=True)
class CanonicalTensorGroup:
    """Raw Arctic Platform tensor names that form one Hugging Face semantic parameter."""

    originals: Tuple[str, ...]
    concatenate_dim: Optional[int] = None


def canonical_dss_tensor_groups(names: Iterable[str]) -> Dict[str, CanonicalTensorGroup]:
    """Group Arctic Platform tensor names by their Hugging Face parameter representation."""
    canonical: Dict[str, CanonicalTensorGroup] = {}
    routed: Dict[str, Dict[str, str]] = {}

    for original in names:
        key = normalize(original)
        match = _ROUTED_EXPERT.match(key)
        if match:
            weight = match.group("weight")
            prefix = match.group("prefix")
            if weight in {"w1", "w3"}:
                routed.setdefault(prefix, {})[weight] = key
            else:
                canonical[f"{prefix}.down_proj"] = CanonicalTensorGroup((key,))
            continue

        match = _SHARED_EXPERT.match(key)
        if match:
            mapped = f"{match.group('layer')}.mlp.shared_expert.{_SHARED_NAMES[match.group('weight')]}"
            canonical[mapped] = CanonicalTensorGroup((key,))
            continue

        if ".mlp.router.gate.weight" in key:
            mapped = key.replace(".mlp.router.gate.weight", ".mlp.gate.weight")
        elif ".shared_expert_gate.weight" in key and ".mlp.shared_expert_gate.weight" not in key:
            mapped = key.replace(".shared_expert_gate.weight", ".mlp.shared_expert_gate.weight")
        else:
            mapped = key
        canonical[mapped] = CanonicalTensorGroup((key,))

    for prefix, pieces in routed.items():
        if set(pieces) == {"w1", "w3"}:
            canonical[f"{prefix}.gate_up_proj"] = CanonicalTensorGroup(
                (pieces["w1"], pieces["w3"]),
                concatenate_dim=1,
            )
        else:
            for weight, original in pieces.items():
                canonical[f"{prefix}.{weight}"] = CanonicalTensorGroup((original,))
    return canonical


def _originals(entries: Dict[str, _Entry], keys: set[str]) -> List[str]:
    return sorted(original for key in keys for original in entries[key].originals)


def align(
    dss: Dict[str, float], reference: Dict[str, float]
) -> Tuple[List[Tuple[str, float, float]], List[str], List[str]]:
    """Pair every semantic tensor and return pairs plus uncovered raw names on either side."""
    dss_by_key = _canonical_dss(dss)
    ref_by_key = _normalized(reference)
    shared = sorted(set(dss_by_key) & set(ref_by_key))
    pairs = [(key, dss_by_key[key].value, ref_by_key[key].value) for key in shared]
    only_dss_keys = set(dss_by_key) - set(ref_by_key)
    only_ref_keys = set(ref_by_key) - set(dss_by_key)
    return pairs, _originals(dss_by_key, only_dss_keys), _originals(ref_by_key, only_ref_keys)
