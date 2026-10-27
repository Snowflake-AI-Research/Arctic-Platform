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


"Canonical tensor names, comparison groups, and optimizer norm schemas."

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
_SHARED_EXPERT = re.compile(
    r"^(?P<layer>(?:.*\.)?layers\.\d+)\.(?:mlp\.)?(?:qwen_)?shared_expert\.(?P<weight>w[123])\.weight$"
)
_BARE_SHARED_EXPERT = re.compile(r"^(?P<layer>(?:.*\.)?layers\.\d+)\.mlp\.shared_expert\.(?P<weight>w[123])$")
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


def _shared_expert_name(name: str) -> Optional[str]:
    match = _SHARED_EXPERT.match(name)
    container = "shared_expert"
    if match is None:
        # GLM-5.3 uses bare Prime projections and a plural HF shared-experts container.
        match = _BARE_SHARED_EXPERT.match(name)
        container = "shared_experts"
    if match is None:
        return None
    return f"{match.group('layer')}.mlp.{container}.{_SHARED_NAMES[match.group('weight')]}"


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

        mapped = _shared_expert_name(key)
        if mapped is not None:
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

        mapped = _shared_expert_name(key)
        if mapped is not None:
            canonical[mapped] = CanonicalTensorGroup((original,))
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


STATE_KEYS = ("parameter_update", "exp_avg", "exp_avg_sq")
COMPARISON_STATE_KEYS = ("parameter_update_residual", "exp_avg", "exp_avg_sq")
OPTIMIZER_NORM_CRITERION = "optimizer-state-l2-norm-v1"


def validate_optimizer_norm_summary(summary: dict) -> dict:
    """Return the canonical parameter mapping after validating the complete step-one norm schema."""
    if not isinstance(summary, dict) or summary.get("criterion") != OPTIMIZER_NORM_CRITERION:
        raise ValueError(f"optimizer norms require criterion {OPTIMIZER_NORM_CRITERION}")
    if type(summary.get("step")) is not int or summary["step"] != 1:
        raise ValueError("optimizer norms must describe step 1")
    if summary.get("state_keys") != list(COMPARISON_STATE_KEYS):
        raise ValueError("optimizer norms have incomplete or incompatible state coverage")
    parameters = summary.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("optimizer norms have no parameter coverage")
    counts = {
        "parameter_count": len(parameters),
        "optimizer_values_count": len(parameters) * len(COMPARISON_STATE_KEYS),
    }
    for key, expected in counts.items():
        if type(summary.get(key)) is not int or summary[key] != expected:
            raise ValueError(f"optimizer norms have inconsistent {key}")
    for name, values in parameters.items():
        if not isinstance(name, str) or not name or normalize(name) != name:
            raise ValueError("optimizer norms require nonempty canonical parameter names")
        if not isinstance(values, dict) or set(values) != {"shape", *COMPARISON_STATE_KEYS}:
            raise ValueError(f"optimizer norms for {name} have incomplete or incompatible state coverage")
        shape = values["shape"]
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError(f"optimizer norms for {name} have an invalid shape")
        for state in COMPARISON_STATE_KEYS:
            value = values[state]
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError(f"optimizer norm for {name}::{state} must be finite and nonnegative")
    return parameters
