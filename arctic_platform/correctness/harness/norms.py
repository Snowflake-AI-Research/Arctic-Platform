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


"""Parameter-name alignment and scalar norm and participation metadata validation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

# Names

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


def _canonical_ap(values: Dict[str, float]) -> Dict[str, _Entry]:
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


def canonical_ap_tensor_groups(names: Iterable[str]) -> Dict[str, CanonicalTensorGroup]:
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
    ap: Dict[str, float], reference: Dict[str, float]
) -> Tuple[List[Tuple[str, float, float]], List[str], List[str]]:
    """Pair every semantic tensor and return pairs plus uncovered raw names on either side."""
    ap_by_key = _canonical_ap(ap)
    ref_by_key = _normalized(reference)
    shared = sorted(set(ap_by_key) & set(ref_by_key))
    pairs = [(key, ap_by_key[key].value, ref_by_key[key].value) for key in shared]
    only_ap_keys = set(ap_by_key) - set(ref_by_key)
    only_ref_keys = set(ref_by_key) - set(ap_by_key)
    return pairs, _originals(ap_by_key, only_ap_keys), _originals(ref_by_key, only_ref_keys)


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


# Participation

PARTICIPATION_SCHEMA = "trainable-parameter-participation-v1"


def validate_participation(record: dict) -> dict:
    if not isinstance(record, dict) or record.get("schema") != PARTICIPATION_SCHEMA:
        raise ValueError("gradient coverage requires a versioned participation inventory")
    parameters = record.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("participation inventory has no trainable parameters")
    for name, entry in parameters.items():
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            raise ValueError("participation inventory has an invalid parameter")
        shape = entry.get("shape")
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError(f"participation shape is invalid for {name}")
        state = entry.get("state")
        if state not in {"active", "inactive"}:
            raise ValueError(f"unexpected missing gradient for {name}")
        if state == "inactive":
            evidence = entry.get("evidence")
            if not isinstance(evidence, dict):
                raise ValueError(f"inactive parameter {name} has no independent evidence")
            if not isinstance(evidence.get("module"), str) or not evidence["module"]:
                raise ValueError(f"inactive parameter {name} has invalid independent evidence")
            kind = evidence.get("kind")
            if kind == "text-decoder-bypass":
                valid = (
                    type(evidence.get("calls")) is int
                    and evidence["calls"] == 0
                    and evidence.get("outside_decoder_and_head") is True
                    and normalize(evidence["module"]) == "visual"
                )
            elif kind == "integer-selector":
                valid = (
                    type(evidence.get("calls")) is int
                    and evidence["calls"] > 0
                    and evidence.get("integer_outputs_only") is True
                    and evidence.get("module_class") == "Glm5NextTextIndexer"
                )
            else:
                valid = False
            digest = evidence.get("forward_sha256")
            if (
                not valid
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not isinstance(evidence.get("module"), str)
                or not evidence["module"]
                or not normalize(name).startswith(normalize(evidence["module"]) + ".")
            ):
                raise ValueError(f"inactive parameter {name} has invalid independent evidence")
    return parameters


def canonical_participation(record: dict, *, backend: str) -> dict:
    if backend not in {"ap", "reference"}:
        raise ValueError(f"unknown participation backend: {backend}")
    parameters = validate_participation(record)
    normalized = {}
    for name, entry in parameters.items():
        key = normalize(name)
        if key in normalized:
            raise ValueError(f"participation inventory has duplicate canonical name {key}")
        normalized[key] = entry
    groups = canonical_ap_tensor_groups(normalized) if backend == "ap" else {name: None for name in normalized}
    covered = [name for group in groups.values() for name in group.originals] if backend == "ap" else list(groups)
    if sorted(covered) != sorted(normalized):
        raise ValueError("participation inventory has a canonical name collision")
    result = {}
    for name, group in groups.items():
        pieces = [normalized[original] for original in group.originals] if group else [normalized[name]]
        if len({entry["state"] for entry in pieces}) != 1:
            raise ValueError(f"participation disagreement within fused parameter {name}")
        shape = list(pieces[0]["shape"])
        if group and group.concatenate_dim is not None:
            dimension = group.concatenate_dim
            if any(
                len(entry["shape"]) != len(shape)
                or any(size != shape[index] for index, size in enumerate(entry["shape"]) if index != dimension)
                for entry in pieces
            ) or dimension >= len(shape):
                raise ValueError(f"participation fused shape mismatch for {name}")
            shape[dimension] = sum(entry["shape"][dimension] for entry in pieces)
        result[name] = {"shape": shape, "state": pieces[0]["state"]}
    return result


def compare_participation(target: dict, reference: dict) -> tuple[int, int]:
    actual = canonical_participation(target, backend="ap")
    expected = canonical_participation(reference, backend="reference")
    if set(actual) != set(expected):
        raise ValueError(
            "incomplete trainable inventory: "
            f"target-only={sorted(set(actual) - set(expected))[:4]}, "
            f"reference-only={sorted(set(expected) - set(actual))[:4]}"
        )
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError(f"gradient participation or shape disagreement for {name}")
    active = sum(entry["state"] == "active" for entry in expected.values())
    return active, len(expected) - active


def active_gradient_norms(record: dict, norms: dict) -> dict:
    parameters = validate_participation(record)
    if set(norms) - set(parameters):
        raise ValueError("gradient norms contain parameters outside the trainable inventory")
    active = {}
    for name, entry in parameters.items():
        value = norms.get(name)
        if value is not None and (type(value) not in {int, float} or not math.isfinite(value) or value < 0):
            raise ValueError(f"gradient norm for {name} must be finite and nonnegative")
        if entry["state"] == "active":
            if name not in norms or value is None:
                raise ValueError(f"active parameter {name} has no reduced gradient norm")
            active[name] = value
        elif name in norms and value != 0:
            raise ValueError(f"inactive parameter {name} has a nonzero gradient")
    return active


def validate_inactive_weights(record: dict, witnesses: dict) -> None:
    parameters = validate_participation(record)
    inactive = {name for name, entry in parameters.items() if entry["state"] == "inactive"}
    if not isinstance(witnesses, dict) or set(witnesses) != inactive:
        raise ValueError("inactive optimizer weights have incomplete coverage")
    for name, witness in witnesses.items():
        if not isinstance(witness, dict) or set(witness) != {"parameter", "master"}:
            raise ValueError(f"inactive optimizer weight witness is incomplete for {name}")
        for kind, values in witness.items():
            if not isinstance(values, dict) or set(values) != {"before", "after"}:
                raise ValueError(f"inactive {kind} weight witness is incomplete for {name}")
            for identity in values.values():
                digest = identity.get("sha256") if isinstance(identity, dict) else None
                if (
                    not isinstance(identity, dict)
                    or set(identity) != {"shape", "dtype", "sha256"}
                    or identity["shape"] != parameters[name]["shape"]
                    or not isinstance(identity["dtype"], str)
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError(f"inactive {kind} weight identity is invalid for {name}")
            if values["before"] != values["after"]:
                raise ValueError(f"inactive {kind} weights changed for {name}")


def validate_optimizer_participation(summary: dict, *, backend: str) -> dict | None:
    record = summary.get("participation")
    if record is None:
        if "inactive_weights" in summary:
            raise ValueError("inactive optimizer weights require participation evidence")
        return None
    validate_inactive_weights(record, summary.get("inactive_weights"))
    canonical = canonical_participation(record, backend=backend)
    active = {name: entry for name, entry in canonical.items() if entry["state"] == "active"}
    norms = summary["parameters"]
    if set(norms) != set(active):
        raise ValueError("optimizer norms do not cover exactly the active trainable inventory")
    for name, entry in active.items():
        if norms[name]["shape"] != entry["shape"]:
            raise ValueError(f"optimizer participation shape mismatch for {name}")
    return canonical
