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

# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Complete trainable inventories and independent backward-participation witnesses."""

from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path

from .names import canonical_dss_tensor_groups
from .names import normalize

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
    if backend not in {"dss", "reference"}:
        raise ValueError(f"unknown participation backend: {backend}")
    parameters = validate_participation(record)
    normalized = {}
    for name, entry in parameters.items():
        key = normalize(name)
        if key in normalized:
            raise ValueError(f"participation inventory has duplicate canonical name {key}")
        normalized[key] = entry
    groups = canonical_dss_tensor_groups(normalized) if backend == "dss" else {name: None for name in normalized}
    covered = [name for group in groups.values() for name in group.originals] if backend == "dss" else list(groups)
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
    actual = canonical_participation(target, backend="dss")
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


def weight_identity(tensor) -> dict:
    from .optimizer_capture import tensor_digest

    if not bool(tensor.isfinite().all()):
        raise ValueError("inactive weight witness contains nonfinite values")
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": tensor_digest(tensor)}


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


def merge_rank_participation(before: list[dict], after: list[dict], shapes: dict, routes: dict) -> tuple[dict, dict]:
    """Union observed backward use across actual ranks, never infer it from zero storage."""
    if not before or len(before) != len(after):
        raise ValueError("participation capture requires every before/after rank")
    entries, witnesses = {}, {}
    for rank, (left, right) in enumerate(zip(before, after, strict=True)):
        initial = left.get("participation", {}).get("parameters")
        observed = right.get("participation", {}).get("parameters")
        if (
            left.get("participation", {}).get("schema") != PARTICIPATION_SCHEMA
            or right.get("participation", {}).get("schema") != PARTICIPATION_SCHEMA
            or not isinstance(initial, dict)
            or not isinstance(observed, dict)
            or set(initial) != set(shapes)
            or set(observed) != set(shapes)
        ):
            raise ValueError(f"rank {rank} participation inventory is incomplete")
        for name, shape in shapes.items():
            if initial[name]["shape"] != list(shape) or observed[name]["shape"] != list(shape):
                raise ValueError(f"rank {rank} participation shape drift for {name}")
            if initial[name]["state"] == "active":
                raise ValueError("participation must be armed before the first backward")
            state = observed[name]["state"]
            if state not in {"active", "inactive", "missing"}:
                raise ValueError(f"invalid rank participation for {name}")
            if state == "inactive":
                validate_participation({"schema": PARTICIPATION_SCHEMA, "parameters": {name: observed[name]}})
    for name, shape in shapes.items():
        observed = [row["participation"]["parameters"][name] for row in after]
        global_shape = list(shape)
        if len(routes[name]) > 1:
            if not global_shape:
                raise ValueError("expert participation requires a nonscalar parameter")
            global_shape[0] *= len(routes[name])
        if any(entry["state"] == "active" for entry in observed):
            entries[name] = {"shape": global_shape, "state": "active", "evidence": None}
            continue
        if any(entry["state"] != "inactive" for entry in observed):
            raise ValueError(f"unexpected missing gradient for {name} on every rank")
        if len(routes[name]) != 1:
            raise ValueError(f"inactive expert weights require global witnesses for {name}")
        identities = []
        for left, right in zip(before, after, strict=True):
            initial = left.get("candidate_weights", {}).get(name)
            final = right.get("candidate_weights", {}).get(name)
            if initial is None or final is None:
                raise ValueError(f"inactive parameter weight witness is incomplete for {name}")
            identities.append({"before": initial, "after": final})
        if any(value != identities[0] for value in identities):
            raise ValueError(f"inactive replicated parameter witnesses disagree for {name}")
        entries[name] = {**observed[0], "shape": global_shape}
        witnesses[name] = {"parameter": identities[0]}
    record = {"schema": PARTICIPATION_SCHEMA, "parameters": entries}
    validate_participation(record)
    return record, witnesses


class ParticipationCapture:
    """Observe actual autograd use; zero storage and Adam moments are not evidence."""

    def __init__(self, model, *, decoder=None, head=None):
        self.model = model
        self.parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.shapes = {name: list(parameter.shape) for name, parameter in self.parameters.items()}
        self.active = set()
        self.handles = []
        self.candidates = {}
        self.source_digests = {}
        self.forwards = []
        try:
            self._arm(model, decoder=decoder, head=head)
        except BaseException:
            self.close()
            raise

    def _arm(self, model, *, decoder, head):
        import torch

        reachable = (
            {id(parameter) for module in (decoder, head) for parameter in module.parameters()}
            if decoder is not None and head is not None
            else None
        )
        for name, parameter in self.parameters.items():
            self.handles.append(parameter.register_hook(self._backward_hook(name)))
        for name, module in model.named_modules():
            kind = None
            if type(module).__name__ == "Glm5NextTextIndexer":
                kind = "integer-selector"
            elif reachable is not None and normalize(name) == "visual":
                kind = "text-decoder-bypass"
            if kind is None:
                continue
            owned = {id(parameter) for parameter in module.parameters()}
            names = [parameter_name for parameter_name, parameter in self.parameters.items() if id(parameter) in owned]
            if kind == "text-decoder-bypass" and any(id(self.parameters[key]) in reachable for key in names):
                continue
            forward = inspect.unwrap(getattr(module.forward, "__func__", module.forward))
            source = inspect.getsourcefile(forward)
            if source is None:
                continue
            if source not in self.source_digests:
                self.source_digests[source] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
            self.forwards.append((module, forward))
            evidence = {
                "kind": kind,
                "module": name,
                "module_class": type(module).__name__,
                "forward_sha256": self.source_digests[source],
                "calls": 0,
                "integer_outputs_only": True,
                "outside_decoder_and_head": kind == "text-decoder-bypass",
            }

            def observe(_module, _inputs, output, *, evidence=evidence):
                evidence["calls"] += 1

                def integer_only(value):
                    if isinstance(value, torch.Tensor):
                        return not value.is_floating_point() and not value.is_complex() and not value.requires_grad
                    if isinstance(value, (tuple, list)):
                        return bool(value) and all(integer_only(item) for item in value)
                    return False

                evidence["integer_outputs_only"] &= integer_only(output)

            self.handles.append(module.register_forward_hook(observe))
            for parameter_name in names:
                self.candidates[parameter_name] = evidence

    def _backward_hook(self, name):
        def record(gradient):
            self.active.add(name)
            return gradient

        return record

    def result(self) -> dict:
        current = {name: parameter for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        if set(current) != set(self.parameters) or any(
            current[name] is not parameter or list(parameter.shape) != self.shapes[name]
            for name, parameter in self.parameters.items()
        ):
            raise ValueError("trainable participation inventory changed during capture")
        for source, digest in self.source_digests.items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
                raise ValueError("participation forward source drift")
        if any(
            inspect.unwrap(getattr(module.forward, "__func__", module.forward)) is not forward
            for module, forward in self.forwards
        ):
            raise ValueError("participation forward implementation changed during capture")
        entries = {}
        for name, parameter in self.parameters.items():
            evidence = self.candidates.get(name)
            justified = evidence is not None and (
                (evidence["kind"] == "text-decoder-bypass" and evidence["calls"] == 0)
                or (
                    evidence["kind"] == "integer-selector"
                    and evidence["calls"] > 0
                    and evidence["integer_outputs_only"]
                )
            )
            entries[name] = {
                "shape": list(parameter.shape),
                "state": "active" if name in self.active else "inactive" if justified else "missing",
                "evidence": dict(evidence) if justified and name not in self.active else None,
            }
        return {"schema": PARTICIPATION_SCHEMA, "parameters": entries}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()
