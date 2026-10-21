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

"""Compact Git references prepared explicitly during onboarding, never during regression."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import tempfile
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_RECORDS_DIR = Path(__file__).resolve().parents[1] / "reference_data"
CRITERIA = {
    "single-step-grads": "gradient-l2-norm-v1",
    "single-step-optimizer": "optimizer-state-l2-norm-v1",
    "e2e-sft-train-validate": "sft-trajectory-ce-v1",
}
MAX_RECORD_BYTES = 8 * 1024 * 1024
_REVIEWED_GRADIENT_NAMES_TRANSITION = {
    "historical": {"sha256": "a5c941eb9592fabe042c8f48a28bfe7e650f5b65d3b8b7a6fd3824397c63ba09", "size": 9905},
    "current": {"sha256": "4d6cd215ef4639ab2d85f5928ca2df83954012878c39366c74e29207cc92725c", "size": 10294},
}


class ReferencePrerequisiteError(ValueError):
    """A missing, incompatible or corrupt baseline is not a numerical mismatch."""


def _json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _digest(value) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


@lru_cache(maxsize=4096)
def _file_hash(path: str, size: int, mtime_ns: int, ctime_ns: int, inode: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path | str) -> dict:
    path = Path(path)
    stat = path.stat()
    sha = _file_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    current = path.stat()
    stable_fields = ("st_size", "st_mtime_ns", "st_ctime_ns", "st_ino", "st_dev")
    if any(getattr(current, name) != getattr(stat, name) for name in stable_fields):
        raise ReferencePrerequisiteError(f"reference source changed while hashing: {path}")
    return {"sha256": sha, "size": stat.st_size}


def model_identity(path: Path | str) -> dict:
    """Hash asset contents, not machine-specific roots or file sizes alone."""
    from .config import NON_MODEL_PROCESSOR_SIDECARS

    root = Path(path)
    if not root.is_dir():
        raise ReferencePrerequisiteError(f"reference model assets unavailable: {root}")
    assets = {
        str(item.relative_to(root)): file_identity(item)
        for item in sorted(root.rglob("*"))
        if item.is_file()
        and item.suffix in {".json", ".safetensors", ".bin", ".model", ".tiktoken", ".txt", ".py"}
        and item.name not in NON_MODEL_PROCESSOR_SIDECARS
    }
    if "config.json" not in assets or not any(name.endswith((".safetensors", ".bin")) for name in assets):
        raise ReferencePrerequisiteError(f"reference requires complete model config and weights: {root}")
    return {"assets": assets, "sha256": _digest(assets)}


def tensor_identity(tensor) -> dict:
    import torch

    value = tensor.detach().to(device="cpu").contiguous()
    metadata = {"shape": list(value.shape), "dtype": str(value.dtype)}
    digest = hashlib.sha256(_json(metadata))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {**metadata, "sha256": digest.hexdigest()}


def batch_identity(batch) -> dict:
    return {
        "name": batch.name,
        "active_tokens": batch.active_tokens,
        "tensors": {name: tensor_identity(getattr(batch, name)) for name in ("input_ids", "position_ids", "labels")},
    }


def workload_identity(ctx, check_id: str, recipe: dict) -> dict:
    """Bind HF semantics and inputs, excluding unrelated target source and local paths."""
    if check_id not in CRITERIA:
        raise ValueError(f"not an HF reference check: {check_id}")
    dependencies = {}
    for package in ("torch", "transformers", "peft", "safetensors", "flash-attn", "liger-kernel"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    # Distribution metadata can omit Torch's CUDA build suffix on an identical runtime.
    import torch

    dependencies["torch"] = str(torch.__version__)
    root = Path(__file__).resolve().parents[1]
    sources = {
        str(path.relative_to(root)): file_identity(path)
        for path in [
            *sorted((root / "reference").glob("*.py")),
            root / "harness" / "batches.py",
            root / "harness" / "sft.py",
            root / "harness" / "names.py",
            *(
                [root / "harness" / "shared_lora_adapter.py"]
                if isinstance(recipe.get("initialization"), dict)
                and recipe["initialization"].get("schema") == "shared-peft-adapter-v1"
                else []
            ),
        ]
    }
    identity = {
        "schema": SCHEMA_VERSION,
        "check_id": check_id,
        "criterion": CRITERIA[check_id],
        "config_id": ctx.config_id,
        "config_checksum": ctx.spec.config_checksum,
        "parameter_name_map": getattr(ctx.spec, "param_name_map", {}),
        "gpu_family": ctx.cfg.gpu_type,
        "reference_sources": sources,
        "dependencies": dependencies,
        "python": platform.python_version(),
        "recipe": recipe,
    }
    if check_id == "single-step-grads":
        identity["gate"] = asdict(ctx.spec.test_tolerances[check_id])
    elif check_id == "single-step-optimizer":
        identity["norm_reduction_source"] = file_identity(root / "checks" / "optimizer_state.py")
        identity["norm_schema_source"] = file_identity(root / "harness" / "names.py")
    elif check_id == "e2e-sft-train-validate":
        # Shared replay/scoring helpers affect the reference semantics. Older
        # records did not bind them, so do not silently relabel those measurements.
        identity["sft_source"] = file_identity(root / "harness" / "sft.py")
    return identity


def _finite(value, trail="measurements") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ReferencePrerequisiteError(f"nonfinite reference measurement: {trail}")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{trail}[{index}]")


def validate_measurements(identity: dict, measurements: dict) -> None:
    if not isinstance(measurements, dict):
        raise ReferencePrerequisiteError("reference measurements must be an object")
    _finite(measurements)
    check = identity["check_id"]
    if check in {"single-step-grads", "single-step-optimizer"}:
        for key in ("loss", "microbatches", "active_tokens"):
            if key not in measurements or not isinstance(measurements[key], (int, float)):
                raise ReferencePrerequisiteError(f"incomplete reference: missing {key}")
        norms = measurements.get("grad_norms")
        if (
            not isinstance(norms, dict)
            or not norms
            or any(
                not isinstance(name, str) or not isinstance(value, (int, float)) or value < 0
                for name, value in norms.items()
            )
        ):
            raise ReferencePrerequisiteError("reference gradient norms must be complete finite nonnegative scalars")
        if check == "single-step-optimizer":
            summary = measurements.get("optimizer_norms", {})
            gate = measurements.get("optimizer_norm_gate", {})
            if summary.get("criterion") != CRITERIA[check] or gate.get("criterion") != CRITERIA[check]:
                raise ReferencePrerequisiteError("optimizer reference requires the versioned norm criterion")
            provenance = gate.get("provenance")
            if (
                gate.get("status") != "reviewed"
                or not isinstance(provenance, (str, dict))
                or not provenance
                or (isinstance(provenance, str) and not provenance.strip())
            ):
                raise ReferencePrerequisiteError("optimizer norm gate has no onboarding review provenance")
            if type(gate.get("absolute")) not in {int, float} or gate["absolute"] < 0:
                raise ReferencePrerequisiteError("optimizer norm gate must be a finite nonnegative scalar")
            if not summary.get("parameters"):
                raise ReferencePrerequisiteError("optimizer reference has no parameter norms")
            from .names import validate_optimizer_norm_summary

            validate_optimizer_norm_summary(summary)
    elif check == "e2e-sft-train-validate":
        for key in ("train_losses", "train_gradient_norms"):
            values = measurements.get(key)
            if (
                not isinstance(values, list)
                or len(values) != 100
                or not all(isinstance(value, (int, float)) for value in values)
            ):
                raise ReferencePrerequisiteError(f"reference requires all 100 scalar {key}")
        if measurements.get("optimizer_steps") != 100:
            raise ReferencePrerequisiteError("reference requires 100 optimizer steps")
        if not isinstance(measurements.get("validation_cross_entropy"), (int, float)):
            raise ReferencePrerequisiteError("reference requires validation cross-entropy")
        if not isinstance(measurements.get("scored_answer_tokens"), int) or measurements["scored_answer_tokens"] <= 0:
            raise ReferencePrerequisiteError("reference requires positive scored-answer-token count")


def _record_path(ctx, identity: dict) -> Path:
    if (
        identity.get("check_id") not in CRITERIA
        or not isinstance(identity.get("config_id"), str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+", identity["config_id"]) is None
        or identity["config_id"] in {".", ".."}
    ):
        raise ReferencePrerequisiteError("invalid reference record config or check identity")
    root = Path(getattr(ctx, "reference_records_dir", DEFAULT_RECORDS_DIR))
    return root / identity["config_id"] / identity["check_id"] / f"{_digest(identity)}.json"


def require_git_records_dir(path: Path | str) -> Path:
    """CLI publication stays in the checkout; transient tensor artifacts cannot use this directory."""
    source_root = Path(__file__).resolve().parents[3]
    root = next((parent for parent in (source_root, *source_root.parents) if (parent / ".git").exists()), source_root)
    destination = Path(path).resolve()
    if not destination.is_relative_to(root) or destination.is_relative_to(root / ".git"):
        raise ReferencePrerequisiteError(f"compact references must belong to the Git checkout: {root}")
    return destination


def _strict_record_json(raw: bytes) -> dict:
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ReferencePrerequisiteError(f"duplicate historical reference JSON key: {key}")
            value[key] = item
        return value

    def invalid_constant(value):
        raise ReferencePrerequisiteError(f"nonfinite historical reference JSON value: {value}")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=invalid_constant)


def _load_historical_gradient_record(ctx, current_identity: dict, binding: dict) -> dict:
    """Reuse reviewed HF bytes without relabeling their identity or widening the gate."""
    try:
        if current_identity.get("check_id") != "single-step-grads":
            raise ReferencePrerequisiteError("historical reference binding supports gradients only")
        if not isinstance(binding, dict) or set(binding) != {"record_path", "record_sha256", "review"}:
            raise ReferencePrerequisiteError("historical reference requires record_path, record_sha256 and review")
        pin, review = binding["record_sha256"], binding["review"]
        if not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{64}", pin) is None:
            raise ReferencePrerequisiteError("historical reference requires a full lowercase raw SHA256")
        if not isinstance(review, str) or not review.strip():
            raise ReferencePrerequisiteError("historical reference requires explicit review provenance")
        _json(review)
        if not isinstance(binding["record_path"], (str, Path)) or not str(binding["record_path"]).strip():
            raise ReferencePrerequisiteError("historical reference requires an explicit record path")
        path = Path(binding["record_path"])
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            raw = stream.read(MAX_RECORD_BYTES + 1)
            after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in fields):
            raise ReferencePrerequisiteError("historical reference changed while reading")
        if len(raw) > MAX_RECORD_BYTES:
            raise ReferencePrerequisiteError("historical reference is oversized")
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ReferencePrerequisiteError("historical reference raw SHA256 mismatch")
        record = _strict_record_json(raw)
        if (
            not isinstance(record, dict)
            or set(record) != {"schema", "identity", "measurements", "payload_sha256"}
            or type(record["schema"]) is not int
            or record["schema"] != SCHEMA_VERSION
        ):
            raise ReferencePrerequisiteError("historical reference schema is invalid")
        historical_identity = record["identity"]
        if not isinstance(historical_identity, dict) or path.name != _record_path(ctx, historical_identity).name:
            raise ReferencePrerequisiteError("historical reference filename does not bind its original identity")
        if (
            type(historical_identity.get("schema")) is not int
            or historical_identity["schema"] != SCHEMA_VERSION
            or historical_identity.get("criterion") != CRITERIA["single-step-grads"]
        ):
            raise ReferencePrerequisiteError("historical reference workload schema or criterion is invalid")
        # This reviewed change affects DSS aliases only, not HF gradient measurements.
        historical_names = historical_identity["reference_sources"]["harness/names.py"]
        current_names = current_identity["reference_sources"]["harness/names.py"]
        transition = _REVIEWED_GRADIENT_NAMES_TRANSITION
        if _json(historical_names) != _json(transition["historical"]) or _json(current_names) != _json(
            transition["current"]
        ):
            raise ReferencePrerequisiteError("historical reference names.py transition is not reviewed")
        comparable = deepcopy(historical_identity)
        comparable["reference_sources"]["harness/names.py"] = deepcopy(current_names)
        if _json(comparable) != _json(current_identity):
            raise ReferencePrerequisiteError("historical reference differs in another workload field")
        if record["payload_sha256"] != _digest(record["measurements"]):
            raise ReferencePrerequisiteError("historical reference payload SHA256 mismatch")
        validate_measurements(historical_identity, record["measurements"])
        names_path = Path(__file__).resolve().with_name("names.py")
        if _json(file_identity(names_path)) != _json(current_names):
            raise ReferencePrerequisiteError("historical reference current names.py source mismatch")
    except ReferencePrerequisiteError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise ReferencePrerequisiteError(f"historical reference binding is invalid: {exc}") from exc
    key = _digest(current_identity)
    provenance = getattr(ctx, "reference_record_provenance", None)
    if provenance is None:
        provenance = {}
        ctx.reference_record_provenance = provenance
    provenance[key] = {
        "historical_identity": deepcopy(historical_identity),
        "current_identity": deepcopy(current_identity),
        "record_path": str(path),
        "record_sha256": pin,
        "review": deepcopy(review),
    }
    paths = getattr(ctx, "reference_record_paths", None)
    if paths is not None:
        paths[key] = str(path)
    print(f"Loaded explicitly reviewed historical HF reference: {path}", flush=True)
    return record["measurements"]


def load_record(ctx, current_identity: dict, *, historical_binding=None) -> dict:
    if historical_binding is not None:
        return _load_historical_gradient_record(ctx, current_identity, historical_binding)
    identity = current_identity
    path = _record_path(ctx, identity)
    selection = ["--test-id", identity["check_id"]]
    recipe = identity.get("recipe", {})
    if identity["check_id"] == "single-step-grads" and recipe.get("optimizer") is not None:
        selection = ["--reference-tests", "single-step-grads", "single-step-optimizer"]
    options = ["--reference-records-dir", str(getattr(ctx, "reference_records_dir", DEFAULT_RECORDS_DIR))]
    if getattr(ctx, "local_gateway_full_model", False):
        options.append("--local-gateway-full-model")
    if identity["check_id"] != "e2e-sft-train-validate" and recipe.get("token_budget") is not None:
        options.extend(["--reference-token-budget", str(recipe["token_budget"])])
    command = (
        "python -m arctic_platform.correctness onboard "
        f"--config {shlex.quote(str(ctx.spec.config_path))} --prepare-references "
        f"{shlex.join(selection + options)}"
    )
    try:
        if path.stat().st_size > MAX_RECORD_BYTES:
            raise ReferencePrerequisiteError(f"reference record is oversized: {path}")
        record = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ReferencePrerequisiteError(f"prepared HF reference missing: {path}; run {command}") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReferencePrerequisiteError(f"prepared HF reference corrupt: {path}; run {command}") from exc
    try:
        valid = (
            isinstance(record, dict)
            and record.get("schema") == SCHEMA_VERSION
            and record.get("identity") == identity
            and record.get("payload_sha256") == _digest(record.get("measurements"))
        )
    except (ValueError, TypeError) as exc:
        raise ReferencePrerequisiteError(f"prepared HF reference corrupt: {path}; run {command}") from exc
    if not valid:
        raise ReferencePrerequisiteError(f"prepared HF reference corrupt or incompatible: {path}; run {command}")
    try:
        validate_measurements(identity, record["measurements"])
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ReferencePrerequisiteError(
            f"prepared HF reference invalid measurements: {path}: {exc}; run {command}"
        ) from exc
    paths = getattr(ctx, "reference_record_paths", None)
    if paths is not None:
        paths[_digest(identity)] = str(path)
    print(f"Loaded prepared HF reference: {path}", flush=True)
    return record["measurements"]


def publish_record(ctx, identity: dict, measurements: dict) -> Path:
    validate_measurements(identity, measurements)
    data = _json(
        {
            "schema": SCHEMA_VERSION,
            "identity": identity,
            "measurements": measurements,
            "payload_sha256": _digest(measurements),
        }
    )
    if len(data) > MAX_RECORD_BYTES:
        raise ReferencePrerequisiteError("reference JSON exceeds compact record size limit")
    path = _record_path(ctx, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A hard link publishes complete bytes atomically without overwriting a concurrent writer.
    fd, temporary = tempfile.mkstemp(prefix=".prepare-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ReferencePrerequisiteError(f"conflicting immutable reference publication: {path}") from None
    finally:
        Path(temporary).unlink(missing_ok=True)
    paths = getattr(ctx, "reference_record_paths", None)
    if paths is not None:
        paths[_digest(identity)] = str(path)
    print(f"Prepared HF reference: {path} ({len(data)} bytes)", flush=True)
    return path
