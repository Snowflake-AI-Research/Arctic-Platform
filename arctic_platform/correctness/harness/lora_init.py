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
"""Versioned, traversal-independent CPU LoRA initialization and exact identity checks."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Mapping

RECIPE_VERSION = "arctic-correctness-lora-init-v1"
_DTYPES = {"float32", "float16", "bfloat16"}
_SEED_DOMAIN = "arctic-correctness-lora-tensor-seed-v1"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_name(name: str) -> str:
    """Strip only activation-checkpoint segments and the terminal default adapter."""
    if not isinstance(name, str) or not name or any(character.isspace() for character in name):
        raise ValueError("LoRA tensor name must be a non-empty dotted name without whitespace")
    parts = [part for part in name.split(".") if part != "_checkpoint_wrapped_module"]
    if any(not part for part in parts):
        raise ValueError(f"invalid LoRA tensor name: {name!r}")
    if len(parts) >= 4 and parts[-3] in {"lora_A", "lora_B"} and parts[-2] == "default":
        del parts[-2]
    if len(parts) < 3 or parts[-2] not in {"lora_A", "lora_B"} or parts[-1] != "weight":
        raise ValueError(f"unsupported LoRA tensor name: {name!r}; only default A/B weights are supported")
    if any(part.startswith("lora_") for part in parts[:-2]):
        raise ValueError(f"ambiguous LoRA tensor name: {name!r}")
    return ".".join(parts)


@dataclass(frozen=True)
class LoraTensorSpec:
    name: str
    shape: tuple[int, int]
    dtype: str

    def __post_init__(self) -> None:
        if canonical_name(self.name) != self.name:
            raise ValueError("LoRA tensor specs require canonical names")
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 2
            or any(type(size) is not int or size <= 0 for size in self.shape)
        ):
            raise ValueError("LoRA initialization supports positive, global two-dimensional A/B shapes")
        if self.dtype not in _DTYPES:
            raise ValueError(f"unsupported LoRA output dtype: {self.dtype!r}")


def _global_seed(seed: int) -> int:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("LoRA global seed must be an integer in [0, 2**63)")
    return seed


def tensor_seed(global_seed: int, name: str) -> int:
    """SHA256 of canonical JSON; first eight big-endian bytes masked to 63 bits."""
    if canonical_name(name) != name:
        raise ValueError("LoRA seed derivation requires a canonical tensor name")
    digest = hashlib.sha256(
        _json_bytes({"domain": _SEED_DOMAIN, "global_seed": _global_seed(global_seed), "name": name})
    ).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)


def generate_tensor(spec: LoraTensorSpec, global_seed: int):
    """Generate a complete global tensor on CPU in FP32, then explicitly cast."""
    import torch

    seed = tensor_seed(global_seed, spec.name)
    tensor = torch.empty(spec.shape, dtype=torch.float32, device="cpu")
    if spec.name.endswith(".lora_A.weight"):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.nn.init.kaiming_uniform_(
            tensor, a=math.sqrt(5.0), mode="fan_in", nonlinearity="leaky_relu", generator=generator
        )
    else:
        tensor.zero_()
    return tensor.to(dtype=getattr(torch, spec.dtype))


def tensor_sha256(name: str, tensor) -> str:
    """Hash name, shape, storage dtype and exact little-endian contiguous bytes."""
    import torch

    if canonical_name(name) != name:
        raise ValueError("LoRA tensor hashes require canonical names")
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type == "meta"
        or tensor.layout != torch.strided
        or str(tensor.dtype).removeprefix("torch.") not in _DTYPES
        or tensor.ndim != 2
        or any(size <= 0 for size in tensor.shape)
    ):
        raise ValueError(f"{name}: hash requires a materialized, dense supported A/B tensor")
    if sys.byteorder != "little":
        raise ValueError("LoRA recipe v1 requires a little-endian host")
    cpu = tensor.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(cpu).all()):
        raise ValueError(f"{name}: LoRA initialization must be finite")
    header = {"name": name, "shape": list(cpu.shape), "dtype": str(cpu.dtype).removeprefix("torch.")}
    digest = hashlib.sha256(_json_bytes(header) + b"\0")
    digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validated_specs(specs: Iterable[LoraTensorSpec]) -> list[LoraTensorSpec]:
    by_name = {}
    for spec in specs:
        if not isinstance(spec, LoraTensorSpec):
            raise ValueError("LoRA initialization requires LoraTensorSpec entries")
        if spec.name in by_name:
            raise ValueError(f"duplicate canonical LoRA tensor: {spec.name}")
        by_name[spec.name] = spec
    if not by_name:
        raise ValueError("LoRA initialization requires at least one complete A/B pair")
    for name, spec in by_name.items():
        prefix, factor, _ = name.rsplit(".", 2)
        partner_name = f"{prefix}.{'lora_B' if factor == 'lora_A' else 'lora_A'}.weight"
        partner = by_name.get(partner_name)
        if partner is None:
            raise ValueError(f"{name}: missing LoRA A/B partner")
        a, b = (spec, partner) if factor == "lora_A" else (partner, spec)
        if a.shape[0] != b.shape[1] or a.dtype != b.dtype:
            raise ValueError(f"{prefix}: LoRA A/B rank or dtype differs")
    return [by_name[name] for name in sorted(by_name)]


def build_manifest(specs: Iterable[LoraTensorSpec], global_seed: int) -> dict:
    """Return only JSON-compatible recipes and hashes, never tensor payloads."""
    import torch

    entries = []
    for spec in _validated_specs(specs):
        is_a = spec.name.endswith(".lora_A.weight")
        entries.append(
            {
                "name": spec.name,
                "shape": list(spec.shape),
                "dtype": spec.dtype,
                "seed": tensor_seed(global_seed, spec.name),
                "distribution": (
                    {
                        "name": "kaiming_uniform",
                        "a": math.sqrt(5.0),
                        "mode": "fan_in",
                        "nonlinearity": "leaky_relu",
                        "fan_in": spec.shape[1],
                    }
                    if is_a
                    else {"name": "zeros"}
                ),
                "sha256": tensor_sha256(spec.name, generate_tensor(spec, global_seed)),
            }
        )
    manifest = {
        "recipe_version": RECIPE_VERSION,
        "global_seed": _global_seed(global_seed),
        "seed_derivation": {"domain": _SEED_DOMAIN, "hash": "sha256", "encoding": "canonical-json-utf8"},
        "generator": {"device": "cpu", "algorithm": "torch-cpu-mt19937", "torch_version": str(torch.__version__)},
        "generation_dtype": "float32",
        "cast": "torch.Tensor.to(output_dtype)",
        "byte_order": "little",
        "tensors": entries,
    }
    manifest["identity_sha256"] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    return manifest


def manifest_specs(manifest: dict) -> list[LoraTensorSpec]:
    """Validate a recipe against this implementation before reconstructing it."""
    if not isinstance(manifest, dict) or manifest.get("recipe_version") != RECIPE_VERSION:
        raise ValueError("unsupported LoRA initialization recipe")
    entries = manifest.get("tensors")
    if not isinstance(entries, list):
        raise ValueError("LoRA initialization recipe requires tensor entries")
    try:
        specs = [
            LoraTensorSpec(entry["name"], tuple(entry["shape"]), entry["dtype"])
            for entry in entries
            if isinstance(entry, dict)
        ]
        if len(specs) != len(entries):
            raise ValueError("invalid LoRA initialization tensor entry")
        expected = build_manifest(specs, manifest["global_seed"])
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete LoRA initialization recipe") from error
    if _json_bytes(expected) != _json_bytes(manifest):
        raise ValueError("LoRA initialization recipe, runtime version, or tensor hash differs")
    return _validated_specs(specs)


def verify_tensors(
    named_tensors: Mapping[str, Any] | Iterable[tuple[str, Any]],
    manifest: dict,
    *,
    allow_exact_float32_widening: bool = False,
) -> dict:
    """Require exact A/B coverage and bytes; permit only explicitly requested lossless widening."""
    import torch

    specs = manifest_specs(manifest)
    actual = {}
    items = named_tensors.items() if isinstance(named_tensors, Mapping) else named_tensors
    for raw_name, tensor in items:
        name = canonical_name(raw_name)
        if name in actual:
            raise ValueError(f"duplicate canonical LoRA tensor: {name}")
        actual[name] = tensor
    expected_names = {spec.name for spec in specs}
    if set(actual) != expected_names:
        raise ValueError(
            f"LoRA initialization coverage differs: missing={sorted(expected_names - set(actual))}, "
            f"extra={sorted(set(actual) - expected_names)}"
        )
    verified = []
    for spec in specs:
        tensor = actual[spec.name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != spec.shape:
            raise ValueError(f"{spec.name}: LoRA initialization shape differs")
        expected = generate_tensor(spec, manifest["global_seed"])
        if tensor.dtype != expected.dtype:
            if (
                not allow_exact_float32_widening
                or tensor.dtype != torch.float32
                or expected.dtype not in (torch.bfloat16, torch.float16)
            ):
                raise ValueError(f"{spec.name}: LoRA initialization dtype differs")
            expected = expected.float()
        digest = tensor_sha256(spec.name, tensor)
        if digest != tensor_sha256(spec.name, expected):
            raise ValueError(f"{spec.name}: LoRA initialization bytes differ")
        verified.append({"name": spec.name, "dtype": str(tensor.dtype).removeprefix("torch."), "sha256": digest})
    return {"identity_sha256": manifest["identity_sha256"], "verified_tensors": verified}
