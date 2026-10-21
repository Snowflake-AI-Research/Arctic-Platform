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

"Shared tensor names and legacy E2E reference adapter recipes."

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Mapping

from .batches import SEED
from .storage import require_data_fast_path

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


IDENTITY_FILENAME = "initialization.json"


def _metadata_sha256(metadata: dict) -> str:
    content = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


class LoraInitializationPrerequisiteError(RuntimeError):
    """The selected target cannot consume and prove the declared LoRA initialization."""


def materialize_adapter(cfg, model_path: str | Path, workdir: str | Path) -> Path | None:
    """Build shape metadata offline, then write an owned temporary PEFT pack under /data-fast."""
    peft_config = cfg.training.get("peft_config")
    if peft_config is None:
        return None

    from arctic_platform.common.peft import normalize_lora_peft_config

    peft_config = normalize_lora_peft_config(peft_config)
    root = require_data_fast_path(workdir, label="deterministic LoRA runtime directory")
    source = Path(model_path).expanduser().resolve(strict=True)
    if not source.is_dir() or not (source / "config.json").is_file():
        raise ValueError("deterministic LoRA requires a resolved local model directory with config.json")
    seed = cfg.sub_job.get("seed")
    seed = SEED if seed is None else seed
    dtype = cfg.sub_job.get("dtype") or cfg.training.get("dtype") or "bfloat16"
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"unsupported deterministic LoRA model dtype: {dtype!r}")
    config_digest = hashlib.sha256((source / "config.json").read_bytes()).hexdigest()

    import peft
    import torch
    import transformers
    from safetensors.torch import save_file

    config = transformers.AutoConfig.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    architectures = config.architectures
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("deterministic LoRA requires one explicit model architecture")
    model_class = getattr(transformers, architectures[0], None)
    if model_class is None:
        raise ValueError("deterministic LoRA model architecture is not locally supported")
    peft_options = dict(peft_config)
    # Older PEFT has no target_parameters argument; an empty list specifies no additional targets.
    if not peft_options["target_parameters"]:
        del peft_options["target_parameters"]
    # Allocate only topology on meta: no full base weights and no global RNG draws.
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        model = model_class(config).to(dtype=getattr(torch, dtype))
        model = peft.get_peft_model(model, peft.LoraConfig(**peft_options))
    specs = []
    raw_names = {}
    for raw_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        name = canonical_name(raw_name)
        if name in raw_names:
            raise ValueError(f"duplicate canonical LoRA tensor: {name}")
        raw_names[name] = raw_name
        specs.append(LoraTensorSpec(name, tuple(parameter.shape), str(parameter.dtype).removeprefix("torch.")))
    manifest = build_manifest(specs, seed)
    state = {raw_names[spec.name]: generate_tensor(spec, seed) for spec in manifest_specs(manifest)}
    verify_tensors(state, manifest)
    adapter_state = peft.get_peft_model_state_dict(model, state_dict=state, adapter_name="default")
    if set(adapter_state) != set(raw_names):
        raise ValueError("PEFT serialization changed the deterministic LoRA tensor coverage")
    verify_tensors(adapter_state, manifest)

    root.mkdir(parents=True, exist_ok=True)
    output = require_data_fast_path(
        tempfile.mkdtemp(prefix="deterministic-lora-", dir=root), label="deterministic LoRA temporary adapter"
    )
    model.peft_config["default"].save_pretrained(output)
    save_file(adapter_state, str(output / "adapter_model.safetensors"), metadata={"format": "pt"})
    metadata = {
        "schema_version": 1,
        "initialization": manifest,
        "model_name": cfg.model_name,
        "model_config_sha256": config_digest,
        "adapter_config_sha256": hashlib.sha256((output / "adapter_config.json").read_bytes()).hexdigest(),
        "peft_config": peft_config,
    }
    if hashlib.sha256((source / "config.json").read_bytes()).hexdigest() != config_digest:
        raise ValueError("deterministic LoRA model configuration changed during materialization")
    metadata["identity_sha256"] = _metadata_sha256(metadata)
    (output / IDENTITY_FILENAME).write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return output


def identity_metadata(adapter_path: str | Path) -> dict:
    """Validate the temporary pack and return compact, path-independent metadata."""
    from safetensors.torch import load_file

    root = require_data_fast_path(adapter_path, label="deterministic LoRA adapter identity")
    metadata = json.loads((root / IDENTITY_FILENAME).read_text())
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or not isinstance(metadata.get("initialization"), dict)
    ):
        raise ValueError("deterministic LoRA adapter requires initialization metadata")
    content = {key: value for key, value in metadata.items() if key != "identity_sha256"}
    if _metadata_sha256(content) != metadata.get("identity_sha256"):
        raise ValueError("deterministic LoRA adapter identity metadata changed")
    config_digest = hashlib.sha256((root / "adapter_config.json").read_bytes()).hexdigest()
    if config_digest != metadata.get("adapter_config_sha256"):
        raise ValueError("deterministic LoRA adapter configuration changed")
    verify_tensors(load_file(str(root / "adapter_model.safetensors"), device="cpu"), metadata["initialization"])
    return metadata


adapter_identity = identity_metadata


def bind_payload(payload: dict, adapter_path: str | Path | None) -> dict:
    """Leave non-LoRA requests untouched; never invent an unsupported target init field."""
    if adapter_path is None:
        if (payload.get("training_config") or {}).get("peft_config") is not None:
            raise LoraInitializationPrerequisiteError("LoRA target requires a deterministic adapter initialization")
        return payload
    raise LoraInitializationPrerequisiteError(
        "deterministic LoRA target initialization is unavailable through the supplied DSS gateway: "
        "no supported public training adapter-load/init route; checkpoint save and zone resumable load "
        "do not bind standalone PEFT adapters. A verified public target initialization route is required "
        "before measuring LoRA regression; no target job was initialized."
    )
