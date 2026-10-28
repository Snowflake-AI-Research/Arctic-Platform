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

"""Deterministic LoRA initialization, shared adapters, and native lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import math
import os
import select
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import MethodType
from typing import Any
from typing import Iterable
from typing import Mapping

from arctic_platform.common.peft import normalize_lora_peft_config
from arctic_platform.correctness.harness.sft import SEED
from arctic_platform.correctness.harness.storage import correctness_workdir
from arctic_platform.correctness.harness.storage import require_data_fast_path

# Lora

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


def _canonicalize_adapter_config(path: Path) -> None:
    config = json.loads(path.read_text())
    modules = config["target_modules"]
    if isinstance(modules, list):
        if any(not isinstance(name, str) or not name for name in modules):
            raise ValueError("PEFT target_modules must contain only non-empty strings")
        # PEFT stores module names as a set; process hash seeds must not affect pack identity.
        config["target_modules"] = sorted(modules)
    elif not isinstance(modules, str) or not modules:
        raise ValueError("PEFT target_modules must be a string or list of strings")
    path.write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n")


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
    _canonicalize_adapter_config(output / "adapter_config.json")
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


lora_adapter_identity = identity_metadata


def bind_payload(payload: dict, adapter_path: str | Path | None, *, native_backend=None) -> dict:
    """Admit the owned native adapter route explicitly; unsupported callers remain rejected."""
    if adapter_path is None:
        if (payload.get("training_config") or {}).get("peft_config") is not None:
            raise LoraInitializationPrerequisiteError("LoRA target requires a deterministic adapter initialization")
        return payload
    if native_backend is not None:
        pass
        pass

        root = Path(adapter_path)
        if (root / "initialization.json").exists():
            return bind_original_adapter(payload, root, native_backend)
        return bind_shared_adapter(payload, root, native_backend)
    raise LoraInitializationPrerequisiteError(
        "deterministic LoRA target initialization is unavailable through the supplied DSS gateway: "
        "no supported public training adapter-load/init route; checkpoint save and zone resumable load "
        "do not bind standalone PEFT adapters. A verified public target initialization route is required "
        "before measuring LoRA regression; no target job was initialized."
    )


# Shared

_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "correctness_recipe.json")
_ORIGINAL_ADAPTER_FILES = (*_ADAPTER_FILES, "initialization.json")
_NATIVE_ADAPTER_SCOPE_LOCK = threading.Lock()


def _topology(config: dict) -> dict:
    if config.get("peft_type") not in {"Lora", "LORA"}:
        raise ValueError("shared adapter must be a standard LoRA adapter")
    for key in (
        "modules_to_save",
        "use_dora",
        "use_rslora",
        "lora_bias",
        "rank_pattern",
        "alpha_pattern",
        "fan_in_fan_out",
        "layer_replication",
        "layers_to_transform",
        "layers_pattern",
        "exclude_modules",
        "use_qalora",
    ):
        if config.get(key):
            raise ValueError(f"shared adapter does not support {key}")
    if config.get("init_lora_weights", True) not in (True, False, "gaussian"):
        raise ValueError("shared adapter does not support initializers that change base weights")
    keys = ("task_type", "r", "lora_alpha", "lora_dropout", "bias", "target_modules", "target_parameters")
    selected = {key: config[key] for key in keys if key in config}
    selected["peft_type"] = "Lora"
    if isinstance(selected.get("target_modules"), (list, set, frozenset)):
        selected["target_modules"] = sorted(selected["target_modules"])
    if selected.get("target_parameters") is None:
        selected["target_parameters"] = []
    topology = normalize_lora_peft_config(selected)
    if topology["target_parameters"]:
        raise ValueError("shared adapter currently supports dense target_modules only")
    return topology


def _tensor_identity(state: dict) -> list[dict]:
    import torch

    pass

    entries = []
    names = set()
    for name, tensor in sorted(state.items()):
        canonical = canonical_name(name)
        if canonical in names:
            raise ValueError("shared adapter contains duplicate canonical tensors")
        names.add(canonical)
        if (
            tensor.ndim != 2
            or min(tensor.shape) <= 0
            or tensor.dtype
            not in {
                torch.float32,
                torch.float16,
                torch.bfloat16,
            }
        ):
            raise ValueError(f"shared adapter has unsupported tensor storage: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"shared adapter has nonfinite values: {name}")
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        entries.append(
            {
                "name": canonical,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if not entries:
        raise ValueError("shared adapter contains no tensors")
    indexed = {entry["name"]: entry for entry in entries}
    for entry in entries:
        prefix, factor, _ = entry["name"].rsplit(".", 2)
        partner = indexed.get(f"{prefix}.{'lora_B' if factor == 'lora_A' else 'lora_A'}.weight")
        if partner is None:
            raise ValueError("shared adapter is missing an A/B partner")
        a, b = (entry, partner) if factor == "lora_A" else (partner, entry)
        if a["shape"][0] != b["shape"][1] or a["dtype"] != b["dtype"]:
            raise ValueError("shared adapter has incompatible A/B shapes or dtypes")
    return sorted(entries, key=lambda entry: entry["name"])


def _read_adapter(path: str | Path, peft_config: dict | None = None) -> tuple[dict, dict]:
    from safetensors.torch import load

    root = require_data_fast_path(path, label="shared PEFT adapter").resolve(strict=True)
    config_bytes = (root / "adapter_config.json").read_bytes()
    tensor_bytes = (root / "adapter_model.safetensors").read_bytes()
    topology = _topology(json.loads(config_bytes))
    if peft_config is not None and topology != _topology(peft_config):
        raise ValueError("shared adapter topology differs from the training configuration")
    state = load(tensor_bytes)
    identity = {
        "schema": "shared-peft-adapter-v1",
        "peft_config": topology,
        "adapter_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "adapter_model_sha256": hashlib.sha256(tensor_bytes).hexdigest(),
        "tensors": _tensor_identity(state),
    }
    recipe_path = root / "correctness_recipe.json"
    if recipe_path.exists():
        identity["generation"] = json.loads(recipe_path.read_bytes())
    return state, identity


def shared_adapter_identity(path: str | Path, peft_config: dict | None = None) -> dict:
    """Content identity independent of the adapter's filesystem location."""
    return _read_adapter(path, peft_config)[1]


def generate_adapter(model_path: str | Path, peft_config: dict, destination: str | Path, *, seed: int = 42) -> Path:
    """Recreate upstream-default CPU factors without loading base-model weight tensors."""
    import peft
    import safetensors
    import torch
    import transformers
    from transformers import AutoConfig

    topology = _topology(peft_config)
    if peft_config.get("init_lora_weights", True) is not True:
        raise ValueError("generated adapter requires upstream default init_lora_weights=True")
    if type(seed) is not int or seed < 0:
        raise ValueError("shared adapter seed must be a nonnegative integer")
    root = require_data_fast_path(destination, label="generated PEFT adapter")
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_class = getattr(transformers, model_config.architectures[0])
    config = {key: value for key, value in topology.items() if key not in {"peft_type", "target_parameters"}}
    # The base architecture lives on meta; only A/B factors ever allocate real storage.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        with torch.device("meta"):
            model = peft.get_peft_model(model_class(model_config).to(dtype=torch.float32), peft.LoraConfig(**config))
        factors = []
        for module in model.modules():
            if isinstance(module, peft.tuners.lora.layer.LoraLayer):
                if "default" not in module.lora_A or "default" not in module.lora_B:
                    raise ValueError("generated adapter requires dense LoRA A/B factors")
                module.lora_A["default"].to_empty(device="cpu")
                module.lora_B["default"].to_empty(device="cpu")
                module.reset_lora_parameters("default", init_lora_weights=True)
                factors.append(module)
        if not factors:
            raise ValueError("generated adapter has no matching target modules")
        root.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(root, safe_serialization=True, save_embedding_layers=False)
    # PEFT otherwise serializes an absolute cache path, making identities node-dependent.
    config_path = root / "adapter_config.json"
    serialized = json.loads(config_path.read_bytes())
    serialized["base_model_name_or_path"] = ""
    if isinstance(serialized.get("target_modules"), list):
        serialized["target_modules"] = sorted(serialized["target_modules"])
    config_path.write_text(json.dumps(serialized, sort_keys=True, indent=2) + "\n")
    recipe = {
        "schema": "upstream-default-lora-v1",
        "seed": seed,
        "initializer": True,
        "peft_config": topology,
        "dependencies": {
            "torch": str(torch.__version__),
            "peft": peft.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
        },
    }
    (root / "correctness_recipe.json").write_text(json.dumps(recipe, sort_keys=True, indent=2) + "\n")
    shared_adapter_identity(root, peft_config)
    return root


def adapter_files(path: str | Path, expected_identity: dict) -> dict[str, bytes]:
    """Read a verified pack for the allocation's existing Ray object transport."""
    root = require_data_fast_path(path, label="shared PEFT adapter")
    if shared_adapter_identity(root) != expected_identity:
        raise ValueError("shared adapter changed before transfer")
    files = {name: (root / name).read_bytes() for name in _ADAPTER_FILES if (root / name).is_file()}
    if (
        hashlib.sha256(files["adapter_config.json"]).hexdigest() != expected_identity["adapter_config_sha256"]
        or hashlib.sha256(files["adapter_model.safetensors"]).hexdigest() != expected_identity["adapter_model_sha256"]
        or (json.loads(files["correctness_recipe.json"]) if "correctness_recipe.json" in files else None)
        != expected_identity.get("generation")
        or shared_adapter_identity(root) != expected_identity
    ):
        raise ValueError("shared adapter changed during transfer")
    return files


def _verify_original_adapter(path: Path, expected_identity: dict, expected_original_metadata: dict) -> dict:
    pass

    metadata = identity_metadata(path)
    if metadata != expected_original_metadata:
        raise ValueError("original adapter initialization metadata differs")
    if (
        shared_adapter_identity(path) != expected_identity
        or _topology(metadata["peft_config"]) != expected_identity["peft_config"]
    ):
        raise ValueError("original adapter shared identity or topology differs")
    return metadata


def original_adapter_files(
    path: str | Path, expected_identity: dict, expected_original_metadata: dict
) -> dict[str, bytes]:
    """Transfer the original deterministic pack, including its authenticated recipe."""
    root = require_data_fast_path(path, label="original deterministic PEFT adapter")
    _verify_original_adapter(root, expected_identity, expected_original_metadata)
    files = adapter_files(root, expected_identity)
    files["initialization.json"] = (root / "initialization.json").read_bytes()
    if json.loads(files["initialization.json"]) != expected_original_metadata:
        raise ValueError("original adapter initialization metadata changed during transfer")
    _verify_original_adapter(root, expected_identity, expected_original_metadata)
    return files


@contextlib.contextmanager
def native_worker_adapter_scope(
    training_module,
    files: dict[str, bytes],
    expected_identity: dict,
    expected_original_metadata: dict,
    *,
    job_id: str,
    rank: int,
):
    """Wrap the actual worker's imported engine initializer before optimizer creation."""
    pass

    if not isinstance(job_id, str) or not job_id.strip() or type(rank) is not int or rank < 0:
        raise ValueError("native adapter scope requires an explicit job identity and nonnegative worker rank")
    if (
        not isinstance(files, dict)
        or not {"adapter_config.json", "adapter_model.safetensors", "initialization.json"}
        <= set(files)
        <= set(_ORIGINAL_ADAPTER_FILES)
        or any(not isinstance(raw, bytes) for raw in files.values())
    ):
        raise ValueError("original adapter transfer must contain only original pack bytes")
    received = dict(files)
    shared_identity = deepcopy(expected_identity)
    original_metadata = deepcopy(expected_original_metadata)
    original = training_module.initialize_training_engine
    signature = inspect.signature(original)
    if not {"model", "rank"} <= set(signature.parameters):
        raise ValueError("native engine initializer must expose model and rank arguments")
    if not _NATIVE_ADAPTER_SCOPE_LOCK.acquire(blocking=False):
        raise ValueError("native adapter worker scope is already active")
    scratch = None
    installed = False
    calls = 0
    completed = None
    receipt = {
        "schema": "original-peft-worker-initialization-v1",
        "job_id": job_id,
        "rank": rank,
        "engine_initializations": 0,
    }
    try:
        scratch = correctness_workdir("original-peft-worker-")
        with tempfile.TemporaryDirectory(prefix="adapter-", dir=scratch) as temporary:
            path = Path(temporary)
            for name, raw in received.items():
                (path / name).write_bytes(raw)
            _verify_original_adapter(path, shared_identity, original_metadata)

            def check_binding():
                if training_module.initialize_training_engine is not initialize_training_engine:
                    raise ValueError("native engine initializer binding changed during adapter scope")

            @wraps(original)
            def initialize_training_engine(*args, **kwargs):
                nonlocal calls, completed

                from peft import get_peft_model_state_dict

                calls += 1
                if calls != 1:
                    raise ValueError("native adapter engine initialization must be single-use per worker")
                check_binding()
                bound = signature.bind(*args, **kwargs)
                if type(bound.arguments.get("rank")) is not int or bound.arguments["rank"] != rank:
                    raise ValueError("native adapter engine rank differs from the selected worker")
                model = bound.arguments["model"]
                _verify_original_adapter(path, shared_identity, original_metadata)
                shared_initialize_adapter(model, path, shared_identity)
                _verify_original_adapter(path, shared_identity, original_metadata)
                verified = verify_tensors(
                    get_peft_model_state_dict(model, adapter_name="default"), original_metadata["initialization"]
                )
                check_binding()
                result = original(*args, **kwargs)
                check_binding()
                _verify_original_adapter(path, shared_identity, original_metadata)
                completed = {
                    "schema": "original-peft-worker-initialization-v1",
                    "job_id": job_id,
                    "rank": rank,
                    "engine_initializations": calls,
                    "original_identity_sha256": original_metadata["identity_sha256"],
                    "shared_identity": deepcopy(shared_identity),
                    "initialization": verified,
                }
                receipt.clear()
                receipt.update(deepcopy(completed))
                return result

            if training_module.initialize_training_engine is not original:
                raise ValueError("native engine initializer binding changed before adapter scope")
            training_module.initialize_training_engine = initialize_training_engine
            installed = True
            yield receipt
            check_binding()
            if calls != 1 or completed is None:
                raise ValueError("native adapter worker requires exactly one successful engine initialization")
            if receipt != completed:
                raise ValueError("native adapter worker initialization receipt changed")
            _verify_original_adapter(path, shared_identity, original_metadata)
    finally:
        # Restore without re-validating source: the constructor may have raised or changed the binding.
        try:
            if installed:
                training_module.initialize_training_engine = original
        finally:
            try:
                if scratch is not None:
                    scratch.rmdir()
            finally:
                _NATIVE_ADAPTER_SCOPE_LOCK.release()


@contextlib.contextmanager
def worker_adapter_scope(object_ref, expected_identity: dict):
    """Receive identical bytes on each worker; no coordinator filesystem access."""
    import ray

    files = ray.get(object_ref)
    if (
        not isinstance(files, dict)
        or not {"adapter_config.json", "adapter_model.safetensors"} <= set(files) <= set(_ADAPTER_FILES)
        or any(not isinstance(value, bytes) for value in files.values())
    ):
        raise ValueError("shared adapter transfer must contain only standard pack bytes")
    scratch = correctness_workdir("shared-peft-worker-")
    try:
        with tempfile.TemporaryDirectory(prefix="adapter-", dir=scratch) as temporary:
            path = Path(temporary)
            for name, raw in files.items():
                (path / name).write_bytes(raw)
            if shared_adapter_identity(path) != expected_identity:
                raise ValueError("transferred shared adapter identity differs")
            with adapter_build_scope(path, expected_identity):
                yield
    finally:
        scratch.rmdir()


def shared_initialize_adapter(model, path: str | Path, expected_identity: dict) -> None:
    """Copy adapter tensors before optimizer construction, then verify their exact bytes."""
    from peft import get_peft_model_state_dict
    from peft import set_peft_model_state_dict

    state, identity = _read_adapter(path)
    if identity != expected_identity:
        raise ValueError("shared adapter changed after reference admission")
    configs = getattr(model, "peft_config", {})
    if set(configs) != {"default"} or _topology(configs["default"].to_dict()) != identity["peft_config"]:
        raise ValueError("target PEFT topology differs from the shared adapter")
    current = get_peft_model_state_dict(model, adapter_name="default")
    before = _tensor_identity(current)
    expected = identity["tensors"]
    if [{key: entry[key] for key in ("name", "shape", "dtype")} for entry in before] != [
        {key: entry[key] for key in ("name", "shape", "dtype")} for entry in expected
    ]:
        raise ValueError("shared adapter target tensor coverage, shapes or dtypes differ")
    # PEFT expects serialized names without the terminal adapter-name segment.
    result = set_peft_model_state_dict(model, state, adapter_name="default")
    if result.unexpected_keys:
        raise ValueError(f"shared adapter load returned unexpected keys: {result.unexpected_keys}")
    if _tensor_identity(get_peft_model_state_dict(model, adapter_name="default")) != expected:
        raise ValueError("shared adapter target tensors differ after loading")
    if shared_adapter_identity(path) != expected_identity:
        raise ValueError("shared adapter changed during loading")


@contextlib.contextmanager
def adapter_build_scope(path: str | Path, expected_identity: dict):
    """Apply the suite-owned loader only inside a correctness worker's initialization."""
    from arctic_platform.common import deepspeed_worker

    original = deepspeed_worker.build_model

    def build_model(*args, **kwargs):
        loaded = original(*args, **kwargs)
        shared_initialize_adapter(loaded.model, path, expected_identity)
        return loaded

    deepspeed_worker.build_model = build_model
    try:
        yield
    finally:
        deepspeed_worker.build_model = original


# Native


def _adapter_parameters(model) -> dict:
    import torch
    from peft.tuners.lora.layer import LoraLayer

    pass

    parameters = {}
    seen = set()
    storage_seen = set()
    base_parameters = set()
    base_storage = set()
    factors = {"lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_magnitude_vector"}
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if factors.intersection(name.split(".")):
            raise ValueError("shared adapter requires registered default dense A/B parameters")
        if buffer.device.type != "meta" and buffer.layout == torch.strided:
            base_storage.add((buffer.device, buffer.untyped_storage().data_ptr()))
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parts = name.split(".")
        if not factors.intersection(parts):
            base_parameters.add(id(parameter))
            if parameter.device.type != "meta" and parameter.layout == torch.strided:
                base_storage.add((parameter.device, parameter.untyped_storage().data_ptr()))
            continue
        if len(parts) < 4 or parts[-3] not in {"lora_A", "lora_B"} or parts[-2:] != ["default", "weight"]:
            raise ValueError("shared adapter requires registered default dense A/B parameters")
        layer = model.get_submodule(".".join(parts[:-3]))
        if (
            not isinstance(layer, LoraLayer)
            or set(layer.lora_A) != {"default"}
            or set(layer.lora_B) != {"default"}
            or getattr(layer, parts[-3])["default"].weight is not parameter
        ):
            raise ValueError("shared adapter requires registered default dense A/B parameters")
        canonical = canonical_name(name)
        if canonical in parameters or id(parameter) in seen:
            raise ValueError("shared adapter contains duplicate canonical tensors or aliased parameters")
        if parameter.device.type == "meta" or parameter.layout != torch.strided or not parameter.is_contiguous():
            raise ValueError(f"shared adapter requires materialized contiguous target storage: {name}")
        storage = (parameter.device, parameter.untyped_storage().data_ptr())
        if storage in storage_seen:
            raise ValueError("shared adapter contains aliased parameter storage")
        parameters[canonical] = parameter
        seen.add(id(parameter))
        storage_seen.add(storage)
    if seen.intersection(base_parameters) or storage_seen.intersection(base_storage):
        raise ValueError("shared adapter parameters alias base-model storage")
    return parameters


def initialize_native_adapter(model, path: str | Path, expected_identity: dict) -> None:
    """Transactionally copy exact native factors before optimizer construction."""
    import torch

    pass

    state, identity = _read_adapter(path)
    if identity != expected_identity:
        raise ValueError("shared adapter changed after reference admission")
    configs = getattr(model, "peft_config", {})
    if set(configs) != {"default"} or _topology(configs["default"].to_dict()) != identity["peft_config"]:
        raise ValueError("target PEFT topology differs from the shared adapter")
    # Checkpoint wrappers strip state-dict paths but retain registered parameter paths.
    current = _adapter_parameters(model)
    before = _tensor_identity(current)
    expected = identity["tensors"]
    if [{key: entry[key] for key in ("name", "shape", "dtype")} for entry in before] != [
        {key: entry[key] for key in ("name", "shape", "dtype")} for entry in expected
    ]:
        raise ValueError("shared adapter target tensor coverage, shapes or dtypes differ")
    if shared_adapter_identity(path) != expected_identity:
        raise ValueError("shared adapter changed before loading")
    admitted = {canonical_name(name): tensor for name, tensor in state.items()}
    originals = {name: parameter.detach().clone() for name, parameter in current.items()}
    try:
        with torch.no_grad():
            for name, parameter in current.items():
                parameter.copy_(admitted[name])
        installed = _adapter_parameters(model)
        if (
            set(installed) != set(current)
            or any(installed[name] is not parameter for name, parameter in current.items())
            or _tensor_identity(installed) != expected
        ):
            raise ValueError("shared adapter target tensors differ after loading")
        if shared_adapter_identity(path) != expected_identity:
            raise ValueError("shared adapter changed during loading")
    except BaseException as primary:
        # Restore every factor even when a copy or postflight fails partway through.
        with torch.no_grad():
            for name, parameter in current.items():
                try:
                    parameter.copy_(originals[name])
                except BaseException as rollback:
                    primary.add_note(f"shared adapter factor rollback failed for {name}: {rollback!r}")
        try:
            restored = _adapter_parameters(model)
            if (
                set(restored) != set(current)
                or any(restored[name] is not parameter for name, parameter in current.items())
                or _tensor_identity(restored) != before
            ):
                raise ValueError("shared adapter factor rollback identity differs")
        except BaseException as rollback:
            primary.add_note(f"shared adapter factor rollback verification failed: {rollback!r}")
        raise


# Native Lora

REQUEST_KEY = "correctness_original_peft_adapter"
SCHEMA = "correctness-original-peft-initialization-v1"
OPENAPI_KEY = "x-correctness-original-peft-initialization"
SHARED_REQUEST_KEY = "correctness_shared_peft_adapter"
SHARED_SCHEMA = "correctness-shared-peft-initialization-v1"
SHARED_OPENAPI_KEY = "x-correctness-shared-peft-initialization"
_MODULE = "arctic_platform.correctness.harness.lora"
_LISTENER_FD_OPTION = "--correctness-listener-fd"
_WORKER_SHUTDOWN_TIMEOUT = 30.0


def _source(callback) -> dict:
    path = inspect.getsourcefile(inspect.unwrap(callback))
    if path is None:
        raise ValueError("native adapter initialization requires a Python source origin")
    return {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def _verify_source(callback, expected: dict) -> None:
    if _source(callback) != expected:
        raise ValueError("native adapter initialization source changed")


def gateway_capability(backend, *, shared: bool = False) -> dict:
    """Discover the owned gateway's explicit capability before creating a target job."""
    import requests

    from arctic_platform.correctness.harness.gateway import require_local_cortex_gateway

    require_local_cortex_gateway(backend)
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(
            f"{backend.base_url.rstrip('/')}/openapi.json",
            timeout=30,
            allow_redirects=False,
        )
        if response.status_code != 200:
            raise ValueError("native adapter initialization requires the owned gateway's OpenAPI document")
        capability = response.json().get(SHARED_OPENAPI_KEY if shared else OPENAPI_KEY)
    if (
        not isinstance(capability, dict)
        or set(capability) != {"schema", "sources"}
        or capability["schema"] != (SHARED_SCHEMA if shared else SCHEMA)
        or not isinstance(capability["sources"], dict)
        or set(capability["sources"]) != {"training", "engine", "adapter", "lora", "composition", "native_adapter"}
        or any(
            not isinstance(value, dict)
            or set(value) != {"path", "sha256"}
            or not isinstance(value["path"], str)
            or not Path(value["path"]).is_absolute()
            or not isinstance(value["sha256"], str)
            or len(value["sha256"]) != 64
            for value in capability["sources"].values()
        )
    ):
        raise ValueError("supplied gateway does not support verified original PEFT initialization")
    return capability


def bind_original_adapter(payload: dict, path: Path, backend) -> dict:
    from arctic_platform.correctness.harness import lora

    adapter_identity = shared_adapter_identity
    pass

    capability = gateway_capability(backend)
    root = require_data_fast_path(path, label="native original PEFT adapter").resolve(strict=True)
    metadata = lora.identity_metadata(root)
    training = deepcopy(payload["training_config"])
    identity = adapter_identity(root, training["peft_config"])
    original_adapter_files(root, identity, metadata)
    debug = dict(training.get("debug") or {})
    if REQUEST_KEY in debug or SHARED_REQUEST_KEY in debug or debug.get("shared_peft_adapter") is not None:
        raise ValueError("native original adapter request cannot replace another initialization")
    debug[REQUEST_KEY] = {
        "schema": SCHEMA,
        "path": str(root),
        "identity": identity,
        "original_metadata": metadata,
        "sources": capability["sources"],
        "training_gpus": training["n_gpus"],
        "model_name": payload["model_name"],
    }
    training["debug"] = debug
    return {**payload, "training_config": training}


def _request(config) -> dict | None:
    raw = config.model_dump(mode="json", by_alias=True)
    request = ((raw.get("training_config") or {}).get("debug") or {}).get(REQUEST_KEY)
    if request is None:
        return None
    if (
        not isinstance(request, dict)
        or set(request)
        != {"schema", "path", "identity", "original_metadata", "sources", "training_gpus", "model_name"}
        or request["schema"] != SCHEMA
        or raw.get("job_type") != "training"
        or raw.get("model_name") != request["model_name"]
        or type(request["training_gpus"]) is not int
        or request["training_gpus"] <= 0
        or request["training_gpus"] != (raw.get("training_config") or {}).get("n_gpus")
        or raw.get("source_checkpoint_info") is not None
        or raw.get("resume_from_checkpoint") is not None
        or raw.get("resume_checkpoint_tag") is not None
        or SHARED_REQUEST_KEY in ((raw.get("training_config") or {}).get("debug") or {})
        or ((raw.get("training_config") or {}).get("debug") or {}).get("shared_peft_adapter") is not None
    ):
        raise ValueError("native original adapter requires one fresh, exactly selected training job")
    require_data_fast_path(request["path"], label="native original PEFT adapter")
    return deepcopy(request)


def _verify_shared_identity(identity: dict) -> None:
    generation = identity.get("generation") if isinstance(identity, dict) else None
    if (
        not isinstance(identity, dict)
        or identity.get("schema") != "shared-peft-adapter-v1"
        or not isinstance(generation, dict)
        or set(generation) != {"schema", "seed", "initializer", "peft_config", "dependencies"}
        or generation["schema"] != "upstream-default-lora-v1"
        or type(generation["seed"]) is not int
        or generation["seed"] < 0
        or generation["initializer"] is not True
        or generation["peft_config"] != identity.get("peft_config")
        or not isinstance(generation["dependencies"], dict)
        or set(generation["dependencies"]) != {"torch", "peft", "transformers", "safetensors"}
        or any(not isinstance(value, str) or not value for value in generation["dependencies"].values())
    ):
        raise ValueError("native shared adapter requires an authenticated upstream-default generation recipe")


def bind_shared_adapter(payload: dict, path: Path, backend) -> dict:
    """Bind an actual upstream-default pack without inventing deterministic metadata."""
    pass
    adapter_identity = shared_adapter_identity

    capability = gateway_capability(backend, shared=True)
    root = require_data_fast_path(path, label="native shared PEFT adapter").resolve(strict=True)
    if (root / "initialization.json").exists():
        raise ValueError("native shared adapter cannot contain deterministic initialization metadata")
    training = deepcopy(payload["training_config"])
    identity = adapter_identity(root, training["peft_config"])
    _verify_shared_identity(identity)
    adapter_files(root, identity)
    debug = dict(training.get("debug") or {})
    if REQUEST_KEY in debug or SHARED_REQUEST_KEY in debug or debug.get("shared_peft_adapter") is not None:
        raise ValueError("native shared adapter request cannot replace another initialization")
    debug[SHARED_REQUEST_KEY] = {
        "schema": SHARED_SCHEMA,
        "path": str(root),
        "identity": identity,
        "sources": capability["sources"],
        "training_gpus": training["n_gpus"],
        "model_name": payload["model_name"],
    }
    training["debug"] = debug
    return {**payload, "training_config": training}


def _shared_request(config) -> dict | None:
    raw = config.model_dump(mode="json", by_alias=True)
    training = raw.get("training_config") or {}
    debug = training.get("debug") or {}
    request = debug.get(SHARED_REQUEST_KEY)
    if request is None:
        return None
    if (
        not isinstance(request, dict)
        or set(request) != {"schema", "path", "identity", "sources", "training_gpus", "model_name"}
        or request["schema"] != SHARED_SCHEMA
        or raw.get("job_type") != "training"
        or raw.get("model_name") != request["model_name"]
        or type(request["training_gpus"]) is not int
        or request["training_gpus"] <= 0
        or request["training_gpus"] != training.get("n_gpus")
        or raw.get("source_checkpoint_info") is not None
        or raw.get("resume_from_checkpoint") is not None
        or raw.get("resume_checkpoint_tag") is not None
        or REQUEST_KEY in debug
        or debug.get("shared_peft_adapter") is not None
    ):
        raise ValueError("native shared adapter requires one fresh, exactly selected training job")
    require_data_fast_path(request["path"], label="native shared PEFT adapter")
    _verify_shared_identity(request["identity"])
    return deepcopy(request)


def _shared_runtime_sources(training) -> dict:
    from arctic_platform.correctness.harness import lora as native_adapter

    return {**_runtime_sources(training), "native_adapter": _source(native_adapter.initialize_native_adapter)}


def _verify_shared_runtime(training, sources: dict) -> None:
    if _shared_runtime_sources(training) != sources:
        raise ValueError("native shared adapter runtime differs from the selected gateway")


def _constructor_engine_binding(training, constructor) -> tuple[dict, object]:
    namespace = getattr(constructor, "__globals__", None)
    expected = training._DeepSpeedWorker.__init__
    if (
        not isinstance(namespace, dict)
        or namespace.get("__name__") != training.__name__
        or getattr(constructor, "__module__", None) != expected.__module__
        or getattr(constructor, "__qualname__", None) != expected.__qualname__
        or _source(constructor) != _source(expected)
        or getattr(constructor, "__code__", None) != getattr(expected, "__code__", None)
    ):
        raise ValueError("native shared adapter constructor namespace differs")
    engine = namespace.get("initialize_training_engine")
    expected_engine = training.initialize_training_engine
    if (
        not inspect.isfunction(engine)
        or getattr(engine, "__module__", None) != expected_engine.__module__
        or getattr(engine, "__qualname__", None) != expected_engine.__qualname__
        or _source(engine) != _source(expected_engine)
        or getattr(engine, "__code__", None) != getattr(expected_engine, "__code__", None)
        or getattr(engine, "_correctness_shared_native_adapter", False)
    ):
        raise ValueError("native shared adapter constructor engine binding differs")
    return namespace, engine


@contextlib.contextmanager
def _shared_adapter_directory():
    from arctic_platform.correctness.harness.storage import correctness_workdir

    scratch = None
    temporary = None
    primary = None
    try:
        scratch = correctness_workdir("shared-peft-native-worker-")
        temporary = tempfile.TemporaryDirectory(prefix="adapter-", dir=scratch)
        yield Path(temporary.name)
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup_failure = None
        callbacks = []
        if temporary is not None:
            callbacks.append(temporary.cleanup)
        if scratch is not None:
            callbacks.append(scratch.rmdir)
        for callback in callbacks:
            try:
                callback()
            except BaseException as cleanup:
                preserved = primary if primary is not None else cleanup_failure
                if preserved is not None:
                    preserved.add_note(
                        f"Shared adapter scratch cleanup also failed: {type(cleanup).__name__}: {cleanup}"
                    )
                else:
                    cleanup_failure = cleanup
        if cleanup_failure is not None:
            raise cleanup_failure


def native_shared_worker_adapter_scope(
    training_module,
    worker_constructor,
    files: dict[str, bytes],
    expected_identity: dict,
    *,
    job_id: str,
    rank: int,
):
    """Bind exact factors in the actual inherited constructor's engine namespace."""
    return _native_worker_adapter_scope(
        training_module, worker_constructor, files, expected_identity, job_id=job_id, rank=rank
    )


def native_original_worker_adapter_scope(
    training_module,
    worker_constructor,
    files: dict[str, bytes],
    expected_identity: dict,
    expected_original_metadata: dict,
    *,
    job_id: str,
    rank: int,
):
    """Load the original deterministic factors using registered native parameter paths."""
    return _native_worker_adapter_scope(
        training_module,
        worker_constructor,
        files,
        expected_identity,
        expected_original_metadata=expected_original_metadata,
        job_id=job_id,
        rank=rank,
    )


@contextlib.contextmanager
def _native_worker_adapter_scope(
    training_module,
    worker_constructor,
    files: dict[str, bytes],
    expected_identity: dict,
    *,
    expected_original_metadata: dict | None = None,
    job_id: str,
    rank: int,
):
    from arctic_platform.correctness.harness import lora as native_adapter
    from arctic_platform.correctness.harness import lora as shared

    pass

    if not isinstance(job_id, str) or not job_id.strip() or type(rank) is not int or rank < 0:
        raise ValueError("native shared adapter requires an explicit job identity and nonnegative worker rank")
    original_metadata = deepcopy(expected_original_metadata)
    if original_metadata is None:
        _verify_shared_identity(expected_identity)
        required_files = allowed_files = set(shared._ADAPTER_FILES)
    else:
        required_files = {"adapter_config.json", "adapter_model.safetensors", "initialization.json"}
        allowed_files = set(shared._ORIGINAL_ADAPTER_FILES)
    if (
        not isinstance(files, dict)
        or not required_files <= set(files) <= allowed_files
        or any(not isinstance(value, bytes) for value in files.values())
    ):
        raise ValueError("native shared adapter transfer must contain only standard generated pack bytes")
    namespace, original = _constructor_engine_binding(training_module, worker_constructor)
    signature = inspect.signature(original)
    if not {"model", "rank"} <= set(signature.parameters):
        raise ValueError("native shared adapter engine initializer must expose model and rank arguments")
    if not shared._NATIVE_ADAPTER_SCOPE_LOCK.acquire(blocking=False):
        raise ValueError("native shared adapter worker scope is already active")
    installed = False
    calls = 0
    completed = None
    identity = deepcopy(expected_identity)
    received = dict(files)
    schema = (
        "shared-peft-worker-initialization-v1"
        if original_metadata is None
        else "original-peft-worker-initialization-v1"
    )
    receipt = {
        "schema": schema,
        "job_id": job_id,
        "rank": rank,
        "engine_initializations": 0,
    }
    try:
        with _shared_adapter_directory() as path:
            for name, raw in received.items():
                (path / name).write_bytes(raw)

            def verify_pack():
                if original_metadata is not None:
                    shared._verify_original_adapter(path, identity, original_metadata)
                elif shared.shared_adapter_identity(path) != identity:
                    raise ValueError("native shared adapter identity changed")

            def verify_binding():
                if namespace.get("initialize_training_engine") is not initialize_training_engine:
                    raise ValueError("native shared adapter constructor engine binding changed during initialization")

            verify_pack()

            @wraps(original)
            def initialize_training_engine(*args, **kwargs):
                nonlocal calls, completed
                calls += 1
                if calls != 1:
                    raise ValueError("native shared adapter engine initialization must be single-use per worker")
                verify_binding()
                bound = signature.bind(*args, **kwargs)
                if type(bound.arguments.get("rank")) is not int or bound.arguments["rank"] != rank:
                    raise ValueError("native shared adapter engine rank differs from the selected worker")
                verify_pack()
                model = bound.arguments["model"]
                native_adapter.initialize_native_adapter(model, path, identity)
                verify_pack()
                verified = (
                    verify_tensors(native_adapter._adapter_parameters(model), original_metadata["initialization"])
                    if original_metadata is not None
                    else None
                )
                verify_binding()
                result = original(*args, **kwargs)
                verify_binding()
                verify_pack()
                completed = {
                    "schema": schema,
                    "job_id": job_id,
                    "rank": rank,
                    "engine_initializations": 1,
                    "shared_identity": deepcopy(identity),
                }
                if original_metadata is not None:
                    completed.update(
                        original_identity_sha256=original_metadata["identity_sha256"],
                        initialization=verified,
                    )
                receipt.clear()
                receipt.update(deepcopy(completed))
                return result

            initialize_training_engine._correctness_shared_native_adapter = True
            if namespace.get("initialize_training_engine") is not original:
                raise ValueError("native shared adapter constructor engine binding changed before installation")
            namespace["initialize_training_engine"] = initialize_training_engine
            installed = True
            yield receipt
            verify_binding()
            if calls != 1 or completed is None:
                raise ValueError("native shared adapter requires exactly one successful engine initialization")
            if receipt != completed:
                raise ValueError("native shared adapter worker initialization receipt changed")
            verify_pack()
    finally:
        try:
            if installed:
                namespace["initialize_training_engine"] = original
        finally:
            shared._NATIVE_ADAPTER_SCOPE_LOCK.release()


def _verify_shared_receipts(receipts: list, request: dict, child: str) -> None:
    if len(receipts) != request["training_gpus"]:
        raise ValueError("native shared adapter requires every worker's initialization receipt")
    for rank, receipt in enumerate(receipts):
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"schema", "job_id", "rank", "engine_initializations", "shared_identity"}
            or receipt.get("schema") != "shared-peft-worker-initialization-v1"
            or receipt.get("job_id") != child
            or type(receipt.get("rank")) is not int
            or receipt["rank"] != rank
            or type(receipt.get("engine_initializations")) is not int
            or receipt["engine_initializations"] != 1
            or receipt.get("shared_identity") != request["identity"]
        ):
            raise ValueError("native shared adapter worker initialization evidence differs")


def _runtime_sources(training) -> dict:
    from arctic_platform.correctness.harness import lora
    from arctic_platform.correctness.harness import lora as native_adapter
    from arctic_platform.correctness.harness import lora as shared_lora_adapter

    return {
        "training": _source(training._DeepSpeedWorker.__init__),
        "engine": _source(training.initialize_training_engine),
        "adapter": _source(shared_lora_adapter.native_worker_adapter_scope),
        "lora": _source(lora.identity_metadata),
        "composition": _source(_runtime_sources),
        "native_adapter": _source(native_adapter.initialize_native_adapter),
    }


def _gateway_adapter_sources(training) -> dict | None:
    try:
        return _runtime_sources(training)
    except AttributeError as error:
        if (
            error.obj is not training
            or error.name not in {"_DeepSpeedWorker", "initialize_training_engine"}
            or error.name in vars(training)
        ):
            raise
        logging.getLogger(__name__).warning(
            "Verified PEFT initialization unavailable: %s lacks %s; adapter capabilities are not advertised",
            training.__name__,
            error.name,
        )
        return None


def _verify_runtime(training, sources: dict) -> None:
    try:
        actual = _runtime_sources(training)
    except (AttributeError, TypeError, ValueError, OSError) as error:
        raise ValueError("native original adapter runtime differs from the selected gateway") from error
    if actual != sources:
        raise ValueError("native original adapter runtime differs from the selected gateway")


def _verify_receipts(receipts: list, request: dict, child: str) -> None:
    if len(receipts) != request["training_gpus"]:
        raise ValueError("native original adapter requires every worker's initialization receipt")
    manifest = request["original_metadata"]["initialization"]
    expected_tensors = [{key: tensor[key] for key in ("name", "dtype", "sha256")} for tensor in manifest["tensors"]]
    for rank, receipt in enumerate(receipts):
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != "original-peft-worker-initialization-v1"
            or receipt.get("job_id") != child
            or type(receipt.get("rank")) is not int
            or receipt["rank"] != rank
            or type(receipt.get("engine_initializations")) is not int
            or receipt["engine_initializations"] != 1
            or receipt.get("original_identity_sha256") != request["original_metadata"]["identity_sha256"]
            or receipt.get("shared_identity") != request["identity"]
            or not isinstance(receipt.get("initialization"), dict)
            or receipt["initialization"].get("identity_sha256") != manifest["identity_sha256"]
            or receipt["initialization"].get("verified_tensors") != expected_tensors
        ):
            raise ValueError("native original adapter worker initialization evidence differs")


def _spawn_worker_binding(spawn, training, sources: dict) -> tuple[dict, type]:
    # Ray's tracing wrapper delegates to the native body, which owns the worker lookup.
    native_spawn = inspect.unwrap(spawn)
    namespace = getattr(native_spawn, "__globals__", None)
    if (
        not inspect.iscoroutinefunction(spawn)
        or not inspect.iscoroutinefunction(native_spawn)
        or not isinstance(namespace, dict)
        or namespace.get("__name__") != training.__name__
    ):
        raise ValueError("native original adapter spawn namespace differs")
    _verify_source(native_spawn, sources["training"])
    worker = namespace.get("_DeepSpeedWorker")
    expected = training._DeepSpeedWorker
    if (
        not inspect.isclass(worker)
        or worker.__module__ != expected.__module__
        or worker.__qualname__ != expected.__qualname__
        or _source(worker.__init__) != sources["training"]
    ):
        raise ValueError("native original adapter spawn worker binding differs")
    return namespace, worker


def _coordinator_class(training, request: dict, files_ref):
    """Use native creation/placement/teardown with an initialization-only worker subclass."""
    import ray

    _verify_runtime(training, request["sources"])
    coordinator_base = training.GPUTrainingJob.__ray_metadata__.modified_class
    worker_base = training._DeepSpeedWorker

    class OriginalAdapterWorker(worker_base):
        def __init__(self, cfg, source_stage_info=None):
            import ray
            from dss.ray_dss.jobs.gpu import training as native_training

            _verify_runtime(native_training, request["sources"])
            rank = int(os.environ["RANK"])
            if cfg.get("job_id") is None or not 0 <= rank < request["training_gpus"]:
                raise ValueError("native original adapter worker has an invalid job or rank")
            with native_original_worker_adapter_scope(
                native_training,
                worker_base.__init__,
                ray.get(files_ref),
                request["identity"],
                request["original_metadata"],
                job_id=cfg["job_id"],
                rank=rank,
            ) as receipt:
                super().__init__(cfg, source_stage_info)
            _verify_runtime(native_training, request["sources"])
            self._correctness_original_adapter_receipt = deepcopy(receipt)

        def original_adapter_receipt(self):
            return deepcopy(self._correctness_original_adapter_receipt)

    class OriginalAdapterCoordinator(coordinator_base):
        async def _spawn_workers(self, cfg, ranks, master_addr, master_port, source_stage_info):
            from dss.ray_dss.jobs.gpu import training as native_training

            _verify_runtime(native_training, request["sources"])
            if self._workers or len(ranks) != request["training_gpus"] or _request(self._info.config) != request:
                raise ValueError("native original adapter coordinator layout or configuration differs")
            native_spawn = super()._spawn_workers
            namespace, original_worker = _spawn_worker_binding(native_spawn, native_training, request["sources"])
            # Ray can deserialize inherited methods with globals separate from the imported module.
            namespace["_DeepSpeedWorker"] = OriginalAdapterWorker
            try:
                try:
                    await native_spawn(cfg, ranks, master_addr, master_port, source_stage_info)
                    if namespace.get("_DeepSpeedWorker") is not OriginalAdapterWorker:
                        raise ValueError("native original adapter worker selection changed during creation")
                finally:
                    namespace["_DeepSpeedWorker"] = original_worker
                _verify_runtime(native_training, request["sources"])
                receipts = await asyncio.wait_for(
                    asyncio.gather(*(worker.original_adapter_receipt.remote() for worker in self._workers)),
                    timeout=60,
                )
                _verify_receipts(receipts, request, self._info.job_id)
                _verify_runtime(native_training, request["sources"])
                self._correctness_original_adapter_receipts = deepcopy(receipts)
                print(
                    json.dumps(
                        {"schema": SCHEMA, "job_id": self._info.job_id, "workers": receipts},
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    flush=True,
                )
            except BaseException as primary:
                # Native spawn handles its own failures; receipt failures also require its owned teardown.
                if self._workers:
                    try:
                        await self._kill_workers([host for host, _ in ranks])
                    except BaseException as cleanup:
                        primary.add_note(
                            f"Owned original-adapter worker teardown also failed: {type(cleanup).__name__}: {cleanup}"
                        )
                raise
            finally:
                namespace["_DeepSpeedWorker"] = original_worker

        def info(self):
            reported = super().info()
            reported["correctness_original_adapter"] = {
                "schema": SCHEMA,
                "workers": deepcopy(getattr(self, "_correctness_original_adapter_receipts", [])),
            }
            return reported

    return ray.remote(num_cpus=1, num_gpus=0)(OriginalAdapterCoordinator)


def _shared_coordinator_class(training, request: dict, files_ref):
    """Keep ordinary native placement and teardown for a separately identified shared pack."""
    import ray

    _verify_shared_runtime(training, request["sources"])
    coordinator_base = training.GPUTrainingJob.__ray_metadata__.modified_class
    worker_base = training._DeepSpeedWorker

    class SharedAdapterWorker(worker_base):
        def __init__(self, cfg, source_stage_info=None):
            import ray
            from dss.ray_dss.jobs.gpu import training as native_training

            _verify_shared_runtime(native_training, request["sources"])
            rank = int(os.environ["RANK"])
            if cfg.get("job_id") is None or not 0 <= rank < request["training_gpus"]:
                raise ValueError("native shared adapter worker has an invalid job or rank")
            with native_shared_worker_adapter_scope(
                native_training,
                worker_base.__init__,
                ray.get(files_ref),
                request["identity"],
                job_id=cfg["job_id"],
                rank=rank,
            ) as receipt:
                super().__init__(cfg, source_stage_info)
            _verify_shared_runtime(native_training, request["sources"])
            self._correctness_shared_adapter_receipt = deepcopy(receipt)

        def shared_adapter_receipt(self):
            return deepcopy(self._correctness_shared_adapter_receipt)

    class SharedAdapterCoordinator(coordinator_base):
        async def _spawn_workers(self, cfg, ranks, master_addr, master_port, source_stage_info):
            from dss.ray_dss.jobs.gpu import training as native_training

            _verify_shared_runtime(native_training, request["sources"])
            if (
                self._workers
                or len(ranks) != request["training_gpus"]
                or _shared_request(self._info.config) != request
            ):
                raise ValueError("native shared adapter coordinator layout or configuration differs")
            native_spawn = super()._spawn_workers
            namespace, original_worker = _spawn_worker_binding(native_spawn, native_training, request["sources"])
            namespace["_DeepSpeedWorker"] = SharedAdapterWorker
            try:
                try:
                    await native_spawn(cfg, ranks, master_addr, master_port, source_stage_info)
                    if namespace.get("_DeepSpeedWorker") is not SharedAdapterWorker:
                        raise ValueError("native shared adapter worker selection changed during creation")
                finally:
                    namespace["_DeepSpeedWorker"] = original_worker
                _verify_shared_runtime(native_training, request["sources"])
                receipts = await asyncio.wait_for(
                    asyncio.gather(*(worker.shared_adapter_receipt.remote() for worker in self._workers)),
                    timeout=60,
                )
                _verify_shared_receipts(receipts, request, self._info.job_id)
                _verify_shared_runtime(native_training, request["sources"])
                self._correctness_shared_adapter_receipts = deepcopy(receipts)
                print(
                    json.dumps(
                        {"schema": SHARED_SCHEMA, "job_id": self._info.job_id, "workers": receipts},
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    flush=True,
                )
            except BaseException as primary:
                if self._workers:
                    try:
                        await self._kill_workers([host for host, _ in ranks])
                    except BaseException as cleanup:
                        primary.add_note(
                            f"Owned shared-adapter worker teardown also failed: {type(cleanup).__name__}: {cleanup}"
                        )
                raise
            finally:
                namespace["_DeepSpeedWorker"] = original_worker

        def info(self):
            reported = super().info()
            reported["correctness_shared_adapter"] = {
                "schema": SHARED_SCHEMA,
                "workers": deepcopy(getattr(self, "_correctness_shared_adapter_receipts", [])),
            }
            return reported

    return ray.remote(num_cpus=1, num_gpus=0)(SharedAdapterCoordinator)


def install_manager(manager) -> None:
    """Select the subclass in the actual zone head, before native create.remote."""
    import ray
    from dss.ray_dss.jobs.gpu import training
    from dss.ray_dss.zones import manager as manager_module

    from arctic_platform.debug.training import maybe_spawn_participation_coordinator

    pass
    pass

    if manager.zone_type.value != "training" or manager.backend.value != "gpu":
        return
    if type(manager) is not manager_module.ZoneManager or "_spawn_actor" in vars(manager):
        raise ValueError("native original adapter requires the actual unmodified zone manager")
    original_spawn = manager._spawn_actor
    actor_key = (manager.zone_type, manager.backend)
    original_actor = manager_module._ACTOR_CLS[actor_key]
    if original_actor is not training.GPUTrainingJob:
        raise ValueError("native original adapter coordinator was already replaced")

    def spawn(owner, info):
        request = _request(info.config)
        shared_request = _shared_request(info.config) if request is None else None
        if request is None and shared_request is None:
            return maybe_spawn_participation_coordinator(
                manager_module, actor_key, original_actor, original_spawn, training, info
            )
        selected_request = shared_request if shared_request is not None else request
        if sum(len(node.gpu_ids) for node in info.layout) != selected_request["training_gpus"]:
            raise ValueError("native original adapter manager allocation differs")
        if shared_request is not None:
            _verify_shared_runtime(training, shared_request["sources"])
            root = Path(shared_request["path"])
            if (root / "initialization.json").exists():
                raise ValueError("native shared adapter cannot contain deterministic initialization metadata")
            files = adapter_files(root, shared_request["identity"])
            selected = _shared_coordinator_class(training, shared_request, ray.put(files))
        else:
            _verify_runtime(training, request["sources"])
            files = original_adapter_files(request["path"], request["identity"], request["original_metadata"])
            selected = _coordinator_class(training, request, ray.put(files))
        # The original spawn is synchronous: no other create can observe this temporary table entry.
        manager_module._ACTOR_CLS[actor_key] = selected
        try:
            return original_spawn(info)
        finally:
            manager_module._ACTOR_CLS[actor_key] = original_actor

    manager._spawn_actor = MethodType(spawn, manager)


def _zone_command(command: str) -> str:
    argv = shlex.split(command)
    if (
        len(argv) >= 4
        and argv[:3] == ["exec", sys.executable, "-m"]
        and argv[3] == "dss.ray_dss.zones.server"
        and "--zone-type" in argv
        and argv[argv.index("--zone-type") + 1] == "training"
        and "--backend" in argv
        and argv[argv.index("--backend") + 1] == "gpu"
    ):
        argv[3:4] = [_MODULE, "zone"]
        return shlex.join(argv)
    return command


def _reserve_listener(host: str, port: int):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind((host, port))
        listener.listen(2048)
        listener.setblocking(False)
        return listener
    except BaseException:
        listener.close()
        raise


def _listener_launch(command: str, listener, kwargs: dict) -> tuple[str, dict]:
    argv = shlex.split(command)
    if _LISTENER_FD_OPTION in argv:
        raise ValueError("owned zone launch already specifies a listener")
    descriptor = listener.fileno()
    argv.extend([_LISTENER_FD_OPTION, str(descriptor)])
    return shlex.join(argv), {
        **kwargs,
        "pass_fds": tuple(dict.fromkeys((*kwargs.get("pass_fds", ()), descriptor))),
    }


@contextlib.contextmanager
def _gateway_listener_scope(local):
    original_port = local.get_free_port
    original_run = local.run_cmd
    reservations = {}

    def close_reservations():
        for listener in reservations.values():
            listener.close()
        reservations.clear()

    def free_port():
        listener = _reserve_listener("0.0.0.0", 0)
        port = listener.getsockname()[1]
        reservations[port] = listener
        return port

    def run(command, host, *args, **kwargs):
        selected = _zone_command(command)
        listener = None
        try:
            if selected != command and local.is_local(host):
                argv = shlex.split(selected)
                port = int(argv[argv.index("--port") + 1])
                listener = reservations.pop(port, None)
                if listener is None:
                    raise ValueError("owned local zone has no reserved HTTP listener")
                selected, kwargs = _listener_launch(selected, listener, kwargs)
            # Worker-health and unrelated launch ports retain DSS's original binding lifecycle.
            close_reservations()
            return original_run(selected, host, *args, **kwargs)
        finally:
            if listener is not None:
                listener.close()
            close_reservations()

    local.get_free_port = free_port
    local.run_cmd = run
    try:
        yield
    finally:
        local.get_free_port = original_port
        local.run_cmd = original_run
        close_reservations()


@contextlib.contextmanager
def _zone_listener():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(_LISTENER_FD_OPTION, type=int)
    options, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv[1:] = remaining
    address_parser = argparse.ArgumentParser(add_help=False)
    address_parser.add_argument("--host", default="0.0.0.0")
    address_parser.add_argument("--port", type=int, default=9001)
    address, _ = address_parser.parse_known_args(remaining)
    descriptor = options.correctness_listener_fd
    if descriptor is not None and descriptor < 0:
        raise ValueError("owned zone listener descriptor must be nonnegative")
    listener = (
        _reserve_listener(address.host, address.port) if descriptor is None else socket.socket(fileno=descriptor)
    )
    try:
        if (
            listener.family != socket.AF_INET
            or listener.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM
            or listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) != 1
            or listener.getsockname() != (address.host, address.port)
        ):
            raise ValueError("owned zone listener does not match its HTTP address")
        listener.set_inheritable(False)
        listener.setblocking(False)
        yield listener
    finally:
        listener.close()


async def _serve_with_listener(server_head, original_serve, args, listener):
    if getattr(args, "http_workers", 1) != 1:
        raise ValueError("owned training listener requires single-process HTTP serving")
    if (args.host, args.port) != listener.getsockname():
        raise ValueError("owned HTTP server changed the reserved listener")
    original_uvicorn = server_head.uvicorn

    class ReservedServer(original_uvicorn.Server):
        async def serve(self, sockets=None):
            if sockets is not None or (self.config.host, self.config.port) != listener.getsockname():
                raise ValueError("owned HTTP server changed the reserved listener")
            return await super().serve(sockets=[listener])

    class Uvicorn:
        Server = ReservedServer

        def __getattr__(self, name):
            return getattr(original_uvicorn, name)

    server_head.uvicorn = Uvicorn()
    try:
        return await original_serve(args)
    finally:
        server_head.uvicorn = original_uvicorn


@contextlib.contextmanager
def gateway_scope(gateway):
    """Keep launch substitution local to the owned correctness gateway process."""
    from dss.ray_dss.jobs.gpu import training
    from dss.ray_dss.placement import local

    original_openapi = gateway.app.openapi
    previous_schema = gateway.app.openapi_schema
    sources = _gateway_adapter_sources(training)
    shared_sources = _shared_runtime_sources(training) if sources is not None else None

    def openapi():
        document = deepcopy(original_openapi())
        if sources is not None:
            document[OPENAPI_KEY] = {"schema": SCHEMA, "sources": deepcopy(sources)}
            document[SHARED_OPENAPI_KEY] = {"schema": SHARED_SCHEMA, "sources": deepcopy(shared_sources)}
        else:
            document.pop(OPENAPI_KEY, None)
            document.pop(SHARED_OPENAPI_KEY, None)
        return document

    gateway.app.openapi = openapi
    try:
        with _gateway_listener_scope(local):
            yield
    finally:
        gateway.app.openapi = original_openapi
        gateway.app.openapi_schema = previous_schema


class _OwnedWorkerLeases:
    """Close every owned worker's stdin before waiting or stopping the head."""

    def __init__(self):
        self.workers = []
        self.errors = []
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.closed = True
        deadline = time.monotonic() + _WORKER_SHUTDOWN_TIMEOUT
        interruption = None
        for process in self.workers:
            try:
                process.stdin.close()
            except Exception as error:
                self.errors.append(f"worker {process.pid} lease close: {error}")
            except BaseException as error:
                self.errors.append(f"worker {process.pid} lease close interrupted: {type(error).__name__}")
                interruption = interruption or error
        for process in self.workers:
            try:
                status = process.wait(timeout=max(0.0, deadline - time.monotonic()))
                if status != 0:
                    self.errors.append(f"worker {process.pid} exit {status}; remote shutdown unconfirmed")
            except Exception as error:
                self.errors.append(f"worker {process.pid} shutdown unconfirmed: {error}")
            except BaseException as error:
                self.errors.append(f"worker {process.pid} shutdown interrupted: {type(error).__name__}")
                interruption = interruption or error
        if interruption is not None:
            raise interruption


class _OwnedZoneProcess:
    def __init__(self, process, leases, *, head=False):
        self.process = process
        self.leases = leases
        self.head = head

    def __getattr__(self, name):
        return getattr(self.process, name)

    def terminate(self):
        self.leases.close()
        if self.head and not self.leases.errors:
            self.process.terminate()

    def kill(self):
        # Killing SSH alone cannot prove that its remote worker stopped.
        self.leases.close()
        if self.head and not self.leases.errors:
            self.process.kill()


def _owned_worker_main(worker):
    descriptor = os.dup(0)
    original_signals = worker.signal
    finished = threading.Event()
    ready = threading.Event()
    handler = []
    failures = []
    watcher = None
    watcher_started = False
    primary = None

    class LeaseSignals:
        def __getattr__(self, name):
            return getattr(original_signals, name)

        def signal(self, signum, callback):
            previous = original_signals.signal(signum, callback)
            if signum == signal.SIGTERM and callable(callback):
                handler[:] = [callback]
                ready.set()
            return previous

    def watch():
        def request_shutdown():
            while not finished.is_set():
                if ready.wait(0.1):
                    handler[0](signal.SIGTERM, None)
                    return

        try:
            while not finished.is_set():
                readable, _, _ = select.select([descriptor], [], [], 0.1)
                if readable and not os.read(descriptor, 4096):
                    request_shutdown()
                    return
        except Exception as error:
            failures.append(error)
            try:
                request_shutdown()
            except Exception as shutdown_error:
                failures.append(shutdown_error)

    try:
        os.set_inheritable(descriptor, False)
        # Ray descendants must not retain the supervisor's lease reader.
        with open(os.devnull, "rb") as null:
            os.dup2(null.fileno(), 0)
        worker.signal = LeaseSignals()
        watcher = threading.Thread(target=watch, name="owned-worker-lease", daemon=True)
        watcher.start()
        watcher_started = True
        worker.main()
    except BaseException as error:
        primary = error
        raise
    finally:
        worker.signal = original_signals
        finished.set()
        cleanup_errors = []
        if watcher_started:
            try:
                watcher.join(timeout=1.0)
                if watcher.is_alive():
                    raise RuntimeError("owned worker lease reader did not stop")
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            os.close(descriptor)
        except BaseException as error:
            cleanup_errors.append(error)
        if primary is not None:
            for error in cleanup_errors:
                primary.add_note(f"Owned worker lease cleanup also failed: {error}")
        if failures:
            detail = f"Owned worker lease failed: {failures[0]}"
            if primary is not None:
                primary.add_note(detail)
            else:
                error = RuntimeError(detail)
                for cleanup_error in cleanup_errors:
                    error.add_note(f"Owned worker lease cleanup also failed: {cleanup_error}")
                raise error from failures[0]
        elif primary is None and cleanup_errors:
            error = cleanup_errors[0]
            for cleanup_error in cleanup_errors[1:]:
                error.add_note(f"Owned worker lease cleanup also failed: {cleanup_error}")
            raise error


def _owned_worker_session(worker):
    """Keep SSH alive until the worker's token-bound Ray descendants have exited."""
    from arctic_platform.correctness.harness.cortex import UnsafeCleanupError
    from arctic_platform.correctness.harness.gateway import _gateway_process_identity
    from arctic_platform.correctness.harness.gateway import _reap_owned_gateway_descendants

    token = os.environ.get("DSS_CORRECTNESS_GATEWAY_OWNER", "")
    if (
        len(token) != 32
        or any(character not in "0123456789abcdef" for character in token)
        or os.environ.get("SPT_NOENV") != "1"
        or os.getsid(0) != os.getpid()
    ):
        raise UnsafeCleanupError("owned worker requires its launch token and isolated process session")
    identity = _gateway_process_identity(os.getpid())
    if identity is None:
        raise UnsafeCleanupError("cannot record the owned worker's live process identity")
    primary = None
    try:
        _owned_worker_main(worker)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            _reap_owned_gateway_descendants(identity, token)
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(f"Owned worker descendant cleanup also failed: {error}")


def main() -> None:
    role = sys.argv.pop(1) if len(sys.argv) > 1 else None
    if role == "zone":
        from dss.ray_dss.zones import server

        original = server._head_argv
        original_run = server.run_cmd
        leases = _OwnedWorkerLeases()

        def head_argv(*args, **kwargs):
            argv = original(*args, **kwargs)
            if argv[:3] != [sys.executable, "-m", "dss.ray_dss.zones.server_head"]:
                raise ValueError("native original adapter zone head entry point changed")
            return argv[:2] + [_MODULE, "head"] + argv[3:] + [_LISTENER_FD_OPTION, str(listener.fileno())]

        def run(command, host, *args, **kwargs):
            argv = shlex.split(command)
            if argv[:3] == [sys.executable, "-m", "dss.ray_dss.zones.server_worker"]:
                if "stdin" in kwargs or leases.closed:
                    raise ValueError("owned worker requires a fresh, exclusive stdin lease")
                selected = [sys.executable, "-m", _MODULE, "worker"] + argv[3:]
                launch = [
                    "setsid",
                    "--wait",
                    "env",
                    f"DSS_CORRECTNESS_GATEWAY_OWNER={os.urandom(16).hex()}",
                    "SPT_NOENV=1",
                    *selected,
                ]
                process = original_run("exec " + shlex.join(launch), host, *args, stdin=subprocess.PIPE, **kwargs)
                leases.workers.append(process)
                return _OwnedZoneProcess(process, leases)
            if argv[:4] != [sys.executable, "-m", _MODULE, "head"]:
                return original_run(command, host, *args, **kwargs)
            if not server.is_local(host) or argv[-2:] != [
                _LISTENER_FD_OPTION,
                str(listener.fileno()),
            ]:
                raise ValueError("owned zone head cannot inherit its local listener")
            kwargs = {
                **kwargs,
                "pass_fds": tuple(dict.fromkeys((*kwargs.get("pass_fds", ()), listener.fileno()))),
            }
            result = original_run("exec " + command, host, *args, **kwargs)
            listener.close()
            return _OwnedZoneProcess(result, leases, head=True)

        with _zone_listener() as listener:
            server._head_argv = head_argv
            server.run_cmd = run
            primary = None
            try:
                server.main()
            except BaseException as error:
                primary = error
                raise
            finally:
                server._head_argv = original
                server.run_cmd = original_run
                try:
                    leases.close()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(f"Owned worker shutdown also failed: {error}")
                if leases.errors:
                    detail = "Owned worker shutdown failed: " + "; ".join(leases.errors)
                    if primary is not None:
                        primary.add_note(detail)
                    else:
                        raise RuntimeError(detail)
    elif role == "head":
        from dss.ray_dss.zones import server_head

        original = server_head._setup_state
        original_serve = server_head._serve

        def setup_state(*args, **kwargs):
            original(*args, **kwargs)
            install_manager(server_head.app.state.manager)

        async def serve(args):
            return await _serve_with_listener(server_head, original_serve, args, listener)

        with _zone_listener() as listener:
            server_head._setup_state = setup_state
            server_head._serve = serve
            try:
                server_head.main()
            finally:
                server_head._setup_state = original
                server_head._serve = original_serve
    elif role == "worker":
        from dss.ray_dss.zones import server_worker

        _owned_worker_session(server_worker)
    else:
        raise ValueError("native original adapter bootstrap requires zone, head or worker")


if __name__ == "__main__":
    main()
