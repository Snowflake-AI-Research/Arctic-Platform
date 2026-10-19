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
"""Temporary deterministic HF adapters; target binding fails closed without a public load route."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from .lora_init import LoraTensorSpec
from .lora_init import build_manifest
from .lora_init import canonical_name
from .lora_init import generate_tensor
from .lora_init import manifest_specs
from .lora_init import verify_tensors
from .seeds import SEED
from .workdir import require_data_fast_path

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
