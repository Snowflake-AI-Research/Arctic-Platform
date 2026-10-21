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

"""Generate or load one standard PEFT adapter for both correctness engines."""

from __future__ import annotations

import contextlib
import hashlib
import json
import tempfile
from pathlib import Path

from arctic_platform.common.peft import normalize_lora_peft_config

from .storage import correctness_workdir
from .storage import require_data_fast_path

_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "correctness_recipe.json")


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

    from .lora import canonical_name

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


def adapter_identity(path: str | Path, peft_config: dict | None = None) -> dict:
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
    adapter_identity(root, peft_config)
    return root


def adapter_files(path: str | Path, expected_identity: dict) -> dict[str, bytes]:
    """Read a verified pack for the allocation's existing Ray object transport."""
    root = require_data_fast_path(path, label="shared PEFT adapter")
    if adapter_identity(root) != expected_identity:
        raise ValueError("shared adapter changed before transfer")
    files = {name: (root / name).read_bytes() for name in _ADAPTER_FILES if (root / name).is_file()}
    if (
        hashlib.sha256(files["adapter_config.json"]).hexdigest() != expected_identity["adapter_config_sha256"]
        or hashlib.sha256(files["adapter_model.safetensors"]).hexdigest() != expected_identity["adapter_model_sha256"]
        or (json.loads(files["correctness_recipe.json"]) if "correctness_recipe.json" in files else None)
        != expected_identity.get("generation")
        or adapter_identity(root) != expected_identity
    ):
        raise ValueError("shared adapter changed during transfer")
    return files


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
            if adapter_identity(path) != expected_identity:
                raise ValueError("transferred shared adapter identity differs")
            with adapter_build_scope(path, expected_identity):
                yield
    finally:
        scratch.rmdir()


def initialize_adapter(model, path: str | Path, expected_identity: dict) -> None:
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
    if adapter_identity(path) != expected_identity:
        raise ValueError("shared adapter changed during loading")


@contextlib.contextmanager
def adapter_build_scope(path: str | Path, expected_identity: dict):
    """Apply the suite-owned loader only inside a correctness worker's initialization."""
    from arctic_platform.common import deepspeed_worker

    original = deepspeed_worker.build_model

    def build_model(*args, **kwargs):
        loaded = original(*args, **kwargs)
        initialize_adapter(loaded.model, path, expected_identity)
        return loaded

    deepspeed_worker.build_model = build_model
    try:
        yield
    finally:
        deepspeed_worker.build_model = original
