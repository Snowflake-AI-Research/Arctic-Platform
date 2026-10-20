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
"""Dense LoRA initialization admission through public SAVE0, before any forward."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
from copy import deepcopy
from pathlib import Path

from arctic_platform.common.peft import normalize_lora_peft_config

from .lora_init import manifest_specs
from .lora_init import verify_tensors
from .workdir import require_data_fast_path


def _dense_training(payload: dict) -> dict:
    training = payload.get("training_config")
    if not isinstance(training, dict) or payload.get("job_type", "training") != "training":
        raise ValueError("LoRA SAVE0 requires a training payload")
    if training.get("n_gpus") != 4 or type(training.get("n_gpus")) is not int:
        raise ValueError("LoRA SAVE0 currently admits original four-GPU dense training only")
    if any(
        type(training.get(key, 1)) is not int or training.get(key, 1) != 1
        for key in ("sp_size", "ep_size", "sequence_parallel_size", "expert_parallel_size")
    ):
        raise ValueError("LoRA SAVE0 currently requires dense EP1/SP1")
    zero = (training.get("ds_config") or {}).get("zero_optimization") or {}
    if not isinstance(zero, dict) or type(zero.get("stage", 0)) is not int or zero.get("stage", 0) not in {0, 1, 2}:
        raise ValueError("LoRA SAVE0 does not admit ZeRO3")
    return training


def bind_initialization(payload: dict, metadata: dict) -> dict:
    """Add only the reviewed AP opt-in; never change the reference topology."""
    training = _dense_training(payload)
    if metadata.get("model_name") != "Qwen/Qwen3-8B":
        raise ValueError("LoRA SAVE0 currently admits the reviewed dense Qwen3-8B reference only")
    manifest = metadata["initialization"]
    specs = manifest_specs(manifest)
    dtypes = {spec.dtype for spec in specs}
    if len(dtypes) != 1:
        raise ValueError("LoRA SAVE0 requires one declared adapter storage dtype")
    topology = normalize_lora_peft_config(training.get("peft_config"))
    if "deterministic_init" in topology or topology != metadata.get("peft_config"):
        raise ValueError("LoRA SAVE0 target topology differs from the genuine reference")
    options = {
        "recipe_version": manifest["recipe_version"],
        "global_seed": manifest["global_seed"],
        "dtype": next(iter(dtypes)),
    }
    bound = deepcopy(payload)
    bound["training_config"]["peft_config"] = normalize_lora_peft_config({**topology, "deterministic_init": options})
    return bound


def _fingerprint(path: Path, *, maximum_bytes: int) -> tuple:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum_bytes:
        raise ValueError(f"LoRA SAVE0 requires a bounded regular file: {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_regular(path: Path, *, maximum_bytes: int) -> bytes:
    before = _fingerprint(path, maximum_bytes=maximum_bytes)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != before:
            raise ValueError(f"LoRA SAVE0 file changed before reading: {path}")
        content = stream.read(maximum_bytes + 1)
    if len(content) != before[2] or _fingerprint(path, maximum_bytes=maximum_bytes) != before:
        raise ValueError(f"LoRA SAVE0 file changed during reading: {path}")
    return content


def verify_saved_initialization(root: str | Path, response: dict, metadata: dict) -> dict:
    """Validate actual saved tensor bytes, not SAVE status, logs, or scalar norms."""
    from safetensors import safe_open

    raw_root = Path(root)
    if not raw_root.is_absolute() or ".." in raw_root.parts:
        raise ValueError("LoRA SAVE0 requires an absolute canonical FAST root")
    resolved = require_data_fast_path(raw_root, label="LoRA SAVE0 checkpoint")
    if raw_root != resolved:
        raise ValueError("LoRA SAVE0 checkpoint root must not traverse symlinks")
    for directory in [resolved, resolved / "global_step0", resolved / "global_step0" / "default"]:
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise ValueError(f"LoRA SAVE0 requires a real directory: {directory}")
    if (
        not isinstance(response, dict)
        or response.get("checkpoint_path") != str(resolved)
        or response.get("checkpoint_tag") != "global_step0"
        or type(response.get("version")) is not int
        or response["version"] != 0
        or response.get("stage_path") is not None
    ):
        raise ValueError("LoRA SAVE0 response must identify this local checkpoint at step/version zero")
    specs = manifest_specs(metadata["initialization"])
    directory = resolved / "global_step0" / "default"
    for entry in directory.iterdir():
        if entry.name not in {"adapter_config.json", "adapter_model.safetensors", "README.md"}:
            raise ValueError(f"LoRA SAVE0 contains an unexpected adapter artifact: {entry.name}")
        _fingerprint(entry, maximum_bytes=1 << 20 if entry.name != "adapter_model.safetensors" else 1 << 30)
    config_path = directory / "adapter_config.json"
    config_bytes = _read_regular(config_path, maximum_bytes=1 << 20)
    config = json.loads(config_bytes)
    if not isinstance(config, dict) or config.get("peft_type") not in {"Lora", "LORA"}:
        raise ValueError("LoRA SAVE0 requires a serialized LoRA configuration")
    topology_keys = set(metadata["peft_config"])
    actual_topology = {key: value for key, value in config.items() if key in topology_keys}
    actual_topology["peft_type"] = "Lora"
    if normalize_lora_peft_config(actual_topology) != metadata["peft_config"]:
        raise ValueError("LoRA SAVE0 serialized topology differs from the reference")
    tensor_path = directory / "adapter_model.safetensors"
    sizes = {"float32": 4, "float16": 2, "bfloat16": 2}
    maximum_bytes = sum(spec.shape[0] * spec.shape[1] * sizes[spec.dtype] for spec in specs) + (1 << 20)
    before = _fingerprint(tensor_path, maximum_bytes=maximum_bytes)
    with safe_open(tensor_path, framework="pt", device="cpu") as saved:
        if len(saved.keys()) != len(specs):
            raise ValueError("LoRA SAVE0 saved tensor coverage differs")
        tensors = [(name, saved.get_tensor(name)) for name in saved.keys()]
        verification = verify_tensors(tensors, metadata["initialization"])
    if (
        _fingerprint(tensor_path, maximum_bytes=maximum_bytes) != before
        or _read_regular(config_path, maximum_bytes=1 << 20) != config_bytes
    ):
        raise ValueError("LoRA SAVE0 artifacts changed during verification")
    return {
        **verification,
        "checkpoint_path": str(resolved),
        "checkpoint_tag": "global_step0",
        "version": 0,
        "adapter_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "proof_scope": "exported-adapter-pack-not-all-rank-equality",
    }


@contextlib.contextmanager
def verified_running_job(session, payload: dict, metadata: dict, *, proof_path: Path):
    """Do not expose the owned client until SAVE0 proves initialization; always close it."""
    from .dss_driver import running_job
    from .hosted import _save_local_checkpoint
    from .hosted import require_local_cortex_gateway

    require_local_cortex_gateway(session.backend)
    proof_path = require_data_fast_path(proof_path, label="LoRA SAVE0 proof")
    if proof_path.exists():
        raise ValueError("LoRA SAVE0 proof destination must be new")
    bound = bind_initialization(payload, metadata)
    with running_job(session, bound) as job:
        root, response = _save_local_checkpoint(job.client, "weights-only")
        proof = verify_saved_initialization(root, response, metadata)
        proof_path.parent.mkdir(parents=True, exist_ok=True)
        with proof_path.open("x") as stream:
            json.dump(proof, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        yield job, proof
