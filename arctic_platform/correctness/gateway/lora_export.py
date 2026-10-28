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

"""Export a live gateway job's LoRA factors as a PEFT adapter the HF reference can load."""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

from ..harness.storage import require_data_fast_path

_PEFT_CONFIG_DROP = ("peft_type", "task_type", "target_parameters")


def zone_url_from_log(log: str | Path) -> str:
    """The newest training zone's v0 server, as announced in its log."""
    text = Path(log).read_text(errors="replace")
    ports = re.findall(r"Uvicorn running on http://0\.0\.0\.0:(\d+)", text)
    if not ports:
        raise RuntimeError(f"no zone server port in {log}")
    return f"http://127.0.0.1:{ports[-1]}"


def zone_save(zone_url: str, job_id: str, checkpoint_type: str, root: str | Path) -> dict:
    """Save through the zone's v0 API; the public v2 save needs stage info that a local gateway lacks."""
    import requests

    root = require_data_fast_path(root, label="gateway checkpoint export")
    zone_job = job_id.removeprefix("p")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    with requests.Session() as session:
        session.trust_env = False
        session.get(f"{zone_url}/api/v0/jobs/{zone_job}", timeout=30).raise_for_status()
        resp = session.post(
            f"{zone_url}/api/v0/jobs/{zone_job}/save",
            json={"checkpoint_type": checkpoint_type, "path": str(root)},
            timeout=60,
        )
        resp.raise_for_status()
        request_id = resp.json()["request_id"]
        while True:
            status = session.get(f"{zone_url}/api/v0/requests/{request_id}", timeout=60).json()
            if status["status"] not in ("pending", "running"):
                break
            time.sleep(1)
    if status["status"] != "completed":
        raise RuntimeError(f"{checkpoint_type} save failed: {status}")
    return status


def lora_factors(module_state: dict) -> dict:
    """LoRA A/B tensors from a DeepSpeed module state, under PEFT's saved names, widened to float32.

    PEFT keeps adapter factors in float32 on the reference, and widening the target's bfloat16 values is exact.
    """
    factors = {}
    for name, tensor in module_state.items():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        key = name.replace("._checkpoint_wrapped_module", "").replace(".default.", ".")
        if key in factors:
            raise ValueError(f"two module-state tensors map to {key}")
        factors[key] = tensor.detach().float().contiguous()
    if not factors:
        raise ValueError("module state holds no LoRA factors")
    return factors


def write_adapter(dest: str | Path, factors: dict, peft_config: dict) -> None:
    import peft
    from safetensors.torch import save_file

    dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True)
    peft.LoraConfig(
        task_type="CAUSAL_LM", **{k: v for k, v in peft_config.items() if k not in _PEFT_CONFIG_DROP}
    ).save_pretrained(str(dest))
    save_file(factors, str(dest / "adapter_model.safetensors"), metadata={"format": "pt"})


def widen_adapter(path: str | Path) -> None:
    """Rewrite an adapter's factors as float32 in place; the shared-adapter loader requires PEFT's dtype."""
    import torch
    from safetensors.torch import load_file
    from safetensors.torch import save_file

    file = Path(path) / "adapter_model.safetensors"
    tensors = load_file(str(file))
    if all(t.dtype == torch.float32 for t in tensors.values()):
        return
    tmp = file.with_name(f".{file.name}.{os.getpid()}")
    save_file({k: t.float().contiguous() for k, t in tensors.items()}, str(tmp), metadata={"format": "pt"})
    os.replace(tmp, file)


def export_adapter(
    zone_url: str, job_id: str, dest: str | Path, peft_config: dict, *, weights_only: bool = True
) -> dict:
    """Write the job's live LoRA factors to ``dest`` as a standard PEFT adapter.

    The weights-only save's ``default/`` adapter is recorded but not used, because it can come out empty.
    The resumable DeepSpeed module state holds every parameter, so the factors come from there.
    ``weights_only=False`` skips that save, for models whose HF export does not exist.
    """
    import torch
    from safetensors import safe_open

    dest = require_data_fast_path(dest, label="LoRA adapter export")
    weights_only_tensors = None
    if weights_only:
        weights_root = dest.with_name(dest.name + "-weights-only")
        zone_save(zone_url, job_id, "weights-only", weights_root)
        packs = sorted(weights_root.glob("**/default/adapter_model.safetensors"))
        if packs:
            with safe_open(packs[-1], framework="pt") as f:
                weights_only_tensors = len(list(f.keys()))
        shutil.rmtree(weights_root, ignore_errors=True)

    resumable_root = dest.with_name(dest.name + "-resumable")
    zone_save(zone_url, job_id, "resumable", resumable_root)
    states = sorted(resumable_root.glob("**/mp_rank_00_model_states.pt"))
    if not states:
        raise RuntimeError(f"resumable save wrote no mp_rank_00_model_states.pt under {resumable_root}")
    factors = lora_factors(torch.load(states[-1], map_location="cpu", weights_only=False)["module"])
    write_adapter(dest, factors, peft_config)
    shutil.rmtree(resumable_root, ignore_errors=True)
    return {
        "source": str(states[-1]),
        "tensors": len(factors),
        "weights_only_adapter_tensors": weights_only_tensors,
        "weights_only_saved": weights_only,
        "lora_B_max_abs": max(float(t.abs().max()) for k, t in factors.items() if ".lora_B." in k),
    }
