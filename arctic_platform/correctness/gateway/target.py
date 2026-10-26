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

"""Run one catalog job config's forward-backward on a local gateway and read back per-parameter grad norms."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Callable

from .lora_export import export_adapter

DEFAULT_GATEWAY = "http://127.0.0.1:7000"
# LoRA B starts at zero, which zeroes every A gradient; one step on another batch moves B off zero first.
WARMUP_LR = 1e-2
WARMUP_SEED = 4321
_TERMINAL = ("cancelled", "canceled", "failed", "completed")


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def training_config(body: dict) -> dict:
    return next(s for s in body["sub_job_configs"] if s["job_type"] == "training")["training_config"]


def vocab_size(fixture: str | Path) -> int:
    config = json.loads((Path(fixture) / "config.json").read_text())
    return int(config.get("text_config", config)["vocab_size"])


def arm_batch(fixture: str | Path, arm: str, n_gpus: int, max_seq_len: int, seed: int | None = None):
    """The reviewed placement arm and its synthetic batch; reference and target must call this identically."""
    from ..harness.batches import SEED
    from ..harness.batches import build_batch
    from ..harness.config import arms_for

    spec = next(a for a in arms_for(max_seq_len, n_gpus) if a.name == arm)
    batch = build_batch(
        spec.name, spec.global_batch_size, spec.max_seq_len, vocab_size(fixture), seed=SEED if seed is None else seed
    )
    return spec, batch


def job_body(
    config: dict,
    fixture: str | Path,
    n_gpus: int,
    max_seq_len: int,
    *,
    fused_cross_entropy: str = "liger",
    max_tokens_per_mb: int | None = None,
) -> dict:
    """The catalog body, training sub-job only, pointed at the fixture with grad-norm debug output.

    The config's own microbatch is kept unless overridden, so long configs run their long packed calls.
    """
    from arctic_platform.rl.processors.microbatch import DEFAULT_MAX_TOKENS_PER_MB

    body = copy.deepcopy(config)
    sub = next(s for s in body["sub_job_configs"] if s["job_type"] == "training")
    sub["model_name"] = str(fixture)
    tc = sub["training_config"]
    tc["n_gpus"] = tc["sp_size"] = tc["ep_size"] = n_gpus
    tc["max_seq_len"] = max_seq_len
    # Full-world SP leaves one data-parallel replica, which DeepSpeed checks the batch size against.
    ds = tc.get("ds_config") or {}
    if "train_batch_size" in ds:
        ds["train_batch_size"] = ds.get("train_micro_batch_size_per_gpu", 1) * ds.get("gradient_accumulation_steps", 1)
    if "train_batch_size" in tc:
        tc["train_batch_size"] = ds.get("train_batch_size", 1)
    configured = (tc.get("mb_spec") or {}).get("max_tokens_per_mb") or DEFAULT_MAX_TOKENS_PER_MB
    tc["mb_spec"] = {"max_tokens_per_mb": int(max_tokens_per_mb or configured)}
    # Plain SFT needs the model to return its loss, which prime_rl only does through fused cross-entropy.
    tc.setdefault("prime_rl", {})["fused_cross_entropy"] = fused_cross_entropy
    tc["debug"] = {**(tc.get("debug") or {}), "gradient_norms_per_param": True, "gradient_sample_max_numel": 0}
    body["sub_job_configs"] = [sub]
    return body


def frame(batch) -> bytes:
    import torch
    from cortex_training import wire

    positions = batch.position_ids
    offsets = torch.arange(positions.shape[1]).unsqueeze(0)
    attention_mask = positions.eq(positions[:, :1] + offsets).to(torch.long).cumprod(dim=1)
    payload = {
        "kwargs": {
            "input_ids": batch.input_ids,
            "position_ids": positions,
            "attention_mask": attention_mask,
            "labels": batch.shifted_labels(),
        },
        "context": {"labels_are_shifted": True},
    }
    return wire.dumps(payload, metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}})


def _rank_owned(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return next((item for item in value if isinstance(item, dict) and item), {})
    return {}


def grad_norms(stepped: dict) -> tuple[dict, dict]:
    """Per-parameter norms, with each expert stack folded to one norm, plus the raw per-expert lists."""
    metrics = stepped.get("metrics") or {}
    per_param = _rank_owned(stepped.get("gradient_norms_per_param") or metrics.get("gradient_norms_per_param"))
    per_expert = _rank_owned(stepped.get("gradient_norms_per_expert") or metrics.get("gradient_norms_per_expert"))
    norms = {k: float(v) for k, v in per_param.items()}
    for name, values in per_expert.items():
        norms[name] = math.sqrt(sum(float(v) ** 2 for v in values))
    return norms, per_expert


def make_client(base_url: str = DEFAULT_GATEWAY):
    try:
        from cortex_training import CortexTrainingClient
    except ImportError as exc:
        raise ImportError("the gateway target needs the cortex-training client (`cortex_training`)") from exc

    client = CortexTrainingClient(
        base_url=base_url,
        database="LOCAL",
        schema="PUBLIC",
        poll_timeout=3600.0,
        poll_interval=0.25,
        poll_backoff_multiplier=1.0,
        poll_max_interval=0.25,
        request_timeout=(30.0, 3600.0),
    )
    client.auto_create_database = False
    return client


def _cancel(client, job_id: str, timeout: float = 900.0) -> None:
    client.cancel_job(job_id)
    deadline = time.time() + timeout
    while time.time() < deadline:
        states = [s.get("status") for s in (client.get_job(job_id).get("sub_jobs") or [])]
        if states and all(str(s).lower() in _TERMINAL for s in states):
            return
        time.sleep(5)


def run_target(
    client,
    body: dict,
    batch,
    *,
    warmup_batch=None,
    adapter_dir: str | Path | None = None,
    zone_url: str | Callable[[], str] | None = None,
    weights_only_save: bool = True,
) -> dict:
    """Create the job, optionally warm up and export its LoRA adapter, then one fwd-bwd and a zero-LR step.

    ``warmup_batch``, ``adapter_dir`` and ``zone_url`` are required together for LoRA configs. A callable
    ``zone_url`` is resolved after the job starts, since a local gateway announces each new zone's port then.
    """
    peft_config = training_config(body).get("peft_config")
    if peft_config and (warmup_batch is None or adapter_dir is None or zone_url is None):
        raise ValueError("a LoRA target needs a warm-up batch, an adapter directory and the zone URL")
    job_id = client.create_job_from_body(body)["job_id"]
    print(f"[target] job {job_id}", flush=True)
    try:
        client.wait_for_job(job_id)
        exported = None
        if peft_config:
            client.poll_request(job_id, client.forward_backward(job_id, frame(warmup_batch)))
            client.poll_request(job_id, client.step(job_id, WARMUP_LR))
            url = zone_url() if callable(zone_url) else zone_url
            exported = export_adapter(url, job_id, adapter_dir, peft_config, weights_only=weights_only_save)
            print(f"[target] adapter {exported} -> {adapter_dir}", flush=True)
        t0 = time.time()
        fwd = client.poll_request(job_id, client.forward_backward(job_id, frame(batch)))
        stepped = client.poll_request(job_id, client.step(job_id, 0.0))
        norms, per_expert = grad_norms(stepped)
        fwd_metrics = fwd.get("metrics") or {}
        return {
            "job_id": job_id,
            "adapter": str(adapter_dir) if peft_config else None,
            "adapter_export": exported,
            "loss": float(fwd.get("avg_loss", fwd_metrics.get("loss"))),
            "grad_norms": norms,
            "per_expert": per_expert,
            "model_calls": fwd_metrics.get("sft_model_calls"),
            "packed_rows": fwd_metrics.get("sft_packed_rows"),
            "grad_norm": stepped.get("grad_norm"),
            "seconds": round(time.time() - t0, 1),
        }
    finally:
        _cancel(client, job_id)
