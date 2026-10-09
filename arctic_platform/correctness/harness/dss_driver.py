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

"""Drive correctness jobs through the Arctic Platform in-process Ray client."""

from __future__ import annotations

import contextlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional

from arctic_platform.client import ArcticClient
from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.client import SamplingConfig
from arctic_platform.client import TrainingConfig
from arctic_platform.client.requests import log_probs_request

from .arms import correctness_microbatch_tokens
from .hostfile import parse_hostfile
from .transport import Checkpoint


@dataclass
class StepResult:
    avg_loss: float
    grad_norms: Dict[str, float]
    model_calls: Optional[int] = None
    packed_rows: Optional[int] = None
    optimizer_state_manifest: Optional[str] = None


@dataclass(frozen=True)
class GatewaySession:
    workdir: Path
    slots: int


@dataclass
class ArcticJob:
    client: ArcticClient
    job_type: str

    @property
    def job_id(self):
        return getattr(self.client.jobs, self.job_type)


def available_gpus() -> int:
    import torch

    local = torch.cuda.device_count() if torch.cuda.is_available() else 0
    hostfile = os.environ.get("DSS_GATEWAY_HOSTFILE") or os.environ.get("ARCTIC_HOSTFILE")
    if not hostfile or not Path(hostfile).exists():
        return local
    return sum(entry.slots for entry in parse_hostfile(hostfile)) or local


_STANDARD_MASTER_PORT = 29500
_RAY_PROCESS_PATTERN = (
    r"([r]aylet|[g]cs_server|[d]ashboard.py|[m]onitor.py|[l]og_monitor.py|[d]efault_worker.py|ray::)"
)


def _allocation_hostfile() -> Path | None:
    value = os.environ.get("DSS_GATEWAY_HOSTFILE") or os.environ.get("ARCTIC_HOSTFILE")
    if value:
        path = Path(value)
        if path.is_file():
            return path
    default = Path("/data-fast/hostfile")
    return default if default.is_file() else None


def _run_on_allocation(script: str) -> subprocess.CompletedProcess[str]:
    hostfile = _allocation_hostfile()
    if hostfile is None:
        command = ["bash", "-lc", script]
    else:
        ds_ssh = Path(sys.executable).resolve().parent / "ds_ssh"
        command = [str(ds_ssh), "-f", str(hostfile), script]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if completed.returncode != 0:
        raise RuntimeError(
            f"allocation runtime command failed ({completed.returncode}): {completed.stdout}{completed.stderr}"
        )
    return completed


def _stop_and_verify_gateway_runtime() -> None:
    """Stop every Ray process in the allocation and prove the standard rendezvous port is reusable."""
    ray_bin = Path(sys.executable).resolve().parent / "ray"
    _run_on_allocation(f"{shlex.quote(str(ray_bin))} stop --force")
    port_probe = (
        "import socket; "
        "s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); "
        f"s.bind(('', {_STANDARD_MASTER_PORT})); s.close()"
    )
    _run_on_allocation(
        f"! pgrep -af {shlex.quote(_RAY_PROCESS_PATTERN)} >/dev/null && "
        f"{shlex.quote(sys.executable)} -c {shlex.quote(port_probe)}"
    )


def _load_gateway_ray_cluster() -> Any:
    from arctic_platform.common import ray_cluster

    return ray_cluster


@contextlib.contextmanager
def _gateway_runtime(slots: int) -> Iterator[None]:
    """Give one correctness job a clean Ray tree sized to its requested GPU topology."""
    import torch

    include_peers = slots > torch.cuda.device_count()
    previous_master_port = os.environ.get("MASTER_PORT")
    previous_auth_mode = os.environ.get("RAY_AUTH_MODE")
    os.environ["MASTER_PORT"] = str(_STANDARD_MASTER_PORT)
    os.environ["RAY_AUTH_MODE"] = "token"
    ray_cluster = None
    try:
        # Ray reads auth mode while importing its C extension; set the harness policy before importing Ray.
        ray_cluster = _load_gateway_ray_cluster()
        _stop_and_verify_gateway_runtime()
        ray_cluster.init_ray_cluster(auto_attach=False, include_peers=include_peers)
        yield
    finally:
        try:
            if ray_cluster is not None:
                ray_cluster._shutdown()
        finally:
            try:
                _stop_and_verify_gateway_runtime()
            finally:
                if previous_master_port is None:
                    os.environ.pop("MASTER_PORT", None)
                else:
                    os.environ["MASTER_PORT"] = previous_master_port
                if previous_auth_mode is None:
                    os.environ.pop("RAY_AUTH_MODE", None)
                else:
                    os.environ["RAY_AUTH_MODE"] = previous_auth_mode


@contextlib.contextmanager
def gateway(tmp_dir: Path, slots: int) -> Iterator[GatewaySession]:
    workdir = Path(tmp_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    yield GatewaySession(workdir=workdir, slots=slots)


def _ds_optimizer(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    value = deepcopy(value)
    name = value.pop("name", value.pop("type", "AdamW"))
    return {"type": name, "params": value}


def _training_client_config(session: GatewaySession, payload: dict) -> ArcticClientConfig:
    tc = deepcopy(payload["training_config"])
    n_gpus = int(tc.pop("n_gpus"))
    max_seq_len = int(tc.pop("max_seq_len", 8192))
    ds = deepcopy(tc.pop("ds_config", {}))
    train_batch_size = tc.pop("train_batch_size", None)
    gradient_clipping = tc.pop("gradient_clipping", None)
    optimizer = tc.pop("optimizer", None)
    peft = tc.pop("peft_config", None)
    if train_batch_size is not None:
        ds.setdefault("train_batch_size", train_batch_size)
    if gradient_clipping is not None:
        ds.setdefault("gradient_clipping", gradient_clipping)
    if optimizer is not None:
        ds.setdefault("optimizer", _ds_optimizer(optimizer))
    activation_checkpointing = tc.pop("activation_checkpointing", None)
    if activation_checkpointing is not None:
        tc.setdefault("enable_gradient_checkpointing", bool(activation_checkpointing))
    debug = tc.get("debug") if isinstance(tc.get("debug"), dict) else {}
    checkpoint_path = str(session.workdir / "checkpoints")
    return ArcticClientConfig(
        model_name=payload["model_name"],
        seed=payload.get("seed"),
        dtype=payload.get("dtype"),
        max_seq_len=max_seq_len,
        training_gpus=n_gpus,
        training=TrainingConfig(
            full_determinism=bool(debug.get("full_determinism", False)),
            checkpoint_path=checkpoint_path,
            ds_config=ds,
            ds_worker_config=tc,
            peft=peft,
        ),
        backend=OnPremConfig(),
    )


def _inference_client_config(payload: dict) -> ArcticClientConfig:
    inf = deepcopy(payload.get("inference_config") or {})
    n_gpus = int(inf.pop("n_gpus"))
    max_seq_len = int(inf.pop("max_seq_len", 8192))
    vllm = deepcopy(inf.pop("vllm_config", {}))
    vllm.setdefault("max_model_len", max_seq_len)
    job_type = payload.get("job_type")
    return ArcticClientConfig(
        model_name=payload["model_name"],
        seed=payload.get("seed"),
        dtype=payload.get("dtype"),
        max_seq_len=max_seq_len,
        sampling_gpus=n_gpus if job_type == "sampling" else 0,
        log_prob_gpus=n_gpus if job_type == "log_prob" else 0,
        sampling=SamplingConfig(vllm=vllm, arctic_inference_config=inf or None),
        backend=OnPremConfig(),
    )


@contextlib.contextmanager
def running_job(session: GatewaySession, payload: dict) -> Iterator[ArcticJob]:
    job_type = str(payload.get("job_type", "training"))
    config = _training_client_config(session, payload) if job_type == "training" else _inference_client_config(payload)
    with _gateway_runtime(session.slots):
        client = ArcticClient(config)
        try:
            yield ArcticJob(client=client, job_type=job_type)
        finally:
            client.shutdown()


def copy_checkpoint_peft_adapter(checkpoint_root: Path, destination: Path) -> Path:
    """Copy the PEFT pack from a weights-only checkpoint when the exporter retained one."""
    candidates = [
        checkpoint_root / "hf",
        checkpoint_root / "default",
        *sorted(checkpoint_root.glob("global_step*/default"), reverse=True),
    ]
    source = next((path for path in candidates if (path / "adapter_model.safetensors").is_file()), None)
    if source is None or not (source / "adapter_config.json").is_file():
        raise RuntimeError(f"initial adapter checkpoint produced no PEFT pack under {checkpoint_root}")
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination)
    return destination


def save_weights_only_checkpoint(session: GatewaySession, job: ArcticJob, path: str | Path) -> str:
    saved = job.client.save_checkpoint(path=str(path), export_hf=True)
    directory = Path(saved.get("hf_path") or saved["path"])
    if not (directory / "config.json").exists():
        raise RuntimeError(f"weights-only save reported {directory}, which holds no model config")
    return str(directory)


def pin_suite_determinism(debug: dict) -> dict:
    debug["full_determinism"] = True
    debug["full_determinism_must_comply"] = False
    return debug


def _max_tokens_per_mb_of(training_config: dict) -> int:
    from arctic_platform.rl.processors.microbatch import DEFAULT_MAX_TOKENS_PER_MB

    return int((training_config.get("mb_spec") or {}).get("max_tokens_per_mb") or DEFAULT_MAX_TOKENS_PER_MB)


def build_payload(
    training_config: dict,
    model_path: str,
    seed: int,
    *,
    attn_implementation: Optional[str] = None,
    optimizer_state_output_dir: Optional[Path] = None,
    lm_head_token_chunk_size: Optional[int] = None,
    determinism: str = "best-effort",
    gradient_norms_per_param: bool = True,
    gradient_accumulation_steps: Optional[int] = None,
) -> dict:
    tc = json.loads(json.dumps(training_config))
    if lm_head_token_chunk_size is not None:
        tc.setdefault("fused_lm_head_token_chunk_size", lm_head_token_chunk_size)
    if attn_implementation is not None:
        tc["attn_implementation"] = attn_implementation
    if gradient_accumulation_steps is not None:
        gas = int(gradient_accumulation_steps)
        if gas < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps!r}")
        ds = dict(tc.get("ds_config") or {})
        ds["gradient_accumulation_steps"] = gas
        micro = ds.get("train_micro_batch_size_per_gpu")
        n_gpus = tc.get("n_gpus")
        if micro is not None and n_gpus is not None:
            sp_size = int(tc.get("sp_size", 1))
            train_batch_size = int(micro) * gas * (int(n_gpus) // sp_size)
            ds["train_batch_size"] = train_batch_size
            if "train_batch_size" in tc:
                tc["train_batch_size"] = train_batch_size
        tc["ds_config"] = ds
    mb_spec = dict(tc.get("mb_spec") or {})
    mb_spec["max_tokens_per_mb"] = correctness_microbatch_tokens(_max_tokens_per_mb_of(tc))
    tc["mb_spec"] = mb_spec
    if int(tc.get("ep_size", 1)) > 1:
        tc.setdefault("deepep_token_chunk_size", mb_spec["max_tokens_per_mb"])
    debug = dict(tc.get("debug") or {})
    debug.update({"gradient_norms_per_param": gradient_norms_per_param, "gradient_sample_max_numel": 0})
    if determinism == "off":
        debug["full_determinism"] = False
        debug["full_determinism_must_comply"] = False
    elif determinism == "best-effort":
        pin_suite_determinism(debug)
    else:
        raise ValueError(f"determinism must be 'best-effort' or 'off', got {determinism!r}")
    debug.setdefault("fp32_precision", False)
    if optimizer_state_output_dir is not None:
        debug["optimizer_state_output_dir"] = str(Path(optimizer_state_output_dir).resolve())
    tc["debug"] = debug
    return {
        "model_name": model_path,
        "job_type": "training",
        "seed": seed,
        "dtype": tc.get("dtype", "bfloat16"),
        "training_config": tc,
    }


def _labels_for_provider(batch, model_provider: str):
    return batch.shifted_labels() if model_provider == "prime_rl" else batch.labels


def _attention_mask(batch):
    return _attention_mask_from_position_ids(batch.position_ids)


def _attention_mask_from_position_ids(position_ids):
    import torch

    offsets = torch.arange(position_ids.shape[1], device=position_ids.device).unsqueeze(0)
    expected_positions = position_ids[:, :1] + offsets
    return position_ids.eq(expected_positions).to(torch.long).cumprod(dim=1)


def _batch_dict(input_ids, position_ids, labels) -> dict:
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": _attention_mask_from_position_ids(position_ids),
        "labels": labels,
    }


def _batch_body_from_wire_batch(
    wire_batch,
    *,
    loss_fn: str = "sft",
    processing_config: Optional[dict] = None,
    labels_are_shifted: bool = False,
) -> dict:
    processing = {"loss_fn": loss_fn}
    if processing_config is not None:
        processing["config"] = deepcopy(processing_config)
    return {
        "batch": wire_batch,
        "meta": {"labels_are_shifted": True} if labels_are_shifted else {},
        "processing": processing,
    }


def _batch_body(
    batch,
    labels,
    *,
    loss_fn: str = "sft",
    processing_config: Optional[dict] = None,
    labels_are_shifted: bool = False,
) -> dict:
    return _batch_body_from_wire_batch(
        _batch_dict(batch.input_ids, batch.position_ids, labels),
        loss_fn=loss_fn,
        processing_config=processing_config,
        labels_are_shifted=labels_are_shifted,
    )


def pack(
    batch,
    *,
    model_provider: str = "huggingface",
    loss_fn: str = "sft",
    processing_config: Optional[dict] = None,
) -> dict:
    labels_are_shifted = model_provider == "prime_rl"
    return _batch_body(
        batch,
        _labels_for_provider(batch, model_provider),
        loss_fn=loss_fn,
        processing_config=processing_config,
        labels_are_shifted=labels_are_shifted,
    )


def pack_microbatches(
    batch,
    microbatches: int,
    *,
    model_provider: str = "huggingface",
    loss_fn: str = "sft",
    processing_config: Optional[dict] = None,
) -> dict:
    """Pack a correctness batch as an AP GAS list when the arm needs several model calls."""
    if microbatches <= 1:
        return pack(batch, model_provider=model_provider, loss_fn=loss_fn, processing_config=processing_config)
    if batch.rows < microbatches:
        raise ValueError(f"cannot split {batch.rows} row(s) into {microbatches} microbatches")
    labels_are_shifted = model_provider == "prime_rl"
    labels = _labels_for_provider(batch, model_provider)
    wire_batches = []
    base_rows, extra_rows = divmod(batch.rows, microbatches)
    start = 0
    for index in range(microbatches):
        rows = base_rows + (1 if index < extra_rows else 0)
        end = start + rows
        wire_batches.append(
            _batch_dict(
                batch.input_ids[start:end].contiguous(),
                batch.position_ids[start:end].contiguous(),
                labels[start:end].contiguous(),
            )
        )
        start = end
    if len(wire_batches) != microbatches:
        raise ValueError(f"split produced {len(wire_batches)} microbatches, expected {microbatches}")
    return _batch_body_from_wire_batch(
        wire_batches,
        loss_fn=loss_fn,
        processing_config=processing_config,
        labels_are_shifted=labels_are_shifted,
    )


def pack_logit_aligned(batch) -> dict:
    body = _batch_body(batch, batch.shifted_labels(), labels_are_shifted=True)
    body["processing"]["return_logprobs"] = True
    return body


def _scalar(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)


def _rank_owned_mapping(value: Any) -> Dict[str, Any]:
    """Return the rank-owned mapping from a step-response field.

    ``merge_dict_shards`` appends each rank's non-list value, so a mapping that only rank zero returns arrives as a
    one-element list when that rank is not the first worker. A mapping is returned unchanged. An empty list or any
    other shape contributes no entries.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item:
                return item
    return {}


def _rank_owned_path(value: Any) -> Optional[str]:
    """Return a rank-owned path metric after Ray response merging."""
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    if isinstance(value, (list, tuple)):
        for item in value:
            path = _rank_owned_path(item)
            if path is not None:
                return path
        return None
    raise TypeError(f"rank-owned path metric must be a string or path-like object, got {type(value).__name__}")


def _globalize_expert_norms(per_parameter: Dict[str, float], per_expert: Dict[str, list[float]]) -> Dict[str, float]:
    result = {key: float(value) for key, value in per_parameter.items()}
    for name, values in per_expert.items():
        result[name] = math.sqrt(sum(float(value) ** 2 for value in values))
    return result


def fwd_bwd_step(session: GatewaySession, job: ArcticJob, body: dict, learning_rate: float = 0.0) -> StepResult:
    response = job.client.fwd_bwd(body)
    stepped = job.client.step(learning_rate)
    step_metrics = stepped.get("metrics") or {}
    response_metrics = response.get("metrics") or {}
    norms = _globalize_expert_norms(
        _rank_owned_mapping(
            stepped.get("gradient_norms_per_param") or step_metrics.get("gradient_norms_per_param") or {}
        ),
        _rank_owned_mapping(
            stepped.get("gradient_norms_per_expert") or step_metrics.get("gradient_norms_per_expert") or {}
        ),
    )
    loss = response.get("avg_loss", response_metrics.get("loss"))
    if loss is None:
        raise RuntimeError(f"forward-backward response carries no loss: {sorted(response)}")
    calls = response_metrics.get("sft_model_calls")
    rows = response_metrics.get("sft_packed_rows")
    return StepResult(
        avg_loss=_scalar(loss),
        grad_norms=norms,
        model_calls=int(calls) if calls is not None else None,
        packed_rows=int(rows) if rows is not None else None,
        optimizer_state_manifest=_rank_owned_path(
            stepped.get("optimizer_state_manifest") or step_metrics.get("optimizer_state_manifest")
        ),
    )


def sampling_payload(model_dir: str | Path, seed: int, *, dtype: str, n_gpus: int, max_seq_len: int) -> dict:
    return {
        "model_name": str(model_dir),
        "job_type": "log_prob",
        "dtype": dtype,
        "seed": seed,
        "inference_config": {
            "n_gpus": n_gpus,
            "max_seq_len": max_seq_len,
            "vllm_config": {
                "tensor_parallel_size": n_gpus,
                "max_model_len": max_seq_len,
                "dtype": dtype,
                "trust_remote_code": True,
                "gpu_memory_utilization": 0.85,
                "disable_custom_all_reduce": True,
                "enforce_eager": True,
                # Qwen GDN training calls FLA chunk_gated_delta_rule. vLLM's auto prefill selects
                # FlashInfer on Hopper; triton selects the FLA prefill.
                "gdn_prefill_backend": "triton",
            },
        },
    }


def log_probs(
    session: GatewaySession,
    job: ArcticJob,
    prompts: List[str],
    completions: List[str],
    *,
    top_k: int,
) -> List[dict]:
    response = job.client._call(log_probs_request(job.client.jobs, prompts, completions, top_k))
    results = response["results"]
    if len(results) != len(prompts):
        raise RuntimeError(f"log-probs returned {len(results)} results for {len(prompts)} prompts")
    return list(results)


def forward_loss(session: GatewaySession, job: ArcticJob, body: dict) -> float:
    """Return the loss from Arctic Platform's forward-only response."""
    response = job.client.fwd_no_grad(body)
    metrics = response.get("metrics") or {}
    loss = response.get("avg_loss", metrics.get("loss"))
    if loss is None:
        raise RuntimeError(f"forward response carries no loss: {sorted(response)}")
    return _scalar(loss)


def forward_logprobs(session: GatewaySession, job: ArcticJob, body: dict):
    import torch

    response = job.client.fwd_no_grad(body)
    if "logprobs" not in response:
        metrics = response.get("metrics") or {}
        if "logprobs" in metrics:
            response = metrics
        else:
            raise RuntimeError(f"forward response carries no logprobs: {sorted(response)}")
    return torch.as_tensor(response["logprobs"], dtype=torch.float32)


@dataclass
class GatewayTrainingJob:
    session: GatewaySession
    job: ArcticJob

    def fwd_bwd_step(self, body: dict, *, learning_rate: float) -> float:
        return fwd_bwd_step(self.session, self.job, body, learning_rate=learning_rate).avg_loss

    def save_resumable(self) -> Checkpoint:
        saved = self.job.client.save_checkpoint()
        path = str(saved["path"])
        return Checkpoint(reference=path, source_job=path)


class GatewayTransport:
    def __init__(
        self,
        workdir: Path,
        training_config: dict,
        model_path: str,
        seed: int,
        *,
        slots: int,
        attn_implementation: Optional[str] = None,
        determinism: str = "best-effort",
    ) -> None:
        self.session = GatewaySession(Path(workdir), slots)
        self.payload = build_payload(
            training_config,
            model_path,
            seed,
            attn_implementation=attn_implementation,
            determinism=determinism,
        )

    def __enter__(self) -> "GatewayTransport":
        self.session.workdir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    @contextlib.contextmanager
    def job(self, *, resume_from: Optional[Checkpoint] = None) -> Iterator[GatewayTrainingJob]:
        with running_job(self.session, self.payload) as job:
            if resume_from is not None:
                job.client.load_checkpoint(path=resume_from.reference)
            yield GatewayTrainingJob(self.session, job)
