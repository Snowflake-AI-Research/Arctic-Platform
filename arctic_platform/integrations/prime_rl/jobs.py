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

"""Cortex job lifecycle over SnowAPI: create, wait for RUNNING, cancel.

The create body is built here rather than through ``arctic_platform``'s config
adapter, which cannot express the ``prime_rl`` model provider the training
sub-job runs on.
"""

import os
import time
from typing import Any

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from prime_rl.utils.logger import get_logger

from .config import CortexConnectionConfig, CortexRLConfig

JOB_TERMINAL_STATES = ("failed", "done", "cancelled", "canceled")
SUB_JOB_TYPES = ("training", "sampling")
ADAMW_EPS = 1e-8  # torch.optim.AdamW's default, which the native trainer uses


def short_state(status: Any) -> str:
    """``JOB_STATE_RUNNING`` -> ``running``."""
    return str(status or "").lower().removeprefix("job_state_")


def sub_job_id(job_id: str, job_type: str) -> str:
    return f"{job_id}:{job_type}:0"


def training_sub_job(config: CortexRLConfig) -> dict[str, Any]:
    model = config.trainer.model
    optim = config.trainer.optim
    training = config.cortex.training
    dp_size = training.num_gpus // training.sp_size
    training_config: dict[str, Any] = {
        "max_seq_len": model.seq_len,
        "train_batch_size": dp_size,
        "n_gpus": training.num_gpus,
        "mb_spec": {"max_tokens_per_mb": training.max_tokens_per_mb or model.seq_len},
        "optimizer": {
            "name": "AdamW",
            "lr": optim.lr,
            "weight_decay": optim.weight_decay,
            "betas": [optim.betas1, optim.betas2],
            "eps": ADAMW_EPS,
        },
        "ds_config": {
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": dp_size,
            "gradient_accumulation_steps": 1,
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": 1, "reduce_scatter": True},
        },
    }
    if model.attn != "auto":
        training_config["attn_implementation"] = model.attn
    if optim.max_norm is not None:
        training_config["gradient_clipping"] = optim.max_norm

    if training.model_provider == "prime_rl":
        # MoE provider: expert parallelism, grouped MoE matmul, DeepEP, and the chunked LM head
        # (the fused cross-entropy head returns only a loss, never the logprobs GRPO needs; the
        # chunked head computes them token chunk by token chunk, as the native trainer does).
        training_config["model_provider"] = "prime_rl"
        training_config["ep_size"] = model.ep
        training_config["prime_rl"] = {
            "seq_len": model.seq_len,
            "ep_comm_backend": "deepep",
            "optimization_dtype": model.optimization_dtype,
            "reduce_dtype": model.reduce_dtype,
            "moe_use_grouped_mm": True,
            "fused_lm_head_token_chunk_size": model.fused_lm_head_token_chunk_size,
            "fp32_lm_head": False,
            "fused_cross_entropy": False,
            **training.prime_rl,
        }
        if model.attn != "auto":
            training_config["attn_impl"] = model.attn
        if training.sp_size > 1:
            training_config["sp_size"] = training.sp_size
        if model.ac is not None:
            ac_config: dict[str, Any] = {"mode": model.ac.mode, "freq": model.ac.freq}
            if model.ac_offloading is not None:
                ac_config["offload_config"] = {"enabled": True}
            training_config["ac_config"] = ac_config
    else:
        # Dense model on Cortex's default provider: the model forward returns logits, so GRPO's
        # compute_logprobs needs no MoE/ep/sp block, no chunked LM head, and no per-token temperature.
        if training.model_provider not in (None, "default"):
            training_config["model_provider"] = training.model_provider
        if model.ac is not None:
            training_config["activation_checkpointing"] = True
    training_config.update(training.extra)
    return {
        "job_type": "training",
        "model_name": model.name,
        "dtype": "bfloat16",
        "training_config": training_config,
    }


def sampling_sub_job(config: CortexRLConfig) -> dict[str, Any]:
    sampling = config.cortex.sampling
    seq_len = config.orchestrator.seq_len
    return {
        "job_type": "sampling",
        "model_name": config.orchestrator.model.name,
        "dtype": "bfloat16",
        "inference_config": {
            "max_seq_len": seq_len,
            "n_gpus": sampling.num_gpus,
            "gpu_memory_utilization": sampling.gpu_memory_utilization,
            "vllm_config": {
                "max_model_len": seq_len,
                "gpu_memory_utilization": sampling.gpu_memory_utilization,
                **sampling.vllm_config,
            },
        },
    }


def job_body(config: CortexRLConfig) -> dict[str, Any]:
    body: dict[str, Any] = {"sub_job_configs": [training_sub_job(config), sampling_sub_job(config)]}
    if config.cortex.image_tag is not None:
        body["debug"] = {"job": {"image_tag": config.cortex.image_tag}}
    return body


def _is_transient(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429 or error.response.status_code >= 500
    return isinstance(error, httpx.TransportError)


class CortexJobs:
    """SnowAPI control plane of one Cortex endpoint, authenticated with a PAT."""

    def __init__(self, connection: CortexConnectionConfig, request_timeout: float):
        pat = os.environ.get(connection.pat_env_var)
        if not pat:
            raise ValueError(f"Cortex needs a programmatic access token in ${connection.pat_env_var}.")
        self.prefix = (
            f"https://{connection.host}/api/v2/databases/{connection.database}"
            f"/schemas/{connection.schema_name}/{connection.endpoint}"
        )
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {pat}",
                "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
            },
            timeout=request_timeout,
        )

    def _send(self, method: str, path: str, *, retry_on=_is_transient, **kwargs: Any) -> dict[str, Any]:
        def attempt() -> dict[str, Any]:
            response = self.client.request(method, f"{self.prefix}{path}", **kwargs)
            if response.is_error:
                # Carry the server's reason (e.g. a model-catalog rejection) into the error.
                raise httpx.HTTPStatusError(
                    f"Cortex {method} {path} returned {response.status_code}: {response.text[:2000]}",
                    request=response.request,
                    response=response,
                )
            return response.json() if response.content.strip() else {}

        retrying = Retrying(
            retry=retry_if_exception(retry_on),
            stop=stop_after_attempt(8),
            wait=wait_exponential_jitter(initial=1.0, max=30.0),
            reraise=True,
        )
        return retrying(attempt)

    def create(self, body: dict[str, Any]) -> str:
        # A create is only retried when it provably never reached the server, so a retry
        # cannot leave a duplicate job holding GPUs.
        created = self._send("POST", "", json=body, retry_on=lambda e: isinstance(e, httpx.ConnectError))
        return created["job_id"]

    def get(self, job_id: str) -> dict[str, Any]:
        job = self._send("GET", f"/{job_id}")
        return job.get("job", job)

    def cancel(self, job_id: str) -> None:
        # Cortex refuses to cancel a session that already ended (409); it holds no GPUs then.
        if short_state(self.get(job_id).get("status")) in JOB_TERMINAL_STATES:
            return
        self._send("POST", f"/{job_id}:cancel")

    def wait_running(self, job_id: str, timeout: float) -> dict[str, Any]:
        logger = get_logger()
        deadline = time.monotonic() + timeout
        last_state = None
        while time.monotonic() < deadline:
            job = self.get(job_id)
            state = short_state(job.get("status"))
            if state != last_state:
                logger.info(f"Cortex job {job_id} is {state}")
                last_state = state
            if state == "running":
                return job
            if state in JOB_TERMINAL_STATES:
                sub_jobs = ", ".join(
                    f"{short_state(sub.get('job_type')).removeprefix('job_type_')}={short_state(sub.get('status'))}"
                    for sub in job.get("sub_jobs") or []
                )
                raise RuntimeError(
                    f"Cortex job {job_id} ended {state} before running (reason={job.get('reason')!r}, "
                    f"sub_jobs=[{sub_jobs}])"
                )
            time.sleep(10)
        raise TimeoutError(f"Cortex job {job_id} did not reach RUNNING within {timeout:.0f}s")

    def close(self) -> None:
        self.client.close()
