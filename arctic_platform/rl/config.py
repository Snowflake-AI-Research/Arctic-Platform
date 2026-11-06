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

"""Configuration models for the Arctic RL client."""

from __future__ import annotations

from typing import Any
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from arctic_platform.common.optimizer import OptimizerSettings
from arctic_platform.common.training_settings import MicroBatchSpec
from arctic_platform.common.training_settings import WorkerTrainingSettings

_LEGACY_OPTIMIZER_KEYS = ("lr", "betas", "beta1", "beta2", "eps", "weight_decay")


class ArcticRLClientConfig(BaseModel):
    checkpoint_path: Optional[str] = None

    model_name: str = Field(description="Model name or HuggingFace ID to load on all engines.")
    ds_config: dict = Field(default_factory=dict, description="DeepSpeed config for training engine.")

    # Generic training worker config — any framework uses this to configure the server's
    # DeepSpeed training engine (optimizer, dtype, gradient_checkpointing, etc.).
    # Standard fields: optimizer (lr, weight_decay, beta1, beta2, eps, lr_scheduler_type,
    #   gradient_clipping, warmup_steps_proportion / warmup_ratio), dtype, gradient_checkpointing,
    #   attn_impl, mb_spec (max_tokens_per_mb). Extra fields are ignored by the server.
    training_config: Optional[dict] = Field(
        default=None, description="Training worker config dict (optimizer, dtype, etc.)."
    )
    vllm_config: Optional[dict] = Field(
        default=None, description="vLLM / ModelConfig overrides for sampling and log-prob engines."
    )
    log_prob_ds_config: Optional[dict] = Field(
        default=None, description="Log-prob DeepSpeed worker config dict (batch size, dtype, etc.)."
    )
    ds_worker_config: Optional[dict] = Field(
        default=None, description="Deepspeed worker config dict (optimizer, dtype, etc.)."
    )
    arctic_inference_config: Optional[dict] = Field(
        default=None, description="Arctic inference config dict (use_fca, spec_model, etc.)."
    )
    full_determinism: bool = Field(
        default=False,
        description="If True, the DeepSpeed worker calls enable_full_determinism for reproducible training.",
    )
    seed: int = Field(default=42, description="Seed used by enable_full_determinism when full_determinism=True.")

    training_gpus: int = Field(default=0, description="Number of GPUs for the DeepSpeed training engine.")
    sampling_gpus: int = Field(default=0, description="Number of GPUs for the vLLM sampling engine.")
    log_prob_gpus: int = Field(default=0, description="Number of GPUs for the log-prob engine.")
    log_prob_engine: Literal["vllm", "deepspeed"] = Field(
        default="vllm", description="Engine backend for the log-prob job."
    )
    colocate: bool = Field(
        default=False,
        description=(
            "Colocate training, sampling, and log-prob workers on the same GPUs using fractional Ray resources."
        ),
    )
    cuda_ipc: bool = Field(
        default=False,
        description=(
            "Colocated weight-sync strategy: push training weights to the sampling engine via zero-copy CUDA IPC "
            "(requires colocate + weights on GPU) instead of the CPU-file path. Optional override on sync_weights()."
        ),
    )
    low_memory: bool = Field(
        default=False,
        description=(
            "With cuda_ipc, stream one gathered param at a time to bound peak extra GPU memory to one param/GPU "
            "instead of the whole model. Optional override on sync_weights()."
        ),
    )

    # How long to wait for each job to reach RUNNING state after /initialize.
    job_ready_timeout: float = Field(
        default=600.0, description="Seconds to wait for each job to become RUNNING after initialization."
    )

    # Reconnect fields — when set, ArcticRLClient skips /initialize and connects
    # to pre-existing jobs. Populated by ArcticRLClient.reconnect_config() and
    # consumed in ArcticRLClient.__init__. Not forwarded to /initialize.
    training_job_id: Optional[int] = Field(default=None, exclude=True)
    sampling_job_id: Optional[int] = Field(default=None, exclude=True)
    log_prob_job_id: Optional[int] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _validate_local_gpu_counts(self) -> "ArcticRLClientConfig":
        if self.training_job_id is not None:
            return self  # skip validation in reconnect mode
        # Any individual engine may be 0 (e.g. a sampling-only generate test or a
        # training-only forward test); require only that at least one job exists.
        if self.training_gpus + self.sampling_gpus + self.log_prob_gpus <= 0:
            raise ValueError(
                "Local backend requires at least one of training_gpus / sampling_gpus / log_prob_gpus > 0."
            )
        return self

    def training_settings(self) -> WorkerTrainingSettings:
        """The typed engine settings for ``ds_config`` plus the legacy ``training_config``.

        ``training_config`` wins over the same setting in ``ds_config``. Its ``lr_scheduler`` over
        ``training_horizon`` steps becomes a DeepSpeed ``scheduler``. Unset clipping stays off.
        """
        ds = dict(self.ds_config)
        optimizer = ds.pop("optimizer", None)
        num_micro_batches = ds.pop("gradient_accumulation_steps", None)
        gradient_clipping = ds.pop("gradient_clipping", None)
        ds.pop("train_micro_batch_size_per_gpu", None)
        ds.pop("train_batch_size", None)
        fp32_masters = _legacy_fp32_masters(ds.pop("bf16", None), ds.pop("fp16", None))
        tc = self.training_config or {}
        legacy_optimizer = tc.get("optimizer")
        if legacy_optimizer is not None:
            for key in ("torch_adam", "adam_w_mode"):
                if key in legacy_optimizer:
                    raise ValueError(f"training_config.optimizer.{key} is no longer supported")
            optimizer = {
                key: legacy_optimizer[key] for key in _LEGACY_OPTIMIZER_KEYS if legacy_optimizer.get(key) is not None
            }
            for clip in (
                legacy_optimizer.get("gradient_clipping"),
                tc.get("gradient_clipping"),
                legacy_optimizer.get("max_grad_norm"),
            ):
                if clip is not None:
                    gradient_clipping = clip
                    break
        elif tc.get("gradient_clipping") is not None:
            gradient_clipping = tc["gradient_clipping"]
        optimizer = OptimizerSettings.model_validate(optimizer or {})
        num_micro_batches = tc.get("gradient_accumulation_steps", num_micro_batches)
        scheduler = _legacy_scheduler(tc.get("lr_scheduler"), int(tc.get("training_horizon") or 0), optimizer.lr)
        if scheduler is not None:
            ds["scheduler"] = scheduler
        return WorkerTrainingSettings(
            optimizer=optimizer,
            gradient_clipping=float(gradient_clipping or 0.0),
            fp32_masters=fp32_masters,
            mb_spec=MicroBatchSpec(num_micro_batches=int(num_micro_batches)) if num_micro_batches else None,
            ds_config=ds or None,
        )

    @property
    def comm_protocol(self) -> str:
        """Read by SkyRL's Arctic entrypoint. On Cortex ``get_server_state`` returns ``None`` either way."""
        return "ray"


def _legacy_fp32_masters(bf16: Optional[dict], fp16: Optional[dict]) -> bool:
    # The on-prem worker filled in ``bf16: {enabled: true}``, and DeepSpeed keeps fp32 masters unless the block
    # asks for bf16 ones.
    bf16 = bf16 or {}
    if (fp16 or {}).get("enabled") or bf16.get("enabled") is False:
        raise ValueError("ds_config may only enable bf16; fp16 and fp32 jobs set the job dtype instead")
    return not (bf16.get("bf16_master_weights_and_grads") and bf16.get("bf16_optimizer_states"))


def _legacy_scheduler(scheduler: Optional[dict], horizon: int, lr: float) -> Optional[dict[str, Any]]:
    if scheduler is None or horizon <= 0:
        return None
    # DeepSpeed's WarmupLR/WarmupCosineLR require an integer warmup_num_steps.
    if "warmup_steps" in scheduler:
        warmup_steps = round(float(scheduler["warmup_steps"]))
    else:
        warmup_steps = round(scheduler.get("warmup_ratio", 0.0) * horizon)
    if scheduler.get("type", "constant") == "cosine":
        return {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": horizon,
                "warmup_num_steps": warmup_steps,
                "warmup_min_ratio": 0.0,
                "cos_min_ratio": scheduler.get("min_lr_ratio") or 0.0,
                "warmup_type": "linear",
            },
        }
    if warmup_steps > 0:
        return {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0.0,
                "warmup_max_lr": lr,
                "warmup_num_steps": warmup_steps,
                "warmup_type": "linear",
            },
        }
    return None


class WeightSyncConfig(BaseModel):
    """NCCL weight-transfer topology between training GPUs and inference replicas.

    Used by :class:`WeightSyncCoordinator` (standalone, not part of the client).
    """

    training_sharding: str = Field(
        default="dp",
        description="Training parallelism strategy: 'dp' or 'fsdp'",
    )
    training_gpus: int = 1
    inference_replicas: int = 1
    inference_tp: int = 1
    base_port: int = 29500
    bucket_size: int = 256 * 1024 * 1024  # 256 MB
