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

from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


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
