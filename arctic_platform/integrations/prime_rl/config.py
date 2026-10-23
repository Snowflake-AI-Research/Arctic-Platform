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

import os
from typing import Any

from pydantic import Field, model_validator
from renderers import RendererConfig

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    CISPOLossConfig,
    ConstantSchedulerConfig,
    IPOLossConfig,
    TrainerConfig,
)
from prime_rl.utils.config import BaseConfig


class CortexConnectionConfig(BaseConfig):
    """SnowAPI coordinates of the Cortex training endpoint. The PAT never enters a
    config: it is read from the environment variable named by ``pat_env_var``."""

    host: str | None = Field(default_factory=lambda: os.environ.get("ARCTIC_CORTEX_HOST"))
    """Snowflake account host. Defaults to ``$ARCTIC_CORTEX_HOST``."""

    database: str | None = Field(default_factory=lambda: os.environ.get("ARCTIC_CORTEX_DATABASE"))
    """Database holding the endpoint. Defaults to ``$ARCTIC_CORTEX_DATABASE``."""

    schema_name: str | None = Field(default_factory=lambda: os.environ.get("ARCTIC_CORTEX_SCHEMA"))
    """Schema holding the endpoint. Defaults to ``$ARCTIC_CORTEX_SCHEMA``."""

    endpoint: str = Field(default_factory=lambda: os.environ.get("ARCTIC_CORTEX_ENDPOINT", "cortex-training"))
    """SnowAPI endpoint name. Defaults to ``$ARCTIC_CORTEX_ENDPOINT``, else ``cortex-training``."""

    pat_env_var: str = "ARCTIC_CORTEX_PAT"
    """Environment variable holding the programmatic access token."""

    max_retries: int = Field(90, ge=0)
    """Retries of one Cortex HTTP call on connection errors and 429/5xx, backing off to 10 s apart.
    90 rides out a control-plane outage of about 15 minutes; any request still failing ends the run."""

    @model_validator(mode="after")
    def validate_complete(self):
        missing = [name for name in ("host", "database", "schema_name") if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"Cortex connection is missing {missing}: set them under [cortex.connection] or export "
                "ARCTIC_CORTEX_HOST / ARCTIC_CORTEX_DATABASE / ARCTIC_CORTEX_SCHEMA."
            )
        return self


class CortexTrainingConfig(BaseConfig):
    num_gpus: int = Field(8, ge=1)
    """GPUs of the training sub-job."""

    sp_size: int = Field(1, ge=1)
    """Ulysses sequence-parallel degree of the training sub-job (the Cortex counterpart of the trainer's cp)."""

    model_provider: str | None = "prime_rl"
    """Cortex training model provider. ``prime_rl`` runs the MoE provider (expert parallelism, grouped
    MoE matmul, DeepEP, chunked LM head). Anything else — ``None`` for the plain default provider, or e.g.
    ``liger`` — runs a dense model through the default provider, with no MoE/ep/sp block."""

    max_tokens_per_mb: int | None = Field(None, ge=1)
    """Token budget of one packed training micro batch. Defaults to ``seq_len``."""

    prime_rl: dict[str, Any] = {}
    """Overrides merged over the training sub-job's ``prime_rl`` model-provider block."""

    extra: dict[str, Any] = {}
    """Extra keys merged into the training sub-job's ``training_config``."""

    @model_validator(mode="after")
    def validate_sp_size(self):
        if self.num_gpus % self.sp_size != 0:
            raise ValueError(f"sp_size ({self.sp_size}) must divide num_gpus ({self.num_gpus}).")
        return self


class CortexSamplingConfig(BaseConfig):
    num_gpus: int = Field(8, ge=1)
    """GPUs of the sampling sub-job."""

    gpu_memory_utilization: float = Field(0.85, gt=0, le=1)
    """vLLM GPU memory fraction of the sampling sub-job."""

    vllm_config: dict[str, Any] = {}
    """vLLM engine settings of the sampling sub-job (e.g. ``tensor_parallel_size``)."""


class CortexConfig(BaseConfig):
    connection: CortexConnectionConfig = Field(default_factory=CortexConnectionConfig)

    training: CortexTrainingConfig = CortexTrainingConfig()

    sampling: CortexSamplingConfig = CortexSamplingConfig()

    job_id: str | None = None
    """Attach to this running Cortex job instead of creating one. An attached job is never cancelled."""

    keep_job: bool = False
    """Leave the created Cortex job running when the launcher exits."""

    image_tag: str | None = None
    """Pin the job's zones to this Cortex backend build (the create body's ``debug.job.image_tag``).
    None runs the endpoint's default build."""

    job_ready_timeout: float = Field(3600.0, gt=0)
    """Seconds to wait for the Cortex job to reach RUNNING."""

    request_timeout: float = Field(600.0, gt=0)
    """Seconds one SnowAPI HTTP call may take."""

    poll_timeout: float = Field(14400.0, gt=0)
    """Seconds one Cortex request (generate, forward-backward, step, weight sync) may take to complete."""

    keepalive: float | None = Field(600.0, gt=0)
    """Send a one-token generation after this many seconds without a Cortex request. Cortex reaps a job
    that sees no request on any of its sub-jobs for its idle timeout. None disables it."""

    host: str = "localhost"
    """Host the Cortex worker serves the policy inference API on."""

    port: int = 8000
    """Port the Cortex worker serves the policy inference API on."""

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


class CortexWorkerConfig(BaseConfig):
    """The Cortex worker stands in for the trainer and the inference server: it trains on the
    orchestrator's batches through the Cortex training sub-job and serves the policy from the
    sampling sub-job."""

    cortex: CortexConfig

    job_id: str
    """Cortex job the worker attaches to."""

    trainer: TrainerConfig

    max_model_len: int
    """Context length of the sampling sub-job (the orchestrator's ``seq_len``)."""

    renderer: RendererConfig
    """Renderer for the chat-completions route (evals), matching the orchestrator's."""


class CortexRLConfig(RLConfig):
    cortex: CortexConfig = Field(default_factory=CortexConfig)

    @model_validator(mode="after")
    def validate_cortex(self):
        if self.inference is not None:
            raise ValueError("Cortex serves the policy: remove the [inference] block.")
        if self.deployment.type != "single_node" or self.slurm is not None:
            raise ValueError("Cortex runs launch locally: use the single_node deployment without [slurm].")
        if self.orchestrator.num_train_workers != 1:
            raise ValueError("Cortex trains from one batch stream: orchestrator.num_train_workers must be 1.")
        if self.weight_broadcast.type != "filesystem":
            raise ValueError("Cortex runs hand weights over through the filesystem broadcast handshake.")
        if self.ckpt is not None or self.resume is not None:
            raise ValueError("Cortex runs do not checkpoint: remove [ckpt] and [resume].")

        model = self.trainer.model
        if model.lora is not None:
            raise ValueError("LoRA is not supported on Cortex.")
        if model.vlm is not None:
            raise ValueError("Multimodal training is not supported on Cortex.")
        if model.cp != 1:
            raise ValueError("Use cortex.training.sp_size for sequence parallelism; trainer.model.cp must be 1.")
        if not isinstance(model.ep, int):
            raise ValueError("Set trainer.model.ep explicitly: it becomes the Cortex training sub-job's ep_size.")
        if self.cortex.training.model_provider != "prime_rl":
            # Dense model on Cortex's default provider: no MoE expert/sequence parallelism.
            if model.ep != 1:
                raise ValueError("The default provider has no expert parallelism: set trainer.model.ep = 1.")
            if self.cortex.training.sp_size != 1:
                raise ValueError("The default provider has no sequence parallelism: set cortex.training.sp_size = 1.")
        if self.trainer.enable_router_replay:
            raise ValueError("Router replay is not wired to Cortex.")

        loss = self.trainer.loss
        if not isinstance(loss, IPOLossConfig | CISPOLossConfig):
            raise ValueError(f"The {loss.type} loss has no Cortex equivalent; use the ipo or cispo loss.")
        if isinstance(loss, CISPOLossConfig) and loss.ratio_low > 0:
            raise ValueError("Cortex CISPO caps the importance weight from above only: set ratio_low = 0.")
        if not isinstance(self.trainer.optim, AdamWConfig):
            raise ValueError("Cortex trains with AdamW only.")
        if not isinstance(self.trainer.scheduler, ConstantSchedulerConfig):
            raise ValueError("Cortex runs use a constant learning rate.")

        max_tokens_per_mb = self.cortex.training.max_tokens_per_mb
        if max_tokens_per_mb is not None and max_tokens_per_mb < model.seq_len:
            raise ValueError(
                f"cortex.training.max_tokens_per_mb ({max_tokens_per_mb}) must fit one sequence ({model.seq_len})."
            )

        client = self.orchestrator.model.client
        if "base_url" in client.model_fields_set and client.base_url != self.cortex.base_url:
            raise ValueError(
                f"orchestrator.model.client.base_url ({client.base_url}) must be the Cortex worker "
                f"({self.cortex.base_url})."
            )
        if client.admin_base_url is not None:
            raise ValueError("The Cortex worker serves the admin routes itself: unset admin_base_url.")
        client.base_url = self.cortex.base_url
        return self
