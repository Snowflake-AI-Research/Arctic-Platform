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

"""SFT client config — training (+ optional sampling for generate_samples).

Mirrors ``ArcticRLClientConfig`` GPU / vLLM fields so SFT can spin a sampling
job the same way RL does when sample generation is enabled.
"""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from arctic_platform.client.config import ArcticRLClientConfig
from arctic_platform.client.config import JobId
from arctic_platform.client.config import OnPremConfig
from arctic_platform.client.config import SamplingConfig
from arctic_platform.client.config import TrainingConfig


class ArcticSFTClientConfig(BaseModel):
    """HTTP-first SFT client config.

    The client process itself must be runnable with ``CUDA_VISIBLE_DEVICES=``
    (empty). When ``launch_local_server=True``, set
    ``server_cuda_visible_devices`` so the server subprocess still sees GPUs.

    Set ``sampling_gpus > 0`` (and typically ``colocate=True``, matching RL e2e)
    to enable vLLM sample generation via weight sync.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    backend: Literal["onprem"] = Field("onprem", description="Deployment target.")
    comm_protocol: Literal["http", "ray"] = Field(
        "http",
        description="onprem transport: HTTP (phase 1) or Ray (phase 2).",
    )
    model_name: str = Field(..., description="HF model id to fine-tune.")
    seed: int | None = Field(None, description="Global seed.")
    max_seq_len: int = Field(2048, description="Max sequence length.")

    training_gpus: int = Field(..., description="Number of training GPUs on the server (>0).")
    # Defaults match ArcticRLClientConfig / RL e2e knobs.
    sampling_gpus: int = Field(0, description="vLLM sampling GPUs (0 = training-only).")
    colocate: bool = Field(
        False,
        description="Share GPUs between training and sampling (RL e2e uses True when sampling on).",
    )
    vllm_config: dict[str, Any] | None = Field(None, description="Forwarded to the sampling job init.")

    host: str = "localhost"
    port: int = 8000
    launch_local_server: bool = Field(False, description="Spawn a local HTTP server before connecting.")
    server_cuda_visible_devices: str | None = Field(
        None,
        description=(
            "CUDA_VISIBLE_DEVICES for the local server subprocess. Required when the "
            "client has CUDA_VISIBLE_DEVICES= (empty) and launch_local_server=True."
        ),
    )
    startup_timeout: float = 600.0
    job_ready_timeout: float = 1800.0
    request_timeout: float = 1800.0

    ds_config: dict[str, Any] | None = None
    ds_worker_config: dict[str, Any] | None = None
    checkpoint_path: str | None = Field(
        None,
        description="Training job checkpoint dir (required by the server for training jobs).",
    )

    training_job_id: JobId | None = Field(None, description="Reconnect to an existing training job.")
    sampling_job_id: JobId | None = Field(None, description="Reconnect to an existing sampling job.")

    @model_validator(mode="after")
    def _require_gpus(self) -> ArcticSFTClientConfig:
        if self.training_gpus <= 0 and self.training_job_id is None:
            raise ValueError("training_gpus must be > 0 (or set training_job_id to reconnect)")
        if self.training_job_id is None and not self.checkpoint_path:
            raise ValueError(
                "checkpoint_path is required to start a new training job "
                "(the server needs it to save checkpoints); set training_job_id to "
                "reconnect to an existing job instead."
            )
        return self

    def to_rl_config(self) -> ArcticRLClientConfig:
        """Adapt to the shared on-prem transport config (nested training/sampling/backend)."""
        return ArcticRLClientConfig(
            model_name=self.model_name,
            seed=self.seed,
            max_seq_len=self.max_seq_len,
            training_gpus=self.training_gpus,
            sampling_gpus=self.sampling_gpus,
            log_prob_gpus=0,
            job_ready_timeout=self.job_ready_timeout,
            request_timeout=self.request_timeout,
            training=TrainingConfig(
                checkpoint_path=self.checkpoint_path,
                ds_config=self.ds_config,
                ds_worker_config=self.ds_worker_config,
            ),
            sampling=SamplingConfig(vllm=dict(self.vllm_config or {})),
            backend_config=OnPremConfig(
                backend=self.backend,
                comm_protocol=self.comm_protocol,
                host=self.host,
                port=self.port,
                colocate=self.colocate,
                launch_local_server=self.launch_local_server,
                server_cuda_visible_devices=self.server_cuda_visible_devices,
                startup_timeout=self.startup_timeout,
            ),
            training_job_id=self.training_job_id,
            sampling_job_id=self.sampling_job_id,
        )
