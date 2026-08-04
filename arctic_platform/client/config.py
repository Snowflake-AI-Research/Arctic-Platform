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
"""One config for every backend. Switching deployment == swapping `backend_config`."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import computed_field

JobId = int | str


class OnPremConfig(BaseModel):
    """Backend-specific settings for the on-prem server (HTTP + in-process Ray)."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    backend: Literal["onprem"] = "onprem"
    comm_protocol: Literal["http", "ray"] = Field("http", description="onprem transport: HTTP or in-process Ray.")
    host: str = Field("localhost", description="onprem: server host.")
    port: int = Field(8000, description="onprem: server port.")
    colocate: bool = Field(False, description="onprem: colocate job types on shared GPUs.")
    launch_local_server: bool = Field(False, description="onprem: spawn a local server before connecting.")
    startup_timeout: float = Field(600.0, description="onprem: seconds to wait for a launched server to become healthy.")
    ds_config: dict[str, Any] | None = Field(None, description="onprem: DeepSpeed config passed to training/log-prob jobs.")
    checkpoint_path: str | None = Field(
        None,
        description=(
            "onprem: training job's base checkpoint dir, set at init (resume-from + weight sync). "
            "This is the default destination used by save_checkpoint()."
        ),
    )


class CortexConfig(BaseModel):
    """Backend-specific settings for the Cortex (SnowAPI) deployment.

    Provide `base_url` for a direct/mock URL (no auth), or `host` + a PAT in the
    env var for Snowflake programmatic-access auth.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    backend: Literal["cortex"] = "cortex"
    base_url: str | None = Field(None, description="cortex: direct/mock GS URL; bypasses PAT auth.")
    host: str | None = Field(None, description="cortex: Snowflake host for PAT auth.")
    pat_env_var: str = Field("CORTEX_PAT", description="cortex: env var holding the programmatic access token.")
    database: str = Field("", description="cortex: Snowflake database.")
    schema_: str = Field("", alias="schema", description="cortex: Snowflake schema.")
    endpoint: str = Field("cortex-training", description="cortex: SnowAPI endpoint name.")
    max_retries: int = Field(10, ge=0, description="cortex: transient-failure retries per HTTP request (tenacity).")


class ArcticRLClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    model_name: str = Field(..., description="HF model id to train/serve.")
    seed: int | None = Field(None, description="Global seed.")
    dtype: str | None = Field(None, description="Parameter dtype override.")
    max_seq_len: int = Field(8192, description="Max sequence length.")

    # GPU allocation — a job type is created only when its count is > 0.
    training_gpus: int = 0
    sampling_gpus: int = 0
    log_prob_gpus: int = 0

    # Shared across backends.
    job_ready_timeout: float = Field(1800.0, description="Seconds to wait for a created job to reach RUNNING.")
    request_timeout: float = Field(
        1800.0, description="Per-request timeout (seconds) applied to every call. Generous for long ops."
    )
    training_config: dict[str, Any] | None = Field(None, description="Training job config.")
    vllm_config: dict[str, Any] | None = Field(None, description="vLLM config for sampling/log-prob jobs.")

    # Backend-specific settings; the concrete type selects the deployment target.
    backend_config: OnPremConfig | CortexConfig = Field(
        default_factory=OnPremConfig, discriminator="backend", description="Backend-specific settings."
    )

    # Reconnect: attach to pre-existing jobs instead of creating new ones.
    training_job_id: JobId | None = None
    sampling_job_id: JobId | None = None
    log_prob_job_id: JobId | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def backend(self) -> Literal["onprem", "cortex"]:
        return self.backend_config.backend

    def gpus_for(self, job_type: str) -> int:
        """GPU count allocated to a job type (0 == the job type is disabled)."""
        return getattr(self, f"{job_type}_gpus")
