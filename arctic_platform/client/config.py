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
"""One config for every backend. Switching deployment == changing `backend`."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

JobId = int | str


class ArcticRLClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    backend: Literal["onprem", "cortex"] = Field("onprem", description="Deployment target.")
    comm_protocol: Literal["http", "ray"] = Field("http", description="onprem transport: HTTP or in-process Ray.")
    model_name: str = Field(..., description="HF model id to train/serve.")
    seed: int | None = Field(None, description="Global seed.")
    dtype: str | None = Field(None, description="Parameter dtype override.")
    max_seq_len: int = Field(8192, description="Max sequence length (cortex sub-jobs).")

    # GPU allocation — a job type is created only when its count is > 0.
    training_gpus: int = 0
    sampling_gpus: int = 0
    log_prob_gpus: int = 0

    # onprem (HTTP + Ray)
    host: str = "localhost"
    port: int = 8000
    colocate: bool = False
    launch_local_server: bool = Field(False, description="onprem: spawn a local server before connecting.")
    startup_timeout: float = 600.0
    job_ready_timeout: float = 1800.0
    request_timeout: float = Field(
        1800.0, description="onprem HTTP: per-request timeout (seconds) applied to every call. Generous for long ops."
    )
    ds_config: dict[str, Any] | None = None
    training_config: dict[str, Any] | None = None
    vllm_config: dict[str, Any] | None = None
    checkpoint_path: str | None = Field(
        None,
        description=(
            "onprem: training job's base checkpoint dir, set at init (resume-from + weight sync). "
            "This is the default destination; a per-call save_checkpoint(path=...) overrides it for "
            "that call and falls back to this dir when path is None."
        ),
    )

    # cortex (SnowAPI)
    cortex_base_url: str | None = Field(None, description="Mock/direct GS URL; bypasses PAT auth.")
    cortex_host: str | None = None
    cortex_pat_env_var: str = "CORTEX_PAT"
    cortex_database: str = ""
    cortex_schema: str = ""
    cortex_endpoint: str = "cortex-training"

    # Reconnect: attach to pre-existing jobs instead of creating new ones.
    training_job_id: JobId | None = None
    sampling_job_id: JobId | None = None
    log_prob_job_id: JobId | None = None

    def gpus_for(self, job_type: str) -> int:
        """GPU count allocated to a job type (0 == the job type is disabled)."""
        return getattr(self, f"{job_type}_gpus")
