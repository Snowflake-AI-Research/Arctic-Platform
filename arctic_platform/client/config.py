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
"""One config for every backend. Switching deployment == swapping `backend_config`.

Canonical nesting (shared across backends)::

    ArcticRLClientConfig
    ├── model_name, seed, dtype, max_seq_len
    ├── training_gpus / sampling_gpus / log_prob_gpus
    ├── training     # DeepSpeed engine (ds_config owns optimizer/scheduler/batch) + checkpoint
    ├── sampling     # sampling / log-prob engines (vllm, log_prob_engine, ...)
    └── backend_config: OnPremConfig  # connection / deployment only

``to_onprem`` is a temporary wire adapter. Drop it once the server accepts this
canonical shape directly.
"""

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
    startup_timeout: float = Field(
        600.0, description="onprem: seconds to wait for a launched server to become healthy."
    )


class SamplingConfig(BaseModel):
    """Sampling / log-prob engine settings shared by every backend.

    Engine knobs live under ``vllm`` — never flattened next to ``max_seq_len`` /
    ``n_gpus``. Those allocation fields stay on ``ArcticRLClientConfig``.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    vllm: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "vLLM engine kwargs (gpu_memory_utilization, enforce_eager, "
            "enable_chunked_prefill, enable_prefix_caching, max_num_seqs, ...)."
        ),
    )
    arctic_inference_config: dict[str, Any] | None = Field(
        None, description="Arctic inference config for the vLLM engines (use_fca, spec_model, ...)."
    )
    log_prob_engine: Literal["vllm", "deepspeed"] = Field(
        "vllm", description="Engine backend for the log-prob job (deepspeed builds a forward-only DS engine)."
    )
    log_prob_ds_config: dict[str, Any] | None = Field(
        None, description="Log-prob DeepSpeed worker config, used only when log_prob_engine='deepspeed'."
    )


class TrainingConfig(BaseModel):
    """On-prem training job: the DeepSpeed engine config + job settings.

    DeepSpeed owns the engine, so optimizer / scheduler / batch size / gradient
    accumulation all live in ``ds_config`` (DeepSpeed config-json format) rather
    than as separate typed fields. Allocation (GPUs, max_seq_len) stays on the client.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    full_determinism: bool = Field(
        False, description="DeepSpeed worker calls enable_full_determinism for reproducible training."
    )
    checkpoint_path: str | None = Field(
        None,
        description=(
            "Training job's base checkpoint dir, set at init (resume-from + weight sync). "
            "The default destination used by save_checkpoint()."
        ),
    )
    ds_config: dict[str, Any] | None = Field(
        None,
        description=(
            "DeepSpeed config-json for the engine (optimizer, scheduler, train_batch_size, zero_optimization, ...)."
        ),
    )
    ds_worker_config: dict[str, Any] | None = Field(
        None,
        description="DeepSpeed worker knobs (attn_implementation, use_liger, enable_gradient_checkpointing, ...).",
    )


class ArcticRLClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    model_name: str = Field(..., description="HF model id to train/serve.")
    seed: int | None = Field(None, description="Global seed.")
    dtype: str | None = Field(None, description="Parameter dtype override.")
    max_seq_len: int = Field(8192, description="Max sequence length (training + sampling).")

    # GPU allocation — a job type is created only when its count is > 0.
    training_gpus: int = 0
    sampling_gpus: int = 0
    log_prob_gpus: int = 0

    # Shared across backends.
    job_ready_timeout: float = Field(1800.0, description="Seconds to wait for a created job to reach RUNNING.")
    request_timeout: float = Field(
        1800.0, description="Per-request timeout (seconds) applied to every call. Generous for long ops."
    )
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training job settings.")
    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig, description="Sampling / log-prob engine settings."
    )

    # Backend-specific settings; the concrete type selects the deployment target.
    backend_config: OnPremConfig = Field(default_factory=OnPremConfig, description="Backend-specific settings.")

    # Reconnect: attach to pre-existing jobs instead of creating new ones.
    training_job_id: JobId | None = None
    sampling_job_id: JobId | None = None
    log_prob_job_id: JobId | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def backend(self) -> Literal["onprem"]:
        return self.backend_config.backend

    def gpus_for(self, job_type: str) -> int:
        """GPU count allocated to a job type (0 == the job type is disabled)."""
        return getattr(self, f"{job_type}_gpus")

    # ── temporary wire adapter ───────────────────────────────────────────────
    # TODO(config-unify): delete to_onprem once the server accepts this nesting
    # (training / sampling.vllm) directly instead of the flat /initialize payload.

    def to_onprem(self, job_type: str) -> dict[str, Any]:
        """Translate into one on-prem ``/initialize`` payload (see ArcticRLHTTPClient)."""
        tc, sc = self.training, self.sampling
        payload: dict[str, Any] = {"model_name": self.model_name, "job_type": job_type, "seed": self.seed}
        # log-prob runs on DeepSpeed only when asked; otherwise it's a vLLM engine.
        use_deepspeed = job_type == "training" or (job_type == "log_prob" and sc.log_prob_engine == "deepspeed")
        if use_deepspeed:
            payload["full_determinism"] = tc.full_determinism
            if tc.ds_config:
                payload["ds_config"] = tc.ds_config
            if tc.ds_worker_config:
                payload["ds_worker_config"] = tc.ds_worker_config
            if job_type == "training":
                if tc.checkpoint_path:
                    payload["checkpoint_path"] = tc.checkpoint_path
            elif sc.log_prob_ds_config:
                payload["log_prob_config"] = sc.log_prob_ds_config
        else:
            if sc.arctic_inference_config:
                payload["arctic_inference_config"] = sc.arctic_inference_config
            if sc.vllm:
                vllm = dict(sc.vllm)
                vllm.setdefault("max_model_len", self.max_seq_len)
                payload["vllm_config"] = vllm
        return payload
