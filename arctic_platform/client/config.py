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
    ├── training.model          # ModelSpec minus path (factory knobs)
    ├── training.optimizer ...  # training-loop settings
    ├── sampling.vllm           # vLLM engine kwargs only
    └── backend_config: OnPremConfig | CortexConfig

``to_cortex`` / ``to_onprem`` are temporary wire adapters. Drop them once both
servers accept this canonical shape directly (see UNIFICATION_NOTES.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import computed_field
from pydantic import model_validator

from arctic_platform.model.config import ParallelismConfig
from arctic_platform.model.config import Patches

if TYPE_CHECKING:
    from arctic_platform.model import ModelSpec

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


class ModelBuildConfig(BaseModel):
    """How to build the training model — ``ModelSpec`` minus path/dtype.

    Path and dtype stay on ``ArcticRLClientConfig`` (shared with sampling).
    Call ``ArcticRLClientConfig.model_spec()`` to assemble a full ``ModelSpec``.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    attn_implementation: str | None = Field(None, description="Attention implementation to request from HF.")
    loader: str | None = Field(None, description="Model factory loader name (maps to Neutrino model_provider).")
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig, description="Loader-specific parallelism.")
    patches: Patches = Field(default_factory=Patches, description="Post-load patches (e.g. liger).")


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True, populate_by_name=True)

    name: str = Field("AdamW", validation_alias=AliasChoices("name", "type"), description="Optimizer name.")
    lr: float = Field(1e-5, gt=0)
    weight_decay: float = Field(0.0, ge=0)
    betas: list[float] = Field(default_factory=lambda: [0.9, 0.999])
    eps: float = Field(1e-8, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_betas(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "beta1" in out or "beta2" in out:
            out["betas"] = [out.pop("beta1", 0.9), out.pop("beta2", 0.999)]
        return out


class TrainingConfig(BaseModel):
    """Shared training-job settings. Allocation (GPUs, max_seq_len) stays on the client."""

    model_config = ConfigDict(extra="allow", validate_default=True, populate_by_name=True)

    model: ModelBuildConfig = Field(default_factory=ModelBuildConfig, description="Model factory build knobs.")
    train_batch_size: int = Field(1, ge=1)
    optimizer: OptimizerConfig | None = Field(default_factory=OptimizerConfig)
    gradient_clipping: float | None = Field(1.0, ge=0)
    activation_checkpointing: bool | int = Field(
        True,
        validation_alias=AliasChoices(
            "activation_checkpointing",
            "gradient_checkpointing",
            "activation_checkpointing_freq",
            "gradient_checkpointing_freq",
        ),
        description="Enable activation/gradient checkpointing (bool) or layer frequency (int).",
    )
    multiplex_job_id: str | None = Field(None, description="Neutrino multiplex job id for the training sub-job.")

    # On-prem worker extras (ignored by Neutrino today).
    lr_scheduler: dict[str, Any] | None = None
    training_horizon: int | None = Field(None, ge=0)
    gradient_accumulation_steps: int | None = Field(None, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_activation_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for key in (
            "activation_checkpointing",
            "gradient_checkpointing",
            "activation_checkpointing_freq",
            "gradient_checkpointing_freq",
        ):
            if key in out:
                out["activation_checkpointing"] = out.pop(key)
                break
        for key in ("gradient_checkpointing", "activation_checkpointing_freq", "gradient_checkpointing_freq"):
            out.pop(key, None)
        # Reject fields owned by ArcticRLClientConfig — adapters inject them.
        for banned in ("max_seq_len", "max_length", "max_model_len", "n_gpus"):
            out.pop(banned, None)
        return out


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
    sampling: SamplingConfig = Field(default_factory=SamplingConfig, description="Sampling / log-prob engine settings.")

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

    def model_spec(self) -> ModelSpec:
        """Assemble a full ``ModelSpec`` from ``model_name`` + ``training.model``."""
        # Import the package (not model.config) so built-in loaders register.
        from arctic_platform.model import ModelSpec

        m = self.training.model
        return ModelSpec(
            model_path_or_name=self.model_name,
            dtype=self.dtype or "bfloat16",
            attn_implementation=m.attn_implementation,
            loader=m.loader,
            parallelism=m.parallelism,
            patches=m.patches,
        )

    # ── temporary wire adapters ──────────────────────────────────────────────
    # TODO(config-unify): delete to_onprem / to_cortex once both servers accept
    # the canonical nesting (training.model / sampling.vllm) without translation.

    def to_onprem(self, job_type: str) -> dict[str, Any]:
        """Translate into one on-prem ``/initialize`` payload.

        Temporary — remove when the on-prem server speaks the canonical shape
        (``build_model(ModelSpec)`` instead of ``ds_worker_config`` knobs).
        """
        payload: dict[str, Any] = {"model_name": self.model_name, "job_type": job_type, "seed": self.seed}
        if job_type in ("training", "log_prob"):
            bc = self.backend_config
            if isinstance(bc, OnPremConfig) and bc.ds_config:
                payload["ds_config"] = bc.ds_config
            if job_type == "training":
                payload["training_config"] = self._onprem_training_config()
                worker = self._onprem_ds_worker_config()
                if worker:
                    payload["ds_worker_config"] = worker
                if isinstance(bc, OnPremConfig) and bc.checkpoint_path:
                    payload["checkpoint_path"] = bc.checkpoint_path
        elif self.sampling.vllm:
            vllm = dict(self.sampling.vllm)
            vllm.setdefault("max_model_len", self.max_seq_len)
            payload["vllm_config"] = vllm
        return payload

    def to_cortex(self) -> list[dict[str, Any]]:
        """Translate into Cortex/SnowAPI ``sub_job_configs``.

        Temporary — remove when Neutrino accepts ``training.model`` / nested
        ``inference_config.vllm_config`` without per-field remapping.
        """
        subs: list[dict[str, Any]] = []
        if self.sampling_gpus > 0:
            subs.append(self._cortex_inference_sub_job("sampling", self.sampling_gpus))
        if self.log_prob_gpus > 0:
            subs.append(self._cortex_inference_sub_job("log_probability", self.log_prob_gpus))
        if self.training_gpus > 0:
            subs.append(self._cortex_training_sub_job())
        return subs

    def _onprem_training_config(self) -> dict[str, Any]:
        tc = self.training
        out: dict[str, Any] = {"train_batch_size": tc.train_batch_size}
        if tc.optimizer is not None:
            out["optimizer"] = tc.optimizer.model_dump()
        if tc.lr_scheduler is not None:
            out["lr_scheduler"] = tc.lr_scheduler
        if tc.training_horizon is not None:
            out["training_horizon"] = tc.training_horizon
        if tc.gradient_accumulation_steps is not None:
            out["gradient_accumulation_steps"] = tc.gradient_accumulation_steps
        if tc.gradient_clipping is not None:
            out["gradient_clipping"] = tc.gradient_clipping
        # Pass through unknown extras (MoE / framework knobs).
        known = set(TrainingConfig.model_fields)
        for k, v in tc.model_dump(exclude_none=True).items():
            if k not in known:
                out.setdefault(k, v)
        return out

    def _onprem_ds_worker_config(self) -> dict[str, Any]:
        """Map ``training.model`` → today's on-prem ``ds_worker_config`` knobs."""
        m = self.training.model
        worker: dict[str, Any] = {}
        if m.attn_implementation is not None:
            worker["attn_implementation"] = m.attn_implementation
        if m.patches.liger:
            worker["use_liger"] = True
        worker["enable_gradient_checkpointing"] = bool(self.training.activation_checkpointing)
        return worker

    def _cortex_training_sub_job(self) -> dict[str, Any]:
        tc = self.training
        training: dict[str, Any] = {
            "max_seq_len": self.max_seq_len,
            "train_batch_size": tc.train_batch_size,
            "n_gpus": self.training_gpus,
            "activation_checkpointing": tc.activation_checkpointing,
        }
        if tc.optimizer is not None:
            training["optimizer"] = tc.optimizer.model_dump()
        if tc.gradient_clipping is not None:
            training["gradient_clipping"] = tc.gradient_clipping
        if tc.multiplex_job_id is not None:
            training["multiplex_job_id"] = tc.multiplex_job_id

        m = tc.model
        # Neutrino still uses model_provider / ep_size; map from the factory surface.
        # Liger-as-provider is Neutrino's old path; prefer it over plain huggingface
        # when patches.liger is set (factory's HF-then-patch isn't on Neutrino yet).
        if m.patches.liger and m.loader in (None, "huggingface"):
            training["model_provider"] = "liger"
        elif m.loader is not None:
            training["model_provider"] = m.loader
        if m.attn_implementation is not None:
            training["attn_implementation"] = m.attn_implementation
        if m.parallelism.expert_parallel > 1:
            training["ep_size"] = m.parallelism.expert_parallel
        if m.parallelism.sequence_parallel > 1:
            training["sp_size"] = m.parallelism.sequence_parallel

        known = set(TrainingConfig.model_fields) | {"model"}
        for k, v in tc.model_dump(exclude_none=True).items():
            if k not in known:
                training.setdefault(k, v)
        return self._cortex_sub_job("training", {"training_config": training})

    def _cortex_inference_sub_job(self, job_type: str, n_gpus: int) -> dict[str, Any]:
        # Neutrino's InferenceConfig reads engine kwargs from nested vllm_config —
        # flattening them onto inference_config is silently ignored (extra="allow").
        vllm = dict(self.sampling.vllm)
        vllm.pop("max_model_len", None)  # owned by top-level max_seq_len
        vllm.pop("tensor_parallel_size", None)  # owned by n_gpus on this wire
        inference: dict[str, Any] = {"max_seq_len": self.max_seq_len, "n_gpus": n_gpus}
        if vllm:
            inference["vllm_config"] = vllm
        return self._cortex_sub_job(job_type, {"inference_config": inference})

    def _cortex_sub_job(self, job_type: str, extra: dict[str, Any]) -> dict[str, Any]:
        sub: dict[str, Any] = {"job_type": job_type, "model_name": self.model_name, **extra}
        if self.dtype is not None:
            sub["dtype"] = self.dtype
        if self.seed is not None:
            sub["seed"] = self.seed
        return sub
