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
"""One config for every backend. Switching deployment == swapping `backend`.

Canonical nesting (shared across backends)::

    ArcticClientConfig
    ├── model_name, seed, dtype, max_seq_len
    ├── training_gpus / sampling_gpus / log_prob_gpus
    ├── training     # DeepSpeed engine (ds_config owns optimizer/scheduler/batch) + checkpoint
    ├── sampling     # sampling / log-prob engines (vllm, log_prob_engine, ...)
    └── backend: OnPremConfig | CortexConfig  # connection / deployment only

``to_onprem`` / ``to_cortex`` are temporary wire adapters. Drop them once both
servers accept this canonical shape directly.
"""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SecretStr
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
from typing_extensions import Self

JobId = int | str


class OnPremConfig(BaseModel):
    """Backend-specific settings for the on-prem server (HTTP + in-process Ray)."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    type: Literal["onprem"] = "onprem"
    protocol: Literal["http", "ray"] = Field("http", description="onprem transport: HTTP or in-process Ray.")
    host: str = Field("localhost", description="onprem: server host.")
    port: int = Field(8000, description="onprem: server port.")
    colocate: bool = Field(False, description="onprem: colocate job types on shared GPUs.")
    launch_local_server: bool = Field(False, description="onprem: spawn a local server before connecting.")
    server_cuda_visible_devices: str | None = Field(
        None,
        description=(
            "CUDA_VISIBLE_DEVICES for the local server subprocess when launch_local_server=True. "
            "Use this when the client process itself has CUDA_VISIBLE_DEVICES= (empty) so the "
            "server child still sees GPUs. None = inherit the client's environment."
        ),
    )
    startup_timeout: float = Field(
        600.0, description="onprem: seconds to wait for a launched server to become healthy."
    )


class CortexConfig(BaseSettings):
    """Cortex protocol settings for the remote backend.

    Provide `base_url` for a direct/mock URL (no auth), or `host` + `pat` for
    Snowflake programmatic-access auth.

    Every field also reads from an ``ARCTIC_CORTEX_``-prefixed env var
    (``ARCTIC_CORTEX_HOST``, ``ARCTIC_CORTEX_PAT``, ``ARCTIC_CORTEX_DATABASE``,
    ``ARCTIC_CORTEX_SCHEMA``, ...), so `CortexConfig()` with no arguments is a
    complete config on a configured shell. Constructor and YAML values win over
    the environment.
    """

    model_config = SettingsConfigDict(
        extra="forbid",
        validate_default=True,
        env_prefix="ARCTIC_CORTEX_",
        # An exported-but-empty var is how a shell says "unset"; without this it
        # would beat the default and fail validation as a present empty string.
        env_ignore_empty=True,
        populate_by_name=True,
    )

    type: Literal["remote"] = "remote"
    protocol: Literal["cortex"] = Field("cortex", description="remote transport protocol.")
    # Present so callers can read `backend.colocate` uniformly across backends.
    # Cortex always splits training and sampling into separate SnowAPI sub-jobs,
    # so it can only ever be False.
    colocate: Literal[False] = Field(False, description="cortex: colocation not supported.")
    base_url: str | None = Field(None, description="cortex: direct/mock GS URL; bypasses PAT auth.")
    host: str | None = Field(None, description="cortex: Snowflake host for PAT auth.")
    # SecretStr so the token cannot ride along into a log line or a serialized
    # config: repr and model_dump render it as `**********`, and reading it
    # takes an explicit `.get_secret_value()`.
    pat: SecretStr | None = Field(None, description="cortex: PAT; also read from ARCTIC_CORTEX_PAT.")
    database: str = Field("", description="cortex: Snowflake database.")
    # `schema` shadows a BaseModel attribute, hence the trailing underscore. An
    # explicit alias opts the field out of `env_prefix`, so the env name has to
    # be spelled out or ARCTIC_CORTEX_SCHEMA is silently ignored.
    schema_: str = Field(
        "",
        validation_alias=AliasChoices("schema", "schema_", "ARCTIC_CORTEX_SCHEMA"),
        serialization_alias="schema",
        description="cortex: Snowflake schema.",
    )
    endpoint: str = Field("cortex-training", description="cortex: SnowAPI endpoint name.")
    max_retries: int = Field(10, ge=0, description="cortex: transient-failure retries per HTTP request (tenacity).")

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not (self.base_url or self.host):
            raise ValueError("cortex: set base_url (direct URL) or host (PAT auth).")
        if self.host and not self.base_url:
            if not (self.database and self.schema_):
                raise ValueError("cortex: database + schema required for host/PAT auth.")
            if not (self.pat and self.pat.get_secret_value()):
                raise ValueError("cortex: no PAT — set `pat` or ARCTIC_CORTEX_PAT for host auth.")
        return self


class SamplingConfig(BaseModel):
    """Sampling / log-prob engine settings shared by every backend.

    Engine knobs live under ``vllm`` — never flattened next to ``max_seq_len`` /
    ``n_gpus``. Those allocation fields stay on ``ArcticClientConfig``.
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
    peft: dict[str, Any] | None = Field(
        None,
        description=(
            "PEFT adapter config (peft_type, r, lora_alpha, lora_dropout, bias, target_modules). "
            "None = dense fine-tuning. Applied to the training job and, when sampling is allocated, "
            "to the sampling engine so it can serve the adapter."
        ),
    )
    cuda_ipc: bool = Field(
        False,
        description=(
            "Colocated weight-sync strategy: push training weights to the sampling engine via zero-copy CUDA IPC "
            "(requires colocate=True and weights resident on GPU) instead of the CPU-file path. Optional override on "
            "sync_weights()."
        ),
    )
    low_memory: bool = Field(
        False,
        description=(
            "With cuda_ipc, stream one gathered param at a time to bound peak extra GPU memory to one param/GPU "
            "instead of the whole model. Optional override on sync_weights()."
        ),
    )


class ArcticClientConfig(BaseModel):
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
    backend: OnPremConfig | CortexConfig = Field(
        default_factory=OnPremConfig, discriminator="type", description="Backend-specific settings."
    )

    # Reconnect: attach to pre-existing jobs instead of creating new ones.
    training_job_id: JobId | None = None
    sampling_job_id: JobId | None = None
    log_prob_job_id: JobId | None = None

    @model_validator(mode="after")
    def _check_backend_supports_peft(self) -> Self:
        # The on-prem server has no PEFT path, and to_onprem() has nowhere to put an
        # adapter config. Silently training dense after asking for LoRA burns a run,
        # so refuse the combination before any job or GPU is claimed.
        if self.training.peft and self.backend.type == "onprem":
            raise ValueError(
                "training.peft is only supported by the remote Cortex backend; "
                "the on-prem server trains dense only. Drop training.peft, or "
                "switch backend to CortexConfig."
            )
        return self

    def gpus_for(self, job_type: str) -> int:
        """GPU count allocated to a job type (0 == the job type is disabled)."""
        return getattr(self, f"{job_type}_gpus")

    # ── temporary wire adapters ──────────────────────────────────────────────
    # TODO(config-unify): delete to_onprem / to_cortex once both servers accept
    # this nesting (training / sampling.vllm) directly.

    def to_onprem(self, job_type: str) -> dict[str, Any]:
        """Translate into one on-prem ``/initialize`` payload (see ArcticRLHTTPClient)."""
        tc, sc = self.training, self.sampling
        payload: dict[str, Any] = {"model_name": self.model_name, "job_type": job_type}
        if self.seed is not None:
            payload["seed"] = self.seed
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
                # Bake the (static) weight-sync strategy onto the training job.
                payload["cuda_ipc"] = tc.cuda_ipc
                payload["low_memory"] = tc.low_memory
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

    def to_cortex(self) -> list[dict[str, Any]]:
        """Translate into Cortex/SnowAPI ``sub_job_configs``.

        Neutrino wants typed sub-job fields, so training knobs are lifted out of the
        DeepSpeed ``ds_config`` / ``ds_worker_config`` here (temporary — drop once
        Neutrino accepts the canonical nesting directly).
        """
        subs: list[dict[str, Any]] = []
        if self.sampling_gpus > 0:
            subs.append(self._cortex_inference_sub_job("sampling", self.sampling_gpus))
        if self.log_prob_gpus > 0:
            subs.append(self._cortex_inference_sub_job("log_probability", self.log_prob_gpus))
        if self.training_gpus > 0:
            subs.append(self._cortex_training_sub_job())
        return subs

    def _cortex_training_sub_job(self) -> dict[str, Any]:
        ds = self.training.ds_config or {}
        worker = self.training.ds_worker_config or {}
        training: dict[str, Any] = {"max_seq_len": self.max_seq_len, "n_gpus": self.training_gpus}
        if "train_batch_size" in ds:
            training["train_batch_size"] = ds["train_batch_size"]
        if "gradient_clipping" in ds:
            training["gradient_clipping"] = ds["gradient_clipping"]
        optimizer = _neutrino_optimizer(ds.get("optimizer"))
        if optimizer:
            training["optimizer"] = optimizer
        if "enable_gradient_checkpointing" in worker:
            training["activation_checkpointing"] = worker["enable_gradient_checkpointing"]
        provider = worker.get("model_provider") or ("liger" if worker.get("use_liger") else None)
        if provider:
            training["model_provider"] = provider
        if worker.get("attn_implementation"):
            training["attn_implementation"] = worker["attn_implementation"]
        # Neutrino-only engine knobs ride along in ds_worker_config; on-prem ignores them.
        for key in ("ep_size", "mb_spec"):
            if key in worker:
                training[key] = worker[key]
        if ds:
            training["ds_config"] = _without_noop_offload(ds)
        if self.training.peft:
            training["peft_config"] = self.training.peft
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
        if self.training.peft:
            inference["peft_config"] = self.training.peft
        return self._cortex_sub_job(job_type, {"inference_config": inference})

    def _cortex_sub_job(self, job_type: str, extra: dict[str, Any]) -> dict[str, Any]:
        sub: dict[str, Any] = {"job_type": job_type, "model_name": self.model_name, **extra}
        if self.dtype is not None:
            sub["dtype"] = self.dtype
        if self.seed is not None:
            sub["seed"] = self.seed
        return sub


def _neutrino_optimizer(ds_optimizer: Any) -> dict[str, Any] | None:
    """DeepSpeed ``{"type", "params": {...}}`` → Neutrino ``{"name", lr, betas, ...}``."""
    if not isinstance(ds_optimizer, dict):
        return None
    params = ds_optimizer.get("params") or {}
    return {"name": ds_optimizer.get("type", "AdamW"), **params}


def _without_noop_offload(ds_config: dict[str, Any]) -> dict[str, Any]:
    """Drop ``offload_optimizer/offload_param: {device: none}`` from a ds_config.

    To DeepSpeed, ``device: none`` and an absent key mean the same thing; to
    Neutrino they don't. It builds the optimizer from the typed ``optimizer``
    field lifted above and settles on ``DeepSpeedCPUAdam``, so forwarding an
    explicit no-op offload block moves only the parameters onto the GPU and the
    first ``step()`` dies with::

        AssertionError: CPUAdam param is on cuda:0 and must be 'cpu'

    SkyRL's arctic_rl config spells the no-op out while the standalone recipes
    omit it, which is why only the framework path hits this. A real
    ``device: cpu`` request is left alone.
    """
    zero = ds_config.get("zero_optimization")
    if not isinstance(zero, dict):
        return ds_config
    kept = {
        key: value
        for key, value in zero.items()
        if key not in ("offload_optimizer", "offload_param")
        or not (isinstance(value, dict) and str(value.get("device", "none")).lower() == "none")
    }
    if len(kept) == len(zero):
        return ds_config
    return {**ds_config, "zero_optimization": kept}
