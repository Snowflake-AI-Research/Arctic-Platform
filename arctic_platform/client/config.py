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
    ├── training     # engine and model settings (optimizer, mb_spec, ds_config, attn, model_options) + checkpoint
    ├── sampling     # sampling / log-prob engines (vllm, log_prob_engine, ...)
    └── backend: OnPremConfig | CortexConfig  # connection / deployment only

``to_onprem`` / ``to_cortex`` are temporary wire adapters. Drop them once both
servers accept this canonical shape directly.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from typing import Literal

from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SecretStr
from pydantic import field_validator
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
from typing_extensions import Self

from arctic_platform.common.config import validate_peft_config
from arctic_platform.common.model_settings import ModelSettings
from arctic_platform.common.model_settings import worker_config_model_keys
from arctic_platform.common.training_settings import WorkerTrainingSettings

JobId = int | str


class OnPremConfig(BaseModel):
    """Backend-specific settings for the on-prem in-process Ray server."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    type: Literal["onprem"] = "onprem"
    colocate: bool = Field(False, description="onprem: colocate job types on shared GPUs.")


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
    verify_ssl: bool = Field(True, description="cortex: verify HTTPS certificates.")
    max_retries: int = Field(10, ge=0, description="cortex: transient-failure retries per HTTP request (tenacity).")
    request_timeout: float = Field(1800.0, gt=0, description="Maximum seconds allowed for one Cortex HTTP request.")
    poll_timeout: float = Field(
        1800.0, gt=0, description="Maximum total seconds to poll a submitted Cortex operation."
    )
    job_ready_timeout: float = Field(
        1800.0, gt=0, description="Maximum total seconds to wait for a Cortex job to reach RUNNING."
    )
    # Every job on a shared account has the same submitter, so put a person or
    # run name here. Existing launchers export CORTEX_JOB_COMMENT.
    job_comment: str | None = Field(
        None,
        validation_alias=AliasChoices("job_comment", "ARCTIC_CORTEX_JOB_COMMENT", "CORTEX_JOB_COMMENT"),
        description="cortex: comment stored on the job at create time.",
    )
    # The server reclaims a job nobody is using, so a client that is killed
    # outright still releases its GPUs. None keeps the server default (30 min).
    idle_timeout_seconds: int | None = Field(
        None,
        description="cortex: seconds a job may sit idle before the server reclaims it; 0 disables, else 300-604800.",
    )
    pending_timeout_seconds: int | None = Field(
        None,
        ge=300,
        le=604_800,
        description="cortex: seconds a job may wait for capacity before the server fails it (default 24h).",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not (self.base_url or self.host):
            raise ValueError("cortex: set base_url (direct URL) or host (PAT auth).")
        if self.host and not self.base_url:
            if not (self.database and self.schema_):
                raise ValueError("cortex: database + schema required for host/PAT auth.")
            if not (self.pat and self.pat.get_secret_value()):
                raise ValueError("cortex: no PAT — set `pat` or ARCTIC_CORTEX_PAT for host auth.")
        idle = self.idle_timeout_seconds
        if idle is not None and idle != 0 and not 300 <= idle <= 604_800:
            raise ValueError("cortex: idle_timeout_seconds must be 0 or between 300 and 604800.")
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


class TrainingConfig(WorkerTrainingSettings):
    """Training job: the engine and model settings every backend reads, plus job settings.

    Each ``step`` applies one optimizer update; its ``learning_rate`` overrides a
    ``ds_config`` scheduler. Allocation (GPUs, max_seq_len) stays on the client.
    PEFT is ``peft``, not ``peft_config``.
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
    ds_worker_config: dict[str, Any] | None = Field(
        None,
        description="On-prem worker runtime knobs (zorro_train_enable, use_autocast, logits_optimization, ...).",
    )
    peft: dict[str, Any] | None = Field(
        None,
        description=(
            "PEFT adapter config (peft_type, r, lora_alpha, lora_dropout, bias, target_modules). "
            "None = dense fine-tuning; an empty config is invalid. On-prem supports training; "
            "Cortex also configures the sampling engine to serve the adapter."
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

    _validate_peft = field_validator("peft")(validate_peft_config)

    @field_validator("ds_worker_config")
    @classmethod
    def _reject_moved_worker_keys(cls, worker: dict[str, Any] | None) -> dict[str, Any] | None:
        if worker and "mb_spec" in worker:
            raise ValueError("training.ds_worker_config.mb_spec moved to training.mb_spec")
        moved = worker_config_model_keys(worker)
        if moved:
            raise ValueError(
                "training.ds_worker_config may not set model settings; set "
                + ", ".join(
                    f"training.{'peft' if field == 'peft_config' else field} instead of {key}"
                    for key, field in moved.items()
                )
            )
        return worker

    @model_validator(mode="after")
    def _reject_peft_config(self) -> Self:
        if self.peft_config is not None:
            raise ValueError("set training.peft, not training.peft_config")
        return self

    def model_settings(self) -> dict[str, Any]:
        """The ``ModelSettings`` the job set, after the ``model_options`` rewrite, with ``peft`` as ``peft_config``.

        Unset and ``None`` fields are left out, so they keep the server's default.
        """
        dumped = self.model_dump(include=set(ModelSettings.model_fields) - {"peft_config"}, exclude_unset=True)
        settings = {key: value for key, value in dumped.items() if value is not None}
        if self.peft is not None:
            settings["peft_config"] = dict(self.peft)
        return settings

    def engine_settings(self) -> dict[str, Any]:
        """The on-prem ``training_config`` wire block: every engine setting explicit, plus ``model_settings``."""
        engine = set(WorkerTrainingSettings.model_fields) - set(ModelSettings.model_fields)
        return {**self.model_dump(include=engine), **self.model_settings()}


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
        if (
            self.training.peft is not None
            and self.backend.type == "onprem"
            and (self.sampling_gpus > 0 or self.sampling_job_id is not None)
        ):
            raise ValueError(
                "On-prem PEFT supports training only; adapter sync to sampling is not implemented. "
                "Use sampling_gpus=0 without a sampling_job_id, or switch to CortexConfig."
            )
        return self

    def gpus_for(self, job_type: str) -> int:
        """GPU count allocated to a job type (0 == the job type is disabled)."""
        return getattr(self, f"{job_type}_gpus")

    # ── temporary wire adapters ──────────────────────────────────────────────
    # TODO(config-unify): delete to_onprem / to_cortex once both servers accept
    # this nesting (training / sampling.vllm) directly.

    def to_onprem(self, job_type: str) -> dict[str, Any]:
        """Translate into one on-prem ``initialize`` payload."""
        tc, sc = self.training, self.sampling
        payload: dict[str, Any] = {"model_name": self.model_name, "job_type": job_type}
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.dtype is not None:
            payload["dtype"] = self.dtype
        # log-prob runs on DeepSpeed only when asked; otherwise it's a vLLM engine.
        use_deepspeed = job_type == "training" or (job_type == "log_prob" and sc.log_prob_engine == "deepspeed")
        if use_deepspeed:
            payload["max_seq_len"] = self.max_seq_len
            payload["full_determinism"] = tc.full_determinism
            if tc.ds_worker_config:
                payload["ds_worker_config"] = dict(tc.ds_worker_config)
            if job_type == "training":
                payload["training_config"] = tc.engine_settings()
                if tc.checkpoint_path:
                    payload["checkpoint_path"] = tc.checkpoint_path
                # Bake the (static) weight-sync strategy onto the training job.
                payload["cuda_ipc"] = tc.cuda_ipc
                payload["low_memory"] = tc.low_memory
            else:
                model_settings = tc.model_settings()
                model_settings.pop("peft_config", None)
                if model_settings:
                    payload["training_config"] = model_settings
                # Inherits the training SP degree unless it sets its own.
                log_prob_config = dict(sc.log_prob_ds_config or {})
                if "sequence_parallel_size" in (tc.ds_config or {}):
                    log_prob_config.setdefault("sequence_parallel_size", tc.ds_config["sequence_parallel_size"])
                payload["log_prob_config"] = log_prob_config
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

        Neutrino reads the engine and model settings as typed ``training_config``
        fields (temporary — drop once Neutrino accepts the canonical nesting directly).
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
        tc = self.training
        worker = tc.ds_worker_config or {}
        training: dict[str, Any] = {
            "max_seq_len": self.max_seq_len,
            "n_gpus": self.training_gpus,
            "optimizer": None if tc.optimizer is None else tc.optimizer.model_dump(exclude_none=True),
            "gradient_clipping": tc.gradient_clipping,
            "fp32_masters": tc.fp32_masters,
        }
        if tc.zero_bucket_size is not None:
            training["zero_bucket_size"] = tc.zero_bucket_size
        if tc.mb_spec is not None:
            if tc.mb_spec.num_micro_batches is not None:
                raise ValueError(
                    "Cortex splits fwd-bwd batches by mb_spec.max_tokens_per_mb; "
                    "mb_spec.num_micro_batches is supported on-prem only"
                )
            training["mb_spec"] = tc.mb_spec.model_dump(exclude_none=True)
        if tc.debug is not None:
            training["debug"] = deepcopy(tc.debug)
        training.update(tc.model_settings())
        # A Neutrino-only engine knob rides along in ds_worker_config; on-prem ignores it.
        if "router_replay" in worker:
            training["router_replay"] = deepcopy(worker["router_replay"])
        if tc.ds_config:
            training["ds_config"] = _without_noop_offload(tc.ds_config)
        return self._cortex_sub_job("training", {"training_config": training})

    def _cortex_inference_sub_job(self, job_type: str, n_gpus: int) -> dict[str, Any]:
        # Neutrino's InferenceConfig reads engine kwargs from nested vllm_config —
        # flattening them onto inference_config is silently ignored (extra="allow").
        vllm = dict(self.sampling.vllm)
        vllm.pop("max_model_len", None)  # owned by top-level max_seq_len
        inference: dict[str, Any] = {
            **(self.sampling.arctic_inference_config or {}),
            "max_seq_len": self.max_seq_len,
            "n_gpus": n_gpus,
        }
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


def _without_noop_offload(ds_config: dict[str, Any]) -> dict[str, Any]:
    """Drop ``offload_optimizer/offload_param: {device: none}`` from a ds_config.

    To DeepSpeed, ``device: none`` and an absent key mean the same thing; to
    Cortex they don't. It builds the optimizer from the typed ``optimizer``
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
