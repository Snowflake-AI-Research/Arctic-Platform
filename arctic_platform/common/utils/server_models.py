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

"""Request models and inference config helpers for the Ray RL server."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from arctic_platform.common.config import SP_SIZE_PATHS
from arctic_platform.common.config import resolve_parallelism_degree
from arctic_platform.common.config import resolve_sp_size

DEFAULT_SEED = 42

__all__ = [
    "SP_SIZE_PATHS",
    "resolve_parallelism_degree",
    "resolve_sp_size",
    "JobConfig",
    "sp_size_from_job_config",
    "GenerateRequest",
    "LogProbsRequest",
    "WeightSyncRequest",
    "parse_arctic_inference_rollout",
    "build_model_config",
]


class JobConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_name: str
    job_type: str = Field(default="training")
    num_devices: int | None = None
    ds_config: dict | None = None
    training_config: dict | None = None
    log_prob_config: dict | None = None
    ds_worker_config: dict | None = None
    vllm_config: dict | None = None
    checkpoint_path: str | None = None
    arctic_inference_config: dict | None = None
    full_determinism: bool = False
    seed: int = DEFAULT_SEED
    # Weight-sync strategy for the source (training) job. A WeightSyncRequest may override either field for one call.
    # Meaningful only on a training job.
    cuda_ipc: bool = False
    low_memory: bool = False

    @field_validator("seed", mode="before")
    @classmethod
    def _coerce_null_seed(cls, value: Any) -> Any:
        return DEFAULT_SEED if value is None else value

    @model_validator(mode="after")
    def _validate_sp_size(self) -> "JobConfig":
        # Resolve at input time so a bad or self-contradicting parallel degree
        # fails when the job is created, not deep inside a worker.
        resolve_sp_size(self.model_dump())
        return self

    @property
    def sp_size(self) -> int:
        """Sequence-parallel degree declared by this job, or 1 when unset.

        The nested config blobs are free-form dicts from several clients, so the
        degree is derived here instead of being re-read at each call site. Not a
        ``computed_field``: ``model_dump()`` is forwarded verbatim to workers and
        must keep the wire shape the client sent.
        """
        return resolve_sp_size(self.model_dump())


def sp_size_from_job_config(job_config: Any) -> int:
    """``JobConfig.sp_size``, also accepting the serialized dict a worker receives."""
    if job_config is None:
        return 1
    if isinstance(job_config, JobConfig):
        return job_config.sp_size
    if hasattr(job_config, "model_dump"):
        job_config = job_config.model_dump()
    if not isinstance(job_config, dict):
        raise TypeError(f"job_config must be a dict or JobConfig; got {type(job_config).__name__}")
    return resolve_sp_size(job_config)


class GenerateRequest(BaseModel):
    # Token-id prompts are what the TRL rollout sends, so the server does not
    # tokenize the text a second time and insert specials the training row lacks.
    prompts: list[str | list[int]]
    sampling_params: dict[str, Any] | None = None
    routing_key: Any = None
    strict: bool = False


class LogProbsRequest(BaseModel):
    prompts: list[str]
    completions: list[str] | None = None
    top_k: int = 1


class WeightSyncRequest(BaseModel):
    # Matches Cortex's `weight_sync(source_sub_job_id, target_sub_job_ids)`.
    # On-prem treats a sub_job_id as its plain job id.
    # cuda_ipc / low_memory: None uses the training job's JobConfig values.
    source_sub_job_id: int
    target_sub_job_ids: list[int]
    cuda_ipc: bool | None = None
    low_memory: bool | None = None


def parse_arctic_inference_rollout(arctic_inference_config, model_config_fields=None):
    if not arctic_inference_config:
        return {}
    out = {}
    fields = model_config_fields or set()

    zorro = arctic_inference_config.get("zorro_inference")
    if isinstance(zorro, dict) and zorro.get("enable"):
        if "use_fca" in fields:
            out["use_fca"] = True

    spec = arctic_inference_config.get("speculative_decoding")
    if isinstance(spec, dict):
        model = (spec.get("model") or "").strip()
        if model and "spec_model" in fields:
            out["spec_model"] = model

    return out


def build_model_config(
    model_name: str,
    vllm_config: dict | None,
    arctic_inference_config: dict | None = None,
):
    """Construct a :class:`ModelConfig` from user-supplied vllm_config dict.

    ``arctic_inference_config`` carries Arctic-platform signals (e.g. use_fca,
    spec_model) that are not vLLM engine args: they are recorded on the
    ModelConfig, which expands them into real engine kwargs in
    ``ModelConfig.to_engine_kwargs()``.
    """
    from arctic_platform.inference.server.config import ModelConfig

    cfg = dict(vllm_config or {})
    cfg["model"] = model_name
    known_fields = set(ModelConfig.model_fields.keys())
    cfg.update(parse_arctic_inference_rollout(arctic_inference_config, known_fields))
    extra = {k: v for k, v in cfg.items() if k not in known_fields}
    base = {k: v for k, v in cfg.items() if k in known_fields}
    if extra:
        base["extra_engine_kwargs"] = extra
    return ModelConfig(**base)
