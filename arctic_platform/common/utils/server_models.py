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

"""Shared request models and inference config helpers for HTTP and Ray RL servers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


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
    seed: int = 42
    # Weight-sync strategy for the source (training) job. Static per run, so it
    # is baked onto the job at init instead of being resent on every weight-sync;
    # a WeightSyncRequest may still override it per call. Meaningful only on a
    # training job (the weight-sync source).
    cuda_ipc: bool = False
    low_memory: bool = False


class GenerateRequest(BaseModel):
    prompts: list[str]
    sampling_params: dict[str, Any] | None = None
    routing_key: Any = None
    strict: bool = False


class LogProbsRequest(BaseModel):
    prompts: list[str]
    completions: list[str] | None = None
    top_k: int = 1


class StepRequest(BaseModel):
    learning_rate: float | None = None


class SaveRequest(BaseModel):
    # On-prem saves to the job's configured checkpoint_path; these mirror
    # Cortex's `save(checkpoint_id, checkpoint_type)` and are accepted (unused).
    checkpoint_id: str | None = None
    checkpoint_type: str = "resumable"
    # SFT extras (optional path/step override, HF export, rotation, stage metadata).
    path: str | None = None
    step: int | None = None
    export_hf: bool = False
    save_total_limit: int | None = None
    stage_info: dict[str, Any] | None = None


class LoadCheckpointRequest(BaseModel):
    path: str | None = None
    step: int | None = None


class ResetPrefixCacheRequest(BaseModel):
    drain: bool = True
    timeout_s: float = 60.0
    retry_interval_s: float = 0.1


class WeightSyncRequest(BaseModel):
    # Matches Cortex's `weight_sync(source_sub_job_id, target_sub_job_ids)`.
    # On-prem treats a sub_job_id as its plain job id.
    #
    # colocate is intentionally absent: it is a server-launch property
    # (app.state.colocate / ArcticRLRayServerState.colocate), so the server never
    # trusts a per-call value. cuda_ipc / low_memory default to None ("unset") so
    # the server falls back to the strategy stored on the training JobConfig at
    # init; a non-None value overrides it for this one call.
    source_sub_job_id: int
    target_sub_job_ids: list[int]
    cuda_ipc: bool | None = None
    low_memory: bool | None = None


class OperationRequest(BaseModel):
    # Generic data-plane envelope, matching Cortex's /{job_id}/operation. On-prem
    # dispatches on operation_type; sub_job_id/sub_job_type are accepted for
    # parity (unused, since on-prem addresses jobs by the job_id query param).
    #
    # sub_job_id accepts int or str: on-prem clients send the plain integer job id
    # while Cortex uses a string handle. Pydantic v2 does not coerce int->str, so a
    # str-only field would 422 the on-prem callers.
    operation_type: str
    sub_job_id: int | str | None = None
    sub_job_type: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class WeightNormRequest(BaseModel):
    training_job_id: int
    sampling_job_id: int


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
    from arctic_inference.server.config import ModelConfig

    cfg = dict(vllm_config or {})
    cfg["model"] = model_name
    known_fields = set(ModelConfig.model_fields.keys())
    cfg.update(parse_arctic_inference_rollout(arctic_inference_config, known_fields))
    extra = {k: v for k, v in cfg.items() if k not in known_fields}
    base = {k: v for k, v in cfg.items() if k in known_fields}
    if extra:
        base["extra_engine_kwargs"] = extra
    return ModelConfig(**base)
