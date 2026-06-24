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

from arctic_inference.server.config import ModelConfig
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


class GenerateRequest(BaseModel):
    prompts: list[str]
    sampling_params: dict[str, Any] | None = None


class LogProbsRequest(BaseModel):
    prompts: list[str]
    completions: list[str] | None = None
    top_k: int = 1


class SyncWeightsRequest(BaseModel):
    training_job_id: int
    sampling_job_id: int
    colocate: bool = False
    cuda_ipc: bool = False
    low_memory: bool = False


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
) -> ModelConfig:
    """Construct a :class:`ModelConfig` from user-supplied vllm_config dict.

    ``arctic_inference_config`` carries Arctic-platform signals (e.g. use_fca,
    spec_model) that are not vLLM engine args: they are recorded on the
    ModelConfig, which expands them into real engine kwargs in
    ``ModelConfig.to_engine_kwargs()``.
    """
    cfg = dict(vllm_config or {})
    cfg["model"] = model_name
    known_fields = set(ModelConfig.model_fields.keys())
    cfg.update(parse_arctic_inference_rollout(arctic_inference_config, known_fields))
    extra = {k: v for k, v in cfg.items() if k not in known_fields}
    base = {k: v for k, v in cfg.items() if k in known_fields}
    if extra:
        base["extra_engine_kwargs"] = extra
    return ModelConfig(**base)
