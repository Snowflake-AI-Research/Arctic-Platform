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
"""Loader for the carved-out Qwen3.5 MoE implementation (meta-init, EP/DeepEP, DeepSpeed)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from typing_extensions import Self

from arctic_platform.model.config import ActivationCheckpointConfig
from arctic_platform.model.config import ModelSpec
from arctic_platform.model.implementations.moe.config_validation import validate_lm_head_fused_ce_config
from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import register_loader

# Declarative options owned by the qwen3_5_moe loader.


class DebugModelOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    random_init: bool = False
    num_layers: int | None = Field(None, gt=0)
    gradient_sample_max_numel: int = Field(0, ge=0)
    full_determinism: bool = False


class Qwen3_5MoeOptions(BaseModel):
    """Validated ``loader_options`` for the qwen3_5_moe loader."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    seq_len: int = Field(4096, gt=0, description="Training sequence length.")
    trust_remote_code: bool = False
    ep_comm_backend: Literal["deepep", "uccl"] = Field("deepep", description="Expert-parallel comm backend.")
    deepep_num_sms: int = Field(20, gt=0, multiple_of=2)
    reduce_dtype: Literal["bfloat16", "float32"] = Field("float32", description="Gradient reduction dtype.")
    moe_use_grouped_mm: bool = Field(True, description="Use grouped matmul for experts.")
    fused_cross_entropy: bool | Literal["liger"] = Field("liger", description="LM-head fused CE backend.")
    fused_lm_head_token_chunk_size: int | Literal["auto", "disabled"] = Field(
        "disabled", description="Chunked LM-head logprobs token size."
    )
    fp32_lm_head: bool = Field(False, description="Compute the LM head in fp32.")
    tiled_mlp_token_chunk_size: int | None = Field(None, gt=0, description="ALST tiled shared-expert MLP token chunk.")
    deepep_token_chunk_size: int | None = Field(None, gt=0, description="DeepEP dispatch token chunk size.")
    # Mirrors ModelConfig.weight_conversion_cache_dir (the implementation's effective default).
    weight_conversion_cache_dir: str | None = Field(
        None, description="Dir for the one-time HF<->Prime weight-conversion cache."
    )
    ac_config: ActivationCheckpointConfig | None = Field(None, description="Activation checkpointing config.")
    debug: DebugModelOptions | None = Field(None, description="Test-only tiny-model overrides.")

    @model_validator(mode="after")
    def _check_lm_head(self) -> Self:
        validate_lm_head_fused_ce_config(self.model_dump())
        return self


def _matches(ctx: LoaderContext) -> bool:
    if ctx.spec.parallelism.expert_parallel <= 1:
        return False
    model_type = getattr(ctx.hf_config, "model_type", "") or ""
    text_config = getattr(ctx.hf_config, "text_config", None)
    return model_type in ("qwen3_5_moe", "qwen3_5_moe_text") or (
        getattr(text_config, "model_type", None) == "qwen3_5_moe_text"
    )


def _validate_spec(spec: ModelSpec) -> None:
    if spec.attn_implementation is None:
        spec.attn_implementation = "flash_attention_3"
    if spec.dtype not in ("bfloat16", "float32"):
        raise ValueError("qwen3_5_moe dtype must be 'bfloat16' or 'float32'")
    if spec.patches.peft is not None:
        raise ValueError("qwen3_5_moe PEFT requires expert adapter integration, which is not yet supported")
    if spec.patches.liger:
        raise ValueError(
            "the qwen3_5_moe loader does not support the liger patch; "
            'use loader_options={"fused_cross_entropy": "liger"} for the LM head instead'
        )
    if spec.patches.gradient_checkpointing or spec.patches.zorro_train:
        raise ValueError("qwen3_5_moe uses loader_options.ac_config and does not support generic forward patches")


@register_loader(
    "qwen3_5_moe",
    matches=_matches,
    options=Qwen3_5MoeOptions,
    validate_spec=_validate_spec,
)
def load_qwen3_5_moe(ctx: LoaderContext) -> LoadedModel:
    parallelism = ctx.spec.parallelism
    groups = ctx.parallel_groups or {}
    if groups.get("ep_group") is None:
        raise ValueError("qwen3_5_moe requires parallel_groups['ep_group'] from the runtime")
    if parallelism.sequence_parallel > 1 and groups.get("sp_group") is None:
        raise ValueError("qwen3_5_moe requires parallel_groups['sp_group'] when sequence_parallel > 1")

    from arctic_platform.model.implementations.qwen35 import load_qwen3_5_moe_model

    options = Qwen3_5MoeOptions.model_validate(ctx.spec.loader_options)
    assert ctx.spec.attn_implementation is not None
    model = load_qwen3_5_moe_model(
        model_name=ctx.spec.model_path_or_name,
        optimization_dtype=ctx.spec.dtype,
        attn_implementation=ctx.spec.attn_implementation,
        ep_size=parallelism.expert_parallel,
        sp_size=parallelism.sequence_parallel,
        sp_group=groups.get("sp_group"),
        ep_group=groups["ep_group"],
        options=options,
    )
    return LoadedModel(model=model)
