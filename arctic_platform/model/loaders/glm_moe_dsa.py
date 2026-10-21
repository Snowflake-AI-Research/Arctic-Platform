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
"""Loader for AP's GLM MoE DSA implementation."""

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


class GlmMoeDsaDebugOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    num_layers: int | None = Field(None, gt=0)
    random_init: bool = False
    init_seed: int = 42
    first_k_dense_replace: int | None = Field(None, ge=0)
    skip_attention: bool = False
    skip_mlp: bool = False
    hidden_size: int | None = Field(None, gt=0)
    intermediate_size: int | None = Field(None, gt=0)
    moe_intermediate_size: int | None = Field(None, gt=0)


class GlmMoeDsaOptions(BaseModel):
    """Validated ``loader_options`` for GLM-5.2/5.3 MoE models."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    seq_len: int = Field(4096, gt=0)
    trust_remote_code: bool = True
    ep_comm_backend: Literal["deepep", "uccl"] = "deepep"
    sparse_mla_backend: Literal["flashmla", "ref", "tilelang", "dense"] = "ref"
    deepep_num_sms: int = Field(20, gt=0, multiple_of=2)
    deepep_token_chunk_size: int | None = Field(None, gt=0)
    reduce_dtype: Literal["bfloat16", "float32"] = "float32"
    moe_use_grouped_mm: bool = False
    fused_cross_entropy: bool | Literal["liger", "quack"] = "liger"
    fused_lm_head_token_chunk_size: int | Literal["auto", "disabled"] = "disabled"
    fp32_lm_head: bool = False
    tiled_mlp_token_chunk_size: int | None = Field(None, gt=0)
    weight_conversion_cache_dir: str | None = None
    ac_config: ActivationCheckpointConfig | None = None
    debug: GlmMoeDsaDebugOptions | None = None

    @model_validator(mode="after")
    def _normalize_lm_head(self) -> Self:
        validate_lm_head_fused_ce_config(self.model_dump())
        if self.fused_cross_entropy == "quack" and self.fp32_lm_head:
            raise ValueError("fp32_lm_head is not supported with fused_cross_entropy='quack'")
        return self


def _matches(ctx: LoaderContext) -> bool:
    if ctx.spec.parallelism.expert_parallel <= 1:
        return False
    return ctx.hf_model_type == "glm_moe_dsa" or ctx.hf_text_model_type == "glm_moe_dsa"


def _validate_spec(spec: ModelSpec) -> None:
    if spec.attn_implementation is None:
        spec.attn_implementation = "flash_attention_2"
    if spec.dtype not in ("bfloat16", "float32"):
        raise ValueError("glm_moe_dsa dtype must be 'bfloat16' or 'float32'")
    if spec.parallelism.sequence_parallel > 1:
        raise ValueError("glm_moe_dsa does not support sequence parallelism")
    if spec.patches.liger:
        raise ValueError(
            "the glm_moe_dsa loader does not support the liger patch; "
            'use loader_options={"fused_cross_entropy": "liger"} for the LM head instead'
        )
    if (
        spec.patches.gradient_checkpointing
        or spec.patches.activation_offload
        or spec.patches.compile
        or spec.patches.tiled_mlp
        or spec.patches.lm_head
        or spec.patches.zorro_train
    ):
        raise ValueError("glm_moe_dsa uses loader_options.ac_config and does not support generic forward patches")


@register_loader(
    "glm_moe_dsa",
    matches=_matches,
    options=GlmMoeDsaOptions,
    validate_spec=_validate_spec,
)
def load_glm_moe_dsa(ctx: LoaderContext) -> LoadedModel:
    parallelism = ctx.spec.parallelism
    groups = ctx.parallel_groups or {}
    if groups.get("ep_group") is None:
        raise ValueError("glm_moe_dsa requires parallel_groups['ep_group'] from the runtime")

    from arctic_platform.model.implementations.glm52 import load_glm_moe_dsa_model

    options = GlmMoeDsaOptions.model_validate(ctx.spec.loader_options)
    assert ctx.spec.attn_implementation is not None
    model = load_glm_moe_dsa_model(
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
