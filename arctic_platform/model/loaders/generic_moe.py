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
"""Loader for generic PrimeRL MoE families ported from DSS #496.

Covers Qwen3 MoE, GLM-4.5, MiniMax M2, AFMoE, and Nemotron H. Expert parallel
uses the shared DeepSpeed path. Sequence parallelism above one is rejected
until a family-specific parity case exists.
"""

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
from arctic_platform.model.loaders.qwen3_5_moe import DebugModelOptions
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

GENERIC_MOE_MODEL_TYPES = frozenset({"qwen3_moe", "glm4_moe", "minimax", "minimax_m2", "afmoe", "nemotron_h"})


class GenericMoeOptions(BaseModel):
    """Validated ``loader_options`` for the generic MoE families."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    seq_len: int = Field(4096, gt=0)
    trust_remote_code: bool = False
    ep_comm_backend: Literal["deepep", "uccl"] = "deepep"
    deepep_num_sms: int = Field(20, gt=0, multiple_of=2)
    reduce_dtype: Literal["bfloat16", "float32"] = "float32"
    moe_use_grouped_mm: bool = True
    fused_cross_entropy: bool | Literal["liger"] = False
    fused_lm_head_token_chunk_size: int | Literal["auto", "disabled"] = "disabled"
    fp32_lm_head: bool = False
    tiled_mlp_token_chunk_size: int | None = Field(None, gt=0)
    deepep_token_chunk_size: int | None = Field(None, gt=0)
    weight_conversion_cache_dir: str | None = None
    ac_config: ActivationCheckpointConfig | None = None
    debug: DebugModelOptions | None = None

    @model_validator(mode="after")
    def _check_lm_head(self) -> Self:
        validate_lm_head_fused_ce_config(self.model_dump())
        return self


def _matches(ctx: LoaderContext) -> bool:
    if ctx.spec.parallelism.expert_parallel <= 1:
        return False
    return ctx.hf_model_type in GENERIC_MOE_MODEL_TYPES or ctx.hf_text_model_type in GENERIC_MOE_MODEL_TYPES


def _validate_spec(spec: ModelSpec) -> None:
    if spec.attn_implementation is None:
        spec.attn_implementation = "flash_attention_3"
    if spec.dtype not in ("bfloat16", "float32"):
        raise ValueError("generic MoE dtype must be 'bfloat16' or 'float32'")
    if spec.parallelism.sequence_parallel > 1:
        raise ValueError("generic MoE families do not support sequence parallelism")
    if spec.patches.peft is not None:
        raise ValueError("generic MoE PEFT requires expert adapter integration, which is not yet supported")
    if spec.patches.liger:
        raise ValueError(
            "the generic MoE loader does not support the liger patch; "
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
        raise ValueError(
            "generic MoE families use loader_options.ac_config and do not support generic forward patches"
        )


@register_loader(
    "generic_moe",
    matches=_matches,
    options=GenericMoeOptions,
    validate_spec=_validate_spec,
)
def load_generic_moe(ctx: LoaderContext) -> LoadedModel:
    parallelism = ctx.spec.parallelism
    groups = ctx.parallel_groups or {}
    if groups.get("ep_group") is None:
        raise ValueError("generic MoE requires parallel_groups['ep_group'] from the runtime")

    from arctic_platform.model.implementations.qwen35.deepspeed_integration import load_generic_moe_model

    options = Qwen3_5MoeOptions.model_validate(ctx.spec.loader_options)
    assert ctx.spec.attn_implementation is not None
    model = load_generic_moe_model(
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
