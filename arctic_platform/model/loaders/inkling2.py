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

"""Inkling2 text training with expert parallelism."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from arctic_platform.model.config import ActivationCheckpointConfig
from arctic_platform.model.config import ModelSpec
from arctic_platform.model.implementations.moe.config import EPCommBackend
from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import register_loader


class Inkling2Options(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    seq_len: int = Field(4096, gt=0)
    ep_comm_backend: EPCommBackend = "deepep"
    deepep_num_sms: int = Field(20, gt=0, multiple_of=2)
    reduce_dtype: Literal["bfloat16", "float32"] = "float32"
    moe_use_grouped_mm: bool = True
    deepep_token_chunk_size: int | None = Field(None, gt=0)
    attention_query_chunk_size: int = Field(2048, gt=0, multiple_of=128)
    fused_lm_head_token_chunk_size: int | Literal["disabled"] = "disabled"
    fp32_lm_head: bool = False
    ac_config: ActivationCheckpointConfig | None = None

    @field_validator("fused_lm_head_token_chunk_size")
    @classmethod
    def _positive_head_chunk(cls, value):
        if isinstance(value, int) and value <= 0:
            raise ValueError("fused_lm_head_token_chunk_size must be positive or disabled")
        return value

    @model_validator(mode="before")
    @classmethod
    def _native_head(cls, value):
        options = dict(value)
        if options.pop("fused_cross_entropy", False) is not False:
            raise ValueError("Inkling2 training uses the native LM head")
        return options


def _matches(ctx: LoaderContext) -> bool:
    return ctx.hf_model_type == "inkling2"


def _validate_spec(spec: ModelSpec) -> None:
    if spec.attn_implementation is None:
        spec.attn_implementation = "flex_attention"
    if spec.attn_implementation != "flex_attention":
        raise ValueError("Inkling2 training requires flex_attention for its relative-position bias and sinks")
    if spec.parallelism.sequence_parallel > 1:
        raise ValueError("Inkling2 training does not support sequence parallelism yet")
    if spec.dtype not in {"bfloat16", "float32"}:
        raise ValueError("Inkling2 training dtype must be bfloat16 or float32")
    patches = spec.patches.model_dump(exclude={"peft"})
    if any(patches.values()):
        raise ValueError("Inkling2 uses loader_options.ac_config and supports only the PEFT model patch")
    options = Inkling2Options.model_validate(spec.loader_options)
    if options.ac_config is not None and options.ac_config.mode != "full":
        raise ValueError("Inkling2 supports full activation checkpointing")


@register_loader("inkling2", matches=_matches, options=Inkling2Options, validate_spec=_validate_spec)
def load_inkling2(ctx: LoaderContext) -> LoadedModel:
    from arctic_platform.model.implementations.inkling2.deepspeed_integration import load_inkling2_model

    groups = ctx.parallel_groups or {}
    if ctx.spec.parallelism.expert_parallel > 1 and groups.get("ep_group") is None:
        from arctic_platform.model.parallel.groups import expert_parallel_group
        from arctic_platform.model.parallel.groups import setup_replicated_expert_groups

        setup_replicated_expert_groups(ctx.spec.parallelism.expert_parallel)
        groups = {**groups, "ep_group": expert_parallel_group(ctx.spec.parallelism.expert_parallel)}
    return LoadedModel(
        model=load_inkling2_model(
            model_name=ctx.spec.model_path_or_name,
            optimization_dtype=ctx.spec.dtype,
            attn_implementation=ctx.spec.attn_implementation,
            ep_size=ctx.spec.parallelism.expert_parallel,
            ep_group=groups.get("ep_group"),
            options=Inkling2Options.model_validate(ctx.spec.loader_options),
        )
    )
