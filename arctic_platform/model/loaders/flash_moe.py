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
"""Loaders for GLM-5.3-Flash and Qwen3.8-Flash-Next.

These families use the shared DeepSpeed MoE lifecycle with their own adapters.
Sequence parallelism above one is rejected. Qwen3.8 expert parallel must divide
512, and its weight sync stays on the Hugging Face iterator.
"""

from __future__ import annotations

from arctic_platform.model.config import ModelSpec
from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import register_loader
from arctic_platform.model.loaders.generic_moe import GenericMoeOptions
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions


def _matches(model_type: str):
    def matches(ctx: LoaderContext) -> bool:
        if ctx.spec.parallelism.expert_parallel <= 1:
            return False
        return ctx.hf_model_type == model_type or ctx.hf_text_model_type == model_type

    return matches


def _validate_common(spec: ModelSpec, family: str) -> None:
    if spec.attn_implementation is None:
        spec.attn_implementation = "sdpa"
    if spec.dtype not in ("bfloat16", "float32"):
        raise ValueError(f"{family} dtype must be 'bfloat16' or 'float32'")
    if spec.parallelism.sequence_parallel > 1:
        raise NotImplementedError(f"Sequence parallelism is not implemented for {family}.")
    if spec.patches.peft is not None:
        raise ValueError(f"{family} PEFT requires expert adapter integration, which is not yet supported")
    if spec.patches.liger:
        raise ValueError(
            f"the {family} loader does not support the liger patch; "
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
        raise ValueError(f"{family} uses loader_options.ac_config and does not support generic forward patches")


def _validate_glm5_next(spec: ModelSpec) -> None:
    _validate_common(spec, "GLM-5.3-Flash")


def _validate_qwen4_exp(spec: ModelSpec) -> None:
    from arctic_platform.model.implementations.qwen38.deepspeed_integration import QWEN38_NUM_EXPERTS

    _validate_common(spec, "Qwen3.8-Flash-Next")
    ep_size = spec.parallelism.expert_parallel
    if QWEN38_NUM_EXPERTS % ep_size:
        raise ValueError(
            f"Qwen3.8-Flash-Next has {QWEN38_NUM_EXPERTS} experts, "
            f"so ep_size={ep_size} must divide {QWEN38_NUM_EXPERTS}."
        )


def _load(ctx: LoaderContext, load_model) -> LoadedModel:
    groups = ctx.parallel_groups or {}
    if groups.get("ep_group") is None:
        raise ValueError("flash MoE requires parallel_groups['ep_group'] from the runtime")
    options = Qwen3_5MoeOptions.model_validate(ctx.spec.loader_options)
    assert ctx.spec.attn_implementation is not None
    model = load_model(
        model_name=ctx.spec.model_path_or_name,
        optimization_dtype=ctx.spec.dtype,
        attn_implementation=ctx.spec.attn_implementation,
        ep_size=ctx.spec.parallelism.expert_parallel,
        sp_size=ctx.spec.parallelism.sequence_parallel,
        sp_group=groups.get("sp_group"),
        ep_group=groups["ep_group"],
        options=options,
    )
    return LoadedModel(model=model)


@register_loader(
    "glm5_next",
    matches=_matches("glm5_next"),
    options=GenericMoeOptions,
    validate_spec=_validate_glm5_next,
)
def load_glm5_next(ctx: LoaderContext) -> LoadedModel:
    from arctic_platform.model.implementations.glm53.deepspeed_integration import load_glm5_next_model

    return _load(ctx, load_glm5_next_model)


@register_loader(
    "qwen4_exp",
    matches=_matches("qwen4_exp"),
    options=GenericMoeOptions,
    validate_spec=_validate_qwen4_exp,
)
def load_qwen4_exp(ctx: LoaderContext) -> LoadedModel:
    from arctic_platform.model.implementations.qwen38.deepspeed_integration import load_qwen4_exp_model

    return _load(ctx, load_qwen4_exp_model)
