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
"""Loader for DeepSeek-V4.1 text (``deepseek_v41`` / ``deepseek_v41_text``)."""

from __future__ import annotations

import json
from pathlib import Path

from arctic_platform.model.config import ModelSpec
from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import register_loader
from arctic_platform.model.loaders.generic_moe import GenericMoeOptions
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

DEEPSEEK_V41_ATTN_BACKEND = "eager"
_MODEL_TYPES = frozenset({"deepseek_v41", "deepseek_v41_text"})


def _matches(ctx: LoaderContext) -> bool:
    if ctx.spec.parallelism.expert_parallel <= 1:
        return False
    return ctx.hf_model_type in _MODEL_TYPES or ctx.hf_text_model_type in _MODEL_TYPES


def _routed_experts(spec: ModelSpec) -> int | None:
    config_path = Path(spec.model_path_or_name) / "config.json"
    if not config_path.is_file():
        return None
    payload = json.loads(config_path.read_text())
    text = payload.get("text_config", payload)
    count = text.get("n_routed_experts")
    return None if count is None else int(count)


def _validate(spec: ModelSpec) -> None:
    if spec.attn_implementation is None:
        spec.attn_implementation = DEEPSEEK_V41_ATTN_BACKEND
    if spec.attn_implementation != DEEPSEEK_V41_ATTN_BACKEND:
        raise ValueError(
            "DeepSeek V4.1 attention is the fused sparse kernel inside the model; "
            f"got attn_implementation={spec.attn_implementation!r}"
        )
    if spec.dtype not in ("bfloat16", "float32"):
        raise ValueError("DeepSeek V4.1 dtype must be 'bfloat16' or 'float32'")
    if spec.patches.liger:
        raise ValueError(
            "the DeepSeek V4.1 loader does not support the liger patch; "
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
        raise ValueError("DeepSeek V4.1 uses loader_options.ac_config and does not support generic forward patches")
    n_experts = _routed_experts(spec)
    ep_size = spec.parallelism.expert_parallel
    if n_experts is not None and n_experts % ep_size:
        raise ValueError(
            f"DeepSeek V4.1 has {n_experts} routed experts, so ep_size={ep_size} must divide {n_experts}."
        )


def _load(ctx: LoaderContext) -> LoadedModel:
    groups = ctx.parallel_groups or {}
    if groups.get("ep_group") is None:
        raise ValueError("DeepSeek V4.1 requires parallel_groups['ep_group'] from the runtime")
    options = Qwen3_5MoeOptions.model_validate(ctx.spec.loader_options)
    assert ctx.spec.attn_implementation is not None
    from arctic_platform.model.implementations.deepseek_v41.deepspeed_integration import load_deepseek_v41_model

    model = load_deepseek_v41_model(
        model_name=ctx.spec.model_path_or_name,
        optimization_dtype=ctx.spec.dtype,
        attn_implementation=ctx.spec.attn_implementation,
        ep_size=ctx.spec.parallelism.expert_parallel,
        sp_size=ctx.spec.parallelism.sequence_parallel,
        sp_group=groups.get("sp_group"),
        ep_group=groups["ep_group"],
        options=options,
        # Packed MXFP4 experts are frozen, so only LoRA trains on them as published.
        dequantize_checkpoint=ctx.spec.patches.peft is None,
    )
    return LoadedModel(model=model)


@register_loader(
    "deepseek_v41",
    matches=_matches,
    options=GenericMoeOptions,
    validate_spec=_validate,
)
def load_deepseek_v41(ctx: LoaderContext) -> LoadedModel:
    return _load(ctx)
