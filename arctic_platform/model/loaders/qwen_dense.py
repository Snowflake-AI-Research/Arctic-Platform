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

"""Qwen dense model loader."""

from __future__ import annotations

from transformers import AutoModelForCausalLM

from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import register_loader


def _text_model_type(ctx: LoaderContext) -> str:
    config = ctx.hf_config
    if config is None:
        return ""
    get_text_config = getattr(config, "get_text_config", None)
    text_config = get_text_config() if callable(get_text_config) else getattr(config, "text_config", config)
    return getattr(text_config, "model_type", "")


def _matches(ctx: LoaderContext) -> bool:
    return _text_model_type(ctx) == "qwen3"


@register_loader("qwen_dense", matches=_matches)
def load_qwen_dense(ctx: LoaderContext) -> LoadedModel:
    parallelism = ctx.spec.parallelism
    if parallelism.expert_parallel != 1:
        raise ValueError(
            f"qwen_dense does not support expert parallelism (got expert_parallel={parallelism.expert_parallel})"
        )
    if parallelism.sequence_parallel > 1 and not (ctx.parallel_groups or {}).get("sp_group"):
        raise ValueError("qwen_dense sequence parallelism requires sp_group")

    model = AutoModelForCausalLM.from_pretrained(
        ctx.spec.model_path_or_name,
        attn_implementation=ctx.spec.attn_implementation,
        dtype=ctx.spec.dtype,
    )
    return LoadedModel(model=model)
