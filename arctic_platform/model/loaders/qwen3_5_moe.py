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

from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import register_loader


def _is_qwen3_5_moe(ctx: LoaderContext) -> bool:
    # Importing the implementation's ``models`` package registers the custom config
    # so AutoConfig can parse a qwen3_5_moe checkpoint. Skip quietly if the installed
    # transformers build lacks the qwen3_5 model family.
    try:
        from arctic_platform.model.implementations.qwen35 import models  # noqa: F401
    except ImportError:
        return False
    model_type = getattr(ctx.hf_config, "model_type", "") or ""
    return model_type.startswith("qwen3_5")


def _matches(ctx: LoaderContext) -> bool:
    # Cheap gate first: only MoE (expert-parallel) specs can use this loader, so
    # HuggingFace specs never trigger the heavier config parse below.
    return ctx.spec.parallelism.expert_parallel > 1 and _is_qwen3_5_moe(ctx)


@register_loader("qwen3_5_moe", matches=_matches)
def load_qwen3_5_moe(ctx: LoaderContext) -> LoadedModel:
    from arctic_platform.model.implementations.qwen35 import load_moe_model_for_dss

    parallelism = ctx.spec.parallelism
    groups = ctx.parallel_groups or {}
    model = load_moe_model_for_dss(
        model_name=ctx.spec.model_path_or_name,
        ep_size=parallelism.expert_parallel,
        sp_size=parallelism.sequence_parallel,
        sp_group=groups.get("sp_group"),
        prl_config=ctx.spec.loader_options,
    )
    return LoadedModel(model=model)
