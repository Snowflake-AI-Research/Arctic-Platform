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
"""Default loader: HuggingFace ``AutoModelForCausalLM.from_pretrained``."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from transformers import AutoModelForCausalLM

from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import register_loader


class HuggingfaceOptions(BaseModel):
    """Validated ``loader_options`` for the huggingface loader."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    fp32_parity: bool = Field(
        True,
        description=(
            "With sequence parallelism, evaluate the GatedDeltaNet beta gate of the layers the sequence-parallel "
            "wrapper adapts as a sigmoid of fp32 b, as vLLM does. False evaluates it in b's own dtype, as the stock "
            "transformers forward does. Without sequence parallelism the stock forward runs either way."
        ),
    )


@register_loader("huggingface", default=True, options=HuggingfaceOptions)
def load_huggingface(ctx: LoaderContext) -> LoadedModel:
    parallelism = ctx.spec.parallelism
    if parallelism.expert_parallel != 1:
        raise ValueError(
            "huggingface loader does not support expert parallelism "
            f"(got expert_parallel={parallelism.expert_parallel})"
        )
    groups = ctx.parallel_groups or {}
    if parallelism.sequence_parallel > 1 and groups.get("sp_group") is None:
        raise ValueError("huggingface sequence parallelism requires parallel_groups['sp_group']")

    options = HuggingfaceOptions.model_validate(ctx.spec.loader_options)
    model = AutoModelForCausalLM.from_pretrained(
        ctx.spec.model_path_or_name,
        attn_implementation=ctx.spec.attn_implementation,
        dtype=ctx.spec.dtype,
    )
    if parallelism.sequence_parallel > 1:
        from arctic_platform.model.implementations.gpu.sp.transformers import (
            configure_transformers_sequence_parallel_model,
        )

        configure_transformers_sequence_parallel_model(model, groups["sp_group"], fp32_beta=options.fp32_parity)
    return LoadedModel(model=model)
