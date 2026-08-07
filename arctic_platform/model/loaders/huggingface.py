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

from transformers import AutoModelForCausalLM

from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import register_loader


@register_loader("huggingface", default=True)
def load_huggingface(ctx: LoaderContext) -> LoadedModel:
    # This loader builds a plain single-process model and can't shard experts or
    # sequences, so it ignores parallelism and refuses a spec that asks for it.
    parallelism = ctx.spec.parallelism
    if parallelism.expert_parallel != 1 or parallelism.sequence_parallel != 1:
        raise ValueError(
            "huggingface loader does not support parallelism "
            f"(got expert_parallel={parallelism.expert_parallel}, "
            f"sequence_parallel={parallelism.sequence_parallel})"
        )

    model = AutoModelForCausalLM.from_pretrained(
        ctx.spec.model_path_or_name,
        attn_implementation=ctx.spec.attn_implementation,
        dtype=ctx.spec.dtype,
    )
    return LoadedModel(model=model)
