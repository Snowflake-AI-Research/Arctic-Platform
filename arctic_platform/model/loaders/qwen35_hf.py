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
"""Dense HF Qwen3.5 loader: AutoModel + Prime LM head + unused-vision freeze."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from transformers import AutoModelForCausalLM
from typing_extensions import Self

from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import register_loader

_QWEN35_HF_TYPES = frozenset({"qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text"})


class Qwen35HfOptions(BaseModel):
    """``loader_options`` for the dense HF Qwen3.5 path (OPD / non-EP)."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    fp32_lm_head: bool = Field(False, description="Compute the LM head in fp32.")
    fused_lm_head_token_chunk_size: int | Literal["auto", "disabled"] = Field(
        "disabled", description="Chunked LM-head logprobs token size."
    )
    fused_cross_entropy: bool | Literal["liger"] = Field(False, description="LM-head fused CE backend.")
    freeze_unused_vision: bool = Field(True, description="Freeze a VLM vision tower unused by text-only jobs.")

    @model_validator(mode="after")
    def _check_lm_head(self) -> Self:
        if self.fused_cross_entropy and (self.fp32_lm_head or isinstance(self.fused_lm_head_token_chunk_size, int)):
            raise ValueError(
                "cannot combine fused_cross_entropy with fp32_lm_head or an integer fused_lm_head_token_chunk_size"
            )
        return self


def _matches(ctx: LoaderContext) -> bool:
    if ctx.spec.parallelism.expert_parallel > 1:
        return False
    model_type = getattr(ctx.hf_config, "model_type", "") or ""
    return model_type in _QWEN35_HF_TYPES


@register_loader("qwen35_hf", matches=_matches, options=Qwen35HfOptions)
def load_qwen35_hf(ctx: LoaderContext) -> LoadedModel:
    parallelism = ctx.spec.parallelism
    if parallelism.expert_parallel != 1 or parallelism.sequence_parallel != 1:
        raise ValueError(
            "qwen35_hf loader does not support parallelism "
            f"(got expert_parallel={parallelism.expert_parallel}, "
            f"sequence_parallel={parallelism.sequence_parallel})"
        )

    model = AutoModelForCausalLM.from_pretrained(
        ctx.spec.model_path_or_name,
        attn_implementation=ctx.spec.attn_implementation,
        dtype=ctx.spec.dtype,
    )
    from arctic_platform.model.implementations.qwen35.hf_training_patches import apply_qwen35_training_patches

    apply_qwen35_training_patches(model, ctx.spec.loader_options)
    return LoadedModel(model=model, applied_patches=frozenset({"qwen35_training"}))
