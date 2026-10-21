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
from pydantic import model_validator
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
            "For the dense Qwen3.5 family (transformers Qwen3_5TextModel), evaluate the residual adds, the partial"
            " rotary embedding and the GatedDeltaNet beta gate in fp32, as vLLM does. Each norm returns the residual"
            " in the dtype of the branch output, so with fp32 weights under bf16 autocast the residual stream and the"
            " final hidden state are bf16, and under full activation checkpointing every decoder layer after the first"
            " holds the residual as a second tensor at its boundary; per-layer hidden states (output_hidden_states)"
            " are not available. The residual add and the following norm are one fused fp32 operation that reads the"
            " norm's weight, so with the liger patch, of the decoder-layer and final norms, only the first layer's"
            " input norm runs Liger's RMSNorm kernel. With sequence parallelism, the beta gate of every GatedDeltaNet"
            " layer the sequence-parallel wrapper adapts follows this option, in any model. False keeps these"
            " operations stock and evaluates that beta gate in b's own dtype. Other models are unaffected. The"
            " GatedDeltaNet output norm of the Qwen3.5 and Qwen3-Next families is evaluated in fp32 and rounded once,"
            " as vLLM does, whatever this option is."
        ),
    )
    fp32_residual: bool = Field(
        False,
        description=(
            "Experimental. With fp32_parity, carry the residual stream of the dense Qwen3.5 family in fp32 between"
            " decoder layers, from the embeddings on, instead of rounding it to the dtype of the branch output after"
            " each fused add; the final norm still returns the hidden state in that dtype. Under full activation"
            " checkpointing the residual every decoder layer after the first holds at its boundary is then fp32"
            " (4 bytes per hidden value instead of 2 for a bf16 model). Requires fp32_parity; other models are"
            " unaffected."
        ),
    )

    @model_validator(mode="after")
    def _fp32_residual_needs_fp32_parity(self) -> HuggingfaceOptions:
        if self.fp32_residual and not self.fp32_parity:
            raise ValueError("fp32_residual requires fp32_parity: only the fp32 parity forward carries the residual")
        return self


def _is_local_qwen3_5_dense(model_path_or_name: str) -> bool:
    """Whether ``model_path_or_name`` is a local checkpoint of the dense Qwen3.5 family, judged by its config.json.

    Reads only a local file, so a hub name is never fetched here; the patch checks again after loading either way.
    """
    import json
    import os

    from arctic_platform.model.implementations.gpu.qwen3_5_dense_parity import QWEN3_5_DENSE_MODEL_TYPES

    config_path = os.path.join(model_path_or_name, "config.json")
    if not os.path.isfile(config_path):
        return False
    with open(config_path) as config_file:
        config = json.load(config_file)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = {}
    return (
        config.get("model_type") in QWEN3_5_DENSE_MODEL_TYPES
        or text_config.get("model_type") in QWEN3_5_DENSE_MODEL_TYPES
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
    if options.fp32_parity and _is_local_qwen3_5_dense(str(ctx.spec.model_path_or_name)):
        from arctic_platform.model.implementations.gpu.qwen3_5_dense_parity import check_stock_classes

        # Fail before the weights load when the installed transformers is not the one the patch is verified for.
        check_stock_classes()
    model = AutoModelForCausalLM.from_pretrained(
        ctx.spec.model_path_or_name,
        attn_implementation=ctx.spec.attn_implementation,
        dtype=ctx.spec.dtype,
    )
    from arctic_platform.model.implementations.gpu.vllm_rounding import round_gated_norms_once

    round_gated_norms_once(model)
    if options.fp32_parity:
        from arctic_platform.model.implementations.gpu.qwen3_5_dense_parity import apply_fp32_parity

        apply_fp32_parity(
            model, sequence_parallel=parallelism.sequence_parallel > 1, fp32_residual=options.fp32_residual
        )
    if parallelism.sequence_parallel > 1:
        from arctic_platform.model.implementations.gpu.sp.transformers import (
            configure_transformers_sequence_parallel_model,
        )

        configure_transformers_sequence_parallel_model(model, groups["sp_group"], fp32_beta=options.fp32_parity)
    return LoadedModel(model=model)
