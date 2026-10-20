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
"""Build the ModelSpec used to load a dense Hugging Face model."""

from __future__ import annotations

from arctic_platform.model.config import ModelSpec
from arctic_platform.model.config import ParallelismConfig
from arctic_platform.model.config import Patches


def dense_model_spec(
    *,
    model_path: str,
    dtype: str,
    attn_implementation: str | None,
    sequence_parallel: int,
    liger: bool,
    training_options: dict,
) -> ModelSpec:
    """Translate dense training options into one Hugging Face ``ModelSpec``."""
    import arctic_platform.model.loaders  # noqa: F401  # registers the huggingface loader

    offload_config = (training_options.get("ac_config") or {}).get("offload_config") or {}
    activation_offload = None
    if offload_config.get("enabled", False):
        activation_offload = {
            key: value for key, value in offload_config.items() if key != "enabled" and value is not None
        }
    token_chunk_size = training_options.get("fused_lm_head_token_chunk_size")
    lm_head = None
    fp32_lm_head = training_options.get("fp32_lm_head", False)
    if fp32_lm_head or token_chunk_size is not None:
        lm_head = {
            "fp32": fp32_lm_head,
            "token_chunk_size": token_chunk_size,
            "vocab_chunk_size": training_options["fused_lm_head_vocab_chunk_size"],
        }
    tiled_mlp_token_chunk_size = training_options.get("tiled_mlp_token_chunk_size")
    tiled_mlp = {"token_chunk_size": tiled_mlp_token_chunk_size} if tiled_mlp_token_chunk_size is not None else None
    return ModelSpec(
        model_path_or_name=model_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        loader="huggingface",
        parallelism=ParallelismConfig(sequence_parallel=sequence_parallel),
        patches=Patches(
            liger=liger,
            gradient_checkpointing=training_options.get("activation_checkpointing", False),
            activation_offload=activation_offload,
            compile=training_options.get("compile"),
            tiled_mlp=tiled_mlp,
            lm_head=lm_head,
            peft=training_options.get("peft_config"),
        ),
    )
