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

"""Profile of the default ``huggingface`` loader: options are ``Patches`` and ``ModelSpec`` fields."""

from __future__ import annotations

from arctic_platform.common.option_registry import register_profile

register_profile(
    "huggingface",
    supports=[
        "checkpointing.freq",
        "checkpointing.offload",
        "tiled_mlp.token_chunk_size",
        "lm_head.fp32",
        "lm_head.token_chunk_size",
        "lm_head.vocab_chunk_size",
        "liger",
        "attention.backend",
        "compile.fullgraph",
        "peft",
        "zorro_train",
    ],
    unsupported={
        "checkpointing.mode": "no equivalent option; `patches.gradient_checkpointing` always recomputes whole layers",
        "checkpointing.targets": "no equivalent option; selective checkpointing exists only in the custom loaders",
        "lm_head.cross_entropy": "no equivalent option; the fused cross-entropy comes only with `patches.liger`",
        "attention.sparse_mla": "no equivalent option; the sparse MLA kernel is chosen only by the glm_moe_dsa loader",
        "numerics.reduce_dtype": "no equivalent option",
    },
    not_applicable={
        "moe.*": "the huggingface loader rejects expert parallelism and has no expert options",
    },
)
