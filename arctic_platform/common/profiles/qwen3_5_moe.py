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

"""Profile of the ``qwen3_5_moe`` loader. Options are ``Qwen3_5MoeOptions`` fields; the ``Patches`` wrappers are
rejected."""

from __future__ import annotations

from arctic_platform.common.option_registry import Registry
from arctic_platform.common.option_registry import register_profile


def register(registry: Registry) -> None:
    """Register the ``qwen3_5_moe`` profile into ``registry``."""
    register_profile(
        "qwen3_5_moe",
        supports=[
            "checkpointing.mode",
            "checkpointing.freq",
            "checkpointing.targets",
            "checkpointing.offload",
            "tiled_mlp.token_chunk_size",
            "lm_head.fp32",
            "lm_head.token_chunk_size",
            "lm_head.cross_entropy",
            "moe.grouped_mm",
            "moe.comm_backend",
            "moe.comm_sms",
            "moe.comm_token_chunk",
            "numerics.reduce_dtype",
        ],
        unsupported={
            "lm_head.vocab_chunk_size": "no equivalent option",
            "liger": "the loader rejects `patches.liger`; set `lm_head.cross_entropy` to liger instead",
            "attention.backend": (
                "no loader option; the backend is `ModelSpec.attn_implementation`, checked by the loader"
            ),
            "compile.fullgraph": "the loader rejects `patches.compile`",
            "peft": "the loader rejects `patches.peft`: expert adapter integration is not yet supported",
            "zorro_train": "the loader rejects `patches.zorro_train`",
        },
        not_applicable={
            "attention.sparse_mla": "Qwen3.5-MoE has no sparse MLA attention",
        },
        registry=registry,
    )
