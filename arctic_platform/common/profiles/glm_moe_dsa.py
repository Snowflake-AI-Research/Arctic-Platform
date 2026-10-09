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

"""Profile of the ``glm_moe_dsa`` loader (GLM-5 with sparse MLA).

Options are ``GlmMoeDsaOptions`` fields, or a ``ModelSpec`` field the loader accepts (``attn_implementation`` for
``attention.backend``). The loader rejects the ``Patches`` wrappers.
"""

from __future__ import annotations

from arctic_platform.common.option_registry import Registry
from arctic_platform.common.option_registry import register_profile


def register(registry: Registry) -> None:
    """Register the ``glm_moe_dsa`` profile into ``registry``."""
    register_profile(
        "glm_moe_dsa",
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
            "attention.sparse_mla",
            "attention.backend",
            "numerics.reduce_dtype",
        ],
        unsupported={
            "lm_head.vocab_chunk_size": "no equivalent option",
            "liger": "the loader rejects `patches.liger`; set `lm_head.cross_entropy` to liger instead",
            "compile.fullgraph": "the loader rejects `patches.compile`",
            "peft": "the loader rejects `patches.peft`: expert adapter integration is not yet supported",
            "zorro_train": "the loader rejects `patches.zorro_train`",
        },
        registry=registry,
    )
