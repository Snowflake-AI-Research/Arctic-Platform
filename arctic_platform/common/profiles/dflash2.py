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

"""Profile of the ``dflash2`` loader (online DFlash2 draft training against a frozen target).

Options are ``DFlash2Options`` fields, or a ``ModelSpec`` field the loader accepts: ``attn_implementation`` for
``attention.backend``, which sets the frozen target's attention (the draft's own is ``loader_options.attention_backend``),
and ``dtype`` for ``numerics.optimization_dtype``. The loader rejects every ``Patches`` wrapper and allows data
parallelism only, so none of the memory and speed options of the full-model loaders apply.
"""

from __future__ import annotations

from arctic_platform.common.option_registry import Registry
from arctic_platform.common.option_registry import register_profile


def register(registry: Registry) -> None:
    """Register the ``dflash2`` profile into ``registry``."""
    register_profile(
        "dflash2",
        supports=[
            "attention.backend",
            "numerics.optimization_dtype",
        ],
        unsupported={
            "checkpointing.mode": (
                "the loader has no activation checkpointing; it rejects `patches.gradient_checkpointing`"
            ),
            "checkpointing.freq": "the loader rejects `patches.gradient_checkpointing`",
            "checkpointing.targets": "the loader has no activation checkpointing",
            "checkpointing.offload": "the loader rejects `patches.activation_offload`",
            "tiled_mlp.token_chunk_size": "the loader rejects `patches.tiled_mlp`",
            "lm_head.fp32": "the draft objective reads the target's own LM head; the loader rejects `patches.lm_head`",
            "lm_head.token_chunk_size": (
                "the loader rejects `patches.lm_head`; `loader_options.objective_chunk_blocks` bounds memory"
            ),
            "lm_head.cross_entropy": "the draft objective computes its own cross-entropy",
            "lm_head.vocab_chunk_size": (
                "the loader rejects `patches.lm_head`; `loader_options.objective_chunk_blocks` bounds memory"
            ),
            "numerics.reduce_dtype": "no equivalent option",
            "numerics.fp32_parity": "this loader has no fp32_parity option",
            "numerics.fp32_residual": "this loader has no fp32_residual option; only huggingface has one",
            "numerics.fp32_routed_weighting": "the draft has no routed experts",
            "liger": "the loader rejects `patches.liger`",
            "compile.fullgraph": "the loader rejects `patches.compile`",
            "peft": "the loader rejects `patches.peft`: the draft is trained in full",
            "zorro_train": "the loader rejects `patches.zorro_train`",
        },
        not_applicable={
            "attention.sparse_mla": "neither the DFlash2 draft nor the target it reads uses sparse MLA attention",
            "moe.*": "the loader rejects expert parallelism and has no expert options",
        },
        registry=registry,
    )
