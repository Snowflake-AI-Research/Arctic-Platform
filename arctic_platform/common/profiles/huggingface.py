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

"""Profile of the default ``huggingface`` loader.

Its options are ``Patches`` and ``ModelSpec`` fields, and the loader's own ``loader_options.fp32_parity`` and
``loader_options.fp32_residual``.

The loader reads ``ModelSpec.dtype`` too, but ``numerics.optimization_dtype`` is unsupported here: on this route
the job sets that dtype, and the key would have no effect.

``loader_options.fp32_parity`` is on by default. It changes the rounding of the dense Qwen3.5 family (transformers
``Qwen3_5TextModel``). Other models load unchanged, except that with sequence parallelism the beta gate of any
GatedDeltaNet layer the sequence-parallel wrapper adapts follows it. Its fused residual-add norm reads each norm's
weight instead of calling the norm, so with ``patches.liger`` only the first layer's input norm, of the decoder-layer
and final norms, runs Liger's RMSNorm kernel. ``numerics.fp32_parity`` is supported here and maps to
``loader_options.fp32_parity``; it changes only the stock dense Qwen3.5 family and, with sequence parallelism, that
beta gate.

``numerics.fp32_residual`` maps to ``loader_options.fp32_residual``, which is off by default, experimental, and
accepted only with ``fp32_parity`` on. It keeps the residual stream of the patched dense forward in fp32 between
decoder layers, so under full activation checkpointing each boundary after the first holds an fp32 residual.
"""

from __future__ import annotations

from arctic_platform.common.option_registry import Registry
from arctic_platform.common.option_registry import register_profile


def register(registry: Registry) -> None:
    """Register the ``huggingface`` profile into ``registry``."""
    register_profile(
        "huggingface",
        supports=[
            "checkpointing.freq",
            "checkpointing.offload",
            "tiled_mlp.token_chunk_size",
            "lm_head.fp32",
            "lm_head.token_chunk_size",
            "lm_head.vocab_chunk_size",
            "numerics.fp32_parity",
            "numerics.fp32_residual",
            "liger",
            "attention.backend",
            "compile.fullgraph",
            "peft",
            "zorro_train",
        ],
        unsupported={
            "checkpointing.mode": (
                "no equivalent option; `patches.gradient_checkpointing` always recomputes whole layers"
            ),
            "checkpointing.targets": "no equivalent option; selective checkpointing exists only in the custom loaders",
            "lm_head.cross_entropy": "no equivalent option; the fused cross-entropy comes only with `patches.liger`",
            "attention.sparse_mla": (
                "no equivalent option; sparse MLA is used only by the GLM loaders glm_moe_dsa and glm5_next"
            ),
            "numerics.reduce_dtype": "no equivalent option",
            "numerics.fp32_routed_weighting": "no equivalent option; only glm5_next and qwen4_exp have one",
            "numerics.optimization_dtype": (
                "the key has no effect on this route: the job sets the model dtype, not this key"
            ),
        },
        not_applicable={
            "moe.*": "the huggingface loader rejects expert parallelism and has no expert options",
        },
        registry=registry,
    )
