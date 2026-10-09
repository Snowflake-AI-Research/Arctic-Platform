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

"""The option entries: one per configurable option that exists in Arctic-Platform today.

Only options with a one-to-one name on the loader side are registered. ``source`` names where each option is set
today: a ``ModelSpec`` or ``Patches`` field for the ``huggingface`` loader; for the custom loaders a
``loader_options`` field or a ``ModelSpec`` field the loader accepts, such as ``attn_implementation``; and an
environment variable for the sampler. ``peft`` and ``zorro_train`` are training modes: the
profiles settle them, but they keep their own config fields.
"""

from __future__ import annotations

from arctic_platform.common.option_registry import Registry
from arctic_platform.common.option_registry import register_option


def register(registry: Registry) -> None:
    """Register the built-in option entries into ``registry``."""
    register_option(
        "checkpointing.mode",
        runs_in="trainer",
        goal="memory",
        why="Recompute whole blocks or only selected submodules in the backward pass.",
        source="`loader_options.ac_config.mode`",
        registry=registry,
    )
    register_option(
        "checkpointing.freq",
        runs_in="trainer",
        goal="memory",
        why="Checkpoint every Nth block.",
        source="`patches.gradient_checkpointing`; `loader_options.ac_config.freq`",
        registry=registry,
    )
    register_option(
        "checkpointing.targets",
        runs_in="trainer",
        goal="memory",
        why="The submodules that selective checkpointing recomputes.",
        source="`loader_options.ac_config.targets`",
        registry=registry,
    )
    register_option(
        "checkpointing.offload",
        runs_in="trainer",
        goal="memory",
        why="Stream checkpointed block boundaries to CPU memory.",
        source="`patches.activation_offload`; `loader_options.ac_config.offload_config`",
        registry=registry,
    )
    register_option(
        "tiled_mlp.token_chunk_size",
        runs_in="trainer",
        goal="memory",
        why="Recompute the MLP in token tiles of this size.",
        source="`patches.tiled_mlp.token_chunk_size`; `loader_options.tiled_mlp_token_chunk_size`",
        registry=registry,
    )
    register_option(
        "lm_head.fp32",
        runs_in="both",
        goal="parity",
        why="Compute the LM-head projection in fp32, so trainer and sampler log-probs agree.",
        source="`patches.lm_head.fp32`; `loader_options.fp32_lm_head`; sampler `ARCTIC_FP32_LM_HEAD`",
        registry=registry,
    )
    register_option(
        "lm_head.token_chunk_size",
        runs_in="trainer",
        goal="memory",
        why="Token tile size of the chunked LM-head projection.",
        source="`patches.lm_head.token_chunk_size`; `loader_options.fused_lm_head_token_chunk_size`",
        registry=registry,
    )
    register_option(
        "lm_head.vocab_chunk_size",
        runs_in="trainer",
        goal="memory",
        why="Vocabulary tile size of the chunked LM-head projection.",
        source="`patches.lm_head.vocab_chunk_size`",
        registry=registry,
    )
    register_option(
        "lm_head.cross_entropy",
        runs_in="trainer",
        goal="memory",
        why="Fuse the LM head with the cross-entropy so the full logits are never materialized.",
        source="`loader_options.fused_cross_entropy`",
        registry=registry,
    )
    register_option(
        "liger",
        runs_in="trainer",
        goal="speed",
        why="Apply the Liger kernels (RMSNorm, SwiGLU and fused linear cross-entropy) to a HuggingFace model.",
        source="`patches.liger`",
        registry=registry,
    )
    register_option(
        "moe.grouped_mm",
        runs_in="trainer",
        goal="speed",
        why="Run the experts as one grouped matmul.",
        source="`loader_options.moe_use_grouped_mm`",
        registry=registry,
    )
    register_option(
        "moe.comm_backend",
        runs_in="trainer",
        goal="speed",
        why="The expert-parallel all-to-all backend.",
        source="`loader_options.ep_comm_backend`",
        registry=registry,
    )
    register_option(
        "moe.comm_sms",
        runs_in="trainer",
        goal="speed",
        why="Streaming multiprocessors reserved for expert-parallel communication.",
        source="`loader_options.deepep_num_sms`",
        registry=registry,
    )
    register_option(
        "moe.comm_token_chunk",
        runs_in="trainer",
        goal="speed",
        why="Token chunk size of the expert-parallel dispatch.",
        source="`loader_options.deepep_token_chunk_size`",
        registry=registry,
    )
    register_option(
        "attention.backend",
        runs_in="trainer",
        goal="speed",
        why="The attention implementation to run.",
        source="`ModelSpec.attn_implementation`",
        registry=registry,
    )
    register_option(
        "attention.sparse_mla",
        runs_in="trainer",
        goal="speed",
        why="The sparse MLA attention kernel.",
        source="`loader_options.sparse_mla_backend`; `ModelSpec.attn_implementation` on glm5_next",
        registry=registry,
    )
    register_option(
        "numerics.reduce_dtype",
        runs_in="trainer",
        goal="parity",
        why="The dtype gradients are reduced in.",
        source="`loader_options.reduce_dtype`",
        registry=registry,
    )
    register_option(
        "compile.fullgraph",
        runs_in="trainer",
        goal="speed",
        why="Compile each transformer layer with torch.compile, as one full graph when set.",
        source="`patches.compile.fullgraph`",
        registry=registry,
    )
    register_option(
        "peft",
        runs_in="trainer",
        goal="mode",
        why="Train adapters (LoRA) instead of the full weights.",
        source="`patches.peft`",
        registry=registry,
    )
    register_option(
        "zorro_train",
        runs_in="trainer",
        goal="mode",
        why="Replace the forward with the ZoRRo Train log-prob and entropy path.",
        source="`patches.zorro_train`",
        registry=registry,
    )
