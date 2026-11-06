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
today: a ``ModelSpec`` or ``Patches`` field, or one of its own ``loader_options``, for the ``huggingface``
loader; for the custom loaders a ``loader_options`` field or a ``ModelSpec`` field the loader accepts, such as
``attn_implementation``; and an environment variable for the sampler. ``peft`` and ``zorro_train`` are training modes: the
profiles settle them, but they keep their own config fields.

``LOADER_OPTION_FIELDS`` names, for the custom loaders, the loader-side field each key sets, so a caller can build
their ``loader_options`` from ``model_options`` keys; ``loader_option_field`` applies the per-loader exceptions.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from arctic_platform.common.option_registry import Registry
from arctic_platform.common.option_registry import register_option

# For the custom loaders: a field path under the loader's options model (``ModelSpec.loader_options``), or a
# ``ModelSpec.``-prefixed path for a ``ModelSpec`` field the loader accepts. None: no loader-side name.
LOADER_OPTION_FIELDS: Mapping[str, str | None] = MappingProxyType(
    {
        "checkpointing.mode": "ac_config.mode",
        "checkpointing.freq": "ac_config.freq",
        "checkpointing.targets": "ac_config.targets",
        "checkpointing.offload": "ac_config.offload_config",
        "tiled_mlp.token_chunk_size": "tiled_mlp_token_chunk_size",
        "lm_head.fp32": "fp32_lm_head",
        "lm_head.token_chunk_size": "fused_lm_head_token_chunk_size",
        "lm_head.vocab_chunk_size": None,
        "lm_head.cross_entropy": "fused_cross_entropy",
        "liger": None,
        "moe.grouped_mm": "moe_use_grouped_mm",
        "moe.sonicmoe": "moe_use_sonicmoe",
        "moe.comm_backend": "ep_comm_backend",
        "moe.comm_sms": "deepep_num_sms",
        "moe.comm_token_chunk": "deepep_token_chunk_size",
        "attention.backend": "ModelSpec.attn_implementation",
        "attention.sparse_mla": "sparse_mla_backend",
        "numerics.reduce_dtype": "reduce_dtype",
        "numerics.fp32_parity": "fp32_parity",
        "numerics.fp32_residual": None,
        "numerics.fp32_routed_weighting": "fp32_routed_weighting",
        "numerics.optimization_dtype": "ModelSpec.dtype",
        "compile.fullgraph": None,
        "peft": None,
        "zorro_train": None,
    }
)

# Per-loader entries that differ from ``LOADER_OPTION_FIELDS``. ``glm5_next`` picks its sparse MLA kernel through
# ``attn_implementation``, which ``_validate_glm5_next`` requires to name a sparse MLA backend.
LOADER_FIELD_OVERRIDES: Mapping[str, Mapping[str, str | None]] = MappingProxyType(
    {
        "glm5_next": MappingProxyType({"attention.sparse_mla": "ModelSpec.attn_implementation"}),
    }
)


def loader_option_field(loader: str, key: str) -> str | None:
    """The loader-side name of option ``key`` for the custom loader ``loader``, or ``None`` when it has none.

    Raises ``KeyError`` for a key that is not registered.
    """
    return LOADER_FIELD_OVERRIDES.get(loader, {}).get(key, LOADER_OPTION_FIELDS[key])


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
        "moe.sonicmoe",
        runs_in="trainer",
        goal="speed",
        why=(
            "Run BF16 routed experts through SonicMoE on the expert-parallel receive buffer when that kernel applies."
            " Off keeps the grouped GEMM path."
        ),
        source="`loader_options.moe_use_sonicmoe`",
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
        "numerics.fp32_parity",
        runs_in="trainer",
        goal="parity",
        why=(
            "Evaluate the residual adds, the partial rotary embedding and the GatedDeltaNet beta gate in fp32, as"
            " vLLM does. Each norm returns the residual in the dtype of the branch output, so with fp32 weights"
            " under bf16 autocast the residual stream and the final hidden state are bf16 rather than fp32, and"
            " under full activation checkpointing every decoder layer after the first holds the residual as a"
            " second tensor at its boundary. Off evaluates all three in the input dtype, the residual adds at the"
            " end of each layer. On GLM-5.3-Flash it rounds each RMSNorm once, after the weight multiply; on"
            " Qwen3.8-Flash-Next it evaluates the QSA rotary embedding, softmax probabilities, P·V product, output"
            " gate and hyper-connection mix and injection in fp32."
        ),
        remove_when="no job needs the residual adds, rotary embedding and beta gate in the input dtype",
        source="`loader_options.fp32_parity`",
        registry=registry,
    )
    register_option(
        "numerics.fp32_residual",
        runs_in="trainer",
        goal="parity",
        why=(
            "Experimental. With fp32_parity on, carry the residual stream of the stock dense Qwen3.5 family in fp32"
            " between decoder layers, from the embeddings on, instead of rounding it to the branch dtype after each"
            " fused add. Checked on CPU against a reference and under checkpointing, offload and compile; its effect"
            " on GPU training is not yet measured. Under full activation checkpointing each decoder layer after the"
            " first holds an fp32 residual at its boundary, twice the bytes of a bf16 one."
        ),
        remove_when="GPU runs show it does not bring training closer to vLLM, or it becomes part of fp32_parity",
        source="`loader_options.fp32_residual`",
        registry=registry,
    )
    register_option(
        "numerics.fp32_routed_weighting",
        runs_in="trainer",
        goal="parity",
        why=(
            "GLM-5.3-Flash and Qwen3.8-Flash-Next BF16 experts multiply the down projection's fp32 accumulator by the"
            " routing weight and round once, as vLLM's Triton fused MoE kernel does, instead of rounding the expert"
            " output and then weighting it. Bit-identical to that kernel where a sampler rank computes whole experts;"
            " under sampler tensor parallelism each rank rounds its own partial sum. Off by default: it runs one"
            " fp32-output GEMM per local expert in place of the grouped GEMM, and on H200 the routed experts' forward"
            " plus backward takes 16-19% longer."
        ),
        remove_when="a grouped GEMM with a weighted fp32 epilogue makes it cheap enough to fold into fp32_parity",
        source="`loader_options.fp32_routed_weighting`",
        registry=registry,
    )
    register_option(
        "numerics.optimization_dtype",
        runs_in="trainer",
        goal="parity",
        why="The dtype the trainer keeps and optimizes the weights in: bfloat16 or float32.",
        source="`ModelSpec.dtype` on the custom MoE loaders",
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
