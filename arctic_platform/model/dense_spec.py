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
"""Build the ModelSpec used to load a dense Hugging Face model.

``dense_model_spec`` is the one translation from dense training options to a ``ModelSpec``. DSS calls it
with its ``training_config``; the on-prem worker calls it through ``ds_worker_dense_model_spec``.
"""

from __future__ import annotations

from typing import Any

from arctic_platform.model.config import ModelSpec
from arctic_platform.model.config import ParallelismConfig
from arctic_platform.model.config import Patches

_LOADER_OPTION_KEYS_IN_TRAINING = ("fp32_parity", "fp32_residual")
_ZORRO_TRAIN_KEYS = (
    "response_len",
    "max_token_len",
    "rollout_n",
    "temperature",
    "use_unpad",
    "world_size",
    "logits_optimization",
    "logits_optimization_peak_mem_size_in_gib",
    "logits_compute_from_fp32_inputs",
    "logits_compute_in_fp32",
)


def dense_model_spec(
    *,
    model_path: str,
    dtype: str,
    attn_implementation: str | None,
    sequence_parallel: int,
    liger: bool,
    training_options: dict,
    loader_options: dict | None = None,
    zorro_train: dict | None = None,
    debug: dict | None = None,
) -> ModelSpec:
    """Translate dense training options into one Hugging Face ``ModelSpec``.

    ``loader_options`` is forwarded whole. The Hugging Face schema validates it;
    this function does not read its keys. Hugging Face loader fields left in
    ``training_options`` are rejected so a misplaced key cannot change numerics
    silently. ``zorro_train`` and ``debug`` are set on the spec only when given.
    """
    import arctic_platform.model.loaders  # noqa: F401  # registers the huggingface loader

    misplaced = [name for name in _LOADER_OPTION_KEYS_IN_TRAINING if name in training_options]
    if misplaced:
        raise ValueError(
            "dense_model_spec takes Hugging Face loader fields only as loader_options, "
            f"not in training_options; move {', '.join(misplaced)}"
        )

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
        vocab_chunk_size = training_options.get("fused_lm_head_vocab_chunk_size")
        if vocab_chunk_size is None:
            raise ValueError("fp32_lm_head and fused_lm_head_token_chunk_size require fused_lm_head_vocab_chunk_size")
        lm_head = {
            "fp32": fp32_lm_head,
            "token_chunk_size": token_chunk_size,
            "vocab_chunk_size": vocab_chunk_size,
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
            **({"zorro_train": zorro_train} if zorro_train is not None else {}),
        ),
        **({"loader_options": loader_options} if loader_options else {}),
        **({"debug": debug} if debug else {}),
    )


def ds_worker_dense_model_spec(
    model_name: str,
    ds_worker_config: dict[str, Any],
    *,
    dtype: str,
    sequence_parallel: int,
    debug: dict | None = None,
) -> ModelSpec:
    """``dense_model_spec`` for the on-prem worker's flat ``ds_worker_config``.

    Only the flat config's own names are mapped here: gradient checkpointing is
    ``enable_gradient_checkpointing`` and defaults on, Liger is either
    ``model_provider="liger"`` or ``use_liger``, ``fp32_parity`` and
    ``fp32_residual`` are loader options, and ``zorro_train_enable`` selects the
    ZoRRo Train keys.
    """
    cfg = ds_worker_config or {}
    training_options = {key: value for key, value in cfg.items() if key not in _LOADER_OPTION_KEYS_IN_TRAINING}
    training_options["activation_checkpointing"] = cfg.get("enable_gradient_checkpointing", True)
    zorro_train = None
    if cfg.get("zorro_train_enable", False):
        zorro_train = {key: cfg[key] for key in _ZORRO_TRAIN_KEYS if key in cfg}
    return dense_model_spec(
        model_path=model_name,
        dtype=dtype,
        attn_implementation=cfg.get("attn_implementation"),
        sequence_parallel=sequence_parallel,
        liger=cfg.get("model_provider") == "liger" or bool(cfg.get("use_liger")),
        training_options=training_options,
        loader_options={key: cfg[key] for key in _LOADER_OPTION_KEYS_IN_TRAINING if key in cfg},
        zorro_train=zorro_train,
        debug=debug,
    )
