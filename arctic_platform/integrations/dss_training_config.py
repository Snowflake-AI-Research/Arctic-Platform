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

"""Explicit AP-to-DSS MoE configuration bridge, without patching DSS."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from typing import Mapping

from arctic_platform.model.implementations.debug.config import training_debug_config

_DSS_MOE_LOADERS = frozenset({"generic_moe", "glm_moe_dsa", "qwen3_5_moe", "glm5_next", "qwen4_exp"})


def _moe_model_spec(payload: Mapping[str, Any], loader_options: dict):
    from arctic_platform.model.config import ModelSpec
    from arctic_platform.model.config import ParallelismConfig

    training = payload["training_config"]
    loader_options = deepcopy(loader_options)
    if isinstance(loader_options.get("debug"), dict):
        # Capacity guards belong to model construction, not loader-default resolution.
        loader_options["debug"].pop("gpu_mem_preallocate_gib", None)
    return ModelSpec(
        model_path_or_name=payload.get("local_model_dir") or payload["model_name"],
        dtype=payload.get("dtype", "bfloat16"),
        attn_implementation=training.get("attn_implementation"),
        parallelism=ParallelismConfig(
            expert_parallel=training["ep_size"],
            sequence_parallel=training.get("sp_size") or 1,
        ),
        loader_options=loader_options,
    )


def dss_moe_initialize_payload(payload: Mapping[str, Any]) -> dict:
    """Copy a single job's flat AP controls into the existing DSS MoE loader channel."""
    transport = deepcopy(dict(payload))
    training = transport["training_config"]
    debug = training_debug_config(training)
    reserve_requested = debug.get("gpu_mem_preallocate_gib", 0) != 0
    chunk = training.get("deepep_token_chunk_size")
    if chunk is not None and (isinstance(chunk, bool) or not isinstance(chunk, int) or chunk <= 0):
        raise ValueError("deepep_token_chunk_size must be a positive integer")
    head_chunk = training.get("fused_lm_head_token_chunk_size")
    if head_chunk is not None and (isinstance(head_chunk, bool) or not isinstance(head_chunk, int) or head_chunk <= 0):
        raise ValueError("fused_lm_head_token_chunk_size must be a positive integer")
    fp32_head = training.get("fp32_lm_head")
    if fp32_head is not None and not isinstance(fp32_head, bool):
        raise ValueError("fp32_lm_head must be a boolean")
    if "model_provider" not in training and (training.get("ep_size") or 1) > 1:
        # AP resolves loaders automatically; DSS otherwise selects its dense HF path.
        spec = _moe_model_spec(transport, training.get("prime_rl") or {})
        if spec.loader not in _DSS_MOE_LOADERS:
            raise ValueError(f"The DSS compatibility bridge cannot map AP loader {spec.loader!r} to MoE training")
        training["model_provider"] = "prime_rl"
    if training.get("model_provider") != "prime_rl" or (training.get("ep_size") or 1) <= 1:
        if reserve_requested or chunk is not None or head_chunk is not None or fp32_head:
            raise ValueError("The DSS compatibility bridge requires a Prime-RL AP MoE loader with EP > 1")
        return transport
    legacy = training.get("prime_rl")
    if legacy is not None and not isinstance(legacy, dict):
        raise TypeError("training_config.prime_rl must be a mapping")
    legacy = dict(legacy or {})
    if chunk is not None:
        nested_chunk = legacy.get("deepep_token_chunk_size")
        if nested_chunk is not None and nested_chunk != chunk:
            raise ValueError("Conflicting top-level and prime_rl.deepep_token_chunk_size")
        legacy["deepep_token_chunk_size"] = chunk
    training.pop("deepep_token_chunk_size", None)
    for key, value in (
        ("fused_lm_head_token_chunk_size", head_chunk),
        ("fp32_lm_head", fp32_head),
    ):
        if value is not None:
            nested = legacy.get(key)
            if nested is not None and (type(nested) is not type(value) or nested != value):
                raise ValueError(f"Conflicting top-level and prime_rl.{key}")
            legacy[key] = value
    # The carved MoE head fixes vocabulary tiles at 8192; it has no loader option for this.
    if isinstance(legacy.get("fused_lm_head_token_chunk_size"), int):
        vocab_chunk = training.get("fused_lm_head_vocab_chunk_size", 8192)
        if isinstance(vocab_chunk, bool) or vocab_chunk != 8192:
            raise ValueError("The AP MoE chunked head requires fused_lm_head_vocab_chunk_size=8192")
    if training.get("max_seq_len") is not None:
        legacy.setdefault("seq_len", training["max_seq_len"])
    if training.get("debug") is not None:
        legacy["debug"] = dict(debug)
    training["prime_rl"] = legacy
    if "attn_implementation" in training and training["attn_implementation"] is None:
        # Resolve before DSS excludes nulls and reparses with its generic default.
        spec = _moe_model_spec(transport, legacy)
        if spec.attn_implementation is None:
            raise ValueError("The AP MoE loader must resolve an explicit-null attention backend")
        training["attn_implementation"] = spec.attn_implementation
    return transport
