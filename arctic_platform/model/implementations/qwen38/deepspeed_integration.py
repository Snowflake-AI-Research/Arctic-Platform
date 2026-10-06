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
"""DeepSpeed adapter for Qwen3.8-Flash-Next (``qwen4_exp``)."""

from __future__ import annotations

from dataclasses import replace

from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from arctic_platform.model.implementations.moe.deepspeed_integration import MoEDeepSpeedAdapter
from arctic_platform.model.implementations.moe.parallel_dims import ParallelDims
from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen_ds
from arctic_platform.model.implementations.qwen35.config import ModelConfig
from arctic_platform.model.implementations.qwen38.qsa_flex import apply_qsa_flex
from arctic_platform.model.implementations.qwen38.qsa_flex import register_qsa_flex_backend
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

QWEN38_NUM_EXPERTS = 512
QWEN38_ATTN_BACKEND = "qsa_flex"


def _build_model_config(
    model_name: str,
    ep_size: int,
    dp_replicate: int,
    optimization_dtype: str,
    attn_implementation: str,
    options: Qwen3_5MoeOptions,
) -> ModelConfig:
    if attn_implementation not in (QWEN38_ATTN_BACKEND, "flex_attention"):
        raise ValueError(
            f"Qwen3.8-Flash-Next training requires the non-SDPA QSA FlexAttention backend; got {attn_implementation!r}"
        )
    register_qsa_flex_backend()
    if QWEN38_NUM_EXPERTS % ep_size:
        raise ValueError(
            f"Qwen3.8-Flash-Next has {QWEN38_NUM_EXPERTS} experts, "
            f"so ep_size={ep_size} must divide {QWEN38_NUM_EXPERTS}."
        )
    model_config = qwen_ds._build_model_config(
        model_name,
        ep_size,
        dp_replicate,
        optimization_dtype,
        QWEN38_ATTN_BACKEND,
        options,
    )
    model_config.attn = QWEN38_ATTN_BACKEND
    return model_config


def _validate_parallelism(sp_size: int, _sp_group=None) -> None:
    if sp_size > 1:
        raise NotImplementedError(
            "Sequence parallelism is not implemented for Qwen3.8-Flash-Next. "
            f"Got sp_size={sp_size}; set training_config.sp_size=1 or omit it."
        )


def _apply_sequence_parallelism(_model: nn.Module, sp_size: int, _sp_group) -> None:
    _validate_parallelism(sp_size)


def _adapter() -> MoEDeepSpeedAdapter:
    return replace(
        qwen_ds._generic_adapter(),
        build_model_config=_build_model_config,
        apply_sequence_parallelism=_apply_sequence_parallelism,
        extra_weight_iterators=(),
    )


def load_qwen4_exp_model_for_deepspeed(
    model_config: ModelConfig,
    parallel_dims: ParallelDims,
    ep_mesh: DeviceMesh,
    ep_group_name: str,
    *,
    fused_cross_entropy: bool | str = False,
    tiled_mlp_token_chunk_size: int | None = None,
    sp_size: int = 1,
    sp_group=None,
) -> nn.Module:
    model = qwen_ds._load_moe_model_for_deepspeed(
        _adapter(),
        model_config,
        parallel_dims,
        ep_mesh,
        ep_group_name,
        fused_cross_entropy=fused_cross_entropy,
        tiled_mlp_token_chunk_size=tiled_mlp_token_chunk_size,
        sp_size=sp_size,
        sp_group=sp_group,
    )
    apply_qsa_flex(model)
    return model


def load_qwen4_exp_model(
    *,
    model_name: str,
    optimization_dtype: str,
    attn_implementation: str,
    ep_size: int,
    sp_size: int = 1,
    sp_group=None,
    ep_group=None,
    options: Qwen3_5MoeOptions,
) -> nn.Module:
    """Load Qwen3.8-Flash-Next. Weight sync stays on the Hugging Face iterator."""
    _validate_parallelism(sp_size)
    model = qwen_ds._load_moe_model(
        _adapter(),
        load_qwen4_exp_model_for_deepspeed,
        model_name=model_name,
        optimization_dtype=optimization_dtype,
        attn_implementation=attn_implementation,
        ep_size=ep_size,
        sp_size=sp_size,
        sp_group=sp_group,
        ep_group=ep_group,
        options=options,
        patch_moe_detection=qwen_ds.patch_deepspeed_moe_detection,
        device_mesh_type=DeviceMesh,
    )
    qwen_ds.maybe_apply_row_invariant_projections(model, options.model_dump())
    return model
