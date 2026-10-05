"""GLM-5.2 adapter for the shared MoE DeepSpeed integration."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from arctic_platform.model.implementations.moe.deepspeed_integration import (
    MoEDeepSpeedAdapter,
    load_moe_model as _load_moe_model,
    load_moe_model_for_deepspeed as _load_moe_model_for_deepspeed,
    patch_deepspeed_moe_detection,
    setup_model_local_no_train,
)
from arctic_platform.model.implementations.moe.layers.lm_head import inject_prime_lm_head
from arctic_platform.model.implementations.moe.layers.moe import BCFeedForward
from arctic_platform.model.implementations.moe.parallel_dims import ParallelDims
from arctic_platform.model.loaders.glm_moe_dsa import GlmMoeDsaOptions

from .config import DebugModelConfig, ModelConfig
from .model_builder import (
    DTYPE_MAP,
    _reset_runtime_moe_buffers,
    apply_ac,
    configure_moe_ep_backend,
    configure_sparse_mla_backend,
    get_model,
    load_dcp_from_hf,
)


def shared_expert_bc_mlp_forward(
    feed_forward: BCFeedForward, hidden_states: torch.Tensor
) -> torch.Tensor:
    gate = torch.matmul(hidden_states, feed_forward.w1.T)
    up = torch.matmul(hidden_states, feed_forward.w3.T)
    return torch.matmul(F.silu(gate) * up, feed_forward.w2.T)


def _validate_sp_size(sp_size: int) -> None:
    if sp_size > 1:
        raise NotImplementedError(
            "Sequence parallelism is not implemented for GLM-5.2. "
            f"Got sp_size={sp_size}; set training_config.sp_size=1 or omit it."
        )


def _apply_sequence_parallelism(_model: nn.Module, sp_size: int, _sp_group) -> None:
    _validate_sp_size(sp_size)


def _build_model_config(
    model_name: str,
    ep_size: int,
    dp_replicate: int,
    optimization_dtype: str,
    attn_implementation: str,
    options: GlmMoeDsaOptions,
) -> ModelConfig:
    return ModelConfig(
        name=model_name,
        weight_conversion_cache_dir=options.weight_conversion_cache_dir,
        trust_remote_code=options.trust_remote_code,
        seq_len=options.seq_len,
        attn=attn_implementation,
        ep=ep_size,
        ep_comm_backend=options.ep_comm_backend,
        sparse_mla_backend=options.sparse_mla_backend,
        deepep_num_sms=options.deepep_num_sms,
        deepep_token_chunk_size=options.deepep_token_chunk_size,
        dp_replicate=dp_replicate,
        cp=1,
        impl="custom",
        optimization_dtype=optimization_dtype,
        reduce_dtype=options.reduce_dtype,
        moe_use_grouped_mm=options.moe_use_grouped_mm,
        ac=options.ac_config,
        fused_lm_head_token_chunk_size=options.fused_lm_head_token_chunk_size,
        fp32_lm_head=options.fp32_lm_head,
        debug=DebugModelConfig(**options.debug.model_dump()) if options.debug is not None else DebugModelConfig(),
    )


def _adapter() -> MoEDeepSpeedAdapter:
    return MoEDeepSpeedAdapter(
        dtype_map=DTYPE_MAP,
        get_model=get_model,
        configure_moe_ep_backend=configure_moe_ep_backend,
        configure_family_backend=configure_sparse_mla_backend,
        inject_lm_head=inject_prime_lm_head,
        apply_sequence_parallelism=_apply_sequence_parallelism,
        apply_ac=apply_ac,
        load_dcp_from_hf=load_dcp_from_hf,
        reset_runtime_moe_buffers=_reset_runtime_moe_buffers,
        shared_expert_type=BCFeedForward,
        shared_expert_forward=shared_expert_bc_mlp_forward,
        build_model_config=_build_model_config,
    )


def _setup_model_local_no_train(
    config: ModelConfig,
    parallel_dims: ParallelDims,
    ep_mesh: DeviceMesh,
    *,
    fused_cross_entropy: bool | str = False,
    tiled_mlp_token_chunk_size: int | None = None,
) -> nn.Module:
    return setup_model_local_no_train(
        _adapter(),
        config,
        parallel_dims,
        ep_mesh,
        fused_cross_entropy=fused_cross_entropy,
        tiled_mlp_token_chunk_size=tiled_mlp_token_chunk_size,
    )


def load_moe_model_for_deepspeed(
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
    _validate_sp_size(sp_size)
    return _load_moe_model_for_deepspeed(
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


def load_glm_moe_dsa_model(
    *,
    model_name: str,
    optimization_dtype: str,
    attn_implementation: str,
    ep_size: int,
    sp_size: int = 1,
    sp_group=None,
    ep_group=None,
    options: GlmMoeDsaOptions,
) -> nn.Module:
    _validate_sp_size(sp_size)
    return _load_moe_model(
        _adapter(),
        load_moe_model_for_deepspeed,
        model_name=model_name,
        optimization_dtype=optimization_dtype,
        attn_implementation=attn_implementation,
        ep_size=ep_size,
        sp_size=sp_size,
        sp_group=sp_group,
        ep_group=ep_group,
        options=options,
        patch_moe_detection=patch_deepspeed_moe_detection,
        device_mesh_type=DeviceMesh,
    )
