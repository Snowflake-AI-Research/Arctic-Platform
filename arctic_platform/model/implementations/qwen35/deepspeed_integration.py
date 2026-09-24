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
"""DeepSpeed integration for the carved-out Qwen3.5 MoE model.

Builds a Prime-RL-style MoE model with expert parallelism but without FSDP,
with local-tensor parameters that DeepSpeed ZeRO can manage.

Public API:
  ``patch_deepspeed_moe_detection``  -- make DeepSpeed recognize the MoE layers.
  ``load_moe_model_for_deepspeed``   -- build a model ready for ``deepspeed.initialize``.
  ``load_moe_model_for_dss``         -- dss-platform entry (model_name + ep_size).

Caller owns ``deepspeed.init_distributed``, ``torch.cuda.set_device``,
``deepspeed.utils.groups._create_expert_and_data_parallel(ep_size)``, optimizer
construction, and ``deepspeed.initialize``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from .config import ActivationCheckpointConfig, DebugModelConfig, ModelConfig
from .sequence_parallel import apply_sequence_parallelism
from .model_builder import (
    DTYPE_MAP,
    _reset_runtime_moe_buffers,
    apply_ac,
    configure_moe_ep_backend,
    get_model,
    load_dcp_from_hf,
)
from arctic_platform.model.implementations.moe.deepspeed_integration import (
    MoEDeepSpeedAdapter,
    apply_ep_with_mesh as _apply_ep_with_mesh,
    build_iter_full_hf_weights as _build_iter_full_hf_weights,
    convert_dtensors_to_local as _convert_dtensors_to_local,
    load_moe_model_for_deepspeed as _load_moe_model_for_deepspeed,
    load_moe_model_for_dss as _load_moe_model_for_dss,
    patch_deepspeed_moe_detection,
    setup_model_local_no_train,
    tag_expert_lora_adapters_for_deepspeed,
    tag_expert_params_for_deepspeed as _tag_expert_params_for_deepspeed,
)
from arctic_platform.model.implementations.moe.config_validation import (
    validate_lm_head_fused_ce_config,
)
from arctic_platform.model.implementations.moe.layers.lm_head import inject_prime_lm_head
from arctic_platform.model.implementations.moe.layers.moe import FeedForward
from arctic_platform.model.implementations.moe.parallel_dims import ParallelDims
from arctic_platform.model.implementations.debug.row_invariant_projection import maybe_apply_row_invariant_projections


def shared_expert_mlp_forward(
    feed_forward: FeedForward, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Un-tiled dense shared-expert FFN compute (``w2(silu(w1 x) * w3 x)``), run per token shard by TiledMLP."""
    return feed_forward.w2(
        F.silu(feed_forward.w1(hidden_states)) * feed_forward.w3(hidden_states)
    )


def _convert_qwen3_5_moe_layer_to_vllm(
    layer_sd: dict,
    layer_idx: int,
    *,
    layer_prefix: str | None = None,
) -> dict:
    """Rename/pack one Qwen3.5 MoE layer from Prime-RL to vLLM bf16 Triton storage."""
    import torch

    if layer_idx < 0:
        return layer_sd

    prefix = layer_prefix if layer_prefix is not None else f"model.layers.{layer_idx}"

    router_key = f"{prefix}.mlp.router.gate.weight"
    if router_key in layer_sd:
        layer_sd[f"{prefix}.mlp.gate.weight"] = layer_sd.pop(router_key)

    w1_key = f"{prefix}.mlp.experts.w1"
    w3_key = f"{prefix}.mlp.experts.w3"
    w2_key = f"{prefix}.mlp.experts.w2"
    if w1_key in layer_sd and w3_key in layer_sd:
        w1 = layer_sd.pop(w1_key)
        w3 = layer_sd.pop(w3_key)
        layer_sd[f"{prefix}.mlp.experts.w13_weight"] = torch.cat([w1, w3], dim=1)
    if w2_key in layer_sd:
        layer_sd[f"{prefix}.mlp.experts.w2_weight"] = layer_sd.pop(w2_key)

    shared_renames = {
        f"{prefix}.shared_expert.w1.weight": f"{prefix}.mlp.shared_expert.gate_proj.weight",
        f"{prefix}.shared_expert.w3.weight": f"{prefix}.mlp.shared_expert.up_proj.weight",
        f"{prefix}.shared_expert.w2.weight": f"{prefix}.mlp.shared_expert.down_proj.weight",
        f"{prefix}.shared_expert_gate.weight": f"{prefix}.mlp.shared_expert_gate.weight",
    }
    for old, new in shared_renames.items():
        if old in layer_sd:
            layer_sd[new] = layer_sd.pop(old)

    qkv_key = f"{prefix}.linear_attn.in_proj_qkv.weight"
    z_key = f"{prefix}.linear_attn.in_proj_z.weight"
    if qkv_key in layer_sd and z_key in layer_sd:
        qkv = layer_sd.pop(qkv_key)
        z = layer_sd.pop(z_key)
        layer_sd[f"{prefix}.linear_attn.in_proj_qkvz.weight"] = torch.cat(
            [qkv, z], dim=0
        )

    b_key = f"{prefix}.linear_attn.in_proj_b.weight"
    a_key = f"{prefix}.linear_attn.in_proj_a.weight"
    if b_key in layer_sd and a_key in layer_sd:
        b = layer_sd.pop(b_key)
        a = layer_sd.pop(a_key)
        layer_sd[f"{prefix}.linear_attn.in_proj_ba.weight"] = torch.cat([b, a], dim=0)

    for buf_key in (f"{prefix}.mlp.expert_bias", f"{prefix}.mlp.tokens_per_expert"):
        layer_sd.pop(buf_key, None)

    return layer_sd


def _to_vllm_vlm_name(name: str) -> str:
    """Rename a Prime-RL VLM parameter key to vLLM's VLM internal layout."""
    if name.startswith("model.visual."):
        return "visual." + name[len("model.visual.") :]
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model.") :]
    if name.startswith("lm_head."):
        return "language_model.lm_head." + name[len("lm_head.") :]
    raise RuntimeError(
        f"_to_vllm_vlm_name: no Prime-RL -> vLLM VLM rule for parameter "
        f"name {name!r}. Either Prime-RL grew a new top-level submodule "
        f"or this model is not the Qwen3.5-MoE VLM shape this helper "
        f"was written for. Extend _to_vllm_vlm_name with a new rule."
    )


def _build_iter_full_vllm_weights(model: nn.Module):
    """Iterator yielding ``(vllm_name, full_tensor)`` on rank 0 for weight sync."""
    cls_name = type(model).__name__
    assert cls_name.startswith("Qwen3_5Moe"), (
        f"_build_iter_full_vllm_weights only supports Qwen3.5 MoE today "
        f"(got {cls_name}); add a per-family layer converter and dispatch."
    )

    import deepspeed.utils.groups as ds_groups
    import torch
    import torch.distributed as dist

    is_vlm = bool(getattr(model, "_is_vlm", False))
    layer_prefix_pattern = (
        "model.language_model.layers.{i}" if is_vlm else "model.layers.{i}"
    )

    def _strip_ac_wrapper(name: str) -> str:
        return name.replace("._checkpoint_wrapped_module", "")

    def _layer_idx(name: str) -> int:
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
            try:
                return int(parts[2])
            except ValueError:
                pass
        if (
            len(parts) >= 4
            and parts[0] == "model"
            and parts[1] == "language_model"
            and parts[2] == "layers"
        ):
            try:
                return int(parts[3])
            except ValueError:
                pass
        return -1

    def _iter():
        is_master = dist.get_rank() == 0
        by_layer: dict[int, list[tuple[str, nn.Parameter]]] = {}
        for name, param in model.named_parameters():
            by_layer.setdefault(_layer_idx(name), []).append((name, param))

        ordered_layers: list[int] = []
        if -1 in by_layer:
            ordered_layers.append(-1)
        ordered_layers.extend(sorted(k for k in by_layer if k >= 0))

        for layer_idx in ordered_layers:
            layer_sd: dict[str, torch.Tensor] = {}
            for name, param in by_layer[layer_idx]:
                if (
                    hasattr(param, "group_name")
                    and getattr(param, "allreduce", True) is False
                ):
                    ep_pg = ds_groups._get_expert_parallel_group(param.group_name)
                    local = param.data.contiguous()
                    shards = [
                        torch.empty_like(local)
                        for _ in range(dist.get_world_size(group=ep_pg))
                    ]
                    dist.all_gather(shards, local, group=ep_pg)
                    if is_master:
                        layer_sd[name] = torch.cat(shards, dim=0)
                elif is_master:
                    layer_sd[name] = param.data

            if not is_master:
                continue

            layer_sd = {_strip_ac_wrapper(k): v for k, v in layer_sd.items()}

            _convert_qwen3_5_moe_layer_to_vllm(
                layer_sd,
                layer_idx,
                layer_prefix=(
                    layer_prefix_pattern.format(i=layer_idx) if layer_idx >= 0 else None
                ),
            )

            for name, tensor in layer_sd.items():
                out_name = _to_vllm_vlm_name(name) if is_vlm else name
                yield out_name, tensor

    return _iter


_validate_lm_head_fused_ce_config = validate_lm_head_fused_ce_config


def _configure_family_backend(_config: ModelConfig) -> None:
    pass


def _apply_sequence_parallelism(model: nn.Module, sp_size: int, sp_group) -> None:
    if sp_size > 1:
        apply_sequence_parallelism(model, sp_size, sp_group)


def _normalize_config(config: dict) -> dict:
    return config


def _build_model_config(
    model_name: str, ep_size: int, dp_replicate: int, config: dict
) -> ModelConfig:
    ac_config = config.get("ac_config")
    debug_kwargs = (
        {"debug": DebugModelConfig(**config["debug"])} if config.get("debug") else {}
    )
    return ModelConfig(
        name=model_name,
        weight_conversion_cache_dir=config.get(
            "weight_conversion_cache_dir", ModelConfig.weight_conversion_cache_dir
        ),
        trust_remote_code=config.get(
            "trust_remote_code", ModelConfig.trust_remote_code
        ),
        seq_len=config.get("seq_len", 4096),
        attn=config.get("attn", "flash_attention_3"),
        ep=ep_size,
        ep_comm_backend=config.get("ep_comm_backend", "deepep"),
        deepep_num_sms=config.get("deepep_num_sms", ModelConfig.deepep_num_sms),
        deepep_token_chunk_size=config.get("deepep_token_chunk_size"),
        dp_replicate=dp_replicate,
        cp=1,
        impl="custom",
        optimization_dtype=config.get("optimization_dtype", "bfloat16"),
        reduce_dtype=config.get("reduce_dtype", "float32"),
        moe_use_grouped_mm=config.get("moe_use_grouped_mm", True),
        ac=ActivationCheckpointConfig(**ac_config) if ac_config else None,
        fused_lm_head_token_chunk_size=config.get(
            "fused_lm_head_token_chunk_size", "disabled"
        ),
        fp32_lm_head=config.get("fp32_lm_head", False),
        **debug_kwargs,
    )


def _adapter() -> MoEDeepSpeedAdapter:
    return MoEDeepSpeedAdapter(
        dtype_map=DTYPE_MAP,
        get_model=get_model,
        configure_moe_ep_backend=configure_moe_ep_backend,
        configure_family_backend=_configure_family_backend,
        inject_lm_head=inject_prime_lm_head,
        apply_sequence_parallelism=_apply_sequence_parallelism,
        apply_ac=apply_ac,
        load_dcp_from_hf=load_dcp_from_hf,
        reset_runtime_moe_buffers=_reset_runtime_moe_buffers,
        shared_expert_type=FeedForward,
        shared_expert_forward=shared_expert_mlp_forward,
        build_model_config=_build_model_config,
        normalize_config=_normalize_config,
        extra_weight_iterators=(
            ("_iter_full_vllm_weights", _build_iter_full_vllm_weights),
        ),
    )


def _setup_model_local_no_train(
    config: ModelConfig,
    parallel_dims: ParallelDims,
    ep_mesh: DeviceMesh,
    *,
    fused_cross_entropy: bool | str = False,
    tiled_mlp_token_chunk_size: int | None = None,
    sp_size: int = 1,
    sp_group=None,
) -> nn.Module:
    return setup_model_local_no_train(
        _adapter(),
        config,
        parallel_dims,
        ep_mesh,
        fused_cross_entropy=fused_cross_entropy,
        tiled_mlp_token_chunk_size=tiled_mlp_token_chunk_size,
        sp_size=sp_size,
        sp_group=sp_group,
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


def load_moe_model_for_dss(
    *,
    model_name: str,
    ep_size: int,
    sp_size: int = 1,
    sp_group=None,
    ep_group=None,
    prl_config: dict | None = None,
    tiled_mlp_token_chunk_size: int | None = None,
) -> nn.Module:
    model = _load_moe_model_for_dss(
        _adapter(),
        load_moe_model_for_deepspeed,
        model_name=model_name,
        ep_size=ep_size,
        sp_size=sp_size,
        sp_group=sp_group,
        ep_group=ep_group,
        prl_config=prl_config,
        tiled_mlp_token_chunk_size=tiled_mlp_token_chunk_size,
        patch_moe_detection=patch_deepspeed_moe_detection,
        device_mesh_type=DeviceMesh,
    )

    maybe_apply_row_invariant_projections(model, prl_config or {})

    return model
