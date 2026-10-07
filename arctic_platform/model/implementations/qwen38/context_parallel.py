# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel training support for Qwen3.8-Flash-Next."""

from __future__ import annotations

import os
import types

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch import nn
from torch.utils.checkpoint import set_checkpoint_early_stop

from arctic_platform.model.implementations.gpu.packing import cu_seqlens_from_position_ids
from arctic_platform.model.implementations.gpu.sp.gated_delta_net import build_gated_delta_net_cp_context
from arctic_platform.model.implementations.moe.logging_utils import get_logger
from arctic_platform.model.implementations.moe.vlm import get_language_model

# TileLang's dqkwg backward kernel can spin indefinitely on H200.
os.environ.setdefault("FLA_TILELANG", "0")
# Keep asynchronous EP backward from overtaking FLA collectives on other ranks.
os.environ.setdefault("ARCTIC_UCCLEP_BACKWARD_BARRIER", "1")

_GLOBAL_CU_SEQLENS = "_dss_sp_global_cu_seqlens"
_FLA_CP_CONTEXT = "_dss_qwen38_fla_cp_context"


def _require_finite(value: torch.Tensor, name: str) -> None:
    if torch.isfinite(value).all():
        return
    finite = value[torch.isfinite(value)]
    finite_range = (float(finite.min()), float(finite.max())) if finite.numel() else (None, None)
    raise FloatingPointError(
        f"Qwen3.8 CP produced non-finite values in {name}; "
        f"shape={tuple(value.shape)}, finite_range={finite_range}"
    )


def _install_finite_check(module: nn.Module, name: str) -> None:
    def check(_module, _args, output):
        values = output if isinstance(output, tuple) else (output,)
        for value in values:
            if torch.is_tensor(value):
                _require_finite(value, name)

    module.register_forward_hook(check)


def _adapt_linear_attention(module: nn.Module, process_group) -> None:
    from fla.modules.conv import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    check_finite = os.environ.get("ARCTIC_QWEN38_CP_CHECK_FINITE") == "1"

    num_key_heads = int(module.num_k_heads)
    num_value_heads = int(module.num_v_heads)

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **_kwargs):
        if cache_params is not None:
            raise RuntimeError("Qwen3.8 context-parallel GatedDeltaNet is a training-only path")
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[..., None].to(hidden_states.dtype)
        if check_finite:
            _require_finite(hidden_states, "GatedDeltaNet hidden states")
        batch_size, local_sequence_length, _ = hidden_states.shape
        cp_context = getattr(self, _FLA_CP_CONTEXT, None)
        if cp_context is None:
            raise RuntimeError("Qwen3.8 GatedDeltaNet requires an FLA context-parallel context")

        mixed_qkv = self.in_proj_qkv(hidden_states)
        if check_finite:
            _require_finite(mixed_qkv, "GatedDeltaNet convolution input")
            _require_finite(self.conv1d.weight, "GatedDeltaNet convolution weight")
        mixed_qkv, _ = causal_conv1d(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            cp_context=cp_context,
        )
        if check_finite:
            _require_finite(mixed_qkv, "GatedDeltaNet convolution output")
        key_width = num_key_heads * int(self.head_k_dim)
        value_width = num_value_heads * int(self.head_v_dim)
        query, key, value = torch.split(
            mixed_qkv,
            [key_width, key_width, value_width],
            dim=-1,
        )
        query = query.view(batch_size, local_sequence_length, num_key_heads, int(self.head_k_dim))
        key = key.view(batch_size, local_sequence_length, num_key_heads, int(self.head_k_dim))
        value = value.view(batch_size, local_sequence_length, num_value_heads, int(self.head_v_dim))
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * torch.nn.functional.softplus(a.float() + self.dt_bias)
        if num_value_heads // num_key_heads > 1:
            replication = num_value_heads // num_key_heads
            query = query.repeat_interleave(replication, dim=2)
            key = key.repeat_interleave(replication, dim=2)
        if check_finite:
            for name, tensor in (
                ("query", query),
                ("key", key),
                ("value", value),
                ("decay", g),
                ("beta", beta),
            ):
                _require_finite(tensor, f"GatedDeltaNet rule {name}")
        output, final_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cp_context=cp_context,
        )
        if final_state is not None:
            raise RuntimeError("Qwen3.8 GatedDeltaNet unexpectedly returned recurrent state")
        if check_finite:
            _require_finite(output, "GatedDeltaNet rule output")
        gate = self.in_proj_z(hidden_states).reshape(-1, int(self.head_v_dim))
        output = self.norm(output.reshape(-1, int(self.head_v_dim)), gate)
        return self.out_proj(output.reshape(batch_size, local_sequence_length, -1))

    module.forward = types.MethodType(forward, module)
    module._dss_gated_delta_net_sequence_parallel = True
    module._dss_gated_delta_net_fla_context_parallel = True


def _previous_rank_context(
    tensor: torch.Tensor,
    width: int,
    *,
    process_group,
    pad_value: int | float,
    differentiable: bool,
) -> torch.Tensor:
    if tensor.shape[1] < width:
        raise ValueError(
            f"Qwen3.8 context parallelism needs local sequence length >= {width}, "
            f"got {tensor.shape[1]}"
        )
    tail = tensor[:, -width:].contiguous()
    if differentiable:
        gathered = dist_nn.all_gather(tail, group=process_group)
    else:
        gathered = [torch.empty_like(tail) for _ in range(dist.get_world_size(process_group))]
        dist.all_gather(gathered, tail, group=process_group)
    rank = dist.get_rank(process_group)
    if rank:
        return gathered[rank - 1]
    previous = torch.full_like(tail, pad_value)
    if differentiable:
        # Rank zero must participate in the all-gather backward with a zero gradient.
        previous = previous + gathered[0] * 0
    return previous


def _adapt_ple_layer(module: nn.Module, process_group) -> None:
    embedding = module.ple_embedding
    embedding_forward = embedding.forward
    context_length = int(embedding.context_len)

    def embedding_cp_forward(self, input_ids, past_key_values):
        if past_key_values is not None:
            raise RuntimeError("Qwen3.8 context-parallel PLE is a training-only path")
        previous = _previous_rank_context(
            input_ids,
            context_length,
            process_group=process_group,
            pad_value=int(self.eos_token_id),
            differentiable=False,
        )
        extended = torch.cat([previous, input_ids], dim=1)
        return embedding_forward(extended, None)[:, context_length:]

    embedding.forward = types.MethodType(embedding_cp_forward, embedding)

    short_conv = module._short_conv
    conv_context_length = int(module.short_conv_state_len)

    def short_conv_cp_forward(self, hidden_states, past_key_values):
        if past_key_values is not None:
            raise RuntimeError("Qwen3.8 context-parallel PLE is a training-only path")
        previous = _previous_rank_context(
            hidden_states,
            conv_context_length,
            process_group=process_group,
            pad_value=0,
            differentiable=True,
        )
        extended = torch.cat([previous, hidden_states], dim=1)
        return short_conv(extended, None)[:, conv_context_length:]

    module._short_conv = types.MethodType(short_conv_cp_forward, module)


def _text_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    if position_ids.ndim == 2:
        return position_ids
    if position_ids.ndim == 3 and position_ids.shape[0] in (1, 4):
        return position_ids[0]
    raise ValueError(
        "Qwen3.8 context parallelism requires [batch, sequence] position_ids "
        f"or Qwen multimodal position_ids, got {tuple(position_ids.shape)}"
    )


def _wrap_backbone_forward(backbone: nn.Module, linear_modules: list[nn.Module], process_group) -> None:
    original_forward = backbone.forward
    world_size = dist.get_world_size(process_group)

    def forward(self, *args, **kwargs):
        position_ids = kwargs.get("position_ids")
        if position_ids is None and len(args) >= 3:
            position_ids = args[2]
        if not torch.is_tensor(position_ids):
            raise ValueError("Qwen3.8 context parallelism requires position_ids")
        local_positions = _text_position_ids(position_ids)
        gathered = [torch.empty_like(local_positions) for _ in range(world_size)]
        dist.all_gather(gathered, local_positions.contiguous(), group=process_group)
        global_cu_seqlens = cu_seqlens_from_position_ids(torch.cat(gathered, dim=1))
        cp_context = build_gated_delta_net_cp_context(
            local_sequence_length=local_positions.shape[-1],
            device=local_positions.device,
            process_group=process_group,
            convolution_kernel_size=int(linear_modules[0].conv_kernel_size),
            global_cu_seqlens=global_cu_seqlens,
        )
        for module in linear_modules:
            setattr(module, _GLOBAL_CU_SEQLENS, global_cu_seqlens)
            setattr(module, _FLA_CP_CONTEXT, cp_context)
        with set_checkpoint_early_stop(False):
            return original_forward(*args, **kwargs)

    backbone.forward = types.MethodType(forward, backbone)


def apply_context_parallelism(model: nn.Module, cp_size: int, cp_group) -> None:
    if cp_size == 1:
        return
    if cp_group is None:
        raise ValueError("Qwen3.8 context parallelism requires an SP process group")
    world_size = dist.get_world_size(cp_group)
    if world_size != cp_size:
        raise ValueError(f"Qwen3.8 CP group size ({world_size}) does not match configured size ({cp_size})")
    model_cp_group = dist.new_group(
        ranks=dist.get_process_group_ranks(cp_group),
        backend=dist.get_backend(cp_group),
        use_local_synchronization=True,
    )

    backbone = get_language_model(model)
    rank = dist.get_rank(model_cp_group)
    qsa_modules = []
    linear_modules = []
    ple_modules = []
    check_finite = os.environ.get("ARCTIC_QWEN38_CP_CHECK_FINITE") == "1"
    for layer_idx, layer in enumerate(backbone.layers):
        if getattr(layer, "layer_type", None) == "linear_attention":
            linear_attention = layer.linear_attn
            _adapt_linear_attention(linear_attention, model_cp_group)
            linear_modules.append(linear_attention)
            if check_finite:
                _install_finite_check(linear_attention, f"layer {layer_idx} GatedDeltaNet")
        elif hasattr(layer, "self_attn"):
            attention = layer.self_attn
            for module in (attention, attention.indexer):
                module._cp_group = model_cp_group
                module._cp_rank = rank
                module._cp_world_size = world_size
            qsa_modules.append(attention)
            if check_finite:
                _install_finite_check(attention, f"layer {layer_idx} QSA")
        if getattr(layer, "ple", None) is not None:
            _adapt_ple_layer(layer.ple, model_cp_group)
            ple_modules.append(layer.ple)
            if check_finite:
                _install_finite_check(layer.ple, f"layer {layer_idx} PLE")
        if check_finite:
            _install_finite_check(layer.mlp, f"layer {layer_idx} MoE")
            _install_finite_check(layer, f"layer {layer_idx}")

    if not qsa_modules or not linear_modules:
        raise TypeError(
            "Qwen3.8 context parallelism requires both QSA and GatedDeltaNet layers; "
            f"found {len(qsa_modules)} and {len(linear_modules)}"
        )
    _wrap_backbone_forward(backbone, linear_modules, model_cp_group)
    get_logger().info(
        "Applied Qwen3.8 context parallelism (cp_size=%d): %d QSA, %d GatedDeltaNet, %d PLE layers",
        world_size,
        len(qsa_modules),
        len(linear_modules),
        len(ple_modules),
    )


__all__ = ["apply_context_parallelism"]
