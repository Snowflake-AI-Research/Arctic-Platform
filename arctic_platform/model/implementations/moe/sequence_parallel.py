"""Sequence parallelism for generic PrimeRL MoE families."""

from __future__ import annotations

import inspect
import types

import torch
import torch.distributed as dist
from torch import nn

from arctic_platform.model.implementations.gpu.packing import cu_seqlens_from_position_ids
from arctic_platform.model.implementations.gpu.sp.collectives import sequence_head_all_to_all
from arctic_platform.model.implementations.moe.logging_utils import get_logger
from arctic_platform.model.implementations.moe.vlm import get_language_model

_FA_IMPLS = ("flash_attention_2", "flash_attention_3", "flash_attention_4")
_SOFTMAX_ATTN_TYPES = ("full_attention", "sliding_attention")


def _ulysses_attn_forward(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    cu_seqlens=None,
    max_seqlen=None,
):
    del attention_mask
    group = self._sp_group
    if hidden_states.size(0) != 1:
        raise NotImplementedError(
            f"Ulysses varlen path assumes a single packed row (B==1), got B={hidden_states.size(0)}"
        )

    projections = self.attn_projections(hidden_states, position_embeddings)
    query_states, key_states, value_states = projections[:3]
    gate = projections[3] if len(projections) == 4 else None

    sp_world_size = dist.get_world_size(group)
    num_kv_heads = key_states.size(2)
    if sp_world_size > num_kv_heads:
        if sp_world_size % num_kv_heads != 0:
            raise NotImplementedError(
                f"Ulysses KV replication needs sp_world_size ({sp_world_size}) to be a "
                f"multiple of num_kv_heads ({num_kv_heads})"
            )
        kv_replication = sp_world_size // num_kv_heads
        key_states = key_states.repeat_interleave(kv_replication, dim=2)
        value_states = value_states.repeat_interleave(kv_replication, dim=2)

    query_states = sequence_head_all_to_all(group, query_states, scatter_dim=2, gather_dim=1)
    key_states = sequence_head_all_to_all(group, key_states, scatter_dim=2, gather_dim=1)
    value_states = sequence_head_all_to_all(group, value_states, scatter_dim=2, gather_dim=1)

    full_cu_seqlens = getattr(self, "_sp_cu_seqlens", cu_seqlens)
    full_max_seqlen = getattr(self, "_sp_max_seqlen", max_seqlen)
    attn_output = self._attention_core(
        query_states,
        key_states,
        value_states,
        cu_seqlens=full_cu_seqlens,
        max_seqlen=full_max_seqlen,
    )

    if attn_output.dim() == 3 and attn_output.shape[0] == hidden_states.shape[0]:
        attn_output = attn_output.view(
            hidden_states.shape[0],
            query_states.shape[1],
            query_states.shape[2],
            query_states.shape[3],
        )
    elif attn_output.dim() == 3:
        attn_output = attn_output.unsqueeze(0)

    attn_output = sequence_head_all_to_all(group, attn_output, scatter_dim=1, gather_dim=2)
    if gate is not None:
        return self.output_proj(attn_output, gate), None
    return self.output_proj(attn_output.flatten(-2)), None


def _make_backbone_sp_forward(original_forward, sp_group, softmax_attn_modules, linear_attn_modules):
    sp_world_size = dist.get_world_size(sp_group)
    accepted = inspect.signature(original_forward).parameters

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        routed_experts=None,
        **kwargs,
    ):
        if position_ids is not None and (softmax_attn_modules or linear_attn_modules):
            gathered_position_ids = [torch.empty_like(position_ids) for _ in range(sp_world_size)]
            dist.all_gather(gathered_position_ids, position_ids.contiguous(), group=sp_group)
            global_position_ids = torch.cat(gathered_position_ids, dim=1)
            cu_seqlens = cu_seqlens_from_position_ids(global_position_ids)
            max_seqlen = int(cu_seqlens.diff().max().item())
            for module in softmax_attn_modules:
                module._sp_cu_seqlens = cu_seqlens
                module._sp_max_seqlen = max_seqlen
            for module in linear_attn_modules:
                module._dss_sp_global_cu_seqlens = cu_seqlens

        call_kwargs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
            "routed_experts": routed_experts,
            "attention_mask": attention_mask,
        }
        call_kwargs = {name: value for name, value in call_kwargs.items() if name in accepted}
        call_kwargs.update(kwargs)
        return original_forward(**call_kwargs)

    return forward


def apply_sequence_parallelism(model: nn.Module, sp_size: int, sp_group) -> None:
    if sp_size == 1 or sp_group is None:
        return

    backbone = get_language_model(model)
    model_type = getattr(backbone.config, "model_type", None)
    if model_type == "nemotron_h":
        raise NotImplementedError(
            "Nemotron H needs Mamba context parallelism in addition to Ulysses attention; "
            "generic sequence parallelism is unsafe."
        )
    if getattr(backbone.config, "_attn_implementation", None) not in _FA_IMPLS:
        raise ValueError("generic MoE sequence parallelism requires flash_attention_2, _3, or _4")

    sp_world_size = dist.get_world_size(sp_group)
    if sp_world_size != sp_size:
        raise ValueError(f"SP group size ({sp_world_size}) does not match configured sp_size ({sp_size})")

    softmax_attn_modules = []
    linear_attn_modules = []
    for layer in backbone.layers:
        layer_type = getattr(layer, "layer_type", None) or getattr(layer, "attention_type", None)
        if layer_type == "linear_attention" and hasattr(layer, "linear_attn"):
            linear_attn = layer.linear_attn
            linear_attn.cp_group = sp_group
            linear_attn_modules.append(linear_attn)
        elif hasattr(layer, "self_attn") and (layer_type is None or layer_type in _SOFTMAX_ATTN_TYPES):
            self_attn = layer.self_attn
            self_attn._sp_group = sp_group
            self_attn._sp_world_size = sp_world_size
            self_attn.forward = types.MethodType(_ulysses_attn_forward, self_attn)
            softmax_attn_modules.append(self_attn)

    if not softmax_attn_modules and not linear_attn_modules:
        raise TypeError(f"{type(backbone).__name__} exposes no supported sequence-parallel attention layers")

    backbone.forward = types.MethodType(
        _make_backbone_sp_forward(
            backbone.forward,
            sp_group,
            softmax_attn_modules,
            linear_attn_modules,
        ),
        backbone,
    )

    get_logger().info(
        "Applied generic MoE SP (sp_size=%d): wrapped %d softmax-attention and %d linear-attention layers",
        sp_world_size,
        len(softmax_attn_modules),
        len(linear_attn_modules),
    )
