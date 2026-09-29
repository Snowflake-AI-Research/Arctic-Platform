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
"""Model-agnostic GatedDeltaNet sequence-parallel integration."""

from __future__ import annotations

import logging
import types
from typing import Any
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ..packing import cu_seqlens_from_position_ids
from .collectives import sequence_head_all_to_all

logger = logging.getLogger(__name__)

_CP_CONTEXT_ATTRIBUTE = "_dss_gated_delta_net_cp_context"
# Set by the model's SP wrapper before the layer loop; see qwen35/sequence_parallel.py.
_GLOBAL_CU_SEQLENS_ATTRIBUTE = "_dss_sp_global_cu_seqlens"


def build_gated_delta_net_cp_context(
    *,
    local_sequence_length: int,
    device: torch.device,
    process_group,
    convolution_kernel_size: int,
    global_cu_seqlens: torch.Tensor,
):
    """Build one FLA context for causal convolution across sequence shards.

    ``global_cu_seqlens`` are the packed sequence boundaries across the whole SP group, and they are required:
    FLA needs the global view to reset the convolution's left pad at the right tokens. A shard-local boundary
    list cannot express a sequence that continues on the next rank, and a packed call holds several sequences
    whose convolution state must not leak into each other. There is no safe default -- assuming one sequence per
    group is wrong for every packed call, and wrong silently, so callers derive the boundaries from all-gathered
    ``position_ids`` instead.
    """
    from fla.ops.cp import build_cp_context

    if not torch.is_tensor(global_cu_seqlens):
        raise ValueError(
            "GatedDeltaNet context parallelism requires the packed group's global cu_seqlens, got "
            f"{global_cu_seqlens!r}"
        )
    del local_sequence_length, device
    return build_cp_context(
        cu_seqlens=global_cu_seqlens,
        group=process_group,
        conv1d_kernel_size=convolution_kernel_size,
    )


def context_parallel_causal_convolution(
    input_tensor: torch.Tensor,
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    cp_context,
) -> torch.Tensor:
    """Run FLA causal convolution on an HF-style ``[B, D, T]`` tensor."""
    from fla.modules.conv import causal_conv1d

    output, _ = causal_conv1d(
        input_tensor.transpose(1, 2),
        weight=weight,
        bias=bias,
        activation=activation,
        cp_context=cp_context,
    )
    return output.transpose(1, 2)


def _validate_global_cu_seqlens(
    global_cu_seqlens: torch.Tensor,
    *,
    global_sequence_length: int,
) -> None:
    if not torch.is_tensor(global_cu_seqlens) or global_cu_seqlens.ndim != 1:
        raise RuntimeError("head-parallel GatedDeltaNet requires one-dimensional global cu_seqlens")
    if global_cu_seqlens.numel() < 2:
        raise RuntimeError("head-parallel GatedDeltaNet requires at least one packed sequence")
    if global_cu_seqlens.device.type == "cpu":
        if int(global_cu_seqlens[0].item()) != 0:
            raise RuntimeError("global cu_seqlens must start at zero")
        if int(global_cu_seqlens[-1].item()) != global_sequence_length:
            raise RuntimeError(
                "global cu_seqlens must cover the complete SP sequence: "
                f"last boundary is {int(global_cu_seqlens[-1].item())}, expected "
                f"{global_sequence_length}"
            )
        if bool(torch.any(global_cu_seqlens[1:] < global_cu_seqlens[:-1]).item()):
            raise RuntimeError("global cu_seqlens must be nondecreasing")


def _pack_and_exchange_head_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    process_group,
) -> dict[str, torch.Tensor]:
    """Exchange heterogeneous head tensors with one all-to-all per dtype."""
    world_size = dist.get_world_size(process_group)
    first = next(iter(tensors.values()))
    if first.ndim not in (3, 4):
        raise RuntimeError("head-parallel GatedDeltaNet tensors must have shape [B, T, H] or [B, T, H, D]")
    batch_size, local_sequence_length = first.shape[:2]

    grouped: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    specs: dict[str, tuple[int, tuple[int, ...]]] = {}
    for name, tensor in tensors.items():
        if tensor.ndim not in (3, 4):
            raise RuntimeError(
                f"head-parallel GatedDeltaNet tensor {name} has unsupported shape {tuple(tensor.shape)}"
            )
        if tensor.shape[:2] != (batch_size, local_sequence_length):
            raise RuntimeError(
                f"head-parallel GatedDeltaNet tensor {name} has batch/sequence "
                f"shape {tuple(tensor.shape[:2])}, expected "
                f"{(batch_size, local_sequence_length)}"
            )
        num_heads = int(tensor.shape[2])
        if num_heads % world_size:
            raise RuntimeError(
                f"head-parallel GatedDeltaNet tensor {name} has {num_heads} heads, "
                f"which is not divisible by SP size {world_size}"
            )
        tail_shape = tuple(tensor.shape[3:])
        specs[name] = (num_heads // world_size, tail_shape)
        grouped.setdefault(tensor.dtype, []).append((name, tensor))

    exchanged: dict[str, torch.Tensor] = {}
    for fields in grouped.values():
        destination_chunks = []
        widths: dict[str, int] = {}
        for destination in range(world_size):
            packed_fields = []
            for name, tensor in fields:
                local_heads, _tail_shape = specs[name]
                start = destination * local_heads
                stop = start + local_heads
                part = tensor[:, :, start:stop]
                width = part[0, 0].numel()
                widths[name] = width
                packed_fields.append(part.reshape(batch_size, local_sequence_length, width))
            destination_chunks.append(torch.cat(packed_fields, dim=-1))

        packed = torch.stack(destination_chunks, dim=2)
        packed = sequence_head_all_to_all(
            process_group,
            packed,
            scatter_dim=2,
            gather_dim=1,
        ).squeeze(2)

        offset = 0
        global_sequence_length = local_sequence_length * world_size
        for name, _tensor in fields:
            width = widths[name]
            local_heads, tail_shape = specs[name]
            field = packed[:, :, offset : offset + width]
            exchanged[name] = field.reshape(
                batch_size,
                global_sequence_length,
                local_heads,
                *tail_shape,
            )
            offset += width
    return exchanged


def _shard_head_parameter(
    value: Any,
    *,
    name: str,
    num_value_heads: int,
    process_group,
) -> Any:
    if value is None or not torch.is_tensor(value):
        return value
    if value.ndim != 1 or value.numel() != num_value_heads:
        raise RuntimeError(
            f"head-parallel GatedDeltaNet expected {name} with shape [{num_value_heads}], got {tuple(value.shape)}"
        )
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    local_heads = num_value_heads // world_size
    return value.narrow(0, rank * local_heads, local_heads)


def _shard_convolution_parameter(
    value: torch.Tensor | None,
    *,
    name: str,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    process_group,
) -> torch.Tensor | None:
    """Select this rank's Q/K/V depthwise-convolution channels."""
    if value is None:
        return None
    key_channels = num_key_heads * key_head_dim
    value_channels = num_value_heads * value_head_dim
    expected_channels = key_channels * 2 + value_channels
    if value.ndim not in (1, 2) or int(value.shape[0]) != expected_channels:
        raise RuntimeError(
            f"head-parallel GatedDeltaNet expected {name} with leading dimension "
            f"{expected_channels}, got {tuple(value.shape)}"
        )

    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    local_key_channels = key_channels // world_size
    local_value_channels = value_channels // world_size
    return torch.cat(
        [
            value.narrow(0, rank * local_key_channels, local_key_channels),
            value.narrow(
                0,
                key_channels + rank * local_key_channels,
                local_key_channels,
            ),
            value.narrow(
                0,
                key_channels * 2 + rank * local_value_channels,
                local_value_channels,
            ),
        ],
        dim=0,
    )


def _packed_sequence_indices(global_cu_seqlens: torch.Tensor) -> torch.Tensor:
    sequence_lengths = global_cu_seqlens[1:] - global_cu_seqlens[:-1]
    return torch.repeat_interleave(
        torch.arange(
            sequence_lengths.numel(),
            dtype=torch.int32,
            device=global_cu_seqlens.device,
        ),
        sequence_lengths,
    ).unsqueeze(0)


def _fallback_depthwise_causal_convolution(
    x: torch.Tensor,
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    global_cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Depthwise packed convolution used only when causal-conv1d is unavailable."""
    kernel_size = int(weight.shape[1])
    outputs = []
    for start, stop in zip(
        global_cu_seqlens[:-1].tolist(),
        global_cu_seqlens[1:].tolist(),
        strict=True,
    ):
        if start == stop:
            continue
        segment = x[:, :, start:stop]
        convolved = F.conv1d(
            segment,
            weight.unsqueeze(1),
            bias=bias,
            padding=kernel_size - 1,
            groups=int(weight.shape[0]),
        )[:, :, : stop - start]
        outputs.append(convolved)
    if not outputs:
        return x[:, :, :0]
    output = torch.cat(outputs, dim=-1)
    if activation in ("silu", "swish"):
        return F.silu(output)
    if activation is None:
        return output
    raise RuntimeError(f"unsupported GatedDeltaNet causal-convolution activation {activation!r}")


def head_parallel_gated_delta_net(
    causal_convolution: Callable[..., torch.Tensor] | None,
    gated_delta_rule: Callable[..., Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    *,
    convolution_weight: torch.Tensor,
    convolution_bias: torch.Tensor | None,
    convolution_activation: str | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    process_group,
    global_cu_seqlens: torch.Tensor,
    num_key_heads: int,
    num_value_heads: int,
    **rule_kwargs,
):
    """Run convolution and recurrence on a full-sequence GDN head shard.

    The first all-to-all exchanges native projected Q/K/V and gate preactivations
    from local-sequence/all-heads to full-sequence/local-heads. The depthwise
    convolution and recurrent rule then share that representation. A final
    all-to-all restores local-sequence/all-value-heads for the norm and output
    projection.
    """
    world_size = dist.get_world_size(process_group)
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise RuntimeError("head-parallel GatedDeltaNet requires q/k/v shaped [B, T, H, D]")
    if q.shape[0] != 1:
        raise RuntimeError(f"head-parallel packed GatedDeltaNet requires B == 1, got B={q.shape[0]}")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise RuntimeError("head-parallel GatedDeltaNet q/k/v batch and token shapes must match")
    if int(q.shape[2]) != num_key_heads or int(k.shape[2]) != num_key_heads:
        raise RuntimeError(
            "head-parallel GatedDeltaNet expects native Q/K heads before exchange: "
            f"expected {num_key_heads}, got {int(q.shape[2])}/{int(k.shape[2])}"
        )
    if int(v.shape[2]) != num_value_heads:
        raise RuntimeError(
            f"head-parallel GatedDeltaNet expected {num_value_heads} value heads, got {int(v.shape[2])}"
        )
    if b.shape != v.shape[:3] or a.shape != v.shape[:3]:
        raise RuntimeError(
            "head-parallel GatedDeltaNet requires b/a to match v's [B, T, H] "
            f"shape {tuple(v.shape[:3])}, got {tuple(b.shape)} and {tuple(a.shape)}"
        )
    if num_key_heads % world_size or num_value_heads % world_size:
        raise RuntimeError(
            f"SP size {world_size} must divide GatedDeltaNet key/value heads ({num_key_heads}/{num_value_heads})"
        )
    if num_value_heads % num_key_heads:
        raise RuntimeError(
            f"GatedDeltaNet value heads ({num_value_heads}) must be divisible by key heads ({num_key_heads})"
        )

    local_sequence_length = int(v.shape[1])
    global_sequence_length = local_sequence_length * world_size
    _validate_global_cu_seqlens(
        global_cu_seqlens,
        global_sequence_length=global_sequence_length,
    )
    exchanged = _pack_and_exchange_head_tensors(
        {"q": q, "k": k, "v": v, "b": b, "a": a},
        process_group=process_group,
    )

    key_head_dim = int(q.shape[3])
    value_head_dim = int(v.shape[3])
    local_key_heads = num_key_heads // world_size
    local_value_heads = num_value_heads // world_size
    local_convolution_weight = _shard_convolution_parameter(
        convolution_weight,
        name="convolution_weight",
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        process_group=process_group,
    )
    local_convolution_bias = _shard_convolution_parameter(
        convolution_bias,
        name="convolution_bias",
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        process_group=process_group,
    )
    mixed_qkv = torch.cat(
        [
            exchanged["q"].reshape(1, global_sequence_length, -1),
            exchanged["k"].reshape(1, global_sequence_length, -1),
            exchanged["v"].reshape(1, global_sequence_length, -1),
        ],
        dim=-1,
    ).transpose(1, 2)
    if causal_convolution is not None:
        mixed_qkv = causal_convolution(
            x=mixed_qkv,
            weight=local_convolution_weight,
            bias=local_convolution_bias,
            activation=convolution_activation,
            seq_idx=_packed_sequence_indices(global_cu_seqlens),
        )
    else:
        mixed_qkv = _fallback_depthwise_causal_convolution(
            mixed_qkv,
            weight=local_convolution_weight,
            bias=local_convolution_bias,
            activation=convolution_activation,
            global_cu_seqlens=global_cu_seqlens,
        )

    mixed_qkv = mixed_qkv.transpose(1, 2)
    local_key_dim = local_key_heads * key_head_dim
    local_value_dim = local_value_heads * value_head_dim
    query, key, value = torch.split(
        mixed_qkv,
        [local_key_dim, local_key_dim, local_value_dim],
        dim=-1,
    )
    query = query.reshape(1, global_sequence_length, local_key_heads, key_head_dim)
    key = key.reshape(1, global_sequence_length, local_key_heads, key_head_dim)
    value = value.reshape(
        1,
        global_sequence_length,
        local_value_heads,
        value_head_dim,
    )

    local_A_log = _shard_head_parameter(
        A_log,
        name="A_log",
        num_value_heads=num_value_heads,
        process_group=process_group,
    )
    local_dt_bias = _shard_head_parameter(
        dt_bias,
        name="dt_bias",
        num_value_heads=num_value_heads,
        process_group=process_group,
    )
    beta = exchanged["b"].sigmoid()
    g = -local_A_log.float().exp() * F.softplus(exchanged["a"].float() + local_dt_bias)

    replication = num_value_heads // num_key_heads
    if replication > 1:
        query = query.repeat_interleave(replication, dim=2)
        key = key.repeat_interleave(replication, dim=2)

    if rule_kwargs.get("initial_state") is not None:
        raise RuntimeError("head-parallel GatedDeltaNet training does not support an initial recurrent state")
    if rule_kwargs.get("output_final_state", False):
        raise RuntimeError("head-parallel GatedDeltaNet training does not support returning recurrent state")
    rule_kwargs["initial_state"] = None
    rule_kwargs["output_final_state"] = False
    rule_kwargs["cu_seqlens"] = global_cu_seqlens
    rule_kwargs["cu_seqlens_cpu"] = None
    rule_kwargs["cp_context"] = None
    output, final_state = gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        **rule_kwargs,
    )
    if final_state is not None:
        raise RuntimeError("head-parallel GatedDeltaNet unexpectedly returned recurrent state")
    return (
        sequence_head_all_to_all(
            process_group,
            output,
            scatter_dim=1,
            gather_dim=2,
        ),
        None,
    )


def head_parallel_gated_delta_rule(
    gated_delta_rule: Callable[..., Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *rule_args,
    process_group,
    global_cu_seqlens: torch.Tensor,
    num_key_heads: int,
    num_value_heads: int,
    **kwargs,
):
    """Run the exact full recurrence on a shard of GDN heads.

    Each rank exchanges its local token window for ``1 / sp`` of the value
    heads, so it temporarily holds the full sequence only for that head shard.
    The total Q/K/V/gate element count per rank remains proportional to
    ``sequence_length * num_heads / sp``; this is not a full-hidden-state
    all-gather.
    """
    world_size = dist.get_world_size(process_group)
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise RuntimeError("head-parallel GatedDeltaNet requires q/k/v shaped [B, T, H, D]")
    if q.shape[0] != 1:
        raise RuntimeError(f"head-parallel packed GatedDeltaNet requires B == 1, got B={q.shape[0]}")
    if num_key_heads < 1 or num_value_heads < 1:
        raise RuntimeError("head-parallel GatedDeltaNet head counts must be positive")
    if num_key_heads % world_size or num_value_heads % world_size:
        raise RuntimeError(
            f"SP size {world_size} must divide GatedDeltaNet key/value heads ({num_key_heads}/{num_value_heads})"
        )
    if num_value_heads % num_key_heads:
        raise RuntimeError(
            f"GatedDeltaNet value heads ({num_value_heads}) must be divisible by key heads ({num_key_heads})"
        )

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if int(tensor.shape[2]) != num_value_heads:
            raise RuntimeError(
                f"head-parallel GatedDeltaNet expected {name} to have "
                f"{num_value_heads} effective value heads, got "
                f"{int(tensor.shape[2])}. Qwen must repeat Q/K from "
                f"{num_key_heads} native key heads before the SP exchange."
            )
    if g.shape != v.shape[:3] or beta.shape != v.shape[:3]:
        raise RuntimeError(
            "head-parallel GatedDeltaNet requires g and beta to match "
            f"v's [B, T, H] shape {tuple(v.shape[:3])}, got "
            f"{tuple(g.shape)} and {tuple(beta.shape)}"
        )

    local_sequence_length = int(v.shape[1])
    global_sequence_length = local_sequence_length * world_size
    _validate_global_cu_seqlens(
        global_cu_seqlens,
        global_sequence_length=global_sequence_length,
    )
    exchanged = _pack_and_exchange_head_tensors(
        {"q": q, "k": k, "v": v, "g": g, "beta": beta},
        process_group=process_group,
    )

    if kwargs.get("initial_state") is not None:
        raise RuntimeError("head-parallel GatedDeltaNet training does not support an initial recurrent state")
    if kwargs.get("output_final_state", False):
        raise RuntimeError("head-parallel GatedDeltaNet training does not support returning recurrent state")
    kwargs["initial_state"] = None
    kwargs["output_final_state"] = False
    kwargs["cu_seqlens"] = global_cu_seqlens
    kwargs["cu_seqlens_cpu"] = None
    kwargs["cp_context"] = None
    for name in ("A_log", "dt_bias"):
        if name in kwargs:
            kwargs[name] = _shard_head_parameter(
                kwargs[name],
                name=name,
                num_value_heads=num_value_heads,
                process_group=process_group,
            )

    output, final_state = gated_delta_rule(
        exchanged["q"],
        exchanged["k"],
        exchanged["v"],
        exchanged["g"],
        exchanged["beta"],
        *rule_args,
        **kwargs,
    )
    if final_state is not None:
        raise RuntimeError("head-parallel GatedDeltaNet unexpectedly returned recurrent state")
    output = sequence_head_all_to_all(
        process_group,
        output,
        scatter_dim=1,
        gather_dim=2,
    )
    return output, None


def _is_gated_delta_net_module(module: nn.Module) -> bool:
    return all(
        hasattr(module, attribute) for attribute in ("causal_conv1d_fn", "chunk_gated_delta_rule", "conv_kernel_size")
    )


def _hidden_states_from_call(args: tuple, kwargs: dict) -> torch.Tensor:
    hidden_states = kwargs.get("hidden_states")
    if hidden_states is None and args:
        hidden_states = args[0]
    if hidden_states is None or not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
        raise RuntimeError("GatedDeltaNet sequence parallelism requires hidden_states with shape [B, T, D]")
    return hidden_states


def _module_head_counts(module: nn.Module) -> tuple[int, int]:
    num_key_heads = getattr(module, "num_k_heads", None)
    if num_key_heads is None:
        num_key_heads = getattr(module, "num_heads", None)
    num_value_heads = getattr(module, "num_v_heads", None)
    if not isinstance(num_key_heads, int) or num_key_heads < 1:
        raise RuntimeError("compatible GatedDeltaNet module is missing a positive num_k_heads/num_heads")
    if not isinstance(num_value_heads, int) or num_value_heads < 1:
        raise RuntimeError("compatible GatedDeltaNet module is missing a positive num_v_heads")
    return num_key_heads, num_value_heads


def _adapt_gated_delta_net_module(module: nn.Module, process_group) -> None:
    required_attributes = (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "conv1d",
        "A_log",
        "dt_bias",
        "norm",
        "out_proj",
        "head_k_dim",
        "head_v_dim",
        "activation",
    )
    missing = [attribute for attribute in required_attributes if not hasattr(module, attribute)]
    if missing:
        raise RuntimeError(
            "compatible GatedDeltaNet module is missing attributes required for "
            f"full temporal-core head parallelism: {missing}"
        )

    original_causal_convolution = module.causal_conv1d_fn
    original_gated_delta_rule = module.chunk_gated_delta_rule
    num_key_heads, num_value_heads = _module_head_counts(module)
    world_size = dist.get_world_size(process_group)
    if num_key_heads % world_size or num_value_heads % world_size:
        raise RuntimeError(
            f"SP size {world_size} must divide GatedDeltaNet key/value heads ({num_key_heads}/{num_value_heads})"
        )
    if num_value_heads % num_key_heads:
        raise RuntimeError(
            f"GatedDeltaNet value heads ({num_value_heads}) must be divisible by key heads ({num_key_heads})"
        )

    def forward(self, *args, **kwargs):
        hidden_states = _hidden_states_from_call(args, kwargs)
        cache_params = kwargs.get("cache_params")
        if cache_params is None and len(args) > 1:
            cache_params = args[1]
        if cache_params is not None:
            raise RuntimeError(
                "head-parallel GatedDeltaNet is a training path and does not support "
                "cached recurrent or convolution state"
            )
        global_cu_seqlens = getattr(self, _GLOBAL_CU_SEQLENS_ATTRIBUTE, None)
        if not torch.is_tensor(global_cu_seqlens):
            raise RuntimeError(
                "GatedDeltaNet sequence parallelism needs the packed group's global cu_seqlens on the module "
                f"as {_GLOBAL_CU_SEQLENS_ATTRIBUTE!r}, and this call has none. The model's forward is supposed "
                "to stash them before the layer loop; reaching a layer without them means that wrapper did not "
                "run."
            )

        batch_size, local_sequence_length, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(
            batch_size,
            local_sequence_length,
            num_value_heads,
            int(self.head_v_dim),
        )
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        key_dim = num_key_heads * int(self.head_k_dim)
        value_dim = num_value_heads * int(self.head_v_dim)
        query, key, value = torch.split(
            mixed_qkv,
            [key_dim, key_dim, value_dim],
            dim=-1,
        )
        query = query.reshape(
            batch_size,
            local_sequence_length,
            num_key_heads,
            int(self.head_k_dim),
        )
        key = key.reshape(
            batch_size,
            local_sequence_length,
            num_key_heads,
            int(self.head_k_dim),
        )
        value = value.reshape(
            batch_size,
            local_sequence_length,
            num_value_heads,
            int(self.head_v_dim),
        )
        core_attn_out, _ = head_parallel_gated_delta_net(
            original_causal_convolution,
            original_gated_delta_rule,
            query,
            key,
            value,
            b,
            a,
            convolution_weight=self.conv1d.weight.squeeze(1),
            convolution_bias=self.conv1d.bias,
            convolution_activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            process_group=process_group,
            global_cu_seqlens=global_cu_seqlens,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out = core_attn_out.reshape(-1, int(self.head_v_dim))
        z = z.reshape(-1, int(self.head_v_dim))
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(
            batch_size,
            local_sequence_length,
            -1,
        )
        return self.out_proj(core_attn_out)

    module.forward = types.MethodType(forward, module)
    module._dss_gated_delta_net_sequence_parallel = True
    module._dss_gated_delta_net_head_parallel = True
    module._dss_gated_delta_net_full_temporal_core = True


def _install_global_boundary_wrapper(model: nn.Module, modules: list, process_group) -> None:
    """Wrap ``model.forward`` to publish the packed group's global boundaries to its linear-attention layers.

    The layers cannot derive the boundaries themselves. Each rank sees one window of the packed call, and a row
    that starts on an earlier rank looks, locally, like a row that starts at the window's first token. Gathering
    the positions across the SP group restores the packed order (shards are contiguous in rank order), and the
    positions then say where every row begins.

    The boundaries are stashed on the modules rather than threaded through the call because the layers are
    reached through the model's own decoder loop, whose signature this code does not control.
    """
    original_forward = model.forward

    def forward(*args, **kwargs):
        position_ids = kwargs.get("position_ids")
        if not torch.is_tensor(position_ids):
            raise ValueError(
                "GatedDeltaNet sequence parallelism needs position_ids on the model call to locate the packed "
                "rows: recurrent state has to reset at each row boundary, and a rank cannot see those "
                "boundaries in its own window."
            )
        world_size = dist.get_world_size(process_group)
        gathered = [torch.empty_like(position_ids) for _ in range(world_size)]
        dist.all_gather(gathered, position_ids.contiguous(), group=process_group)
        # SP shards are contiguous in rank order, so concatenation rebuilds the packed call's positions.
        global_cu_seqlens = cu_seqlens_from_position_ids(torch.cat(gathered, dim=1))
        for module in modules:
            setattr(module, _GLOBAL_CU_SEQLENS_ATTRIBUTE, global_cu_seqlens)
        return original_forward(*args, **kwargs)

    model.forward = forward


def apply_gated_delta_net_sequence_parallelism(model: nn.Module, process_group) -> int:
    """Adapt every HF-style GatedDeltaNet module in ``model`` and return the count."""
    if process_group is None:
        return 0

    compatible_modules = 0
    adapted = []
    for module in model.modules():
        if not _is_gated_delta_net_module(module):
            continue
        compatible_modules += 1
        if getattr(module, "_dss_gated_delta_net_sequence_parallel", False):
            continue
        _adapt_gated_delta_net_module(module, process_group)
        adapted.append(module)
    adapted_modules = len(adapted)
    if adapted:
        _install_global_boundary_wrapper(model.base_model, adapted, process_group)

    text_config = model.config.get_text_config() if hasattr(model, "config") else None
    layer_types = getattr(text_config, "layer_types", ())
    if "linear_attention" in layer_types and compatible_modules == 0:
        raise RuntimeError(
            "model configuration contains linear_attention layers, but no compatible GatedDeltaNet modules were found"
        )

    if adapted_modules:
        logger.info(
            "Applied GatedDeltaNet SP to %d linear-attention modules "
            "(full-sequence/local-head convolution and recurrence)",
            adapted_modules,
        )
    return adapted_modules
