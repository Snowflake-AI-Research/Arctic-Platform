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

import importlib
import inspect
import logging
import types
import weakref
from dataclasses import dataclass
from typing import Any
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from arctic_platform.common.packing.layout import PREFIX_CONTINUATION_ATTR
from arctic_platform.common.packing.layout import PackedLayout
from arctic_platform.common.packing.layout import get_sharing_layout
from arctic_platform.common.packing.layout import get_stashed_layout
from arctic_platform.common.packing.layout import require_stash_inside_context

from ..packing import cu_seqlens_from_position_ids
from .collectives import sequence_head_all_to_all

logger = logging.getLogger(__name__)

_CP_CONTEXT_ATTRIBUTE = "_dss_gated_delta_net_cp_context"


def _group_size(process_group) -> int:
    """SP size of ``process_group``; ``None`` is SP=1, which needs no initialized process group."""
    return 1 if process_group is None else dist.get_world_size(process_group)


def kernel_groups_value_heads(rule: Callable[..., Any] | None) -> bool:
    """Whether ``rule`` applies grouped value attention when there are more value heads than key heads.

    FLA's chunk rule does: value head ``hv`` reads key head ``hv // (HV // H)``, the assignment
    ``repeat_interleave`` would build, and it sums dq/dk in fp32. The pure-PyTorch fallback assumes one head
    count for q, k, v, beta and g, so q and k still have to be expanded for it.
    """
    fn = getattr(rule, "__func__", rule)
    fn = getattr(fn, "__wrapped__", fn)
    return str(getattr(fn, "__module__", "")).startswith("fla.")


def repeat_query_key_for_value_heads(
    query: torch.Tensor,
    key: torch.Tensor,
    num_key_heads: int,
    num_value_heads: int,
    rule: Callable[..., Any] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand q and k to the value-head count unless ``rule`` groups value heads itself."""
    replication = num_value_heads // num_key_heads
    if replication > 1 and not kernel_groups_value_heads(rule):
        query = query.repeat_interleave(replication, dim=2)
        key = key.repeat_interleave(replication, dim=2)
    return query, key


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
        widths: dict[str, int] = {}
        views = []
        for name, tensor in fields:
            # Each destination's heads are contiguous, so ``[B, T, world, width]`` is a view of the
            # field; one cat packs every field, and its backward is views rather than one
            # zero-filled full-size gradient per destination.
            widths[name] = tensor[0, 0].numel() // world_size
            views.append(tensor.reshape(batch_size, local_sequence_length, world_size, widths[name]))
        packed = torch.cat(views, dim=-1)
        packed = sequence_head_all_to_all(
            process_group,
            packed,
            scatter_dim=2,
            gather_dim=1,
        ).squeeze(2)

        global_sequence_length = local_sequence_length * world_size
        # split, not slicing: its backward is one cat instead of a zero-filled packed-size gradient per field.
        for (name, _tensor), field in zip(fields, torch.split(packed, [widths[name] for name, _ in fields], dim=-1)):
            local_heads, tail_shape = specs[name]
            exchanged[name] = field.reshape(
                batch_size,
                global_sequence_length,
                local_heads,
                *tail_shape,
            )
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
    world_size = _group_size(process_group)
    if world_size == 1:
        return value
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

    world_size = _group_size(process_group)
    if world_size == 1:
        return value
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


@dataclass(frozen=True)
class GatedDeltaContinuation:
    """Index plan that runs one sharing :class:`PackedLayout` call through GatedDeltaNet.

    Attributes:
        carry_index: Long ``[T + (k-1) * C]``. Packed tokens fed to the convolution: every root, then each
            continuation preceded by its parent root's last ``k-1`` tokens, which seed its left context.
        carry_cu_seqlens: Int32 ``[S + 1]`` segment boundaries of ``carry_index``.
        keep_index: Long ``[T]``. Convolution outputs that belong to packed tokens, in packed order; the seed
            outputs are dropped.
        root_cu_seqlens: Int32 ``[R + 1]`` boundaries of the roots, tokens ``[0, root_tokens)``.
        continuation_cu_seqlens: Int32 ``[C + 1]`` boundaries of the continuations, counted from ``root_tokens``.
        parent: Long ``[C]``. Root whose final recurrent state each continuation starts from.
        root_tokens: Packed tokens in roots.
    """

    carry_index: torch.Tensor
    carry_cu_seqlens: torch.Tensor
    keep_index: torch.Tensor
    root_cu_seqlens: torch.Tensor
    continuation_cu_seqlens: torch.Tensor
    parent: torch.Tensor
    root_tokens: int


_CONTINUATION_PLANS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _build_continuation(layout: PackedLayout, kernel_size: int) -> GatedDeltaContinuation:
    cu = layout.cu_seqlens.tolist()
    parents = layout.parent.tolist()
    num_roots = layout.num_roots
    if any(parent != -1 for parent in parents[:num_roots]) or any(
        not 0 <= parent < num_roots for parent in parents[num_roots:]
    ):
        raise ValueError("PackedLayout segments must be roots first, then continuations whose parent is a root")
    seed = kernel_size - 1
    carry = [torch.arange(0, cu[num_roots])]
    keep = [torch.arange(0, cu[num_roots])]
    carry_lengths = [cu[s + 1] - cu[s] for s in range(num_roots)]
    offset = cu[num_roots]
    for segment in range(num_roots, layout.num_segments):
        parent = parents[segment]
        root_length = cu[parent + 1] - cu[parent]
        if root_length < seed:
            raise ValueError(
                f"GatedDeltaNet conv carry needs every shared root to hold at least kernel_size - 1 = {seed} "
                f"tokens, but root {parent} has {root_length}"
            )
        length = cu[segment + 1] - cu[segment]
        carry.append(torch.arange(cu[parent + 1] - seed, cu[parent + 1]))
        carry.append(torch.arange(cu[segment], cu[segment + 1]))
        keep.append(torch.arange(offset + seed, offset + seed + length))
        carry_lengths.append(seed + length)
        offset += seed + length
    root_tokens = cu[num_roots]
    return GatedDeltaContinuation(
        carry_index=torch.cat(carry),
        carry_cu_seqlens=F.pad(torch.tensor(carry_lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0)),
        keep_index=torch.cat(keep),
        root_cu_seqlens=layout.cu_seqlens[: num_roots + 1].to(torch.int32),
        continuation_cu_seqlens=(layout.cu_seqlens[num_roots:] - root_tokens).to(torch.int32),
        parent=layout.parent[num_roots:].to(torch.long),
        root_tokens=root_tokens,
    )


def gated_delta_continuation(
    layout: PackedLayout, kernel_size: int, device: torch.device | str
) -> GatedDeltaContinuation:
    """The cached :class:`GatedDeltaContinuation` of a sharing ``layout`` for a ``kernel_size`` convolution."""
    if not layout.has_sharing:
        raise ValueError("gated_delta_continuation needs a sharing PackedLayout")
    plans = _CONTINUATION_PLANS.setdefault(layout, {})
    key = (int(kernel_size), str(torch.device(device)))
    plan = plans.get(key)
    if plan is None:
        cpu = plans.get((int(kernel_size), "cpu")) or _build_continuation(layout, int(kernel_size))
        plans[(int(kernel_size), "cpu")] = cpu
        plan = GatedDeltaContinuation(
            carry_index=cpu.carry_index.to(device),
            carry_cu_seqlens=cpu.carry_cu_seqlens.to(device),
            keep_index=cpu.keep_index.to(device),
            root_cu_seqlens=cpu.root_cu_seqlens.to(device),
            continuation_cu_seqlens=cpu.continuation_cu_seqlens.to(device),
            parent=cpu.parent.to(device),
            root_tokens=cpu.root_tokens,
        )
        plans[key] = plan
    return plan


def packed_causal_convolution(
    causal_convolution: Callable[..., torch.Tensor] | None,
    x: torch.Tensor,
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    cu_seqlens: torch.Tensor,
    continuation: GatedDeltaContinuation | None,
) -> torch.Tensor:
    """Depthwise causal convolution of a packed ``[1, C, T]`` call, restarting at every ``cu_seqlens`` boundary.

    With ``continuation`` each continuation segment instead starts from its parent root's last ``k-1`` inputs,
    as the unshared row would. The gather's backward scatter-adds the continuation's gradient into those root
    inputs.
    """
    if continuation is not None:
        # causal-conv1d takes seq_idx only with a channel-last x; gathering along tokens of [1, T, C] keeps it so.
        x = x.transpose(1, 2).index_select(1, continuation.carry_index).transpose(1, 2)
        cu_seqlens = continuation.carry_cu_seqlens
    if causal_convolution is not None:
        output = causal_convolution(
            x=x,
            weight=weight,
            bias=bias,
            activation=activation,
            seq_idx=_packed_sequence_indices(cu_seqlens),
        )
    else:
        output = _fallback_depthwise_causal_convolution(
            x,
            weight=weight,
            bias=bias,
            activation=activation,
            global_cu_seqlens=cu_seqlens,
        )
    if continuation is not None:
        output = output.transpose(1, 2).index_select(1, continuation.keep_index).transpose(1, 2)
    return output


def packed_gated_delta_rule(
    gated_delta_rule: Callable[..., Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    continuation: GatedDeltaContinuation | None,
    **rule_kwargs,
) -> torch.Tensor:
    """Gated delta rule over a packed ``[1, T, H, D]`` call; returns the ``[1, T, HV, Dv]`` output.

    With ``continuation`` the roots run first and return their final recurrent states; the continuations then
    run from their parent's state. A root with several continuations receives the sum of their state
    gradients. The state is passed on in the dtype the rule returns it in (fp32 for FLA), never cast.
    """
    for name in ("initial_state", "output_final_state", "cu_seqlens_cpu", "cp_context"):
        value = rule_kwargs.pop(name, None)
        if value is not None and value is not False:
            raise RuntimeError(f"packed GatedDeltaNet sets {name} itself; callers may not pass it")
    if continuation is None:
        output, final_state = gated_delta_rule(
            q, k, v, g, beta, initial_state=None, output_final_state=False, cu_seqlens=cu_seqlens, **rule_kwargs
        )
        if final_state is not None:
            raise RuntimeError("packed GatedDeltaNet unexpectedly returned recurrent state")
        return output

    split = continuation.root_tokens

    def roots(tensor):
        return tensor[:, :split]

    def continuations(tensor):
        return tensor[:, split:]

    root_output, state = gated_delta_rule(
        roots(q),
        roots(k),
        roots(v),
        roots(g),
        roots(beta),
        initial_state=None,
        output_final_state=True,
        cu_seqlens=continuation.root_cu_seqlens,
        **rule_kwargs,
    )
    num_roots = int(continuation.root_cu_seqlens.numel() - 1)
    expected = (num_roots, int(v.shape[2]), int(k.shape[3]), int(v.shape[3]))
    if not torch.is_tensor(state) or tuple(state.shape) != expected or state.device != v.device:
        raise RuntimeError(
            f"packed GatedDeltaNet expected root final states shaped {expected} on {v.device}, got "
            f"{None if state is None else (tuple(state.shape), state.device)}"
        )
    if state.dtype != torch.promote_types(v.dtype, torch.float32):
        raise RuntimeError(
            f"packed GatedDeltaNet expected {torch.promote_types(v.dtype, torch.float32)} state, got {state.dtype}"
        )
    continuation_output, final_state = gated_delta_rule(
        continuations(q),
        continuations(k),
        continuations(v),
        continuations(g),
        continuations(beta),
        initial_state=state.index_select(0, continuation.parent),
        output_final_state=False,
        cu_seqlens=continuation.continuation_cu_seqlens,
        **rule_kwargs,
    )
    if final_state is not None:
        raise RuntimeError("packed GatedDeltaNet unexpectedly returned recurrent state")
    return torch.cat([root_output, continuation_output], dim=1)


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
    global_cu_seqlens: torch.Tensor | None,
    num_key_heads: int,
    num_value_heads: int,
    fp32_beta: bool,
    layout: PackedLayout | None = None,
    repeat_query_key: bool = False,
    **rule_kwargs,
):
    """Run convolution and recurrence on a full-sequence GDN head shard.

    ``fp32_beta`` evaluates the beta gate as a sigmoid of fp32 ``b``; ``False`` evaluates it in ``b``'s own dtype. It
    has no default, so every caller chooses.

    The first all-to-all exchanges native projected Q/K/V and gate preactivations
    from local-sequence/all-heads to full-sequence/local-heads. The depthwise
    convolution and recurrent rule then share that representation. A final
    all-to-all restores local-sequence/all-value-heads for the norm and output
    projection. With ``process_group=None`` (or a one-rank group) there is no exchange.

    With ``layout`` (the caller's stashed :class:`PackedLayout`, global across the SP group) the boundaries come
    from the layout rather than ``global_cu_seqlens``, and under sharing each continuation resumes its parent
    root's convolution inputs and recurrent state. ``repeat_query_key`` expands Q/K to the value-head count even
    for a rule that groups value heads itself, as the stock transformers forward does.
    """
    world_size = _group_size(process_group)
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
    continuation = None
    if layout is not None:
        if global_sequence_length != layout.num_tokens:
            raise RuntimeError(
                f"GatedDeltaNet's PackedLayout has {layout.num_tokens} packed tokens, but this call covers "
                f"{global_sequence_length} ({local_sequence_length} per rank x SP {world_size}); padded calls "
                "are not supported inside using_layout"
            )
        global_cu_seqlens = layout.cu_seqlens.to(device=v.device, dtype=torch.int32)
        if layout.has_sharing:
            continuation = gated_delta_continuation(layout, int(convolution_weight.shape[-1]), v.device)
    _validate_global_cu_seqlens(
        global_cu_seqlens,
        global_sequence_length=global_sequence_length,
    )
    if world_size == 1:
        exchanged = {"q": q, "k": k, "v": v, "b": b, "a": a}
    else:
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
    mixed_qkv = packed_causal_convolution(
        causal_convolution,
        mixed_qkv,
        weight=local_convolution_weight,
        bias=local_convolution_bias,
        activation=convolution_activation,
        cu_seqlens=global_cu_seqlens,
        continuation=continuation,
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
    # vLLM evaluates the beta gate as a sigmoid of fp32 ``b`` and keeps it in fp32; a sigmoid of low-precision
    # ``b`` would round the gate before the recurrence reads it. This function is shared by the custom Qwen3.5-MoE
    # model and by the generic transformers wrapper (``_adapt_gated_delta_net_module``); each passes its loader's
    # ``fp32_parity`` setting.
    beta = exchanged["b"].float().sigmoid() if fp32_beta else exchanged["b"].sigmoid()
    g = -local_A_log.float().exp() * F.softplus(exchanged["a"].float() + local_dt_bias)

    if repeat_query_key and local_value_heads > local_key_heads:
        replication = local_value_heads // local_key_heads
        query = query.repeat_interleave(replication, dim=2)
        key = key.repeat_interleave(replication, dim=2)
    else:
        query, key = repeat_query_key_for_value_heads(query, key, local_key_heads, local_value_heads, gated_delta_rule)

    if rule_kwargs.get("initial_state") is not None:
        raise RuntimeError("head-parallel GatedDeltaNet training does not support an initial recurrent state")
    if rule_kwargs.get("output_final_state", False):
        raise RuntimeError("head-parallel GatedDeltaNet training does not support returning recurrent state")
    output = packed_gated_delta_rule(
        gated_delta_rule,
        query,
        key,
        value,
        g,
        beta,
        cu_seqlens=global_cu_seqlens,
        continuation=continuation,
        **rule_kwargs,
    )
    if world_size == 1:
        return output, None
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


_KERNEL_DISPATCH = {
    "causal_conv1d_fn": (
        "causal_conv1d_fn",
        "causal_conv1d",
        "causal_conv1d_fn",
    ),
    "chunk_gated_delta_rule": (
        "torch_chunk_gated_delta_rule",
        "fla.ops.gated_delta_rule",
        "chunk_gated_delta_rule",
    ),
}


def _resolve_gated_delta_net_kernel(module: nn.Module, attribute: str):
    kernel = getattr(module, attribute, None) or getattr(module, f"_{attribute}", None)
    if kernel is not None:
        return kernel

    dispatch_name, package_name, package_attribute = _KERNEL_DISPATCH[attribute]
    implementation = importlib.import_module(type(module).__module__)
    dispatch = getattr(implementation, dispatch_name, None)
    if dispatch is None or not hasattr(dispatch, "__wrapped__"):
        return dispatch
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        return dispatch
    return getattr(package, package_attribute, dispatch)


def _has_gated_delta_net_kernel_slot(module: nn.Module, attribute: str) -> bool:
    if hasattr(module, attribute) or hasattr(module, f"_{attribute}"):
        return True
    dispatch_name = _KERNEL_DISPATCH[attribute][0]
    implementation = importlib.import_module(type(module).__module__)
    return hasattr(implementation, dispatch_name)


def _is_gated_delta_net_module(module: nn.Module) -> bool:
    return (
        hasattr(module, "conv_kernel_size")
        and _has_gated_delta_net_kernel_slot(module, "causal_conv1d_fn")
        and _has_gated_delta_net_kernel_slot(module, "chunk_gated_delta_rule")
    )


def validate_packed_gated_delta_net_kernels(model: nn.Module) -> None:
    """Reject HF hybrid fallbacks that silently ignore packed boundaries."""
    found = False
    for name, module in model.named_modules():
        if not _is_gated_delta_net_module(module):
            continue
        found = True
        for attribute, boundary in (
            ("causal_conv1d_fn", "seq_idx"),
            ("chunk_gated_delta_rule", "cu_seqlens"),
        ):
            kernel = _resolve_gated_delta_net_kernel(module, attribute)
            try:
                parameters = inspect.signature(kernel).parameters if callable(kernel) else {}
            except (TypeError, ValueError):
                parameters = {}
            # The HF torch fallback accepts **kwargs but ignores cu_seqlens.
            # Require the explicit boundary argument of the supported kernels.
            if boundary not in parameters:
                raise RuntimeError(
                    f"Packed hybrid attention requires boundary-aware {attribute} "
                    f"with explicit {boundary} at {name or '<root>'}. Install the "
                    "supported causal-conv1d and flash-linear-attention kernels; "
                    "the Hugging Face torch fallbacks do not isolate packed rows."
                )
    if not found:
        raise RuntimeError(
            "model configuration contains linear_attention layers, but no "
            "compatible packed GatedDeltaNet modules were found"
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


GATED_DELTA_CHUNK_SIZE = 64
# Exact versions whose initial_state / final-state handoff has been checked on GPU against an unshared run.
_PREFIX_CONTINUATION_VERSIONS = {
    "fla-core": "0.6.0",
    "flash-linear-attention": "0.6.0",
    "causal-conv1d": "1.7.0",
}
_SPLIT_PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
_FUSED_PROJECTIONS = ("in_proj_qkvz", "in_proj_ba", "fix_query_key_value_ordering")


def _installed_version(distribution: str) -> str | None:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _kernel_parameters(kernel) -> set[str]:
    try:
        return set(inspect.signature(kernel).parameters) if callable(kernel) else set()
    except (TypeError, ValueError):
        return set()


def prefix_continuation_kernel_blocker(causal_convolution, gated_delta_rule) -> str | None:
    """Why these resolved kernels cannot continue a shared prefix, or ``None`` if they can."""
    if not kernel_groups_value_heads(gated_delta_rule):
        return "prefix sharing needs flash-linear-attention's chunk_gated_delta_rule, not a fallback rule"
    missing = {"cu_seqlens", "initial_state", "output_final_state"} - _kernel_parameters(gated_delta_rule)
    if missing:
        return f"the gated delta rule takes no {sorted(missing)}"
    if "seq_idx" not in _kernel_parameters(causal_convolution):
        return (
            "prefix sharing needs causal-conv1d's causal_conv1d_fn with seq_idx; the torch fallback ignores boundaries"
        )
    checked = ["fla-core", "flash-linear-attention"]
    if str(getattr(causal_convolution, "__module__", "")).startswith("causal_conv1d"):
        checked.append("causal-conv1d")
    for distribution in checked:
        installed = _installed_version(distribution)
        wanted = _PREFIX_CONTINUATION_VERSIONS[distribution]
        if installed is not None and installed != wanted:
            return f"prefix sharing is verified on {distribution}=={wanted}, found {installed}"
    if _installed_version("fla-core") is None and _installed_version("flash-linear-attention") is None:
        return "prefix sharing needs flash-linear-attention installed"
    return None


def gated_delta_net_prefix_continuation_blocker(
    layout: PackedLayout | None, kernel_size: int, causal_convolution, gated_delta_rule
) -> str | None:
    """Why a GatedDeltaNet with these kernels cannot run ``layout`` (any layout if ``None``), or ``None``."""
    reason = prefix_continuation_kernel_blocker(causal_convolution, gated_delta_rule)
    if reason is not None or layout is None or not layout.has_sharing:
        return reason
    cu = layout.cu_seqlens.tolist()
    for root in sorted(set(layout.parent[layout.num_roots :].tolist())):
        length = cu[root + 1] - cu[root]
        if length % GATED_DELTA_CHUNK_SIZE:
            return (
                f"shared root {root} has {length} tokens; GatedDeltaNet roots must be a multiple of its "
                f"{GATED_DELTA_CHUNK_SIZE}-token chunk (shared_prefix_length(first_weighted, "
                f"{GATED_DELTA_CHUNK_SIZE}))"
            )
        if length < kernel_size - 1:
            return f"shared root {root} has {length} tokens, fewer than the conv carry's {kernel_size - 1}"
    return None


def gated_delta_net_prefix_continuation_check(module: nn.Module, causal_convolution, gated_delta_rule):
    """``check(layout)`` for :data:`~arctic_platform.common.packing.layout.PREFIX_CONTINUATION_ATTR`."""
    kernel_size = int(module.conv_kernel_size)

    def check(layout: PackedLayout | None) -> str | None:
        return gated_delta_net_prefix_continuation_blocker(layout, kernel_size, causal_convolution, gated_delta_rule)

    return check


def _project_gated_delta_net_inputs(module: nn.Module, hidden_states: torch.Tensor, num_key_heads, num_value_heads):
    """Native-head ``q, k [B, T, H, K]``, ``v, z [B, T, HV, V]``, ``b, a [B, T, HV]`` of split or fused projections.

    Qwen3-Next's fused ``in_proj_qkvz`` / ``in_proj_ba`` interleave heads by key-head group;
    ``fix_query_key_value_ordering`` returns them in the split layout, value head ``hv`` under key head
    ``hv // (HV // H)``, so every rank's contiguous head slice lines up across q, k, v, z, b and a.
    """
    batch_size, sequence_length, _ = hidden_states.shape
    key_head_dim = int(module.head_k_dim)
    value_head_dim = int(module.head_v_dim)
    if hasattr(module, "in_proj_qkvz"):
        query, key, value, z, b, a = module.fix_query_key_value_ordering(
            module.in_proj_qkvz(hidden_states), module.in_proj_ba(hidden_states)
        )
    else:
        key_dim = num_key_heads * key_head_dim
        value_dim = num_value_heads * value_head_dim
        query, key, value = torch.split(module.in_proj_qkv(hidden_states), [key_dim, key_dim, value_dim], dim=-1)
        z = module.in_proj_z(hidden_states)
        b = module.in_proj_b(hidden_states)
        a = module.in_proj_a(hidden_states)
    shape = (batch_size, sequence_length)
    return (
        query.reshape(*shape, num_key_heads, key_head_dim),
        key.reshape(*shape, num_key_heads, key_head_dim),
        value.reshape(*shape, num_value_heads, value_head_dim),
        z.reshape(*shape, num_value_heads, value_head_dim),
        b.reshape(*shape, num_value_heads),
        a.reshape(*shape, num_value_heads),
    )


def _adapt_gated_delta_net_module(
    module: nn.Module, process_group, *, fp32_beta: bool, stock_numerics: bool = False
) -> None:
    """Replace ``module.forward`` with the head-parallel forward.

    ``fp32_beta`` is passed to :func:`head_parallel_gated_delta_net`: ``True`` evaluates the beta gate as a sigmoid of
    fp32 ``b``, ``False`` as ``b.sigmoid()`` in ``b``'s own dtype, which is what the stock transformers forward does.

    ``process_group=None`` is SP=1. There a call with a sharing :class:`PackedLayout` stashed, or an unshared one
    that is exactly this call (``[1, num_tokens]``, no cache), runs the packed path on the layout's segments with
    ``stock_numerics``, which repeats Q/K to the value heads as the stock forward does. Any other call runs the stock
    forward unchanged.
    """
    fused = all(hasattr(module, attribute) for attribute in _FUSED_PROJECTIONS)
    required_attributes = (
        *(_FUSED_PROJECTIONS if fused else _SPLIT_PROJECTIONS),
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

    original_causal_convolution = _resolve_gated_delta_net_kernel(module, "causal_conv1d_fn")
    original_gated_delta_rule = _resolve_gated_delta_net_kernel(module, "chunk_gated_delta_rule")
    num_key_heads, num_value_heads = _module_head_counts(module)
    world_size = _group_size(process_group)
    if num_key_heads % world_size or num_value_heads % world_size:
        raise RuntimeError(
            f"SP size {world_size} must divide GatedDeltaNet key/value heads ({num_key_heads}/{num_value_heads})"
        )
    if num_value_heads % num_key_heads:
        raise RuntimeError(
            f"GatedDeltaNet value heads ({num_value_heads}) must be divisible by key heads ({num_key_heads})"
        )
    stock_forward = module.forward

    def forward(self, *args, **kwargs):
        hidden_states = _hidden_states_from_call(args, kwargs)
        cache_params = kwargs.get("cache_params")
        if cache_params is None and len(args) > 1:
            cache_params = args[1]
        layout = get_sharing_layout(self)
        if layout is None:
            stash = get_stashed_layout(self)
            require_stash_inside_context(type(self).__name__, stash)
            if process_group is None:
                # The stock forward resets state only at cu_seq_lens_q the caller passes; an unshared stash that is
                # exactly this call gives its own segments instead.
                if (
                    stash is None
                    or cache_params is not None
                    or tuple(hidden_states.shape[:2]) != (1, stash.num_tokens)
                ):
                    return stock_forward(*args, **kwargs)
                layout = stash
        if cache_params is not None:
            raise RuntimeError(
                "the packed GatedDeltaNet path (sequence parallelism or prefix sharing) is a training path and "
                "does not support cached recurrent or convolution state"
            )
        # Under a sharing layout the boundaries are the layout's; seq_idx / cu_seq_lens_* the caller passed describe
        # the unshared rows and are not read.
        global_cu_seqlens = None if layout is not None else getattr(self, _GLOBAL_CU_SEQLENS_ATTRIBUTE, None)
        if layout is None and not torch.is_tensor(global_cu_seqlens):
            raise RuntimeError(
                "GatedDeltaNet sequence parallelism needs the packed group's global cu_seqlens on the module "
                f"as {_GLOBAL_CU_SEQLENS_ATTRIBUTE!r}, and this call has none. The model's forward is supposed "
                "to stash them before the layer loop; reaching a layer without them means that wrapper did not "
                "run."
            )

        batch_size, local_sequence_length, _ = hidden_states.shape
        query, key, value, z, b, a = _project_gated_delta_net_inputs(
            self, hidden_states, num_key_heads, num_value_heads
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
            fp32_beta=fp32_beta,
            layout=layout,
            repeat_query_key=stock_numerics,
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
    setattr(
        module,
        PREFIX_CONTINUATION_ATTR,
        gated_delta_net_prefix_continuation_check(module, original_causal_convolution, original_gated_delta_rule),
    )


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
        stash = get_stashed_layout(model)
        if stash is not None and stash.has_sharing:
            # The layers read the stashed global layout; this rank's positions cover only its window.
            return original_forward(*args, **kwargs)
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
        global_cu_seqlens = cu_seqlens_from_position_ids(torch.cat(gathered, dim=1), stash)
        for module in modules:
            setattr(module, _GLOBAL_CU_SEQLENS_ATTRIBUTE, global_cu_seqlens)
        return original_forward(*args, **kwargs)

    model.forward = forward


def apply_gated_delta_net_sequence_parallelism(model: nn.Module, process_group, *, fp32_beta: bool) -> int:
    """Adapt every HF-style GatedDeltaNet module in ``model`` and return the count.

    ``fp32_beta`` selects the beta gate's precision in every adapted module (see ``_adapt_gated_delta_net_module``).
    """
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
        _adapt_gated_delta_net_module(module, process_group, fp32_beta=fp32_beta)
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


def apply_gated_delta_net_prefix_continuation(model: nn.Module) -> int:
    """Install the GatedDeltaNet adapter at SP=1 and return the count of adapted modules.

    Calls without a sharing :class:`PackedLayout` stashed still run each module's stock forward. Calls under a
    sharing layout run the packed path in stock numerics (Q/K repeated to the value heads, beta as
    ``b.sigmoid()`` in ``b``'s dtype, so an fp32 ``in_proj_b`` output hook still applies), with boundaries from
    the layout and each continuation resuming its root.
    """
    adapted = 0
    for name, module in model.named_modules():
        if not _is_gated_delta_net_module(module) or getattr(module, "_dss_gated_delta_net_sequence_parallel", False):
            continue
        try:
            _adapt_gated_delta_net_module(module, None, fp32_beta=False, stock_numerics=True)
        except RuntimeError as exc:
            # The stock forward still runs; prefix sharing refuses an unadapted GatedDeltaNet.
            logger.info("GatedDeltaNet %s keeps its stock forward without prefix continuation: %s", name, exc)
            continue
        adapted += 1
    return adapted
