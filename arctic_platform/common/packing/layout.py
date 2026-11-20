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
# See the License for the specific language governing rights and
# limitations under the License.

"""Prefix-continuation packed layout: shared roots first, then per-row continuations.

Pack/unpack gathers from one source row per packed token so a shared position is not
last-write-wins and is not gradient-counted once per row. ``layout.cu_seqlens`` are
attention segments, not loss rows. Until the loss-alignment PR, sharing-on is
token-mean only. MoE shared tokens are counted once; groups whose router-replay ids
disagree are not shared.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .walk import DROP
from .walk import map_batch_values

LAYOUT_ATTR = "_arctic_packed_layout"
# Shared prefix tokens contribute once to tokens_per_expert / expert_bias / aux.
MOE_SHARED_TOKEN_COUNT = "once"


def shared_prefix_length(first_weighted: int, chunk: int | None = None) -> int:
    """Share unweighted predict-next positions, optionally rounded down to a GDN chunk."""
    if first_weighted < 0:
        raise ValueError(f"first_weighted must be >= 0, got {first_weighted}")
    if chunk is None:
        return first_weighted
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    return (first_weighted // chunk) * chunk


def require_token_mean_if_sharing(sharing: bool, loss_agg_mode: str) -> None:
    if sharing and loss_agg_mode != "token-mean":
        raise ValueError(
            "prefix sharing only supports loss_agg_mode='token-mean' until the loss-alignment PR; "
            f"got {loss_agg_mode!r}"
        )


def active_layout_requires_token_mean(loss_agg_mode: str, prefix_sharing: bool = False) -> None:
    """Enforce token-mean from the stashed layout so callers cannot forget ``prefix_sharing=True``."""
    layout = get_active_packed_layout()
    sharing = prefix_sharing or (layout is not None and layout.has_sharing)
    require_token_mean_if_sharing(sharing, loss_agg_mode)


def _valid_prefix(values: Tensor, attention_mask: Tensor, row: int, n: int) -> Tensor:
    cols = attention_mask[row].bool().nonzero(as_tuple=False).flatten()
    if n > int(cols.numel()):
        raise ValueError(f"shared length {n} exceeds row {row}'s {int(cols.numel())} valid tokens")
    return values[row].index_select(0, cols[:n].to(device=values.device))


def require_identical_prefixes(
    input_ids: Tensor,
    groups: Sequence[Sequence[int]],
    shared: Sequence[int],
    attention_mask: Tensor,
) -> None:
    """``shared[i]`` is a count of valid tokens, not raw columns."""
    for rows, n in zip(groups, shared):
        if n <= 0 or len(rows) < 2:
            continue
        first = _valid_prefix(input_ids, attention_mask, rows[0], n)
        for row in rows[1:]:
            if not torch.equal(_valid_prefix(input_ids, attention_mask, row, n), first):
                raise ValueError(f"prompt group {list(rows)} is not token-identical on the first {n} valid tokens")


def skip_share_if_router_disagree(
    groups: Sequence[Sequence[int]],
    shared: Sequence[int],
    routed_experts: Tensor,
    attention_mask: Tensor,
) -> list[int]:
    """Zero the shared length of any group whose router-replay ids disagree on the valid prefix."""
    out = list(shared)
    for i, (rows, n) in enumerate(zip(groups, shared)):
        if n <= 0 or len(rows) < 2:
            continue
        first = _valid_prefix(routed_experts, attention_mask, rows[0], n)
        if any(not torch.equal(_valid_prefix(routed_experts, attention_mask, row, n), first) for row in rows[1:]):
            out[i] = 0
    return out


@dataclass(frozen=True)
class PackedLayout:
    """One packed model call. Segments ``[0, num_roots)`` are roots; later segments continue a parent.

    ``src_row`` / ``src_col`` gather from one client cell per packed token. ``row_index`` maps each
    client ``[B, S]`` cell onto a packed position (shared prefix cells point at the root). Pre-pack
    width ``S`` is ``packed_loss_scale_factor``.
    """

    cu_seqlens: Tensor
    parent: Tensor
    num_roots: int
    position_ids: Tensor
    row_index: Tensor
    src_row: Tensor
    src_col: Tensor
    logical_row: Tensor
    packed_loss_scale_factor: int
    conv_kernel_size: int | None = None

    @classmethod
    def build(
        cls,
        attention_mask: Tensor,
        groups: Sequence[Sequence[int]],
        shared: Sequence[int],
        position_ids: Tensor | None = None,
        conv_kernel_size: int | None = None,
        input_ids: Tensor | None = None,
        routed_experts: Tensor | None = None,
    ) -> "PackedLayout":
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must be [B, S], got {tuple(attention_mask.shape)}")
        if len(groups) != len(shared):
            raise ValueError("groups and shared must have the same length")
        batch_size, sequence_length = attention_mask.shape
        covered = [row for rows in groups for row in rows]
        if sorted(covered) != list(range(batch_size)):
            raise ValueError("groups must partition the batch rows")
        if routed_experts is not None:
            shared = skip_share_if_router_disagree(groups, shared, routed_experts, attention_mask)
        sharing = any(n > 0 and len(list(rows)) > 1 for rows, n in zip(groups, shared))
        if sharing:
            if input_ids is None:
                raise ValueError("input_ids are required when sharing a prefix")
            require_identical_prefixes(input_ids, groups, shared, attention_mask)
        bool_mask = attention_mask.bool()
        cols = [bool_mask[row].nonzero(as_tuple=False).flatten() for row in range(batch_size)]
        if position_ids is None:
            position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
            for row, col in enumerate(cols):
                position_ids[row, col] = torch.arange(col.numel(), device=attention_mask.device)

        roots: list[tuple[int, int, int, int]] = []  # src_row, start, end, parent
        conts: list[tuple[int, int, int, int]] = []
        for rows, n in zip(groups, shared):
            rows = list(rows)
            if n < 0:
                raise ValueError(f"shared length must be >= 0, got {n}")
            if n > 0 and any(n > cols[row].numel() for row in rows):
                raise ValueError("shared length exceeds a group's valid tokens")
            if n > 0 and len(rows) > 1 and any(n >= cols[row].numel() for row in rows):
                raise ValueError("shared length must leave at least one continuation token")
            if n == 0 or len(rows) == 1:
                for row in rows:
                    roots.append((row, 0, int(cols[row].numel()), -1))
                continue
            parent = len(roots)
            roots.append((rows[0], 0, n, -1))
            for row in rows:
                conts.append((row, n, int(cols[row].numel()), parent))
        segments = roots + conts
        lengths = torch.tensor([end - start for _, start, end, _ in segments], dtype=torch.int32)
        cu = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        num_tokens = int(cu[-1])
        row_index = torch.full((batch_size, sequence_length), -1, dtype=torch.long)
        src_row = torch.empty(num_tokens, dtype=torch.long)
        src_col = torch.empty(num_tokens, dtype=torch.long)
        logical_row = torch.empty(num_tokens, dtype=torch.long)
        packed_pos = torch.empty(num_tokens, dtype=torch.long)
        for s, (row, start, end, parent) in enumerate(segments):
            packed_range = torch.arange(int(cu[s]), int(cu[s + 1]))
            token_cols = cols[row][start:end]
            src_row[packed_range] = row
            src_col[packed_range] = token_cols
            logical_row[packed_range] = row
            packed_pos[packed_range] = packed_range
            row_index[row, token_cols] = packed_range
            if parent >= 0:
                parent_range = torch.arange(int(cu[parent]), int(cu[parent + 1]))
                row_index[row, cols[row][:start]] = parent_range
        packed_positions = position_ids[src_row, src_col]
        return cls(
            cu_seqlens=cu,
            parent=torch.tensor([p for *_, p in segments], dtype=torch.long),
            num_roots=len(roots),
            position_ids=packed_positions,
            row_index=row_index,
            src_row=src_row,
            src_col=src_col,
            logical_row=logical_row,
            packed_loss_scale_factor=sequence_length,
            conv_kernel_size=conv_kernel_size,
        )

    @property
    def num_segments(self) -> int:
        return int(self.cu_seqlens.numel() - 1)

    @property
    def has_sharing(self) -> bool:
        return self.num_roots < self.num_segments

    @property
    def num_tokens(self) -> int:
        return int(self.cu_seqlens[-1])

    @cached_property
    def root_tokens(self) -> int:
        return int(self.cu_seqlens[self.num_roots])

    @cached_property
    def kv_index(self) -> Tensor:
        cu = self.cu_seqlens.tolist()
        spans = []
        for s, parent in enumerate(self.parent.tolist()):
            if parent >= 0:
                spans.append(torch.arange(cu[parent], cu[parent + 1], device=self.cu_seqlens.device))
            spans.append(torch.arange(cu[s], cu[s + 1], device=self.cu_seqlens.device))
        if not spans:
            return self.cu_seqlens.new_zeros((0,), dtype=torch.long)
        return torch.cat(spans)

    @cached_property
    def cu_seqlens_k(self) -> Tensor:
        lengths = self.cu_seqlens.diff()
        extra = torch.zeros_like(lengths)
        has_parent = self.parent >= 0
        extra[has_parent] = lengths[self.parent[has_parent]]
        return F.pad((lengths + extra).cumsum(0, dtype=torch.int32), (1, 0))

    @cached_property
    def max_len_q(self) -> int:
        return int(self.cu_seqlens.diff().max()) if self.num_tokens else 0

    @cached_property
    def max_len_k(self) -> int:
        return int(self.cu_seqlens_k.diff().max()) if self.num_tokens else 0

    def to(self, device: torch.device | str) -> "PackedLayout":
        return PackedLayout(
            cu_seqlens=self.cu_seqlens.to(device),
            parent=self.parent.to(device),
            num_roots=self.num_roots,
            position_ids=self.position_ids.to(device),
            row_index=self.row_index.to(device),
            src_row=self.src_row.to(device),
            src_col=self.src_col.to(device),
            logical_row=self.logical_row.to(device),
            packed_loss_scale_factor=self.packed_loss_scale_factor,
            conv_kernel_size=self.conv_kernel_size,
        )

    def pack(self, rows: Tensor) -> Tensor:
        """Gather ``[B, S, ...]`` onto packed tokens from one source cell each."""
        if rows.ndim < 2:
            raise ValueError(f"pack expects [B, S, ...], got {tuple(rows.shape)}")
        return rows[self.src_row.to(rows.device), self.src_col.to(rows.device)]

    def unpack(self, packed: Tensor, pad_value: float = 0.0) -> Tensor:
        """Scatter packed ``[T, ...]`` back to ``[B, S, ...]`` via ``row_index``."""
        if packed.ndim >= 2 and packed.shape[0] == 1:
            packed = packed.squeeze(0)
        batch_size, sequence_length = self.row_index.shape
        trailing = tuple(packed.shape[1:])
        out = packed.new_full((batch_size, sequence_length, *trailing), pad_value)
        valid = self.row_index >= 0
        out[valid] = packed[self.row_index[valid]]
        return out

    def require_unweighted_shared(self, weights: Tensor) -> None:
        if weights.ndim >= 2 and weights.shape[0] == 1:
            weights = weights.squeeze(0)
        valid = self.row_index >= 0
        index = self.row_index[valid]
        shared = torch.bincount(index.cpu(), minlength=self.num_tokens) > 1
        if weights[valid][shared.to(index.device)[index]].bool().any():
            raise ValueError("a shared prefix position carries loss weight")


def unshared_groups(batch_size: int) -> tuple[list[list[int]], list[int]]:
    return [[row] for row in range(batch_size)], [0] * batch_size


def pack_with_layout(data: Dict[str, Any], layout: PackedLayout) -> Dict[str, Any]:
    """Pack ``[B, S, ...]`` leaves with ``layout``; drop ``attention_mask`` like ``pack_microbatch``."""
    batch_size, sequence_length = layout.row_index.shape

    def packed_value(key: str, value: Any) -> Any:
        if key == "attention_mask":
            return DROP
        if torch.is_tensor(value) and value.ndim >= 2 and tuple(value.shape[:2]) == (batch_size, sequence_length):
            packed = layout.pack(value)
            return packed.unsqueeze(0)
        return value

    packed = map_batch_values(data, packed_value)
    packed["position_ids"] = layout.position_ids.unsqueeze(0)
    return packed


_ACTIVE_LAYOUT: PackedLayout | None = None


def _wrap_hf_flash_forward(orig):
    if getattr(orig, "_arctic_prefix", False) or orig is None:
        return orig

    def wrapped(query_states, key_states, value_states, *args, **kwargs):
        layout = get_active_packed_layout()
        if layout is not None and layout.has_sharing:
            attention_mask = args[0] if args else kwargs.get("attention_mask")
            if attention_mask is not None:
                raise ValueError("prefix sharing requires a packed flash call with attention_mask=None")
            k, v = key_states, value_states
            if k.ndim != 4:
                raise ValueError(f"HF prefix gather expects [B, T, H, D], got {tuple(k.shape)}")
            if k.shape[0] != 1:
                raise ValueError(f"HF prefix gather expects packed batch 1, got {k.shape[0]}")
            k, v, cu_k, max_k = apply_prefix_kv(k.squeeze(0), v.squeeze(0), None, None, layout)
            key_states, value_states = k.unsqueeze(0), v.unsqueeze(0)
            kwargs["cu_seq_lens_q"] = layout.cu_seqlens.to(device=query_states.device, dtype=torch.int32)
            kwargs["cu_seq_lens_k"] = cu_k
            kwargs["max_length_q"] = layout.max_len_q
            kwargs["max_length_k"] = max_k
        return orig(query_states, key_states, value_states, *args, **kwargs)

    wrapped._arctic_prefix = True
    return wrapped


def install_prefix_attention_hooks() -> None:
    """Make HF flash-attn honour the active PackedLayout (AP FlashAttention already gathers)."""
    try:
        import transformers.integrations.flash_attention as fa_int
        import transformers.modeling_flash_attention_utils as fa_utils
    except ImportError:
        return
    if not hasattr(fa_int, "_flash_attention_forward") or not hasattr(fa_utils, "_flash_attention_forward"):
        raise ValueError("transformers flash attention is missing _flash_attention_forward; cannot share prefixes")
    wrapped = _wrap_hf_flash_forward(fa_int._flash_attention_forward)
    fa_int._flash_attention_forward = wrapped
    fa_utils._flash_attention_forward = wrapped


def stash_layout(module: torch.nn.Module, layout: PackedLayout | None) -> None:
    global _ACTIVE_LAYOUT
    _ACTIVE_LAYOUT = layout
    if layout is not None:
        install_prefix_attention_hooks()
    for child in module.modules():
        setattr(child, LAYOUT_ATTR, layout)


@contextmanager
def using_layout(module: torch.nn.Module, layout: PackedLayout | None) -> Iterator[PackedLayout | None]:
    """Stash a layout for forward and backward; restore the previous layout on exit."""
    previous = get_active_packed_layout()
    stash_layout(module, layout)
    try:
        yield layout
    finally:
        stash_layout(module, previous)


def get_stashed_layout(module: torch.nn.Module) -> PackedLayout | None:
    return getattr(module, LAYOUT_ATTR, None)


def get_active_packed_layout() -> PackedLayout | None:
    return _ACTIVE_LAYOUT


def apply_prefix_kv(
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor | None,
    max_seqlen: int | None,
    layout: PackedLayout | None,
) -> tuple[Tensor, Tensor, Tensor | None, int | None]:
    """Gather K/V so each continuation segment attends to its parent root then itself."""
    if layout is None or not layout.has_sharing:
        return k, v, cu_seqlens, max_seqlen
    if k.ndim != 3:
        raise ValueError(f"prefix K/V gather expects [T, H, D], got {tuple(k.shape)}")
    idx = layout.kv_index.to(device=k.device)
    k = k.index_select(0, idx)
    v = v.index_select(0, idx)
    dtype = cu_seqlens.dtype if cu_seqlens is not None else torch.int32
    return k, v, layout.cu_seqlens_k.to(device=k.device, dtype=dtype), layout.max_len_k


_PREFIX_ATTENTION = {
    "FlashAttention",
    "AfmoeFlashAttention",
    "Qwen3Attention",
    "Qwen3MoeAttention",
    "Qwen2Attention",
    "Qwen2MoeAttention",
}
_NATIVE_FLASH = {"FlashAttention", "AfmoeFlashAttention"}
_SEQUENCE_MIXER_NEEDLES = ("Attention", "DeltaNet", "GatedDelta", "LinearAttention", "KDA")


def sequence_mixers(model: torch.nn.Module) -> list[torch.nn.Module]:
    found = []
    for module in model.modules():
        name = type(module).__name__
        if any(needle in name for needle in _SEQUENCE_MIXER_NEEDLES):
            found.append(module)
    return found


def supports_prefix_continuation(module: torch.nn.Module) -> bool:
    name = type(module).__name__
    if name in _NATIVE_FLASH:
        return True
    if name not in _PREFIX_ATTENTION:
        return False
    config = getattr(module, "config", None)
    impl = getattr(config, "_attn_implementation", None)
    return isinstance(impl, str) and impl.startswith("flash")


def require_prefix_continuation(model: torch.nn.Module) -> None:
    mixers = sequence_mixers(model)
    if not mixers:
        raise ValueError("zorro/prefix sharing found no sequence mixers to validate")
    unsupported = sorted({type(module).__name__ for module in mixers if not supports_prefix_continuation(module)})
    if unsupported:
        raise ValueError(
            "zorro/prefix sharing needs prefix continuation in every sequence mixer; "
            f"unsupported in this PR: {unsupported}"
        )
