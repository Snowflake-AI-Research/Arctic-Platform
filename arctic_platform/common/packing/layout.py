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

"""Prefix-continuation packed layout: shared roots first, then per-row continuations.

Pack/unpack gathers from one source row per packed token so a shared position is not
last-write-wins and is not gradient-counted once per row. ``layout.cu_seqlens`` are
attention segments, not loss rows. Until the loss-alignment PR, sharing-on is
token-mean only.

Lifetime: :func:`using_layout` is the only way to activate a layout. It must wrap forward,
loss, and backward of one model call. Activation checkpointing re-runs forward inside
backward, and that recompute reads the same stash; a backward outside the context
recomputes without the prefix gather. Entering with a sharing layout validates every
sequence mixer (:func:`require_prefix_continuation`) and the router-replay decision for
MoE models, and raises before anything runs. Calling it per model call is the caller's
job. Training does not use this API yet: that is deliberate until the executor PR
(``plan_calls`` / ``execute_calls``). PR1 users are the CPU tests and the on-prem GPU gate.

Attention reads the layout stashed on its own module. Arctic-Platform ``FlashAttention``
gathers K/V itself. Hugging Face flash attention gets the gather from a wrapper that
``using_layout`` puts on ``transformers``' attention registry for the duration of the
context only; the wrapper is a pass-through for modules without a sharing stash and
refuses already-gathered K/V. Code with no module at hand (``cu_seqlens_from_position_ids``
and the loss guards) reads the active layout, which the context also restores on exit.

MoE policy: a shared root token goes through the router and experts once per call, so
``tokens_per_expert``, expert-bias updates, and load-balance / aux statistics count it once,
not once per row. Expert and router weight gradients still match an unshared run (the root
receives the sum of every row's gradient), but any term that depends on token counts does
not; share-on vs share-off is not an all-parameter match when such terms are active.
With router replay, groups whose replayed ids disagree on the prefix are not shared.
"""

from __future__ import annotations

import sys
from contextlib import ExitStack
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace
from functools import cached_property
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .walk import DROP
from .walk import map_batch_values

LAYOUT_ATTR = "_arctic_packed_layout"
_HF_PREFIX_MARK = "_arctic_prefix_attention"
_MISSING = object()


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
    """Enforce token-mean from the active layout so callers cannot forget ``prefix_sharing=True``."""
    layout = get_active_packed_layout()
    sharing = prefix_sharing or (layout is not None and layout.has_sharing)
    require_token_mean_if_sharing(sharing, loss_agg_mode)


def require_no_prefix_sharing(what: str) -> None:
    """Raise if a sharing layout is active: ``what`` infers segments from ``position_ids == 0``."""
    layout = get_active_packed_layout()
    if layout is not None and layout.has_sharing:
        raise ValueError(
            f"{what} infers segments from position_ids == 0, which splits a continuation from its root "
            "under prefix sharing; read the active PackedLayout instead"
        )


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

    Construct with :meth:`build`; activate with :func:`using_layout`. Do not infer these
    boundaries from ``position_ids == 0``. All index tensors are built on CPU; move them
    with :meth:`to`.

    Attributes:
        cu_seqlens: Int32 ``[num_segments + 1]`` exclusive prefix sums of packed query tokens.
            Attention segments, not loss rows. ``cu_seqlens[s]:cu_seqlens[s+1]`` is segment ``s``.
        parent: Int64 ``[num_segments]``. ``parent[s] == -1`` for a root; otherwise the root
            segment index this continuation attends into.
        num_roots: Count of root segments. They occupy ``[0, num_roots)`` in segment order.
        position_ids: Int64 ``[T]`` client RoPE positions for packed tokens (not regenerated
            ``arange``). ``T`` is ``cu_seqlens[-1]``.
        row_index: Int64 ``[B, S]``. Packed token index for each client cell, or ``-1`` if the
            cell is padding. Shared-prefix cells on every row of a group point at the root.
        src_row: Int64 ``[T]``. Client batch row gathered for each packed token (one source
            row per token; the first row of a shared group for the root).
        src_col: Int64 ``[T]``. Client column gathered for each packed token (valid-token
            columns, so left padding is skipped).
        logical_row: Int64 ``[T]``. Client row that owns this packed token for later loss
            reduction. Not ``layout.cu_seqlens``. It equals ``src_row`` in this PR because
            every root is gathered from, and attributed to, its group's first row; it is a
            separate field because a root copied for a split group (executor PR) is still
            gathered from one row but owned by the rows of its fragment.
        packed_loss_scale_factor: Pre-pack width ``S`` (including padding). Seq-mean-token-sum-norm
            uses this, not packed ``T``.
        router_replay: ``True`` if :meth:`build` checked router-replay ids, ``False`` if the
            caller declared there is no replay in this job, ``None`` if neither was stated.
            :func:`using_layout` refuses a sharing layout with ``None`` on an MoE model.
        conv_kernel_size: Optional GatedDeltaNet conv width for a later ``conv_carry`` guard.
            Unused by full attention.
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
    router_replay: bool | None = None
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
        router_replay: bool | None = None,
    ) -> "PackedLayout":
        """Build a layout from a padded ``[B, S]`` micro-batch.

        Inputs may live on any device; they are read on CPU and the layout is returned on CPU.

        Args:
            attention_mask: ``[B, S]`` 1 on valid tokens, 0 on pad. ``shared`` counts valid
                tokens, not raw columns.
            groups: Partition of batch rows. Each inner sequence is one prompt group and gets at
                most one root. A singleton is packed as its own root (no sharing).
            shared: One integer per group: how many **valid** tokens to store once as the
                root. ``0`` means no sharing. Must be ``<`` each row's valid length so a
                continuation remains. Typical full-attn value is ``prompt_len - 1``.
            position_ids: Optional ``[B, S]`` client positions. Packed positions are gathered
                from these. If omitted, valid tokens in each row get ``0 .. length-1``.
            conv_kernel_size: Optional conv width stored on the layout for GDN. Ignored here.
            input_ids: ``[B, S]`` token ids. Required when any group shares (``shared > 0``
                and at least two rows). The first ``shared[i]`` valid tokens must match.
            routed_experts: Optional ``[B, S, ...]`` router-replay ids. Groups whose valid
                prefix disagrees are not shared (``shared`` for that group becomes 0). Passing
                them sets ``router_replay=True``.
            router_replay: Pass ``False`` to declare that this job does not replay routes, which
                lets an MoE model share without ``routed_experts``. ``True`` requires
                ``routed_experts``. Leave ``None`` for dense models.

        Returns:
            A layout whose packed width is the sum of root tokens plus continuation tokens.
        """
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must be [B, S], got {tuple(attention_mask.shape)}")
        if len(groups) != len(shared):
            raise ValueError("groups and shared must have the same length")
        if routed_experts is not None:
            if router_replay is False:
                raise ValueError("routed_experts were passed with router_replay=False")
            router_replay = True
        elif router_replay:
            raise ValueError("router_replay=True needs routed_experts")
        attention_mask = attention_mask.cpu()
        batch_size, sequence_length = attention_mask.shape
        covered = [row for rows in groups for row in rows]
        if sorted(covered) != list(range(batch_size)):
            raise ValueError("groups must partition the batch rows")
        if routed_experts is not None:
            if tuple(routed_experts.shape[:2]) != (batch_size, sequence_length):
                raise ValueError(f"routed_experts must be [B, S, ...], got {tuple(routed_experts.shape)}")
            shared = skip_share_if_router_disagree(groups, shared, routed_experts.cpu(), attention_mask)
        sharing = any(n > 0 and len(list(rows)) > 1 for rows, n in zip(groups, shared))
        if sharing:
            if input_ids is None:
                raise ValueError("input_ids are required when sharing a prefix")
            require_identical_prefixes(input_ids.cpu(), groups, shared, attention_mask)
        bool_mask = attention_mask.bool()
        cols = [bool_mask[row].nonzero(as_tuple=False).flatten() for row in range(batch_size)]
        if position_ids is None:
            position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
            for row, col in enumerate(cols):
                position_ids[row, col] = torch.arange(col.numel())
        position_ids = position_ids.cpu()

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
        for s, (row, start, end, parent) in enumerate(segments):
            packed_range = torch.arange(int(cu[s]), int(cu[s + 1]))
            token_cols = cols[row][start:end]
            src_row[packed_range] = row
            src_col[packed_range] = token_cols
            row_index[row, token_cols] = packed_range
            if parent >= 0:
                parent_range = torch.arange(int(cu[parent]), int(cu[parent + 1]))
                row_index[row, cols[row][:start]] = parent_range
        return cls(
            cu_seqlens=cu,
            parent=torch.tensor([p for *_, p in segments], dtype=torch.long),
            num_roots=len(roots),
            position_ids=position_ids[src_row, src_col],
            row_index=row_index,
            src_row=src_row,
            src_col=src_col,
            logical_row=src_row.clone(),
            packed_loss_scale_factor=sequence_length,
            router_replay=router_replay,
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
        return replace(
            self,
            cu_seqlens=self.cu_seqlens.to(device),
            parent=self.parent.to(device),
            position_ids=self.position_ids.to(device),
            row_index=self.row_index.to(device),
            src_row=self.src_row.to(device),
            src_col=self.src_col.to(device),
            logical_row=self.logical_row.to(device),
        )

    def pack(self, rows: Tensor) -> Tensor:
        """Gather ``[B, S, ...]`` onto packed tokens from one source cell each."""
        if rows.ndim < 2:
            raise ValueError(f"pack expects [B, S, ...], got {tuple(rows.shape)}")
        return rows[self.src_row.to(rows.device), self.src_col.to(rows.device)]

    def _tokens_first(self, packed: Tensor) -> Tensor:
        """Accept ``[T, ...]`` or ``[1, T, ...]``. ``[T, ...]`` wins when both read the same (``T == 1``)."""
        if packed.ndim >= 1 and packed.shape[0] == self.num_tokens:
            return packed
        if packed.ndim >= 2 and packed.shape[0] == 1 and packed.shape[1] == self.num_tokens:
            return packed.squeeze(0)
        raise ValueError(f"expected [T, ...] or [1, T, ...] with T={self.num_tokens}, got {tuple(packed.shape)}")

    def unpack(self, packed: Tensor, pad_value: float = 0.0) -> Tensor:
        """Scatter packed ``[T, ...]`` (or ``[1, T, ...]``) back to ``[B, S, ...]`` via ``row_index``."""
        packed = self._tokens_first(packed)
        batch_size, sequence_length = self.row_index.shape
        trailing = tuple(packed.shape[1:])
        out = packed.new_full((batch_size, sequence_length, *trailing), pad_value)
        row_index = self.row_index.to(packed.device)
        valid = row_index >= 0
        out[valid] = packed[row_index[valid]]
        return out

    def require_unweighted_shared(self, weights: Tensor) -> None:
        """Raise if any ``[B, S]`` loss weight sits on a cell that more than one row reads."""
        if tuple(weights.shape[:2]) != tuple(self.row_index.shape):
            raise ValueError(f"weights must be [B, S]={tuple(self.row_index.shape)}, got {tuple(weights.shape)}")
        row_index = self.row_index.to(weights.device)
        valid = row_index >= 0
        index = row_index[valid]
        shared = torch.bincount(index, minlength=self.num_tokens) > 1
        if weights[valid][shared[index]].bool().any():
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


def get_stashed_layout(module: torch.nn.Module) -> PackedLayout | None:
    return getattr(module, LAYOUT_ATTR, None)


def get_active_packed_layout() -> PackedLayout | None:
    """Layout of the innermost :func:`using_layout`, for code that has no module to read."""
    return _ACTIVE_LAYOUT


@contextmanager
def using_layout(model: torch.nn.Module, layout: PackedLayout | None) -> Iterator[PackedLayout | None]:
    """Stash ``layout`` on every submodule of ``model`` for one forward + loss + backward.

    With a sharing layout this first raises unless every sequence mixer supports prefix
    continuation and, for MoE models, the router-replay decision is explicit. On exit
    (including on an exception) each module's previous stash, the previous active layout,
    and any Hugging Face attention registry entry are restored, so contexts nest.
    """
    global _ACTIVE_LAYOUT
    if layout is not None and layout.has_sharing:
        require_prefix_continuation(model)
        require_router_replay_decision(model, layout)
    previous_active = _ACTIVE_LAYOUT
    previous_stash = [(module, module.__dict__.get(LAYOUT_ATTR, _MISSING)) for module in model.modules()]
    with ExitStack() as stack:
        if layout is not None and layout.has_sharing:
            stack.enter_context(_hf_prefix_attention(model))
        try:
            for module, _ in previous_stash:
                setattr(module, LAYOUT_ATTR, layout)
            _ACTIVE_LAYOUT = layout
            yield layout
        finally:
            _ACTIVE_LAYOUT = previous_active
            for module, previous in previous_stash:
                if previous is _MISSING:
                    module.__dict__.pop(LAYOUT_ATTR, None)
                else:
                    setattr(module, LAYOUT_ATTR, previous)


def prefix_varlen_args(
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor | None,
    max_seqlen: int | None,
    layout: PackedLayout | None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, int | None, int | None]:
    """Return ``k, v, cu_q, cu_k, max_q, max_k`` for one varlen call over ``[T, H, D]`` K/V.

    Without a layout this is the caller's own ``cu_seqlens``. With one, segments come from the
    layout, and under sharing each continuation's K/V is its parent root followed by itself.
    """
    if layout is None:
        return k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
    if k.ndim != 3:
        raise ValueError(f"prefix K/V gather expects [T, H, D], got {tuple(k.shape)}")
    if k.shape[0] != layout.num_tokens or v.shape[0] != layout.num_tokens:
        raise ValueError(
            f"layout expects {layout.num_tokens} packed K/V tokens, got {k.shape[0]} / {v.shape[0]} "
            "(already gathered, or padded past the layout)"
        )
    cu_q = layout.cu_seqlens.to(device=k.device, dtype=torch.int32)
    if not layout.has_sharing:
        return k, v, cu_q, cu_q, layout.max_len_q, layout.max_len_q
    idx = layout.kv_index.to(device=k.device)
    cu_k = layout.cu_seqlens_k.to(device=k.device, dtype=torch.int32)
    return k.index_select(0, idx), v.index_select(0, idx), cu_q, cu_k, layout.max_len_q, layout.max_len_k


def _hf_prefix_attention_fn(orig: Callable) -> Callable:
    """Wrap a Hugging Face flash attention function with the prefix K/V gather of the caller's stash."""

    def prefix_attention(module, query, key, value, attention_mask, *args, **kwargs):
        layout = get_stashed_layout(module)
        if layout is None or not layout.has_sharing:
            return orig(module, query, key, value, attention_mask, *args, **kwargs)
        name = type(module).__name__
        if args:
            raise ValueError(f"{name}: prefix sharing expects keyword arguments after attention_mask")
        if attention_mask is not None:
            raise ValueError(f"{name}: prefix sharing requires a packed flash call with attention_mask=None")
        if kwargs.get("sliding_window") is not None:
            # transformers widens the window to (w-1, w-1) once K is longer than w; not a causal prefix window.
            raise ValueError(f"{name}: Hugging Face sliding-window attention is not supported under prefix sharing")
        if query.ndim != 4 or query.shape[0] != 1:
            raise ValueError(f"{name}: prefix sharing expects packed [1, H, T, D] queries, got {tuple(query.shape)}")
        if query.shape[2] != layout.num_tokens or key.shape[2] != layout.num_tokens:
            raise ValueError(
                f"{name}: layout expects {layout.num_tokens} packed tokens, got q={query.shape[2]} "
                f"k={key.shape[2]} (already gathered, or padded past the layout)"
            )
        idx = layout.kv_index.to(device=key.device)
        key = key.index_select(2, idx)
        value = value.index_select(2, idx)
        kwargs["cu_seq_lens_q"] = layout.cu_seqlens.to(device=query.device, dtype=torch.int32)
        kwargs["cu_seq_lens_k"] = layout.cu_seqlens_k.to(device=query.device, dtype=torch.int32)
        kwargs["max_length_q"] = layout.max_len_q
        kwargs["max_length_k"] = layout.max_len_k
        return orig(module, query, key, value, None, **kwargs)

    setattr(prefix_attention, _HF_PREFIX_MARK, True)
    return prefix_attention


@contextmanager
def _hf_prefix_attention(model: torch.nn.Module) -> Iterator[None]:
    """Point the flash entries of ``transformers``' attention registry at the prefix wrapper, then restore them."""
    hf_modules = [module for module in model.modules() if type(module).__name__ in _HF_PREFIX_ATTENTION]
    if not hf_modules:
        yield
        return
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as registry
    except ImportError as exc:
        raise ValueError("prefix sharing on Hugging Face attention needs transformers' AttentionInterface") from exc
    local = getattr(registry, "_local_mapping", None)
    if not isinstance(local, dict):
        raise ValueError(
            "transformers' AttentionInterface has no per-instance mapping; cannot scope the prefix wrapper"
        )
    for module in hf_modules:
        if getattr(sys.modules.get(type(module).__module__), "ALL_ATTENTION_FUNCTIONS", None) is not registry:
            raise ValueError(f"{type(module).__name__} does not dispatch through transformers.ALL_ATTENTION_FUNCTIONS")
    saved: list[tuple[str, Any]] = []
    try:
        for impl in sorted({module.config._attn_implementation for module in hf_modules}):
            if impl not in registry:
                raise ValueError(f"attention implementation {impl!r} is not registered with transformers")
            current = registry[impl]
            if getattr(current, _HF_PREFIX_MARK, False):
                continue
            saved.append((impl, local.get(impl, _MISSING)))
            registry[impl] = _hf_prefix_attention_fn(current)
        yield
    finally:
        for impl, previous in reversed(saved):
            if previous is _MISSING:
                del registry[impl]
            else:
                registry[impl] = previous


_HF_PREFIX_ATTENTION = {"Qwen3Attention", "Qwen3MoeAttention"}
_NATIVE_FLASH = {"FlashAttention", "AfmoeFlashAttention"}
_SEQUENCE_MIXER_NEEDLES = ("Attention", "DeltaNet", "GatedDelta", "LinearAttention", "KDA")
_MOE_NAME_NEEDLES = ("MoE", "Moe")
_MOE_CONFIG_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")


def sequence_mixers(model: torch.nn.Module) -> list[torch.nn.Module]:
    found = []
    for module in model.modules():
        name = type(module).__name__
        if any(needle in name for needle in _SEQUENCE_MIXER_NEEDLES):
            found.append(module)
    return found


def _prefix_continuation_blocker(module: torch.nn.Module) -> str | None:
    """Why ``module`` cannot run prefix continuation, or ``None`` if it can."""
    name = type(module).__name__
    if name in _NATIVE_FLASH:
        return None
    if name not in _HF_PREFIX_ATTENTION:
        return f"{name} (no prefix-continuation path)"
    impl = getattr(getattr(module, "config", None), "_attn_implementation", None)
    if not (isinstance(impl, str) and impl.startswith("flash")):
        return f"{name} (attn_implementation={impl!r}, needs flash)"
    if getattr(module, "sliding_window", None) is not None:
        return f"{name} (Hugging Face sliding window)"
    return None


def supports_prefix_continuation(module: torch.nn.Module) -> bool:
    return _prefix_continuation_blocker(module) is None


def require_prefix_continuation(model: torch.nn.Module) -> None:
    mixers = sequence_mixers(model)
    if not mixers:
        raise ValueError("zorro/prefix sharing found no sequence mixers to validate")
    unsupported = sorted({reason for module in mixers if (reason := _prefix_continuation_blocker(module))})
    if unsupported:
        raise ValueError(
            "zorro/prefix sharing needs prefix continuation in every sequence mixer; "
            f"unsupported in this PR: {unsupported}"
        )


def is_moe_model(model: torch.nn.Module) -> bool:
    for module in model.modules():
        if any(needle in type(module).__name__ for needle in _MOE_NAME_NEEDLES):
            return True
        config = getattr(module, "config", None)
        for key in _MOE_CONFIG_KEYS:
            value = getattr(config, key, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return True
    return False


def require_router_replay_decision(model: torch.nn.Module, layout: PackedLayout) -> None:
    if layout.has_sharing and layout.router_replay is None and is_moe_model(model):
        raise ValueError(
            "prefix sharing on an MoE model needs PackedLayout.build(routed_experts=...) so disagreeing groups "
            "are not shared, or router_replay=False to declare that this job does not replay routes"
        )
