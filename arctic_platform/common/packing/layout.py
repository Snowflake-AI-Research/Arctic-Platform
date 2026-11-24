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
loss, and backward of one model call. Entering with a sharing layout validates every
sequence mixer (:func:`require_prefix_continuation`) and the router-replay decision for
MoE models, and raises before anything runs. Calling it per model call is the caller's
job. Training does not use this API yet: that is deliberate until the executor PR
(``plan_calls`` / ``execute_calls``). PR1 users are the CPU tests and the on-prem GPU gate.

Hold the context through backward. Activation checkpointing re-runs forward inside
backward and that recompute reads the same stash. If the context has exited by then,
non-reentrant checkpointing (``use_reentrant=False``) raises ``CheckpointError``, but
reentrant checkpointing (``use_reentrant=True``, which ``transformers<5``
``gradient_checkpointing_enable()`` picks by default) recomputes without the prefix gather
and returns finite, wrong gradients. :meth:`PackedLayout.unpack` hooks backward and raises
if its layout's context is gone, which catches that case only when the loss flows through
``unpack``. Prefer ``use_reentrant=False``.

One thread per process. Contexts nest, but only on the thread that entered the outermost
one; another thread entering while any context is active raises. The active-layout stack
and the Hugging Face registry override are process globals. Reads are not thread-checked:
autograd runs backward, and checkpoint recompute, on its own device threads.

Every model that runs while a context is active needs its own :func:`using_layout` (an
unshared layout is fine). Arctic-Platform attention (flash or SDPA) with no stash raises while
any context is active, because ``cu_seqlens_from_position_ids`` would hand it another call's
segments, and SDPA also raises on a layout with several segments. Hugging Face attention that
dispatches through ``transformers``' registry (every implementation but eager) raises with no
stash while a sharing context is active. With nested
contexts the loss and position-id guards fire if any layout in the stack shares, and
``cu_seqlens_from_position_ids`` raises when the stack holds more than one layout, since it
cannot tell which call it is serving.

Attention reads the layout stashed on its own module. Arctic-Platform ``FlashAttention``
gathers K/V itself. Hugging Face flash attention gets the gather from a wrapper that
``using_layout`` puts on ``transformers``' attention registry for the duration of the
context only; the wrapper passes modules with an unshared stash through and refuses
already-gathered K/V and attention dropout, which would draw one mask for a shared root.

Only flash-attn 2 (>= 2.1, the ``flash_attn`` package), FA3 (the ``flash_attn_3`` package)
and FA4 (the ``flash-attn-4`` package, not the copy flash_attn 2.8 bundles, which has no
varlen backward) have been checked on GPU against an unshared run. A sharing layout refuses
other ``FlashAttention`` versions and Hugging Face implementations other than
``flash_attention_2`` / ``_3`` / ``_4``, and needs the kernel each mixer calls installed.

Mixers are found by class name and by a direct ``Conv1d`` child; any that is not one of the
flash attentions above (Mamba / SSM, GatedDeltaNet, short convolutions) refuses a sharing
layout, because it would restart its state at each continuation instead of carrying the root's.

Not supported inside :func:`using_layout` until the GatedDeltaNet and executor PRs define
them: sequence parallelism (SP >= 2, a shard's ``position_ids`` do not cover the layout)
and padded calls (``pad_packed_microbatch``). Both fail the packed-token-count checks in
``cu_seqlens_from_position_ids`` and :func:`prefix_varlen_args`.

MoE policy (documentation only; nothing here adjusts token counts): a shared root token goes
through the router and experts once per call, so ``tokens_per_expert``, expert-bias updates,
and load-balance / aux statistics count it once, not once per row. Expert and router weight
gradients still match an unshared run (the root receives the sum of every row's gradient),
but any term that depends on token counts does not; share-on vs share-off is not an
all-parameter match when such terms are active. With router replay, groups whose replayed
ids disagree on the prefix are not shared.
"""

from __future__ import annotations

import inspect
import re
import sys
import threading
from contextlib import ExitStack
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from functools import cached_property
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import Sequence
from typing import TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor

from .pack import IGNORE_INDEX
from .pack import ROW_BROADCAST_KEYS
from .walk import DROP
from .walk import map_batch_values

LAYOUT_ATTR = "_arctic_packed_layout"
# Per-token loss masks the providers in this stack send; nonzero marks a cell the loss reads. ECHO's
# ``sft_mask`` / ``echo_observation_mask`` sit outside ``loss_mask`` but still weight a loss term.
LOSS_WEIGHT_KEYS = frozenset({"loss_mask", "response_mask", "sft_mask", "echo_observation_mask"})
_HF_PREFIX_MARK = "_arctic_prefix_attention"
# Instance attribute: ``check(layout) -> reason | None``. A mixer that continues prefixes itself (the GatedDeltaNet
# adapter) sets it on each module it installs on; a class attribute does not count.
PREFIX_CONTINUATION_ATTR = "_arctic_prefix_continuation"
_MISSING = object()
_ModuleType = TypeVar("_ModuleType", bound=type)


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
    """Enforce token-mean from the active layouts so callers cannot forget ``prefix_sharing=True``."""
    require_token_mean_if_sharing(prefix_sharing or prefix_sharing_active(), loss_agg_mode)


def require_no_prefix_sharing(what: str) -> None:
    """Raise if any active layout shares: ``what`` infers segments from ``position_ids == 0``."""
    if prefix_sharing_active():
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
    position_ids: Tensor | None = None,
) -> None:
    """``shared[i]`` is a count of valid tokens, not raw columns.

    With ``position_ids`` the shared tokens must also sit at the same positions in every row: the
    root is gathered from one row, so every row's continuation attends to that row's RoPE.
    """
    for rows, n in zip(groups, shared):
        if n <= 0 or len(rows) < 2:
            continue
        first = _valid_prefix(input_ids, attention_mask, rows[0], n)
        for row in rows[1:]:
            if not torch.equal(_valid_prefix(input_ids, attention_mask, row, n), first):
                raise ValueError(f"prompt group {list(rows)} is not token-identical on the first {n} valid tokens")
        if position_ids is None:
            continue
        first = _valid_prefix(position_ids, attention_mask, rows[0], n)
        for row in rows[1:]:
            if not torch.equal(_valid_prefix(position_ids, attention_mask, row, n), first):
                raise ValueError(
                    f"prompt group {list(rows)} has different position_ids on the first {n} valid tokens; "
                    f"the shared root would give row {row} the positions of row {rows[0]}"
                )


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


@dataclass(frozen=True, eq=False)
class PackedLayout:
    """One packed model call. Segments ``[0, num_roots)`` are roots; later segments continue a parent.

    Construct with :meth:`build`; activate with :func:`using_layout`. Do not infer these
    boundaries from ``position_ids == 0``. All index tensors are built on CPU; move them
    with :meth:`to`. Sequence-parallel shards and padded calls cannot use a layout yet (see
    the module docstring).

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
            :func:`using_layout` refuses a sharing layout with ``None`` on an MoE model, and
            :func:`pack_with_layout` checks it against the batch's ``routed_experts``.
        conv_kernel_size: Optional GatedDeltaNet conv width for a later ``conv_carry`` guard.
            Unused by full attention.
        origin: Identity shared by a layout and its :meth:`to` copies, so backward can tell
            whether this layout's context is still active.
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
    origin: object = field(default_factory=object, repr=False)

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
                tokens, not raw columns. Every row needs at least one valid token: an all-pad
                row would be an empty segment, which flash-attn rejects.
            groups: Partition of batch rows. Each inner sequence is one prompt group and gets at
                most one root. A singleton is packed as its own root (no sharing).
            shared: One integer per group: how many **valid** tokens to store once as the
                root. ``0`` means no sharing. Must be ``<`` each row's valid length so a
                continuation remains. Typical full-attn value is ``prompt_len - 1``.
            position_ids: Optional ``[B, S]`` client positions. Packed positions are gathered
                from these. If omitted, valid tokens in each row get ``0 .. length-1``. In a
                sharing group the first ``shared[i]`` valid positions must be equal in every
                row (e.g. ``cumsum(mask) - 1``, not column indices under uneven left padding),
                because the root's positions come from one row.
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

        Raises:
            ValueError: on a malformed partition, an all-pad row, or a sharing group whose
                valid prefix differs across rows in ``input_ids`` or ``position_ids``.
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
        bool_mask = attention_mask.bool()
        cols = [bool_mask[row].nonzero(as_tuple=False).flatten() for row in range(batch_size)]
        empty = [row for row, col in enumerate(cols) if col.numel() == 0]
        if empty:
            raise ValueError(
                f"rows {empty} have no valid tokens; flash-attn rejects empty sequences, so drop them before packing"
            )
        if position_ids is not None:
            if tuple(position_ids.shape) != (batch_size, sequence_length):
                raise ValueError(
                    f"position_ids must be [B, S]={batch_size, sequence_length}, got {tuple(position_ids.shape)}"
                )
            position_ids = position_ids.cpu()
        if routed_experts is not None:
            if tuple(routed_experts.shape[:2]) != (batch_size, sequence_length):
                raise ValueError(f"routed_experts must be [B, S, ...], got {tuple(routed_experts.shape)}")
            shared = skip_share_if_router_disagree(groups, shared, routed_experts.cpu(), attention_mask)
        sharing = any(n > 0 and len(list(rows)) > 1 for rows, n in zip(groups, shared))
        if sharing:
            if input_ids is None:
                raise ValueError("input_ids are required when sharing a prefix")
            require_identical_prefixes(input_ids.cpu(), groups, shared, attention_mask, position_ids)
        if position_ids is None:
            position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
            for row, col in enumerate(cols):
                position_ids[row, col] = torch.arange(col.numel())

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

    @cached_property
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

    def _tokens_first(self, packed: Tensor, batched: bool | None) -> Tensor:
        tokens = self.num_tokens
        reads_batched = packed.ndim >= 2 and packed.shape[0] == 1 and packed.shape[1] == tokens
        reads_tokens = packed.ndim >= 1 and packed.shape[0] == tokens
        if batched is None:
            # With T == 1 a [1, 1, ...] tensor reads both ways; model outputs are [1, T, ...], so that wins.
            batched = reads_batched
        if batched and reads_batched:
            return packed.squeeze(0)
        if not batched and reads_tokens:
            return packed
        expected = "[1, T, ...]" if batched else "[T, ...] or [1, T, ...]"
        raise ValueError(f"expected {expected} with T={tokens}, got {tuple(packed.shape)}")

    def unpack(self, packed: Tensor, pad_value: float = 0.0, batched: bool | None = None) -> Tensor:
        """Scatter packed ``[T, ...]`` (or ``[1, T, ...]``) back to ``[B, S, ...]`` via ``row_index``.

        ``batched=None`` reads a leading ``[1, T]`` as a batch dimension, including ``[1, 1, ...]``
        when ``T == 1``; pass ``batched=False`` for a ``[T, ...]`` tensor whose second dim is ``T``
        or ``1`` by coincidence. With a sharing layout and a tensor that requires grad, backward
        raises if this layout's :func:`using_layout` has exited (see the module docstring).
        """
        packed = self._tokens_first(packed, batched)
        if self.has_sharing and packed.requires_grad and torch.is_grad_enabled():
            packed.register_hook(_require_context_at_backward(self.origin))
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
    """Pack ``[B, S, ...]`` leaves with ``layout``; drop ``attention_mask`` like ``pack_microbatch``.

    ``ROW_BROADCAST_KEYS`` leaves sent as ``[B]`` become ``[1, T]`` by each packed token's
    ``logical_row``, as ``pack_microbatch`` does; a shared root token takes its source row's
    value, which reaches no loss as long as shared positions carry no loss weight. A loss
    computed on packed tokens would count a weighted root once instead of once per row, so
    ``[B, S]`` ``LOSS_WEIGHT_KEYS`` leaves (or ``labels != IGNORE_INDEX`` when none is sent)
    must be zero on shared cells (:meth:`PackedLayout.require_unweighted_shared`). A batch's
    ``routed_experts`` must agree with ``layout.router_replay`` and, when replayed, with the
    layout's shared cells. A ``[B, S]`` ``attention_mask``, ``position_ids`` or ``input_ids``
    must be the batch the layout was built from: the same valid cells, the layout's positions
    (which replace the batch's), and tokens that agree on every shared cell.
    """
    batch_size, sequence_length = layout.row_index.shape
    routed: list[Tensor] = []
    weights: list[Tensor] = []
    labels: list[Tensor] = []
    identity: dict[str, Tensor] = {}

    def packed_value(key: str, value: Any) -> Any:
        if key in ("attention_mask", "position_ids", "input_ids") and torch.is_tensor(value):
            if tuple(value.shape) == (batch_size, sequence_length):
                identity[key] = value
        if key == "attention_mask":
            return DROP
        if key == "routed_experts" and torch.is_tensor(value):
            routed.append(value)
        if torch.is_tensor(value) and tuple(value.shape) == (batch_size, sequence_length):
            if key in LOSS_WEIGHT_KEYS:
                weights.append(value)
            elif key == "labels":
                labels.append(value != IGNORE_INDEX)
        if key in ROW_BROADCAST_KEYS and torch.is_tensor(value) and tuple(value.shape) == (batch_size,):
            return value[layout.logical_row.to(value.device)].unsqueeze(0)
        if torch.is_tensor(value) and value.ndim >= 2 and tuple(value.shape[:2]) == (batch_size, sequence_length):
            packed = layout.pack(value)
            return packed.unsqueeze(0)
        return value

    packed = map_batch_values(data, packed_value)
    _require_batch_matches_layout(layout, identity)
    _require_routed_experts_match(layout, routed)
    if layout.has_sharing:
        for value in weights or labels:
            layout.require_unweighted_shared(value)
    packed["position_ids"] = layout.position_ids.unsqueeze(0)
    return packed


def _require_batch_matches_layout(layout: PackedLayout, identity: Dict[str, Tensor]) -> None:
    row_index = layout.row_index
    valid = row_index >= 0
    mask = identity.get("attention_mask")
    if mask is not None and not torch.equal(mask.bool().cpu(), valid.cpu()):
        raise ValueError("the batch's attention_mask has different valid cells from the layout's")
    positions = identity.get("position_ids")
    if positions is not None:
        index = row_index[valid].to(positions.device)
        expected = layout.position_ids.to(positions.device, torch.long)[index]
        if not torch.equal(expected, positions[valid.to(positions.device)].long()):
            raise ValueError("the batch's position_ids differ from the positions the layout was built with")
    ids = identity.get("input_ids")
    if ids is not None and layout.has_sharing:
        index = row_index[valid].to(ids.device)
        if not torch.equal(layout.pack(ids)[index], ids[valid.to(ids.device)]):
            raise ValueError("the batch's input_ids differ across rows on a shared prefix cell")


def _require_routed_experts_match(layout: PackedLayout, routed: Sequence[Tensor]) -> None:
    if not routed:
        if layout.router_replay:
            raise ValueError("the layout was built with routed_experts but the batch carries none")
        return
    if layout.router_replay is False:
        raise ValueError("the batch carries routed_experts but the layout was built with router_replay=False")
    if layout.router_replay is None and layout.has_sharing:
        raise ValueError(
            "the batch carries routed_experts but PackedLayout.build was not given them, so groups whose "
            "replayed routes disagree on the prefix were not checked"
        )
    for value in routed:
        if tuple(value.shape[:2]) != tuple(layout.row_index.shape):
            raise ValueError(
                f"routed_experts must be [B, S, ...]={tuple(layout.row_index.shape)}, got {tuple(value.shape)}"
            )
        row_index = layout.row_index.to(value.device)
        valid = row_index >= 0
        if not torch.equal(layout.pack(value)[row_index[valid]], value[valid]):
            raise ValueError(
                "the batch's routed_experts disagree on a shared prefix cell with those the layout was built from"
            )


# Innermost last. Process-global: read from autograd threads, written only by the owning thread.
_ACTIVE_LAYOUTS: list[PackedLayout | None] = []
_CONTEXT_LOCK = threading.Lock()
_CONTEXT_OWNER: int | None = None
_CONTEXT_DEPTH = 0


def get_stashed_layout(module: torch.nn.Module) -> PackedLayout | None:
    return getattr(module, LAYOUT_ATTR, None)


def get_active_packed_layout() -> PackedLayout | None:
    """Layout of the innermost :func:`using_layout`, for code that has no module to read."""
    return _ACTIVE_LAYOUTS[-1] if _ACTIVE_LAYOUTS else None


def prefix_sharing_active() -> bool:
    """``True`` if any active :func:`using_layout`, not just the innermost, shares a prefix."""
    return any(layout is not None and layout.has_sharing for layout in _ACTIVE_LAYOUTS)


def layout_for_position_ids() -> PackedLayout | None:
    """The layout whose segments replace boundaries read from ``position_ids``, or ``None`` to read them.

    The caller has no module, so it cannot tell which context's model it serves. One layout
    (possibly entered more than once) is unambiguous. With several, any choice can hand one
    model's segments to another model's mixers, and the packed-token-count check only catches
    that when the widths differ, so this raises instead.
    """
    if not _ACTIVE_LAYOUTS:
        return None
    if any(layout is not _ACTIVE_LAYOUTS[0] for layout in _ACTIVE_LAYOUTS):
        raise ValueError(
            "nested using_layout contexts with different layouts are active, so segments cannot be "
            "read from position_ids for an unknown caller; run each model call in its own, non-nested context"
        )
    return _ACTIVE_LAYOUTS[0]


def _require_context_at_backward(origin: object) -> Callable[[Tensor], None]:
    def check(grad: Tensor) -> None:
        if not any(layout is not None and layout.origin is origin for layout in _ACTIVE_LAYOUTS):
            raise RuntimeError(
                "backward of a prefix-sharing call ran after its using_layout exited; hold the context "
                "through backward (activation-checkpoint recompute would run without the prefix gather)"
            )

    return check


@contextmanager
def _owning_thread() -> Iterator[None]:
    global _CONTEXT_OWNER, _CONTEXT_DEPTH
    me = threading.get_ident()
    with _CONTEXT_LOCK:
        if _CONTEXT_DEPTH and _CONTEXT_OWNER != me:
            raise RuntimeError(
                f"using_layout is already active on thread {_CONTEXT_OWNER}; PackedLayout contexts are "
                "process-global, so only one thread may hold them at a time"
            )
        _CONTEXT_OWNER = me
        _CONTEXT_DEPTH += 1
    try:
        yield
    finally:
        with _CONTEXT_LOCK:
            _CONTEXT_DEPTH -= 1
            if not _CONTEXT_DEPTH:
                _CONTEXT_OWNER = None


@contextmanager
def using_layout(model: torch.nn.Module, layout: PackedLayout | None) -> Iterator[PackedLayout | None]:
    """Stash ``layout`` on every submodule of ``model`` for one forward + loss + backward.

    With a sharing layout this first raises unless every sequence mixer supports prefix
    continuation, the router-replay decision is explicit for MoE models, and any installed
    flash-attn 2 aligns causal masks bottom-right. On exit (including on an exception) each
    module's previous stash, the active-layout stack, and any Hugging Face attention registry
    entry are restored, so contexts nest on one thread; another thread entering while a
    context is active raises.

    Keep the context open through backward. With activation checkpointing, recompute after
    exit raises ``CheckpointError`` for ``use_reentrant=False`` and silently drops the prefix
    gather for ``use_reentrant=True`` (``transformers<5`` ``gradient_checkpointing_enable()``
    default); :meth:`PackedLayout.unpack` raises on the latter only if the loss flows through it.
    """
    if layout is not None and layout.has_sharing:
        require_prefix_continuation(model, layout)
        require_router_replay_decision(model, layout)
        require_bottom_right_causal_flash(model)
    previous_stash = [(module, module.__dict__.get(LAYOUT_ATTR, _MISSING)) for module in model.modules()]
    with ExitStack() as stack:
        stack.enter_context(_owning_thread())
        if layout is not None and layout.has_sharing:
            stack.enter_context(_hf_prefix_attention(model))
        pushed = False
        try:
            for module, _ in previous_stash:
                setattr(module, LAYOUT_ATTR, layout)
            _ACTIVE_LAYOUTS.append(layout)
            pushed = True
            yield layout
        finally:
            if pushed:
                _ACTIVE_LAYOUTS.pop()
            for module, previous in previous_stash:
                if previous is _MISSING:
                    module.__dict__.pop(LAYOUT_ATTR, None)
                else:
                    setattr(module, LAYOUT_ATTR, previous)


def _unstashed_error(name: str) -> ValueError:
    return ValueError(
        f"{name} ran with no PackedLayout stashed while a using_layout context is active in this process; "
        "the active layout's segments may have reached it through cu_seqlens_from_position_ids. Wrap this "
        "model in its own using_layout (an unshared layout is fine)"
    )


def require_stash_inside_context(name: str, layout: PackedLayout | None) -> None:
    """Raise if mixer ``name`` has no stash while any context is active: its boundaries may be another call's."""
    if layout is None and any(active is not None for active in _ACTIVE_LAYOUTS):
        raise _unstashed_error(name)


def require_dense_attention_layout(name: str, layout: PackedLayout | None) -> None:
    """Raise unless dense (non-varlen) attention ``name`` can run on a call with stash ``layout``.

    It masks one causal rectangle per batch row, so a layout with several segments would let them
    attend to each other, and with no stash inside a context it cannot tell whether it got a packed batch.
    """
    if layout is None:
        if any(active is not None for active in _ACTIVE_LAYOUTS):
            raise _unstashed_error(name)
        return
    if layout.num_segments > 1:
        raise ValueError(
            f"{name} has no varlen path, so its causal mask would span the layout's {layout.num_segments} "
            "segments; packed layouts need FlashAttention"
        )


def prefix_varlen_args(
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor | None,
    max_seqlen: int | None,
    layout: PackedLayout | None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, int | None, int | None]:
    """Return ``k, v, cu_q, cu_k, max_q, max_k`` for one varlen call over ``[T, H, D]`` K/V.

    ``layout`` is the caller's own stash. Without one this is the caller's own ``cu_seqlens``,
    unless some context is active: then those segments may be another call's, so this raises.
    With one, segments come from the layout, and under sharing each continuation's K/V is its
    parent root followed by itself.
    """
    if layout is None:
        if any(active is not None for active in _ACTIVE_LAYOUTS):
            raise _unstashed_error("attention")
        return k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
    if k.ndim != 3:
        raise ValueError(f"prefix K/V gather expects [T, H, D], got {tuple(k.shape)}")
    if k.shape[0] != layout.num_tokens or v.shape[0] != layout.num_tokens:
        raise ValueError(
            f"layout expects {layout.num_tokens} packed K/V tokens, got {k.shape[0]} / {v.shape[0]} "
            "(already gathered, or a sequence-parallel shard or padded call, which using_layout does not "
            "support until the GatedDeltaNet / executor PRs)"
        )
    cu_q = layout.cu_seqlens.to(device=k.device, dtype=torch.int32)
    if not layout.has_sharing:
        return k, v, cu_q, cu_q, layout.max_len_q, layout.max_len_q
    idx = layout.kv_index.to(device=k.device)
    cu_k = layout.cu_seqlens_k.to(device=k.device, dtype=torch.int32)
    return k.index_select(0, idx), v.index_select(0, idx), cu_q, cu_k, layout.max_len_q, layout.max_len_k


def _installed_version(distribution: str) -> tuple[int, int] | None:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version

    try:
        raw = version(distribution)
    except PackageNotFoundError:
        return None
    match = re.match(r"(\d+)\.(\d+)", raw)
    if match is None:
        raise ValueError(f"cannot parse the installed {distribution} version {raw!r}")
    return int(match.group(1)), int(match.group(2))


# Flash kernel generation -> (distribution, oldest version that aligns causal masks bottom-right when a
# segment has more keys than queries). flash-attn < 2.1 aligns them top-left; FA3 is bottom-right throughout.
# FA4 is the flash-attn-4 package: the copy bundled with flash_attn 2.8 has no varlen backward.
_FLASH_KERNELS = {2: ("flash_attn", (2, 1)), 3: ("flash_attn_3", (3, 0)), 4: ("flash_attn_4", (4, 0))}
_HF_FLASH_KERNELS = {"flash_attention_2": 2, "flash_attention_3": 3, "flash_attention_4": 4}


def _flash_kernel(module: torch.nn.Module) -> int:
    if type(module) in _NATIVE_PREFIX_ATTENTION:
        return getattr(module, "_flash_attn_version", 2)
    return _HF_FLASH_KERNELS[module.config._attn_implementation]


def require_bottom_right_causal_flash(model: torch.nn.Module) -> None:
    """Every flash kernel ``model``'s mixers call must be installed and align the causal mask bottom-right."""
    attention = [module for module in sequence_mixers(model) if _prefix_continuation_check(module) is None]
    for kernel in sorted({_flash_kernel(module) for module in attention}):
        distribution, oldest = _FLASH_KERNELS[kernel]
        installed = _installed_version(distribution)
        if installed is None:
            raise ValueError(
                f"prefix sharing needs the {distribution} package (>= {oldest[0]}.{oldest[1]}); it is not installed"
            )
        if installed < oldest:
            raise ValueError(
                f"prefix sharing needs {distribution} >= {oldest[0]}.{oldest[1]} (bottom-right causal alignment);"
                f" found {installed[0]}.{installed[1]}"
            )
        if kernel == 4 and not _fa4_varlen_backward_available():
            raise ValueError(
                "prefix sharing on FA4 needs flash-attn-4's flash_attn.cute, but the importable one is missing or is"
                " flash_attn 2.8's bundled copy, which has no varlen backward; reinstall flash-attn-4 after flash_attn"
            )


def _fa4_varlen_backward_available() -> bool:
    """flash-attn-4's ``flash_attn_varlen_func`` takes ``max_seqlen_k``; the copy flash_attn 2.8 bundles does not."""
    try:
        from flash_attn.cute import flash_attn_varlen_func
    # Some flash_attn builds raise AttributeError, not ImportError, while importing ``flash_attn.cute``.
    except (ImportError, AttributeError):
        return False
    return "max_seqlen_k" in inspect.signature(flash_attn_varlen_func).parameters


def _hf_prefix_attention_fn(orig: Callable, impl: str = "flash_attention_2") -> Callable:
    """Wrap Hugging Face attention function ``impl`` with the prefix K/V gather of the caller's stash.

    Modules with an unshared stash pass through; with no stash they raise while a sharing context
    is active; only a flash ``impl`` runs a sharing call.
    """

    def prefix_attention(module, query, key, value, attention_mask, *args, **kwargs):
        layout = get_stashed_layout(module)
        name = type(module).__name__
        if layout is None and prefix_sharing_active():
            raise _unstashed_error(name)
        if layout is None or not layout.has_sharing:
            return orig(module, query, key, value, attention_mask, *args, **kwargs)
        if impl not in _HF_FLASH_KERNELS:
            raise ValueError(f"{name}: prefix sharing needs a flash attention implementation, got {impl!r}")
        if kwargs.get("dropout"):
            # One mask for a shared root where unshared rows would each draw their own.
            raise ValueError(f"{name}: attention dropout is not supported under prefix sharing")
        if args:
            raise ValueError(f"{name}: prefix sharing expects keyword arguments after attention_mask")
        if attention_mask is not None:
            raise ValueError(f"{name}: prefix sharing requires a packed flash call with attention_mask=None")
        if kwargs.get("sliding_window") is not None:
            # The window path through transformers' flash kwargs is unverified for prefix continuation.
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
    """Point every entry of ``transformers``' attention registry at the prefix wrapper, then restore them.

    Every entry, not just the ones ``model`` uses, so another model's Hugging Face attention with no
    stash raises instead of reading the packed batch as one sequence. Eager attention skips the registry.
    """
    hf_modules = [module for module in model.modules() if type(module).__name__ in _HF_PREFIX_ATTENTION]
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as registry
    except ImportError as exc:
        raise ValueError("prefix sharing needs transformers' AttentionInterface") from exc
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
        for impl in sorted(registry.valid_keys()):
            current = registry[impl]
            if getattr(current, _HF_PREFIX_MARK, False):
                continue
            saved.append((impl, local.get(impl, _MISSING)))
            registry[impl] = _hf_prefix_attention_fn(current, impl)
        yield
    finally:
        for impl, previous in reversed(saved):
            if previous is _MISSING:
                del registry[impl]
            else:
                registry[impl] = previous


_HF_PREFIX_ATTENTION = {"Qwen3Attention", "Qwen3MoeAttention"}
_NATIVE_PREFIX_ATTENTION: set[type] = set()
_SEQUENCE_MIXER_NEEDLES = (
    "Attention",
    "DeltaNet",
    "GatedDelta",
    "LinearAttention",
    "KDA",
    "Mamba",
    "SSM",
    "Mixer",
    "Convolution",
    "ShortConv",
)
_MOE_NAME_NEEDLES = ("MoE", "Moe")
_MOE_CONFIG_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")


def register_native_prefix_attention(cls: _ModuleType) -> _ModuleType:
    """Accept ``cls`` as a mixer that runs :func:`prefix_varlen_args` on its own stash.

    Matched by exact type: a subclass may override the kernel call and must register itself.
    """
    _NATIVE_PREFIX_ATTENTION.add(cls)
    return cls


def _is_sequence_mixer(module: torch.nn.Module) -> bool:
    name = type(module).__name__
    if any(needle in name for needle in _SEQUENCE_MIXER_NEEDLES):
        return True
    # A causal Conv1d over tokens (Mamba, GatedDeltaNet, short-conv mixers) carries state past a segment start,
    # whatever its parent is called.
    return any(isinstance(child, torch.nn.Conv1d) for child in module.children())


def sequence_mixers(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [module for module in model.modules() if _is_sequence_mixer(module)]


def _prefix_continuation_check(module: torch.nn.Module) -> Callable[[PackedLayout | None], str | None] | None:
    check = module.__dict__.get(PREFIX_CONTINUATION_ATTR)
    return check if callable(check) else None


def _prefix_continuation_blocker(module: torch.nn.Module, layout: PackedLayout | None = None) -> str | None:
    """Why ``module`` cannot run prefix continuation (on ``layout``, if given), or ``None`` if it can."""
    name = type(module).__name__
    check = _prefix_continuation_check(module)
    if check is not None:
        reason = check(layout)
        return None if reason is None else f"{name} ({reason})"
    if type(module) in _NATIVE_PREFIX_ATTENTION:
        version = getattr(module, "_flash_attn_version", 2)
        if version not in _FLASH_KERNELS:
            return f"{name} (flash-attn {version}; only flash-attn 2, 3 and 4 are verified for prefix continuation)"
        return None
    if name not in _HF_PREFIX_ATTENTION:
        return f"{name} (no prefix-continuation path)"
    impl = getattr(getattr(module, "config", None), "_attn_implementation", None)
    if impl not in _HF_FLASH_KERNELS:
        return (
            f"{name} (attn_implementation={impl!r}, needs flash_attention_2, flash_attention_3 or flash_attention_4)"
        )
    if getattr(module, "sliding_window", None) is not None:
        return f"{name} (Hugging Face sliding window)"
    return None


def supports_prefix_continuation(module: torch.nn.Module) -> bool:
    return _prefix_continuation_blocker(module) is None


def require_prefix_continuation(model: torch.nn.Module, layout: PackedLayout | None = None) -> None:
    """Raise unless every sequence mixer of ``model`` can continue a shared prefix (on ``layout``, if given)."""
    mixers = sequence_mixers(model)
    if not mixers:
        raise ValueError("zorro/prefix sharing found no sequence mixers to validate")
    unsupported = sorted({reason for module in mixers if (reason := _prefix_continuation_blocker(module, layout))})
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
