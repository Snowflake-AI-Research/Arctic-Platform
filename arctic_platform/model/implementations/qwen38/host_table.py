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
"""Frozen embedding rows kept in pinned host memory and gathered by the GPU over unified addressing.

Qwen3.8's n-gram table (51.2B BF16 parameters, frozen) is sharded over the EP group, so each GPU holds 1/EP of it:
12.8 GB at EP8. A lookup reads only the rows its tokens hash to (16 rows of 320 bytes per token), so the shard can
live in pinned host memory: a kernel on the GPU reads the rows straight from it over PCIe (pinned host memory is
addressable from the device under UVA), about 42 MB per lookup at 8K tokens per rank, and the GPU memory is free
for activations. The table has no gradient, so nothing comes back.

A table that moved keeps an empty parameter. ``HostTables`` registers the tables as a holding of the model's frozen
weights (``moe/frozen_held.py``), so a checkpoint taken or loaded inside ``full_state_dict`` (dss's DeepSpeed
checkpoint and expert side files) writes and restores the host rows.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import AbstractContextManager

import torch
import triton
import triton.language as tl
from torch import nn

# The job's ``loader_options.ngram_table_on_host`` (the qwen4_exp loader); None: off.
_configured: bool | None = None


def configure(enabled: bool | None) -> None:
    """Set this process's choice from the job's loader options (None: off)."""
    global _configured
    _configured = enabled


def host_table_enabled() -> bool:
    return bool(_configured)


@triton.jit
def _gather_rows_kernel(table_ptr, ids_ptr, out_ptr, dim, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    index = tl.load(ids_ptr + row).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < dim
    values = tl.load(table_ptr + index * dim + cols, mask=mask)
    tl.store(out_ptr + row * dim + cols, values, mask=mask)


def move_to_host(weight: torch.Tensor) -> torch.Tensor:
    """A pinned host copy of ``weight`` (row-major), for :func:`gather_rows`."""
    host = torch.empty(weight.shape, dtype=weight.dtype, pin_memory=True)
    host.copy_(weight)
    print(
        f"[ngram-host] {host.numel() * host.element_size() / 1e9:.1f} GB table shard in pinned host memory",
        file=sys.stderr,
        flush=True,
    )
    return host


def gather_rows(host_table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``host_table[ids]`` on ``ids.device``, read by the GPU directly from pinned host memory."""
    if not host_table.is_pinned():
        raise ValueError("gather_rows needs a pinned host table")
    rows, dim = ids.numel(), host_table.shape[1]
    out = torch.empty((rows, dim), dtype=host_table.dtype, device=ids.device)
    if rows:
        flat = ids.reshape(-1).contiguous()
        low, high = torch.stack([flat.min(), flat.max()]).tolist()  # a stray id would read past the host buffer
        if low < 0 or high >= host_table.shape[0]:
            raise IndexError(f"row ids {low}..{high} outside the host table's {host_table.shape[0]} rows")
        _gather_rows_kernel[(rows,)](host_table, flat, out, dim, BLOCK=triton.next_power_of_2(dim))
    return out.reshape(*ids.shape, dim)


class HostTables:
    """The n-gram tables a model keeps in pinned host memory, as a ``moe/frozen_held.py`` holding of its frozen weights.

    Each table (an ``EPShardedEmbedding``) moves its shard to ``_host_weight`` at its first forward and keeps an empty
    parameter. Inside ``whole_on_host`` that parameter views the host shard: the state dict, DeepSpeed's checkpoint and
    dss's expert side files, which read and write ``parameter.data``, see its rows, and a load writes them into the host
    shard. ``materialize`` points it there for the weight export too (``build_iter_full_hf_weights`` gathers a pinned
    shard through the GPU, a slice at a time). Outside them the parameter is empty again. A table that has not moved yet
    holds its rows in its parameter and needs neither.
    """

    def __init__(self, embeddings: Sequence[nn.Embedding]) -> None:
        self.embeddings = list(embeddings)
        self.whole = 0  # depth of whole_on_host

    def _moved(self) -> list[tuple[nn.Embedding, torch.Tensor]]:
        """``(embedding, host shard)`` of each table that moved."""
        return [(e, host) for e in self.embeddings if (host := getattr(e, "_host_weight", None)) is not None]

    @property
    def held(self) -> bool:
        return not self.whole and bool(self._moved())

    @contextlib.contextmanager
    def _on_host(self, moved: list[tuple[nn.Embedding, torch.Tensor]]) -> Iterator[None]:
        emptied = [(e, e.weight.data) for e, _ in moved]
        for e, host in moved:
            e.weight.data = host
        try:
            yield
        finally:
            for e, empty in emptied:
                e.weight.data = empty

    def materialize(self, tensors: Sequence[torch.Tensor]) -> AbstractContextManager[None]:
        wanted = {id(t) for t in tensors}
        return self._on_host([] if self.whole else [(e, host) for e, host in self._moved() if id(e.weight) in wanted])

    @contextlib.contextmanager
    def whole_on_host(self) -> Iterator[None]:
        moved = [] if self.whole else self._moved()
        self.whole += 1
        try:
            with self._on_host(moved):
                yield
        finally:
            self.whole -= 1


def hold_host_tables(root: nn.Module, embeddings: Sequence[nn.Embedding]) -> HostTables:
    """Register ``embeddings``' host tables with ``root``'s checkpoint and weight export (``moe/frozen_held.py``)."""
    from arctic_platform.model.implementations.moe.frozen_held import hold_frozen_weights

    tables = HostTables(embeddings)
    hold_frozen_weights(root, tables)
    return tables
