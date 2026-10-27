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
"""

from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl

HOST_TABLE_ENV = "ARL_NGRAM_TABLE_ON_HOST"


def host_table_enabled() -> bool:
    return os.environ.get(HOST_TABLE_ENV, "0") == "1"


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
    print(f"[ngram-host] {host.numel() * host.element_size() / 1e9:.1f} GB table shard in pinned host memory",
          file=sys.stderr, flush=True)
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
