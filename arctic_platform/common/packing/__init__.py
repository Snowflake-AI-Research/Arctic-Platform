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

"""Token-budget packing: group rows into model calls, pack them varlen, split them across sequence ranks.

Every training path that packs rows has to agree on where a row begins and ends, so there is one contract
here. Entry points are pure functions of their arguments -- no process group, no rank lookups, no config
objects -- which is what lets dispatch and a worker call the same code from opposite sides of the wire.

This is a library copy of ``dss.ray_dss.jobs.gpu.packing``. DSS dispatch and the zone worker still import
that package. Nothing in ``run_pipeline`` calls this one. ``arctic_platform.rl.processors.packing``
(``pack_sequences``) is a different layout: it page-aligns, rewrites ``position_ids``, and does not walk a
nested ``context``. ``arctic_platform.model.implementations.gpu.packing`` re-exports ``IGNORE_INDEX`` and
``cu_seqlens_from_position_ids`` from here so the model stack and this package share one definition.

The contract, in the order the pieces get used:

1. ``token_budget_groups(valid_lengths, max_tokens_per_mb)`` decides which rows share a model call, and
   ``split_groups_to_count`` raises a plan to a required number of calls. Both read only the rows' token counts,
   so every rank derives the same plan from the same shard without communicating.
   ``token_budget_microbatch_groups(batch, max_tokens_per_mb, min_groups)`` applies the pair to a whole batch.
2. ``pack_microbatch(data)`` turns that group's valid tokens into one ``[1, T, ...]`` model call plus a
   ``PackMetadata`` recording where each row lives. Padding is gone: ``T`` is the sum of the rows' real lengths.
3. ``pad_packed_microbatch(packed, metadata, multiple)`` right-pads ``T`` to a multiple, which a token-axis split
   needs in order to hand every shard the same width.
4. ``token_shard(packed, metadata, shard_count, shard_index)`` takes one shard's contiguous token window.
   Concatenating the shards in index order reproduces the packed call exactly, which is what lets attention
   rebuild the global varlen boundaries from gathered ``position_ids``.
5. ``window_row_pieces`` cuts one shard's window at the row boundaries, which is how a worker answers with row
   pieces; ``unpack_output`` is the whole-call inverse, mapping a complete ``[1, T, ...]`` call back to
   ``[B, S, ...]`` rows.

``shard_count == 1`` is the case where one shard owns the whole packed call. Sequence-parallel and plain
data-parallel training share this code.

Invariants: rows are left-aligned; ``labels`` pad with ``IGNORE_INDEX`` and ``temperature`` pads with ``1.0``;
``attention_mask`` is dropped on pack; ``cu_seqlens`` live on ``PackMetadata``, not in the packed dict; a
window has one segment per packed row, empty when the shard misses that row, with tail padding folded into
the last segment.
"""

from .groups import MicrobatchSplit
from .groups import batch_input_ids
from .groups import batch_num_rows
from .groups import batch_position_ids
from .groups import batch_seqlen
from .groups import batch_tensor
from .groups import select_rows
from .groups import shard_valid_lengths
from .groups import singleton_microbatch_groups
from .groups import split_groups_to_count
from .groups import split_microbatches
from .groups import token_budget_groups
from .groups import token_budget_microbatch_groups
from .groups import token_validity_mask
from .pack import IGNORE_INDEX
from .pack import PackMetadata
from .pack import pack_microbatch
from .pack import packed_token_index
from .pack import pad_packed_microbatch
from .pack import sequence_pad_value
from .pack import unpack_output
from .walk import DROP
from .walk import map_batch_values
from .windows import cu_seqlens_from_position_ids
from .windows import token_shard
from .windows import token_window
from .windows import window_cu_seqlens
from .windows import window_row_pieces

__all__ = [
    "DROP",
    "cu_seqlens_from_position_ids",
    "IGNORE_INDEX",
    "MicrobatchSplit",
    "PackMetadata",
    "batch_input_ids",
    "batch_num_rows",
    "batch_position_ids",
    "batch_seqlen",
    "batch_tensor",
    "map_batch_values",
    "pack_microbatch",
    "packed_token_index",
    "pad_packed_microbatch",
    "select_rows",
    "shard_valid_lengths",
    "split_groups_to_count",
    "sequence_pad_value",
    "singleton_microbatch_groups",
    "split_microbatches",
    "token_budget_groups",
    "token_budget_microbatch_groups",
    "token_shard",
    "token_validity_mask",
    "token_window",
    "unpack_output",
    "window_cu_seqlens",
    "window_row_pieces",
]
