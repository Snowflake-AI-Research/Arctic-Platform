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
import torch

IGNORE_INDEX = -100


def cu_seqlens_from_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Varlen boundaries of a packed sequence, read back out of its ``position_ids``.

    Packing writes each row's positions starting from 0, so a 0 anywhere but the first token marks the start of
    the next row. That makes the positions a self-describing layout: whoever holds them can rebuild the segment
    boundaries without being told the row lengths, which is what lets a rank recover the *global* boundaries
    from all-gathered positions after its own window has lost sight of where its rows began.

    Padding appended by ``pad_packed_microbatch`` carries position 0, so each pad token reads as its own
    single-token segment rather than extending the row it follows.
    """
    if not torch.is_tensor(position_ids) or position_ids.ndim < 1:
        raise ValueError(f"position_ids must be a tensor with at least one dimension, got {position_ids!r}")
    flat = position_ids[0] if position_ids.ndim == 3 else position_ids
    flat = flat.reshape(-1)
    if flat.numel() == 0:
        return torch.zeros(1, dtype=torch.int32, device=position_ids.device)
    segment_lengths = torch.cat([flat[0:1], flat[:-1][(flat == 0)[1:]] + 1, flat[-1:] + 1])
    return segment_lengths.cumsum(dim=0, dtype=torch.int32)
