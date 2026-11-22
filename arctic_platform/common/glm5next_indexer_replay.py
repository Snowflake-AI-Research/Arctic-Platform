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

"""Layout of the GLM-5.3 pool selections appended to routed-expert records."""

import torch

SPARSE_LAYER_TYPES = ("deepseek_sparse_attention", "indexed_attention")


def pool_rows(text_config) -> int | None:
    """Capture rows per sparse-attention layer; None when the selected pools do not fill whole rows."""
    if text_config.index_topk is None:
        return None
    pools = text_config.index_topk // text_config.index_kpool
    rows, remainder = divmod(pools, text_config.num_experts_per_tok)
    return None if remainder else rows


def sparse_layers(text_config) -> list[int]:
    return [i for i, kind in enumerate(text_config.layer_types) if kind in SPARSE_LAYER_TYPES]


def capture_pools(capture, first_row: int, topk_indices: torch.Tensor, topk_tokens: int, kpool: int, top_k: int):
    """Read each complete pool's last token. One-based pool IDs reserve zero for missing capture rows."""
    last_tokens = topk_indices[:, kpool - 1 : topk_tokens : kpool]
    pools = torch.div(last_tokens, kpool, rounding_mode="floor") + 1
    for row, ids in enumerate(pools.split(top_k, dim=1)):
        capture(first_row + row, ids)
