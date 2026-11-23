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

"""Attend to each rank's interleaved bf16 KV shard and return its LSE for the DCP merge."""

from functools import cache

from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend, FlashMLASparseImpl
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_filter_and_convert_dcp_index,
)

from arctic_platform.inference.patching import ArcticPatch


class _SparseImpl(FlashMLASparseImpl):
    def __init__(self, *args, **kwargs):
        # The upstream bf16 refusal is the only constructor reader. forward_mqa needs the original flag.
        returns_lse = self.need_to_return_lse_for_decode
        self.need_to_return_lse_for_decode = False
        super().__init__(*args, **kwargs)
        self.need_to_return_lse_for_decode = returns_lse

    def _forward_bf16_kv(self, q, kv_c_and_k_pe_cache, topk_indices, attn_metadata, actual_num_heads):
        if self.dcp_world_size == 1:
            return super()._forward_bf16_kv(q, kv_c_and_k_pe_cache, topk_indices, attn_metadata, actual_num_heads)
        kv_rows, block_stride_rows = flat_kv_row_view(kv_c_and_k_pe_cache, attn_metadata.block_size)
        local_topk, local_count = triton_filter_and_convert_dcp_index(
            attn_metadata.req_id_per_token[: topk_indices.shape[0]],
            attn_metadata.block_table,
            topk_indices,
            dcp_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
            cp_kv_cache_interleave_size=attn_metadata.cp_kv_cache_interleave_size,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        # A row selecting none of this rank's tokens gets zero output and +inf LSE, which the DCP merge weighs zero.
        out, lse = self._bf16_flash_mla_kernel(q, kv_rows, local_topk, local_count, actual_num_heads)
        return out.contiguous(), lse


class _FlashMLABackendPatch(ArcticPatch[FlashMLASparseBackend]):
    # Subclassed, not patched in place: the HY V4 sink impl inherits FlashMLASparseImpl and keeps its DCP refusal.
    @staticmethod
    def get_impl_cls():
        return _SparseImpl


@cache
def ensure_flashmla_sparse_bf16_dcp() -> None:
    _FlashMLABackendPatch.apply_patch()
