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

"""Serve a bf16 sparse-MLA KV cache under decode context parallelism (DCP).

Under DCP each rank of the group stores every DCP-size-th token of a request,
and the attention layer merges the ranks' partial results by their
log-sum-exp. vLLM's FlashMLA sparse backend does this for its fp8_ds_mla cache
only: its bf16 path refuses DCP > 1 at construction and, past that, converts
each token's top-k positions through the full-sequence slot rule. Here the
bf16 path keeps the top-k positions this rank stores, as the fp8 path does,
attends them with the same bf16 kernel, and returns the kernel's log-sum-exp.
DCP = 1 is unchanged.
"""

from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_filter_and_convert_dcp_index,
)

from arctic_platform.inference.patching import ArcticPatch


class FlashMLASparseBf16DcpPatch(ArcticPatch[FlashMLASparseImpl]):

    _orig_init = FlashMLASparseImpl.__init__
    _orig_forward_bf16_kv = FlashMLASparseImpl._forward_bf16_kv

    def __init__(self, *args, **kwargs):
        # The bf16 refusal is the constructor's only reader of this flag.
        returns_lse = self.need_to_return_lse_for_decode
        self.need_to_return_lse_for_decode = False
        try:
            self._orig_init(*args, **kwargs)
        finally:
            self.need_to_return_lse_for_decode = returns_lse

    def _forward_bf16_kv(self, q, kv_c_and_k_pe_cache, topk_indices, attn_metadata, actual_num_heads):
        if self.dcp_world_size == 1:
            return self._orig_forward_bf16_kv(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata, actual_num_heads
            )
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
        out, lse = self._bf16_flash_mla_kernel(q, kv_rows, local_topk, local_count, actual_num_heads)
        # (0, -inf) drops this rank from the merge of a token whose selection it holds none of.
        empty = local_count == 0
        out.masked_fill_(empty.view(-1, 1, 1), 0.0)
        lse.masked_fill_(empty.view(-1, 1), float("-inf"))
        return out.contiguous(), lse


_PATCHED = False


def ensure_flashmla_sparse_bf16_dcp() -> None:
    global _PATCHED
    if not _PATCHED:
        FlashMLASparseBf16DcpPatch.apply_patch()
        _PATCHED = True
