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

"""The platform's attention core, which plugin attention blocks inherit instead of writing their own."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from arctic_platform.model_plugin.parallel import FULL_SHARD
from arctic_platform.model_plugin.parallel import HeadShard


def segment_ids(cu_seqlens: torch.Tensor, length: int) -> torch.Tensor:
    """The packed-row index of each of ``length`` tokens."""
    positions = torch.arange(length, device=cu_seqlens.device)
    return torch.bucketize(positions, cu_seqlens[1:].to(positions.dtype), right=True)


def varlen_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Causal attention within each packed row. ``q`` is ``[1, T, H, D]``; ``k``/``v`` are ``[1, T, Hkv, D]``.

    With ``sliding_window`` a token attends to itself and the ``sliding_window - 1`` tokens before it.
    """
    length, heads, kv_heads = q.shape[1], q.shape[2], k.shape[2]
    if heads % kv_heads != 0:
        raise ValueError(f"{heads} query heads are not a multiple of {kv_heads} key/value heads")
    k = k.repeat_interleave(heads // kv_heads, dim=2)
    v = v.repeat_interleave(heads // kv_heads, dim=2)
    rows = segment_ids(cu_seqlens, length)
    causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
    if sliding_window is not None:
        causal = causal.triu(-(sliding_window - 1))
    mask = causal & (rows[:, None] == rows[None, :])
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, scale=scale
    )
    return out.transpose(1, 2)


class AttentionModule(nn.Module):
    """Base class for a plugin's full-attention block.

    A subclass sets ``ap_layer_type``, ``num_heads``, ``num_kv_heads``, ``head_dim`` and optionally
    ``sliding_window``, computes q/k/v (including any positional encoding) from its local tokens, and calls
    ``self.mix(q, k, v, cu_seqlens=cu_seqlens)``. It must not override :meth:`mix`: the platform owns that core,
    swaps in its fused kernels, and makes it sequence-parallel.
    """

    ap_layer_type: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    sliding_window: int | None = None

    def mix(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        shard: HeadShard = FULL_SHARD,
    ) -> torch.Tensor:
        return varlen_causal_attention(
            q, k, v, cu_seqlens, scale=self.head_dim**-0.5, sliding_window=self.sliding_window
        )
