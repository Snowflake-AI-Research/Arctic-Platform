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

"""DeepSeek-V4.1 shared sparse attention.

Ratio 0 attends over the sliding window only. Positive ratios also read a compressed KV that
only ``kv_source_layer_ids`` produce: ratio 2 pools disjoint token pairs with an fp32 gate,
ratio 1 is ``norm(wkv(x))``. Later layers reuse that latent. Indexers live only on
``index_source_layer_ids`` and score the source latent with ``wq_b`` / ``weights_proj``; the
owning indexer also maps it through ``wk`` / ``k_norm``. The candidate source publishes a
block-max mask that later indexers apply before their own top-k.

The window and the selected compressed entries are one sparse attention. On GPU, when the
differentiable V4 kernel can run the shape, that kernel does the gather. CPU and every other
shape use an eager softmax with the same sinks. Shared tensors are returned to the caller;
nothing is stashed on the module, so a checkpoint recompute receives them as inputs.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._pytree import register_dataclass

from arctic_platform.model.implementations.deepseek_v4.attention import PackedContext
from arctic_platform.model.implementations.deepseek_v4.attention import SparseAttnInputs
from arctic_platform.model.implementations.deepseek_v4.cp import CPContext
from arctic_platform.model.implementations.deepseek_v4.cp import gather_for_cp
from arctic_platform.model.implementations.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from arctic_platform.model.implementations.deepseek_v41.configuration_deepseek_v41 import DeepseekV41Config
from arctic_platform.model.implementations.kernels.deepseek_v4 import IGNORE_SLOT
from arctic_platform.model.implementations.kernels.deepseek_v4.index_scores import index_scores
from arctic_platform.model.implementations.kernels.deepseek_v4.index_scores import index_scores_applies
from arctic_platform.model.implementations.kernels.deepseek_v4.index_scores import split_keys
from arctic_platform.model.implementations.layers.grouped_linear import GroupedLinear
from arctic_platform.model.implementations.layers.norms import RMSNorm
from arctic_platform.model.implementations.layers.norms import RMSNormConfig

try:
    from arctic_platform.model.implementations.kernels.deepseek_v4.dsv4_sparse_attn import dsv4_sparse_attn
    from arctic_platform.model.implementations.kernels.deepseek_v4.dsv4_sparse_attn import sparse_attn_shape_error
except ImportError:
    dsv4_sparse_attn = None
    sparse_attn_shape_error = None


@dataclass
class _WindowConfig:
    """The slice of a config ``PackedContext.build`` reads."""

    sliding_window: int
    max_position_embeddings: int
    layer_types: tuple[str, ...] = ()
    compress_rates: dict | None = None

    def __post_init__(self) -> None:
        if self.compress_rates is None:
            self.compress_rates = {}


@dataclass
class _WindowRotary:
    config: _WindowConfig


@dataclass
class SharedAttentionState:
    """Compressed KV, index keys, top-k, and the candidate mask published by earlier layers.

    Keys are source layer ids. Values are this rank's tensors: compressed KV and index keys
    cover the whole packed sequence, top-k and candidates cover this rank's queries.
    """

    compress_kv: dict[int, torch.Tensor]
    index_k: dict[int, torch.Tensor]
    topk: dict[int, torch.Tensor]
    candidates: torch.Tensor | None = None

    @classmethod
    def empty(cls) -> "SharedAttentionState":
        return cls(compress_kv={}, index_k={}, topk={}, candidates=None)

    def copy(self) -> "SharedAttentionState":
        return SharedAttentionState(
            compress_kv=dict(self.compress_kv),
            index_k=dict(self.index_k),
            topk=dict(self.topk),
            candidates=self.candidates,
        )


register_dataclass(SharedAttentionState)


def build_packed(
    config: DeepseekV41Config,
    seq_lens: torch.Tensor,
    device: torch.device,
    *,
    cp_rank: int = 0,
    cp_world_size: int = 1,
) -> PackedContext:
    """Document-clipped window, positions, and per-ratio entry layout for one packed row."""
    rates = {str(rate): rate for rate in sorted({rate for rate in config.compress_ratios if rate > 0})}
    rotary = _WindowRotary(
        _WindowConfig(
            sliding_window=config.sliding_window,
            max_position_embeddings=config.max_position_embeddings,
            layer_types=tuple(rates),
            compress_rates=rates,
        )
    )
    return PackedContext.build(
        rotary_emb=rotary,
        seq_lens=seq_lens,
        device=device,
        cp_rank=cp_rank,
        cp_world_size=cp_world_size,
    )


def apply_interleaved_rope(
    hidden: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate the trailing rotary pairs of each head. ``inverse`` undoes that rotation."""
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    positions = position_ids.reshape(-1).to(dtype=torch.long)
    cache = cos_sin_cache[positions].view(*position_ids.shape, rope_dim)
    cos = cache[..., :half].unsqueeze(-2)
    sin = cache[..., half:].unsqueeze(-2)
    if inverse:
        sin = -sin
    rope = hidden[..., -rope_dim:].float()
    first = rope[..., 0::2]
    second = rope[..., 1::2]
    rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1).flatten(-2)
    if rope_dim < hidden.shape[-1]:
        prefix = hidden[..., :-rope_dim].float()
    else:
        prefix = hidden.new_empty(*hidden.shape[:-1], 0, dtype=torch.float32)
    return torch.cat((prefix, rotated), dim=-1).to(dtype=hidden.dtype)


def _rotate(hidden, cache, position_ids, *, inverse: bool = False) -> torch.Tensor:
    if hidden.is_cuda and hidden.numel() > 0:
        from arctic_platform.model.implementations.kernels.deepseek_v4.dsv4_rope import dsv4_rope

        return dsv4_rope(hidden, cache, position_ids, inverse=inverse)
    return apply_interleaved_rope(hidden, cache, position_ids, inverse=inverse)


def _latest_source(layer_ids: list[int], layer_idx: int) -> int:
    earlier = [layer_id for layer_id in layer_ids if layer_id <= layer_idx]
    if not earlier:
        raise ValueError(f"layer {layer_idx} has no source at or below it in {layer_ids}")
    return max(earlier)


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Keep the highest-scoring blocks per query. The block holding the newest position is pinned."""
    width = logits.size(-1)
    if width == 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    pad = (-width) % block_size
    scores = F.pad(logits, (0, pad), value=float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    block_ids = torch.arange(num_blocks, device=logits.device)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(block_ids == last.unsqueeze(-1), float("inf"))
    chosen = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, chosen.indices, chosen.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def packed_candidate_mask(
    scores: torch.Tensor,
    layout,
    tok_doc_idx: torch.Tensor,
    causal_len: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Per-document block mask, scattered back onto the packed entry axis. ``scores`` is ``(queries, entries)``."""
    queries, entries = scores.shape
    max_entries = layout.max_entries_per_doc
    if max_entries == 0 or queries == 0:
        return torch.zeros(queries, entries, dtype=torch.bool, device=scores.device)
    counts = torch.zeros(layout.first_entry_of_doc.shape[0], dtype=torch.long, device=scores.device)
    counts.scatter_add_(0, layout.entry_doc_idx, torch.ones_like(layout.entry_doc_idx))
    doc_counts = counts[tok_doc_idx]
    local = torch.arange(max_entries, device=scores.device)
    index = layout.first_entry_of_doc[tok_doc_idx][:, None] + local[None, :]
    valid = local[None, :] < doc_counts[:, None]
    gathered = scores.gather(1, index.clamp(max=max(entries - 1, 0))).masked_fill(~valid, float("-inf"))
    gathered = gathered.masked_fill(local[None, :] >= causal_len[:, None], float("-inf"))
    local_mask = select_candidate_blocks(gathered, causal_len, topk_blocks, block_size)
    # Slots past a document's entries all land in a spare column, so the real columns of a row are distinct.
    mask = torch.zeros(queries, entries + 1, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, index.masked_fill(~valid, entries), local_mask & valid)
    return mask[:, :entries]


def _fp32_weight(linear: nn.Linear) -> None:
    linear.weight = nn.Parameter(linear.weight.detach().to(dtype=torch.float32))


class DeepseekV41Compressor(nn.Module):
    """Pre-RoPE compressed latent. Ratio 2 softmax-pools disjoint pairs; ratio 1 projects each token."""

    def __init__(self, config: DeepseekV41Config, compress_ratio: int):
        super().__init__()
        if compress_ratio not in (1, 2):
            raise ValueError(f"DeepSeek V4.1 compresses at ratio 1 or 2, got {compress_ratio}")
        self.compress_ratio = compress_ratio
        self.head_dim = config.head_dim
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.head_dim, eps=config.rms_norm_eps))
        if compress_ratio > 1:
            self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False)
            _fp32_weight(self.wkv)
            _fp32_weight(self.wgate)

    def forward(self, hidden: torch.Tensor, packed: PackedContext, gather) -> torch.Tensor:
        layout = packed.compression_layouts[self.compress_ratio]
        entries = layout.entry_tok_idx.shape[0]
        if self.compress_ratio == 1:
            projected = gather(self.wkv(hidden))
            if entries == 0:
                return projected.new_empty(projected.shape[0], 0, self.head_dim)
            return self.norm(projected[:, layout.entry_tok_idx[:, 0]])

        kv = gather(F.linear(hidden.float(), self.wkv.weight.float()))
        gate = gather(F.linear(hidden.float(), self.wgate.weight.float()))
        if entries == 0:
            return hidden.new_empty(hidden.shape[0], 0, self.head_dim)
        pooled = kv[:, layout.entry_tok_idx] * gate[:, layout.entry_tok_idx].softmax(dim=2)
        return self.norm(pooled.sum(dim=2).to(dtype=hidden.dtype))


class DeepseekV41Indexer(nn.Module):
    """Top-k compressed positions. ``owns_k`` projects the shared pre-RoPE latent; other indexers reuse it."""

    def __init__(self, config: DeepseekV41Config, layer_idx: int, compress_ratio: int, owns_k: bool):
        super().__init__()
        if config.qk_rope_head_dim > config.index_head_dim:
            raise ValueError(
                f"qk_rope_head_dim ({config.qk_rope_head_dim}) exceeds index_head_dim ({config.index_head_dim})"
            )
        self.layer_idx = layer_idx
        self.compress_ratio = compress_ratio
        self.owns_k = owns_k
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.kv_source_layer_id = _latest_source(config.kv_source_layer_ids, layer_idx)
        self.is_candidate_source = layer_idx == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_idx
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        if self.is_candidate_source and (self.candidate_block_size < 1 or self.candidate_topk_blocks < 1):
            raise ValueError("the candidate source needs a positive block size and block top-k")
        self.score_scale = self.head_dim**-0.5 * self.num_heads**-0.5
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        if owns_k:
            self.wk = nn.Linear(config.head_dim, self.head_dim, bias=False)
            self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))

    def forward(
        self,
        hidden: torch.Tensor,
        query_residual: torch.Tensor,
        latent: torch.Tensor | None,
        packed: PackedContext,
        shared: SharedAttentionState,
        rope_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        layout = packed.compression_layouts[self.compress_ratio]
        if self.owns_k:
            if latent is None:
                raise RuntimeError(f"layer {self.layer_idx} owns index keys but produced no latent")
            keys = self.k_norm(self.wk(latent))
            positions = (layout.entry_local_idx * self.compress_ratio).view(1, -1)
            keys = _rotate(keys.unsqueeze(2), rope_cache, positions).squeeze(2)
        else:
            keys = shared.index_k.get(self.kv_source_layer_id)
            if keys is None:
                raise RuntimeError(
                    f"layer {self.layer_idx} is missing index keys from layer {self.kv_source_layer_id}"
                )

        batch, queries, _ = hidden.shape
        query = self.wq_b(query_residual).view(batch, queries, self.num_heads, self.head_dim)
        query = _rotate(query, rope_cache, packed.position_ids)
        weights = self.weights_proj(hidden) * self.score_scale
        prior_candidates = shared.candidates
        if self.uses_candidates and not self.is_candidate_source and prior_candidates is None:
            raise RuntimeError(f"layer {self.layer_idx} expected a candidate mask from an earlier indexer")
        # Every per-query tensor below spans the whole sequence's entries, so at long context
        # they are only ever materialized for a slice of queries.
        entries = keys.shape[1]
        fused = batch == 1 and index_scores_applies(query, weights)
        if fused:
            # The fused scores keep no per-head intermediate, so the slice is sized by the scores themselves.
            chunk = max(1, _INDEX_CHUNK_BYTES // max(1, entries * 4))
            score_keys = split_keys(keys[0])
        else:
            chunk = max(1, _INDEX_CHUNK_BYTES // max(1, batch * self.num_heads * entries * 4))
            score_keys = keys.float()
        topk = torch.empty(queries, self.index_topk, dtype=torch.long, device=hidden.device)
        candidates = torch.empty(
            batch, queries, entries if self.is_candidate_source else 0, dtype=torch.bool, device=hidden.device
        )
        for start in range(0, queries, chunk):
            rows = slice(start, min(start + chunk, queries))
            if fused:
                scores = index_scores(query[0, rows], score_keys, weights[0, rows])
            else:
                scores = _index_scores(query[:, rows], score_keys, weights[:, rows])[0]
            tok_doc_idx = packed.tok_doc_idx[rows]
            visible, causal_len = _visible_entries(
                layout, packed.position_ids[0, rows], tok_doc_idx, self.compress_ratio
            )
            if self.is_candidate_source:
                candidates[0, rows] = packed_candidate_mask(
                    scores,
                    layout,
                    tok_doc_idx,
                    causal_len,
                    self.candidate_topk_blocks,
                    self.candidate_block_size,
                )
            elif self.uses_candidates and prior_candidates is not None:
                visible = visible & prior_candidates[0, rows]
            topk[rows] = _topk_indices(scores, visible, self.index_topk)
        return topk.unsqueeze(0), keys if self.owns_k else None, candidates if self.is_candidate_source else None


_INDEX_CHUNK_BYTES = 1 << 30


def _index_scores(query: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Head-weighted ReLU scores ``(b, q, e)``."""
    scores = torch.einsum("bqhd,bed->bqhe", query.float(), keys.float()).relu()
    return (scores * weights.float().unsqueeze(-1)).sum(dim=2)


def _visible_entries(
    layout, position_ids: torch.Tensor, tok_doc_idx: torch.Tensor, ratio: int
) -> tuple[torch.Tensor, torch.Tensor]:
    causal_len = (position_ids + 1) // ratio
    same_doc = layout.entry_doc_idx[None, :] == tok_doc_idx[:, None]
    reached = layout.entry_local_idx[None, :] < causal_len[:, None]
    return same_doc & reached, causal_len


def _topk_indices(scores: torch.Tensor, visible: torch.Tensor, k: int) -> torch.Tensor:
    masked = scores.masked_fill(~visible, float("-inf"))
    queries, entries = masked.shape
    picks = torch.full((queries, k), IGNORE_SLOT, dtype=torch.long, device=scores.device)
    if k == 0 or entries == 0:
        return picks
    take = min(k, entries)
    values, indices = masked.topk(take, dim=-1)
    picks[:, :take] = torch.where(torch.isfinite(values), indices, IGNORE_SLOT)
    return picks


def _eager_sparse(
    query: torch.Tensor,
    kv_buf: torch.Tensor,
    indices: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    slots = indices[0, :, 0].to(dtype=torch.long)
    valid = slots >= 0
    gathered = kv_buf[0, slots.clamp(min=0), 0] * valid.unsqueeze(-1)
    scores = torch.einsum("thd,tsd->ths", query[0].float(), gathered.float()) * scale
    scores = scores.masked_fill(~valid[:, None, :], torch.finfo(scores.dtype).min)
    sink = sinks.float()
    row_max = torch.maximum(scores.amax(dim=-1), sink.view(1, -1))
    exp_scores = torch.exp(scores - row_max.unsqueeze(-1)) * valid[:, None, :]
    denom = exp_scores.sum(dim=-1) + torch.exp(sink.view(1, -1) - row_max)
    attended = torch.einsum("ths,tsd->thd", exp_scores / denom.unsqueeze(-1), gathered.float())
    return attended.view_as(query).to(dtype=query.dtype)


class DeepseekV41Attention(nn.Module):
    """Sliding window plus, when the ratio is positive, top-k compressed entries from the shared source."""

    def __init__(self, config: DeepseekV41Config, layer_idx: int, rotary_emb: DeepseekV4RotaryEmbedding):
        super().__init__()
        if config.attention_dropout != 0.0:
            raise ValueError("DeepSeek V4.1 attention implements no dropout")
        ratio = int(config.compress_ratios[layer_idx])
        if ratio not in (0, 1, 2):
            raise ValueError(f"layer {layer_idx} compress_ratio must be 0, 1, or 2, got {ratio}")
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = ratio
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_layer_type = "main" if ratio == 0 else "compress"
        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids
        if ratio == 0 and (self.is_kv_source or self.is_index_source):
            raise ValueError(f"layer {layer_idx} is a sparse source but its compress ratio is 0")
        self.kv_source_layer_id = _latest_source(config.kv_source_layer_ids, layer_idx) if ratio else None
        self.index_source_layer_id = _latest_source(config.index_source_layer_ids, layer_idx) if ratio else None
        self.dsa_backend = getattr(config, "dsa_backend", "tilelang")
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(RMSNormConfig(hidden_size=config.q_lora_rank, eps=config.rms_norm_eps))
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
        self.o_a_proj = GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))
        self.rotary_emb = rotary_emb
        self.cp_context = CPContext()
        self.compressor = DeepseekV41Compressor(config, ratio) if self.is_kv_source else None
        self.indexer = (
            DeepseekV41Indexer(config, layer_idx, ratio, owns_k=self.is_kv_source) if self.is_index_source else None
        )

    def _gather_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.cp_context.cp_enabled:
            return tensor
        if self.cp_context.cp_group is None:
            raise RuntimeError("context parallel is enabled but no process group is installed")
        return gather_for_cp(tensor, self.cp_context.cp_group)

    def _attend(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        compressed_kv: torch.Tensor | None,
        topk: torch.Tensor | None,
        packed: PackedContext,
    ) -> torch.Tensor:
        inputs = SparseAttnInputs.build(
            kv=kv,
            compressed_kv=None if compressed_kv is None else compressed_kv.unsqueeze(1),
            top_k_indices=topk,
            window_indices=packed.window_indices,
        )
        if _kernel_ready(query, inputs.kv_buf, self.dsa_backend):
            attended, _ = dsv4_sparse_attn(
                query.contiguous(),
                inputs.kv_buf.contiguous(),
                inputs.indices.contiguous(),
                self.sinks,
                self.scaling,
                backend=self.dsa_backend,
                max_slots=self.config.sliding_window + self.config.index_topk,
            )
            return attended
        return _eager_sparse(query, inputs.kv_buf, inputs.indices, self.sinks, self.scaling)

    def forward(
        self,
        hidden_states: torch.Tensor,
        packed: PackedContext,
        shared: SharedAttentionState | None = None,
    ) -> tuple[torch.Tensor, SharedAttentionState]:
        batch, seq_len, _ = hidden_states.shape
        if batch != 1:
            raise ValueError(f"DeepSeek V4.1 attention expects a packed batch of size 1, got {batch}")
        shared = SharedAttentionState.empty() if shared is None else shared.copy()
        cache = self.rotary_emb.cos_sin_cache(self.rope_layer_type)
        query_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(query_residual).view(batch, seq_len, self.num_heads, self.head_dim)
        query = _rotate(query, cache, packed.position_ids)
        kv = self.kv_norm(self.kv_proj(hidden_states)).view(batch, seq_len, 1, self.head_dim)
        kv = _rotate(kv, cache, packed.position_ids)
        kv = self._gather_sequence(kv).transpose(1, 2)

        compressed = None
        topk = None
        if self.compress_ratio:
            kv_source_layer_id = self.kv_source_layer_id
            index_source_layer_id = self.index_source_layer_id
            if kv_source_layer_id is None or index_source_layer_id is None:
                raise RuntimeError(f"layer {self.layer_idx} compresses without a KV or index source")
            layout = packed.compression_layouts[self.compress_ratio]
            latent = None
            if self.is_kv_source:
                if self.compressor is None:
                    raise RuntimeError(f"layer {self.layer_idx} is a KV source without a compressor")
                latent = self.compressor(hidden_states, packed, self._gather_sequence)
            if self.is_index_source:
                if self.indexer is None:
                    raise RuntimeError(f"layer {self.layer_idx} is an index source without an indexer")
                with torch.no_grad():
                    topk, index_k, candidates = self.indexer(
                        hidden_states, query_residual, latent, packed, shared, cache
                    )
                if index_k is not None:
                    shared.index_k[kv_source_layer_id] = index_k
                shared.topk[self.layer_idx] = topk
                if candidates is not None:
                    shared.candidates = candidates
            else:
                topk = shared.topk.get(index_source_layer_id)
                if topk is None:
                    raise RuntimeError(
                        f"layer {self.layer_idx} is missing top-k from index source {index_source_layer_id}"
                    )
            if self.is_kv_source:
                if latent is None:
                    raise RuntimeError(f"layer {self.layer_idx} produced no compressed latent")
                positions = (layout.entry_local_idx * self.compress_ratio).view(1, -1)
                compressed = _rotate(latent.unsqueeze(2), cache, positions).squeeze(2)
                shared.compress_kv[self.layer_idx] = compressed
            else:
                compressed = shared.compress_kv.get(kv_source_layer_id)
                if compressed is None:
                    raise RuntimeError(
                        f"layer {self.layer_idx} is missing compressed KV from layer {kv_source_layer_id}"
                    )
            if compressed.shape[1] != layout.entry_tok_idx.shape[0]:
                raise RuntimeError(
                    f"layer {self.layer_idx} expected {layout.entry_tok_idx.shape[0]} entries from "
                    f"layer {kv_source_layer_id}, got {compressed.shape[1]}"
                )

        attended = self._attend(query, kv, compressed, topk, packed)
        attended = _rotate(attended, cache, packed.position_ids, inverse=True)
        grouped = self.o_a_proj(attended.reshape(batch, seq_len, self.config.o_groups, -1)).flatten(2)
        return self.o_b_proj(grouped), shared

    def init_weights(self, init_std: float) -> None:
        del init_std
        nn.init.zeros_(self.sinks)


def _kernel_ready(query: torch.Tensor, kv_buf: torch.Tensor, backend: str) -> bool:
    if dsv4_sparse_attn is None or sparse_attn_shape_error is None:
        return False
    if not query.is_cuda or query.dtype != torch.bfloat16:
        return False
    heads, kv_group, dim = query.shape[2], kv_buf.shape[2], query.shape[3]
    return sparse_attn_shape_error(heads, kv_group, dim, backend) is None


__all__ = [
    "DeepseekV41Attention",
    "DeepseekV41Compressor",
    "DeepseekV41Indexer",
    "SharedAttentionState",
    "apply_interleaved_rope",
    "build_packed",
    "packed_candidate_mask",
    "select_candidate_blocks",
]
