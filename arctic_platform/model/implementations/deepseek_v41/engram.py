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

"""DeepSeek-V4.1 Engram: n-gram hashes and a frozen expert-parallel embedding.

Hashes follow the V4.1 / vLLM 0.31 layout. Each engram layer owns one row-sharded
table. Quantized tables keep float8 weights and UE8M0 block scales; only the rows
a caller asks for are dequantized. Unquantized float tables are the tiny-test path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch import nn

from arctic_platform.model.implementations.deepseek_v41.configuration_deepseek_v41 import DeepseekV41Config

DEAD_ID = -1
_FP8_BLOCK = 32
_MILLER_RABIN_BASES = (2, 7, 61)


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for the prime range these tables use."""
    if n < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % prime == 0:
            return n == prime
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for base in _MILLER_RABIN_BASES:
        x = pow(base, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """Smallest prime strictly above ``start`` that has not been handed out."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def compute_hash_multipliers(layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int) -> Tensor:
    """Odd per-layer multipliers. The seed is ``10007 * layer_id`` and the bound uses the compressed vocab."""
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1, dtype=torch.long))
    return torch.stack(rows)


def ep_row_shard(num_rows: int, world_size: int, rank: int) -> tuple[int, int]:
    """Row range ``DeepEPShardParallel`` keeps on ``rank``.

    DTensor ``Shard(0)`` uses a ceil chunk and leaves the short tail on the last rank.
    """
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid EP rank {rank} of world {world_size}")
    chunk = (num_rows + world_size - 1) // world_size
    start = min(num_rows, chunk * rank)
    end = min(num_rows, start + chunk)
    return start, end


def dequantize_block_fp8(rows: Tensor, scales: Tensor, block_size: int = _FP8_BLOCK) -> Tensor:
    """Dequantize already-gathered FP8 rows. ``scales`` is ``float8_e8m0fnu`` and decodes with ``.float()``."""
    if rows.shape[-1] % block_size != 0:
        raise ValueError(f"engram head dim {rows.shape[-1]} is not a multiple of {block_size}")
    if scales.shape[-1] != rows.shape[-1] // block_size or scales.shape[:-1] != rows.shape[:-1]:
        raise ValueError(f"scale shape {tuple(scales.shape)} does not match rows {tuple(rows.shape)}")
    values = rows.float().unflatten(-1, (-1, block_size))
    restored = values * scales.float().unsqueeze(-1)
    return restored.flatten(-2).to(dtype=torch.bfloat16)


def _index_packed(param: Tensor, local_ids: Tensor) -> Tensor:
    return param.view(torch.uint8).index_select(0, local_ids).view(param.dtype)


@dataclass(frozen=True)
class EngramLayout:
    """Prime buckets for 2-gram .. max-gram, ``n_heads`` each, disjoint across engram layers."""

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    offsets: tuple[tuple[int, ...], ...]
    n_heads: int
    head_dim: int
    compressed_vocab_size: int

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @classmethod
    def from_config(cls, config: DeepseekV41Config) -> EngramLayout | None:
        layer_ids = tuple(int(layer_id) for layer_id in config.engram_layer_ids)
        if not layer_ids:
            return None
        num_embeddings = tuple(int(size) for size in config.engram_num_embeddings)
        if len(num_embeddings) != len(layer_ids):
            raise ValueError(
                f"engram_num_embeddings ({len(num_embeddings)}) must match engram_layer_ids ({len(layer_ids)})"
            )
        max_ngram_size = int(config.engram_max_ngram_size)
        n_heads = int(config.engram_n_heads)
        if max_ngram_size < 2 or n_heads < 1:
            raise ValueError(
                f"engram n-grams need max_ngram_size >= 2 and n_heads >= 1, got {max_ngram_size}, {n_heads}"
            )
        primes: list[tuple[tuple[int, ...], ...]] = []
        seen: set[int] = set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes: list[int] = []
                current = int(config.engram_vocab_size) - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        flat = [[prime for per_ngram in layer for prime in per_ngram] for layer in primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=num_embeddings,
            primes=tuple(primes),
            offsets=tuple(tuple(int(value) for value in row) for row in offsets),
            n_heads=n_heads,
            head_dim=int(config.engram_head_dim),
            compressed_vocab_size=int(config.engram_compressed_vocab_size),
        )


def hash_ngrams(tokens: Tensor, multipliers: Tensor, primes: Tensor, offsets: Tensor, max_ngram_size: int) -> Tensor:
    """XOR lookbacks into the prime buckets. ``tokens`` is ``[B, L, max_ngram]`` and already pad-filled."""
    products = tokens.unsqueeze(2) * multipliers
    rolling = products[..., 0]
    hashes = []
    for index in range(1, max_ngram_size):
        rolling = torch.bitwise_xor(rolling, products[..., index])
        hashes.append(rolling.unsqueeze(-1) % primes[:, index - 1])
    return torch.cat(hashes, dim=-1) + offsets


def _document_ids(seq_lens: Tensor | None, batch: int, length: int, device: torch.device) -> Tensor:
    if seq_lens is None:
        return torch.zeros(batch, length, dtype=torch.long, device=device)
    lengths = seq_lens.to(device=device, dtype=torch.long).reshape(-1)
    if int(lengths.sum()) != length:
        raise ValueError(f"seq_lens covers {int(lengths.sum())} tokens, hashes have {length}")
    ids = torch.repeat_interleave(torch.arange(lengths.numel(), device=device), lengths)
    return ids.view(1, length).expand(batch, length)


def lookback_tokens(
    compressed: Tensor,
    dead: Tensor,
    pad_id: int,
    max_ngram_size: int,
    seq_lens: Tensor | None = None,
) -> Tensor:
    """Tokens ending at each position. A pad, image, or other document blocks every older lookback."""
    batch, length = compressed.shape
    positions = torch.arange(length, device=compressed.device)
    document_ids = _document_ids(seq_lens, batch, length, compressed.device)
    marked = torch.where(dead, DEAD_ID, compressed)
    blocked = torch.zeros(batch, length, dtype=torch.bool, device=compressed.device)
    columns = []
    for shift in range(max_ngram_size):
        source_pos = positions - shift
        in_range = source_pos >= 0
        safe = source_pos.clamp(min=0)
        source = marked[:, safe]
        crossed = document_ids[:, safe] != document_ids
        blocked = blocked | (~in_range) | (source == DEAD_ID) | crossed
        columns.append(torch.where(blocked, pad_id, source))
    return torch.stack(columns, dim=-1)


def resolve_compressed_tokens(
    input_ids: Tensor,
    *,
    compressed_vocab_size: int,
    engram_pad_token_id: int,
    pad_token_id: int,
    image_token_id: int,
    compressed_token_map: Tensor | None = None,
    compressed_token_ids: Tensor | None = None,
    compressed_pad_id: int | None = None,
) -> tuple[Tensor, Tensor, int]:
    """Map tokenizer ids, or accept precomputed compressed ids. Neither one is a silent raw-id hash."""
    if compressed_token_map is None and compressed_token_ids is None:
        raise RuntimeError(
            "DeepSeek V4.1 Engram needs compressed_token_map or compressed_token_ids. "
            "Hashing raw tokenizer ids would silently disagree with the checkpoint tables."
        )
    if compressed_token_map is not None:
        compressed_token_map = compressed_token_map.to(dtype=torch.long)
        if compressed_token_map.ndim != 1 or compressed_token_map.numel() == 0:
            raise ValueError("compressed_token_map must be a non-empty 1-D lookup")
        if int(compressed_token_map.min()) < 0:
            raise ValueError("compressed_token_map ids must be non-negative")
        covered = int(compressed_token_map.max()) + 1
        if covered != compressed_vocab_size:
            raise ValueError(
                f"compressed_token_map covers ids 0..{covered - 1} but engram_compressed_vocab_size "
                f"is {compressed_vocab_size}. Multipliers are derived from that size, so a mismatch would "
                "rehash the tables."
            )
        if int(input_ids.max()) >= compressed_token_map.numel() or int(input_ids.min()) < 0:
            raise ValueError("input_ids fall outside compressed_token_map")
        if compressed_pad_id is None:
            if engram_pad_token_id < 0 or engram_pad_token_id >= compressed_token_map.numel():
                raise ValueError("engram_pad_token_id is outside compressed_token_map")
            compressed_pad_id = int(compressed_token_map[engram_pad_token_id])
        if compressed_token_ids is None:
            compressed_token_ids = compressed_token_map.to(device=input_ids.device)[input_ids]
    elif compressed_pad_id is None:
        raise RuntimeError(
            "compressed_token_ids needs compressed_pad_id, the compressed id of the engram pad token. "
            "Pass compressed_token_map to derive it."
        )
    if compressed_token_ids is None or compressed_pad_id is None:
        raise RuntimeError("compressed token ids were not resolved")
    if compressed_token_ids.shape != input_ids.shape:
        raise ValueError(
            f"compressed_token_ids shape {tuple(compressed_token_ids.shape)} != input_ids {tuple(input_ids.shape)}"
        )
    compressed = compressed_token_ids.to(device=input_ids.device, dtype=torch.long)
    if compressed.numel() and (int(compressed.min()) < 0 or int(compressed.max()) >= compressed_vocab_size):
        raise ValueError(f"compressed token ids must lie in [0, {compressed_vocab_size})")
    dead = (input_ids == pad_token_id) | (input_ids == image_token_id)
    return compressed, dead, int(compressed_pad_id)


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Collapse tokenizer ids that normalize alike. The returned size is the hash multiplier bound."""
    from tokenizers import Regex
    from tokenizers import normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


class EngramHasher(nn.Module):
    """Per-layer primes, offsets, and multipliers. The compressed map is installed from the tokenizer."""

    def __init__(self, config: DeepseekV41Config):
        super().__init__()
        layout = EngramLayout.from_config(config)
        if layout is None:
            raise ValueError("EngramHasher requires engram_layer_ids")
        self.layout = layout
        self.engram_pad_token_id = int(config.engram_pad_token_id)
        self.pad_token_id = int(config.pad_token_id)
        self.image_token_id = int(config.image_token_id)
        primes, offsets, multipliers = self._cpu_tables()
        self.register_buffer("primes", primes, persistent=False)
        self.register_buffer("offsets", offsets, persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("token_map", torch.empty(0, dtype=torch.long), persistent=False)

    def install_compressed_token_map(self, token_map: Tensor) -> None:
        token_map = token_map.detach().to(dtype=torch.long, device=self.primes.device).reshape(-1).contiguous()
        if token_map.numel() == 0 or int(token_map.min()) < 0:
            raise ValueError("compressed token map must be a non-empty lookup of non-negative ids")
        covered = int(token_map.max()) + 1
        if covered != self.layout.compressed_vocab_size:
            raise ValueError(
                f"compressed token map covers ids 0..{covered - 1} but engram_compressed_vocab_size "
                f"is {self.layout.compressed_vocab_size}. Multipliers are derived from that size."
            )
        self.token_map = token_map

    def _cpu_tables(self) -> tuple[Tensor, Tensor, Tensor]:
        layout = self.layout
        primes = torch.tensor(layout.primes, dtype=torch.long)
        offsets = torch.tensor(layout.offsets, dtype=torch.long)
        multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, layout.compressed_vocab_size)
        return primes, offsets, multipliers

    def init_buffers_post_meta(self) -> None:
        if self.primes.device.type == "meta":
            return
        primes, offsets, multipliers = self._cpu_tables()
        self.primes.copy_(primes.to(device=self.primes.device))
        self.offsets.copy_(offsets.to(device=self.offsets.device))
        self.multipliers.copy_(multipliers.to(device=self.multipliers.device))

    def forward(
        self,
        input_ids: Tensor,
        *,
        compressed_token_map: Tensor | None = None,
        compressed_token_ids: Tensor | None = None,
        compressed_pad_id: int | None = None,
        seq_lens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``[B, L, n_engram_layers, n_hash_cols]`` ids and a live-token mask ``[B, L]``."""
        if compressed_token_map is None and self.token_map.numel() > 0:
            compressed_token_map = self.token_map
        compressed, dead, pad_id = resolve_compressed_tokens(
            input_ids,
            compressed_vocab_size=self.layout.compressed_vocab_size,
            engram_pad_token_id=self.engram_pad_token_id,
            pad_token_id=self.pad_token_id,
            image_token_id=self.image_token_id,
            compressed_token_map=compressed_token_map,
            compressed_token_ids=compressed_token_ids,
            compressed_pad_id=compressed_pad_id,
        )
        tokens = lookback_tokens(compressed, dead, pad_id, self.layout.max_ngram_size, seq_lens)
        hashes = hash_ngrams(
            tokens,
            self.multipliers,
            self.primes,
            self.offsets,
            self.layout.max_ngram_size,
        )
        return hashes, ~dead


class EngramEmbedding(nn.Module):
    """Row-sharded hash table. FP8 rows stay packed; callers receive rows in the original id order."""

    _dss_shard_on_ep = True
    _dss_skip_weight_sync = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        quantized: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if num_embeddings < 1 or embedding_dim < 1:
            raise ValueError(f"engram table shape must be positive, got {(num_embeddings, embedding_dim)}")
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.quantized = quantized
        if quantized:
            if embedding_dim % _FP8_BLOCK != 0:
                raise ValueError(f"quantized engram head dim {embedding_dim} must be a multiple of {_FP8_BLOCK}")
            weight = torch.zeros(num_embeddings, embedding_dim, dtype=torch.float8_e4m3fn)
            scale = torch.zeros(num_embeddings, embedding_dim // _FP8_BLOCK, dtype=torch.float8_e8m0fnu)
            self.scale = nn.Parameter(scale, requires_grad=False)
            self.scale._dss_skip_weight_sync = True
        else:
            weight = torch.zeros(num_embeddings, embedding_dim, dtype=dtype)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight._dss_skip_weight_sync = True

    def _apply(self, fn, recurse=True):
        if not self.quantized:
            return super()._apply(fn, recurse=recurse)
        probe = fn(torch.empty(0, dtype=torch.float32, device=self.weight.device))
        if probe.dtype == torch.float32:
            return super()._apply(fn, recurse=recurse)

        def move_only(tensor: Tensor) -> Tensor:
            return tensor.to(device=probe.device)

        return super()._apply(move_only, recurse=recurse)

    def _lookup(self, local_ids: Tensor) -> Tensor:
        out_dtype = torch.bfloat16 if self.quantized else self.weight.dtype
        if local_ids.numel() == 0:
            return torch.empty(0, self.embedding_dim, dtype=out_dtype, device=local_ids.device)
        if self.quantized:
            return dequantize_block_fp8(_index_packed(self.weight, local_ids), _index_packed(self.scale, local_ids))
        return self.weight.index_select(0, local_ids)

    def _exchange(self, flat: Tensor) -> Tensor:
        world = self._ep_world_size
        group = self._ep_group
        rank = self._ep_rank
        chunk = (self.num_embeddings + world - 1) // world
        owners = torch.div(flat, chunk, rounding_mode="floor")
        order = torch.argsort(owners, stable=True)
        send_ids = flat[order].contiguous()
        send_counts_t = torch.bincount(owners, minlength=world).to(dtype=torch.long)
        recv_counts_t = torch.empty_like(send_counts_t)
        dist.all_to_all_single(recv_counts_t, send_counts_t, group=group)
        send_counts = send_counts_t.tolist()
        recv_counts = recv_counts_t.tolist()
        recv_ids = torch.empty(sum(recv_counts), dtype=send_ids.dtype, device=send_ids.device)
        dist.all_to_all_single(
            recv_ids,
            send_ids,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=group,
        )
        start, end = ep_row_shard(self.num_embeddings, world, rank)
        if recv_ids.numel() and (int(recv_ids.min()) < start or int(recv_ids.max()) >= end):
            raise RuntimeError("Engram all_to_all delivered an id this rank does not own")
        local = self._lookup(recv_ids - start)
        gathered = torch.empty(flat.shape[0], self.embedding_dim, dtype=local.dtype, device=flat.device)
        dist.all_to_all_single(
            gathered,
            local.contiguous(),
            output_split_sizes=send_counts,
            input_split_sizes=recv_counts,
            group=group,
        )
        output = torch.empty_like(gathered)
        output[order] = gathered
        return output

    def forward(self, indices: Tensor) -> Tensor:
        world = getattr(self, "_ep_world_size", 1)
        if world > 1 and getattr(self, "_ep_group", None) is None:
            raise RuntimeError("Engram embedding is expert-parallel but no process group is installed")
        flat = indices.reshape(-1).to(dtype=torch.long)
        if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= self.num_embeddings):
            raise ValueError(f"engram ids must lie in [0, {self.num_embeddings})")
        start, end = ep_row_shard(self.num_embeddings, world, getattr(self, "_ep_rank", 0))
        if self.weight.shape[0] != end - start:
            raise RuntimeError(
                f"engram shard rows {self.weight.shape[0]} != owned range {end - start} "
                f"(rank {getattr(self, '_ep_rank', 0)} of {world})"
            )
        if self.quantized and self.scale.shape[0] != end - start:
            raise RuntimeError("engram scale shard does not match the weight shard")
        rows = self._lookup(flat - start) if world == 1 else self._exchange(flat)
        return rows.reshape(*indices.shape, self.embedding_dim)


class DeepseekV41Engram(nn.Module):
    """Gated n-gram residual on the full hyper-connection stream, before that layer's mixes."""

    def __init__(self, config: DeepseekV41Config, layer_id: int, layout: EngramLayout):
        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        self.embed = EngramEmbedding(
            layout.num_embeddings[self.layer_hash_index],
            layout.head_dim,
            quantized=bool(config.engram_quantized),
        )
        in_features = layout.n_hash_cols * layout.head_dim
        self.wkv = nn.Linear(in_features, config.hidden_size * (config.hc_mult + 1), bias=False)
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))

    def init_weights(self) -> None:
        nn.init.ones_(self.q_weight)
        nn.init.ones_(self.k_weight)

    def forward(self, hidden: Tensor, hash_ids: Tensor, token_mask: Tensor | None = None) -> Tensor:
        """``hidden`` is ``[B, L, hc, dim]``. ``hash_ids`` is ``[B, L, n_hash_cols]``. False mask zeros the gate."""
        rows = self.embed(hash_ids)
        kv = self.wkv(rows.flatten(-2).to(dtype=self.wkv.weight.dtype))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        stream = hidden.float()
        rstd = torch.rsqrt(stream.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (stream * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (stream + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(dtype=hidden.dtype)
