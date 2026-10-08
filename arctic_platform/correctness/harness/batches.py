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

"""Generate the batches, materialize them, and assert the properties the comparison depends on.

Both engines load the same file rather than seeding their own generators. Arctic Platform and the reference are
separate processes that may hold different library versions, and any unrelated draw between them shifts
global RNG state; loading bytes removes the question entirely.

The invariants below are asserted on the materialized tensor rather than assumed from the sampling. Padding
in particular is load-bearing: it is what makes rows contribute unequal token counts, and with sequence
parallelism it gives each rank a different active-token count. A batch of uniformly full rows cannot
distinguish a request-wide loss denominator from a per-row or per-rank one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .seeds import SEED

IGNORE_INDEX = -100


@dataclass
class Batch:
    name: str
    input_ids: "object"
    position_ids: "object"
    labels: "object"

    @property
    def rows(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def total_tokens(self) -> int:
        return self.rows * self.seq_len

    def shifted_labels(self):
        """Labels aligned to the logits that predict them.

        Requests carry HuggingFace-convention labels, where ``labels[b, s]`` is the token *at* position s.
        Arctic Platform converts them once at dispatch, before packing, because a model-side shift inside a packed
        window would take the next row's first token as this row's last target. The reference applies the
        same conversion so both engines optimize against the same targets.
        """
        import torch

        out = torch.full_like(self.labels, IGNORE_INDEX)
        out[:, :-1] = self.labels[:, 1:]
        return out

    @property
    def active_tokens(self) -> int:
        """Counted after the shift, since that is the set that carries loss weight."""
        return int((self.shifted_labels() != IGNORE_INDEX).sum())

    @property
    def pad_fraction(self) -> float:
        return 1.0 - self.active_tokens / self.total_tokens


# Every arm must carry real padding, not the single position the label shift masks. Padding is what gives
# rows unequal token counts and, under sequence parallelism, gives each rank a different active-token
# count; without it a request-wide loss denominator is indistinguishable from a per-row or per-rank one.
MIN_PAD_FRACTION = 0.05


def row_lengths(rows: int, max_len: int, seed: int, min_frac: float = 0.15) -> List[int]:
    """Deliberately non-uniform lengths spanning short rows to the arm maximum.

    With more than one row the first is pinned to ``max_len`` so the arm reaches its stated maximum and the
    second to the floor, which guarantees padding. A single-row arm cannot do both: a full-width row leaves
    nothing padded, so the row is shortened instead. The tensor is still ``max_len`` wide, which is what
    sequence parallelism shards, and the tail is padding.
    """
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    low = max(8, int(max_len * min_frac))
    if rows == 1:
        return [max(low, int(max_len * 0.75))]
    lengths = torch.randint(low, max_len + 1, (rows,), generator=gen).tolist()
    lengths[0] = max_len
    lengths[1] = low
    return [int(x) for x in lengths]


def build_batch(name: str, rows: int, max_len: int, vocab_size: int, seed: int = SEED) -> Batch:
    """One arm, with labels in HuggingFace convention and pad positions masked out."""
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (rows, max_len), generator=gen)
    # HuggingFace convention: labels[b, s] is the token at position s, not the one predicted from it. The
    # shift to logit alignment belongs to whoever consumes the batch, and both engines apply it identically.
    labels = torch.full((rows, max_len), IGNORE_INDEX, dtype=torch.long)
    for r, length in enumerate(row_lengths(rows, max_len, seed)):
        labels[r, :length] = input_ids[r, :length]
    position_ids = torch.arange(max_len).unsqueeze(0).expand(rows, -1).contiguous()

    batch = Batch(name=name, input_ids=input_ids, position_ids=position_ids, labels=labels)
    assert_invariants(batch)
    return batch


def assert_invariants(batch: Batch) -> None:
    """Fail loudly at generation time rather than producing a test that silently proves less than it claims."""

    if batch.active_tokens == 0:
        raise AssertionError(f"arm {batch.name}: no active tokens")
    if batch.pad_fraction < MIN_PAD_FRACTION:
        raise AssertionError(
            f"arm {batch.name}: padding is {batch.pad_fraction:.2%}, below the {MIN_PAD_FRACTION:.0%} floor. "
            "A batch whose only masked position is the one the label shift consumes cannot distinguish a "
            "request-wide loss denominator from a per-row or per-rank one."
        )
    unique_rows = {tuple(batch.input_ids[r].tolist()) for r in range(batch.rows)}
    if len(unique_rows) != batch.rows:
        raise AssertionError(
            f"arm {batch.name}: rows must all differ, found {len(unique_rows)} distinct of {batch.rows}"
        )


def save(batch: Batch, path: Path) -> str:
    """Materialize so both engines read identical bytes, and return a hash for the report."""
    import hashlib

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"input_ids": batch.input_ids, "position_ids": batch.position_ids, "labels": batch.labels, "name": batch.name},
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load(path: Path) -> Batch:
    import torch

    blob = torch.load(path, weights_only=False)
    return Batch(
        name=blob["name"], input_ids=blob["input_ids"], position_ids=blob["position_ids"], labels=blob["labels"]
    )
