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
"""How a sampler with prefix caching computed a multi-turn row, as a per-token plan the aligned forward replays.

A multi-turn row is one conversation: each turn's prompt is the previous prompt plus its completion and the next
user or tool message. The sampler computes turn ``t`` as one request that resumes at its prefix-cache hit ``H_t``:
it reads the attention pages of ``[0, H_t)`` that earlier requests wrote, starts its GatedDeltaNet layers from the
conv and recurrent state an earlier request saved at ``H_t``, prefills the rest of its prompt in scheduled chunks
and decodes its completion. The same positions are computed by several requests, each in its own way, and each
logprob comes from one of them, so the trainer replays every request as its own span of the row.

``vllm_aligned_plan`` is an integer tensor ``[batch, seq, K, 3]``, one column ``k`` per request (computation) of the
row, in an order where every column only reads columns before it:

* plane 0, role of position ``p`` in column ``k``: ``0`` not used; ``1`` prefilled; ``2`` prefilled, the last token
  of a scheduled chunk; ``3`` decoded; ``-(c + 1)`` before the column's hit, whose attention page column ``c``
  wrote. A column's prefilled and decoded positions are one run ``[H, E)``, with ``H`` a multiple of the attention
  page size and every position before ``H`` read from some column, page by page.
* plane 1, at the column's first position ``H > 0``: ``c + 1``, the column whose GatedDeltaNet state after ``H``
  tokens the request resumed from; that column has a chunk or decode run ending at ``H``.
* plane 2: ``1`` where this column's logprob of the next token is the one the row returns.

A position no column marks in plane 2 returns the logprob of the first column that computes it. The sampler runs a
one-token prefill chunk as a decode step, and so does the replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

PLAN_KWARG = "vllm_aligned_plan"
ABSENT, PREFILL, CHUNK_END, DECODE = 0, 1, 2, 3
ROLE, RESUME, OUTPUT = 0, 1, 2


@dataclass(frozen=True)
class TurnSpan:
    """One column of one row: the request computed ``[start, end)`` of the row, resuming at ``start``."""

    row: int
    column: int
    start: int
    end: int
    segments: tuple[tuple[bool, int, int], ...]
    page_columns: tuple[int, ...]
    resume_column: int | None
    outputs: tuple[int, ...]


def parse_row(plan: torch.Tensor, *, row: int, block_size: int) -> list[TurnSpan]:
    """The columns of one row's ``[seq, K, 3]`` plan, as spans with absolute positions; checks the plan's rules."""
    plan = plan.to("cpu", torch.long)
    seq_len, num_columns = plan.shape[0], plan.shape[1]
    spans = []
    for column in range(num_columns):
        role = plan[:, column, ROLE]
        computed = torch.nonzero(role > 0, as_tuple=False).flatten()
        if computed.numel() == 0:
            continue
        start, end = int(computed[0]), int(computed[-1]) + 1
        where = f"row {row} column {column}"
        if computed.numel() != end - start:
            raise ValueError(f"{where}: computed positions are not one run")
        if bool((role[end:] != ABSENT).any()) or bool((role[:start] >= 0).any()):
            raise ValueError(f"{where}: every position before {start} must name the column that wrote it")
        if start % block_size:
            raise ValueError(f"{where}: resumes at {start}, not a multiple of the page size {block_size}")
        pages = []
        for page in range(start // block_size):
            readers = -role[page * block_size : (page + 1) * block_size] - 1
            if bool((readers != readers[0]).any()) or not 0 <= int(readers[0]) < column:
                raise ValueError(f"{where}: page {page} must be read from one earlier column")
            pages.append(int(readers[0]))
        resume = None
        if start > 0:
            resume = int(plan[start, column, RESUME]) - 1
            if not 0 <= resume < column:
                raise ValueError(f"{where}: resumes at {start} without an earlier column's state")
        segments: list[tuple[bool, int, int]] = []
        lo = start
        roles = role[start:end].tolist()
        for offset, value in enumerate(roles):
            p = start + offset
            decode = value == DECODE
            last = offset + 1 == len(roles)
            if decode:
                if not last and roles[offset + 1] == DECODE:
                    continue
                segments.append((True, lo, p + 1))
                lo = p + 1
            elif value == CHUNK_END or last or roles[offset + 1] == DECODE:
                segments.append((False, lo, p + 1))
                lo = p + 1
        outputs = torch.nonzero(plan[start:end, column, OUTPUT] != 0, as_tuple=False).flatten() + start
        if bool((plan[:start, column, OUTPUT] != 0).any()) or bool((plan[end:, column, OUTPUT] != 0).any()):
            raise ValueError(f"{where}: an output outside the positions it computes")
        spans.append(TurnSpan(row, column, start, end, tuple(segments), tuple(pages), resume, tuple(outputs.tolist())))
    by_column = {span.column: span for span in spans}
    for span in spans:
        where = f"row {row} column {span.column}"
        for page, column in enumerate(span.page_columns):
            writer = by_column.get(column)
            if writer is None or writer.start > page * block_size or writer.end < (page + 1) * block_size:
                raise ValueError(f"{where}: column {column} did not compute page {page}")
        if span.resume_column is not None:
            provider = by_column.get(span.resume_column)
            if provider is None or span.start not in {hi for _, _, hi in provider.segments}:
                raise ValueError(f"{where}: column {span.resume_column} has no state at {span.start}")
    if seq_len and spans and max(span.end for span in spans) > seq_len:
        raise ValueError(f"row {row}: a column runs past the row")
    return spans


@dataclass(frozen=True)
class TurnLayout:
    """A packed batch with its rows' spans laid out one after another.

    ``expand_index`` gives each span token's packed position, ``positions`` its position in its row, and
    ``output_index`` the span token whose logprob each packed position returns (``-1`` for none).
    """

    expand_index: torch.Tensor
    positions: torch.Tensor
    output_index: torch.Tensor
    attention: object
    gdn: object


def build_turn_layout(
    plan: torch.Tensor, cu_seqlens: torch.Tensor, *, block_size: int, conv_width: int, device: torch.device
) -> TurnLayout:
    """Spans for every row of the packed ``[tokens, K, 3]`` plan whose rows ``cu_seqlens`` bounds."""
    from arctic_platform.model.implementations.vllm_aligned import ops
    from arctic_platform.model.implementations.vllm_aligned.qwen3_5 import GdnLayout
    from arctic_platform.model.implementations.vllm_aligned.qwen3_5 import GdnSpan

    plan = plan.to("cpu", torch.long)
    bounds = cu_seqlens.to(torch.long).tolist()
    num_tokens = bounds[-1]
    spans: list[TurnSpan] = []
    row_starts: list[int] = []
    for row, (row_start, row_end) in enumerate(zip(bounds[:-1], bounds[1:])):
        row_spans = parse_row(plan[row_start:row_end], row=row, block_size=block_size)
        if not row_spans:
            row_spans = [TurnSpan(row, 0, 0, row_end - row_start, ((False, 0, row_end - row_start),), (), None, ())]
        spans += row_spans
        row_starts += [row_start] * len(row_spans)

    span_id = {(span.row, span.column): i for i, span in enumerate(spans)}
    lengths = [span.end - span.start for span in spans]
    starts = torch.cumsum(torch.tensor([0] + lengths), 0).tolist()
    expand_index = torch.cat(
        [torch.arange(row_start + span.start, row_start + span.end) for span, row_start in zip(spans, row_starts)]
    )
    positions = torch.cat([torch.arange(span.start, span.end) for span in spans])

    output_index = torch.full((num_tokens,), -1, dtype=torch.long)
    for i in reversed(range(len(spans))):
        span, row_start = spans[i], row_starts[i]
        output_index[row_start + span.start : row_start + span.end] = torch.arange(starts[i], starts[i] + lengths[i])
    for i, (span, row_start) in enumerate(zip(spans, row_starts)):
        if span.outputs:
            outputs = torch.tensor(span.outputs)
            output_index[row_start + outputs] = starts[i] + outputs - span.start

    pages_per_span = [-(-length // block_size) for length in lengths]
    first_page = torch.cumsum(torch.tensor([0] + pages_per_span), 0).tolist()
    max_pages = max(-(-span.end // block_size) for span in spans)
    block_table = torch.zeros((len(spans), max_pages), dtype=torch.int32)
    key_index, slots = [], []
    for i, span in enumerate(spans):
        own = span.start // block_size
        for page in range(-(-span.end // block_size)):
            if page < own:
                writer = span_id[(span.row, span.page_columns[page])]
                block_table[i, page] = first_page[writer] + page - spans[writer].start // block_size
            else:
                block_table[i, page] = first_page[i] + page - own
        slots.append(first_page[i] * block_size + torch.arange(lengths[i]))
        context = [
            starts[span_id[(span.row, span.page_columns[page])]] - spans[span_id[(span.row, span.page_columns[page])]].start
            + torch.arange(page * block_size, (page + 1) * block_size)
            for page in range(own)
        ]  # fmt: skip
        key_index.append(torch.cat(context + [torch.arange(starts[i], starts[i] + lengths[i])]))
    key_lens = torch.tensor([span.end for span in spans], dtype=torch.int32)
    seq_lens = torch.tensor(lengths, dtype=torch.int32)
    zero = torch.zeros(1, dtype=torch.int32)
    attention = ops.PackedLayout(
        positions=positions.to(device),
        cu_seqlens=torch.cat((zero, torch.cumsum(seq_lens, 0, dtype=torch.int32))).to(device),
        seq_lens=seq_lens.to(device),
        max_seqlen=max(lengths),
        block_size=block_size,
        block_table=block_table.to(device),
        slots=torch.cat(slots).to(device),
        num_blocks=first_page[-1],
        key_lens=key_lens.to(device),
        max_key_len=int(key_lens.max()),
        key_index=torch.cat(key_index).to(device),
        key_cu_seqlens=torch.cat((zero, torch.cumsum(key_lens, 0, dtype=torch.int32))).to(device),
    )
    gdn_spans = []
    for span in spans:
        resume = None
        if span.resume_column is not None:
            provider = span_id[(span.row, span.resume_column)]
            resume = (provider, span.start - spans[provider].start)
        segments = tuple((decode, lo - span.start, hi - span.start) for decode, lo, hi in span.segments)
        gdn_spans.append(GdnSpan(starts[span_id[(span.row, span.column)]], span.end - span.start, segments, resume))
    gdn = GdnLayout.from_spans(gdn_spans, num_tokens=starts[-1], conv_width=conv_width, device=device)
    return TurnLayout(
        expand_index=expand_index.to(device),
        positions=positions.to(device),
        output_index=output_index.to(device),
        attention=attention,
        gdn=gdn,
    )


def encode_turns(seq_len: int, turns: list[dict], *, block_size: int) -> torch.Tensor:
    """A row's ``[seq_len, K, 3]`` plan from its requests, in order, for clients and tests.

    Each request is a dict with ``start`` (its hit ``H``), ``chunk_ends`` (where its scheduled prefill chunks
    ended), ``prompt_end`` (where decoding began), ``end`` (one past the last token it computed or sampled),
    ``outputs`` (the positions whose next-token logprob the row returns from it), and optionally ``pages`` (the
    request that wrote each page before ``start``) and ``resume`` (the request whose state it resumed from);
    both default to the first earlier request that computed them, the copy vLLM's prefix cache returns while
    nothing is evicted.
    """
    plan = torch.zeros((seq_len, len(turns), 3), dtype=torch.int32)
    for k, turn in enumerate(turns):
        start, prompt_end, end = int(turn["start"]), int(turn["prompt_end"]), int(turn["end"])
        plan[start:prompt_end, k, ROLE] = PREFILL
        for chunk_end in turn.get("chunk_ends", ()):
            if start < chunk_end <= prompt_end:
                plan[chunk_end - 1, k, ROLE] = CHUNK_END
        plan[prompt_end:end, k, ROLE] = DECODE
        pages = turn.get("pages")
        for page in range(start // block_size):
            if pages is not None:
                writer = pages[page]
            else:
                writer = min(
                    j for j in range(k) if turns[j]["start"] <= page * block_size and turns[j]["end"] > (page + 1) * block_size
                )  # fmt: skip
            plan[page * block_size : (page + 1) * block_size, k, ROLE] = -(writer + 1)
        if start > 0:
            resume = turn.get("resume")
            if resume is None:
                resume = min(
                    j for j in range(k)
                    if turns[j]["start"] < start and (start in turns[j].get("chunk_ends", ()) or start == turns[j]["prompt_end"])
                )  # fmt: skip
            plan[start, k, RESUME] = resume + 1
        for p in turn.get("outputs", ()):
            plan[p, k, OUTPUT] = 1
    return plan
