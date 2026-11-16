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


"""Supervised batches, tokenization, and loss accounting."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

from arctic_platform.correctness.harness.storage import require_data_fast_path

# Synthetic

SEED = 1234


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


def build_synthetic_batch(name: str, rows: int, max_len: int, vocab_size: int, seed: int = SEED) -> Batch:
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
    path = require_data_fast_path(path, label="batch tensor output")
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


# Sft

# The prompt ends without a trailing space and the completion supplies it, so the boundary falls where a
# tokenizer is free to merge across it. ``answer_boundary`` is what makes that harmless.
PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
COMPLETION_TEMPLATE = " {answer}"


@dataclass(frozen=True)
class Example:
    """One dataset row, tokenized once as prompt-plus-completion with the answer span located in it."""

    question: str
    prompt: str
    completion: str
    input_ids: Tuple[int, ...]
    # Index into ``input_ids`` where the answer begins. Every id at or after it is scored.
    n_prompt: int

    @property
    def length(self) -> int:
        return len(self.input_ids)

    @property
    def answer_tokens(self) -> int:
        return self.length - self.n_prompt


def read_rows(path: str | Path, *, start: int = 0, count: int | None = None) -> List[Tuple[str, str]]:
    """Question and answer strings from a GSM8K split, in file order.

    File order is the whole selection rule. A shuffle would need a seed, and the two splits are already
    disjoint by being two files, so ordering buys nothing and costs reproducibility.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(Path(path))
    limit = table.num_rows - start if count is None else count
    if start < 0 or limit < 0 or start + limit > table.num_rows:
        raise ValueError(f"{path} holds {table.num_rows} rows; asked for {limit} beginning at {start}")
    return [
        (row["extra_info"]["question"], row["extra_info"]["answer"]) for row in table.slice(start, limit).to_pylist()
    ]


def answer_boundary(prompt_ids: Sequence[int], full_ids: Sequence[int]) -> int:
    """Length of the common prefix of the two tokenizations.

    Not ``len(prompt_ids)``. The prompt does not end on a token boundary of the joined text, so tokenizing
    it alone can produce a last id that the joined text never contains, and the two lengths then differ.
    Scoring ``len(prompt_ids)`` positions would put the training side and the serving side on different
    spans, which is a difference in what is measured rather than in the model.
    """
    limit = min(len(prompt_ids), len(full_ids))
    index = 0
    while index < limit and prompt_ids[index] == full_ids[index]:
        index += 1
    return index


def load_tokenizer(model_path: str | Path):
    """The tokenizer stored beside a model's weights, checked against the embedding table it indexes.

    ``AutoTokenizer.from_pretrained`` on a directory that carries no tokenizer sidecars neither raises nor
    returns nothing: it returns a tokenizer whose vocabulary holds one entry and which encodes every
    string to the empty list. Every batch built from it would be empty, and a comparison of two empty
    batches reports perfect agreement, so the size is asserted rather than assumed.
    """
    from transformers import AutoConfig
    from transformers import AutoTokenizer

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    embedding_rows = int(getattr(config, "text_config", config).vocab_size)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    assert_vocabulary_matches_model(len(tokenizer), embedding_rows, model_path)
    return tokenizer


def assert_vocabulary_matches_model(vocabulary_size: int, embedding_rows: int, model_path) -> None:
    """Both directions are real constraints, and between them they exclude the empty tokenizer.

    A vocabulary wider than the embedding table can emit an id the model cannot look up. A vocabulary of a
    single entry cannot represent text at all, which is what an absent sidecar set produces.
    """
    if vocabulary_size > embedding_rows:
        raise ValueError(
            f"tokenizer at {model_path} has {vocabulary_size:,} entries but the model config declares "
            f"{embedding_rows:,} embedding rows, so it can emit an id the model cannot look up"
        )
    if vocabulary_size <= 1:
        raise ValueError(
            f"tokenizer at {model_path} has {vocabulary_size} vocabulary entr(ies) and cannot represent "
            "text; the directory carries no tokenizer sidecars, and such a tokenizer encodes every "
            "string to an empty id list instead of failing"
        )


def tokenize_examples(tokenizer, rows: Sequence[Tuple[str, str]]) -> List[Example]:
    """One tokenizer call per example on the joined text, plus one on the prompt to find the boundary.

    The joined text is tokenized in a single call because that is what a serving engine does to
    ``prompt + completion``; tokenizing the two halves separately and concatenating would produce a
    different id sequence wherever the boundary merges.
    """
    examples: List[Example] = []
    for question, answer in rows:
        prompt = PROMPT_TEMPLATE.format(question=question)
        completion = COMPLETION_TEMPLATE.format(answer=answer)
        full_ids = list(tokenizer(prompt + completion, add_special_tokens=True)["input_ids"])
        prompt_ids = list(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        n_prompt = answer_boundary(prompt_ids, full_ids)
        if n_prompt < 1:
            raise ValueError(f"prompt and joined tokenizations share no prefix for question {question!r}")
        if n_prompt >= len(full_ids):
            raise ValueError(f"no answer tokens remain after the prompt for question {question!r}")
        examples.append(
            Example(
                question=question, prompt=prompt, completion=completion, input_ids=tuple(full_ids), n_prompt=n_prompt
            )
        )
    return examples


def padded_width(examples: Sequence[Example]) -> int:
    return max(example.length for example in examples)


def length_distribution(examples: Sequence[Example]) -> Dict[str, int]:
    """What the selection below is sized against, reported wherever the selected count is reported."""
    lengths = sorted(example.length for example in examples)
    return {
        "examples": len(lengths),
        "minimum_tokens": lengths[0],
        "median_tokens": int(statistics.median(lengths)),
        "maximum_tokens": lengths[-1],
        "answer_tokens": sum(example.answer_tokens for example in examples),
    }


def select_by_token_budget(
    examples: Sequence[Example], token_budget: int, minimum_rows: int
) -> Tuple[List[Example], int]:
    """The longest leading run of examples whose padded rectangle fits one forward's token budget.

    A single forward has to hold ``rows * padded width`` token slots, and the width is the longest row in
    the selection, so extending the selection can only grow the rectangle. The largest count that fits is
    therefore the last one that fits, and the scan stops at the first that does not.
    """
    if token_budget < 1:
        raise ValueError(f"token budget must be positive, got {token_budget}")
    chosen, width = 0, 0
    for count, example in enumerate(examples, start=1):
        candidate_width = max(width, example.length)
        if candidate_width * count > token_budget:
            break
        chosen, width = count, candidate_width
    if chosen < minimum_rows:
        raise ValueError(
            f"a {token_budget:,}-token forward holds {chosen} of these examples at a padded width of "
            f"{width:,}, but every one of the config's {minimum_rows} data-parallel shards needs a row"
        )
    return list(examples[:chosen]), width


def assert_disjoint(training: Sequence[Example], validation: Sequence[Example]) -> None:
    """State the held-out property on the examples instead of trusting the two file names.

    The two slices are read from the dataset's own train and test files, so they are disjoint by
    construction; this turns that into a checked property, because a validation loss measured on rows the
    run trained on says nothing about either engine.
    """
    shared = {example.question for example in training} & {example.question for example in validation}
    if shared:
        raise ValueError(
            f"{len(shared)} question(s) appear in both the training and the validation slice; the first is "
            f"{sorted(shared)[0]!r}"
        )


def build_batch(name: str, examples: Sequence[Example], pad_id: int) -> Batch:
    """Right-padded ids with HuggingFace-convention labels over the answer span only.

    ``labels[b, s]`` is the token at ``s`` when ``s`` is inside the answer, and ``IGNORE_INDEX``
    everywhere else -- over the prompt and over the padded tail. That is the convention Arctic Platform requests carry,
    and ``Batch.shifted_labels`` converts it to logit alignment for whoever needs that instead.
    """
    import torch

    if not examples:
        raise ValueError(f"{name}: refusing to build a batch from no examples")
    rows = len(examples)
    width = padded_width(examples)

    input_ids = torch.full((rows, width), pad_id, dtype=torch.long)
    position_ids = torch.zeros((rows, width), dtype=torch.long)
    labels = torch.full((rows, width), IGNORE_INDEX, dtype=torch.long)
    for row, example in enumerate(examples):
        ids = torch.tensor(example.input_ids, dtype=torch.long)
        input_ids[row, : example.length] = ids
        position_ids[row, : example.length] = torch.arange(example.length, dtype=torch.long)
        labels[row, example.n_prompt : example.length] = ids[example.n_prompt :]

    return Batch(name=name, input_ids=input_ids, position_ids=position_ids, labels=labels)


def padding_token_id(tokenizer) -> int:
    """The id the padded columns carry.

    Padding is masked out of the loss on both label conventions, so the value only has to be an id the
    model can look up. A tokenizer without a pad token has an end-of-sequence token that satisfies that.
    """
    for token_id in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if token_id is not None:
            return int(token_id)
    raise ValueError("tokenizer declares neither a pad token nor an end-of-sequence token")


TRAIN_SPLIT = Path("/code/shared/gsm8k/train.parquet")
VALIDATION_SPLIT = Path("/code/shared/gsm8k/test.parquet")


def configured_learning_rate(cfg) -> float:
    """The rate the config's own optimizer block declares.

    Not the check's to choose. Ten steps at another rate reach a different set of weights and therefore a
    different validation loss than the operator's job would reach, and the point of the comparison is that
    both engines read the weights this config produces.
    """
    optimizer = cfg.training.get("optimizer")
    rate = optimizer.get("lr") if isinstance(optimizer, dict) else None
    if rate is None:
        raise ValueError(
            f"{cfg.config_id}: the training config declares no optimizer learning rate. This check trains "
            "the config as written, so there is nothing to run rather than a rate to pick."
        )
    return float(rate)


def configured_rows_per_step(cfg) -> int:
    """The global batch size the config trains at, which is how many dataset rows a step consumes."""
    rows = cfg.training.get("train_batch_size")
    if rows is None:
        raise ValueError(f"{cfg.config_id}: the training config declares no train_batch_size")
    return int(rows)


def cross_entropy(per_row_answer_logprobs: Sequence[Sequence[float]]) -> Tuple[float, int]:
    """Negative mean log-probability over every answer token in the batch.

    The one reduction both sides use. Each side is responsible only for producing, per row, that row's
    answer-token log-probabilities in order; the sign, the denominator and the pooling across rows are
    decided here, once.
    """
    total = 0.0
    count = 0
    for row in per_row_answer_logprobs:
        for value in row:
            total += float(value)
            count += 1
    if count == 0:
        raise ValueError("no answer tokens were scored; a cross entropy over nothing is not a measurement")
    return -total / count, count


def reference_answer_logprobs(logprobs, examples: Sequence[Example]) -> List[List[float]]:
    """The training zone's log-probabilities over each row's answer span.

    The route returns one entry per position, scoring the label at that position. The labels it was sent
    are logit aligned, so the entry at ``s`` is the log-probability of the token at ``s + 1``. A row of
    length ``L`` whose answer begins at ``p`` therefore has its answer scored by entries ``p - 1`` through
    ``L - 2``.
    """
    if logprobs.ndim != 2 or logprobs.shape[0] != len(examples):
        raise RuntimeError("forward-only logprobs must have one two-dimensional row per validation example")
    rows: List[List[float]] = []
    for index, example in enumerate(examples):
        _assert_answer_span(example, index)
        span = [float(value) for value in logprobs[index, example.n_prompt - 1 : example.length - 1].tolist()]
        if len(span) != example.answer_tokens:
            raise RuntimeError(
                f"row {index}: the forward-only response covers {len(span)} of the row's "
                f"{example.answer_tokens} answer tokens"
            )
        if not all(math.isfinite(value) for value in span):
            raise RuntimeError(f"row {index}: forward-only answer log-probabilities must be finite")
        rows.append(span)
    return rows


def _assert_answer_span(example: Example, index: int) -> None:
    if not 1 <= example.n_prompt < example.length:
        raise RuntimeError(f"row {index}: answer span must follow a prompt token and contain at least one token")
