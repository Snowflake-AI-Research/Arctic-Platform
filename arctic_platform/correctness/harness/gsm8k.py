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

"""Turn a GSM8K parquet split into a batch whose loss covers answer tokens only.

Distinct from ``batches``, which draws uniform random token ids and asserts the padding invariants the
gradient comparison depends on. Nothing here is drawn: the ids come from a dataset file through the
model's own tokenizer, and the masked span is decided by where the prompt ends rather than by a seeded
length. A check that scores real answer tokens cannot be built out of random ids, and the invariants that
make a synthetic case useful -- unique rows, a padding floor -- are not properties a dataset slice has.

A scored example carries its prompt and completion as strings as well as ids. A training engine is handed
tensors and a serving engine is handed text, and both have to score the same span, so the span is located
once here and both representations are kept beside it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

from .batches import IGNORE_INDEX
from .batches import Batch

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
