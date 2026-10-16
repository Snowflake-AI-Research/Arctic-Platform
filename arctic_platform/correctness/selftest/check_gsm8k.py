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

"""GSM8K intake: the answer boundary, the answer-only labels, the budget-sized slice, and the tokenizer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from arctic_platform.correctness.harness import gsm8k
from arctic_platform.correctness.harness.batches import IGNORE_INDEX


def example(prompt: str, completion: str, input_ids, n_prompt: int) -> gsm8k.Example:
    return gsm8k.Example(
        question=prompt, prompt=prompt, completion=completion, input_ids=tuple(input_ids), n_prompt=n_prompt
    )


def test_the_boundary_is_the_common_prefix_not_the_prompt_length() -> None:
    """A prompt whose last token merges with the completion's first tokenizes one way alone and another
    way joined. Taking ``len(prompt_ids)`` would score a position the joined sequence does not have there.
    """
    prompt_ids = [9, 4, 7, 88]
    full_ids = [9, 4, 7, 91, 12, 13]

    assert gsm8k.answer_boundary(prompt_ids, full_ids) == 3
    assert gsm8k.answer_boundary(prompt_ids, full_ids) != len(prompt_ids)


def test_a_prompt_that_survives_whole_gives_its_own_length() -> None:
    assert gsm8k.answer_boundary([9, 4, 7], [9, 4, 7, 12, 13]) == 3


def test_labels_cover_the_answer_span_only() -> None:
    rows = [example("q1", " a1", [5, 6, 7, 8], 2), example("q2", " a2", [1, 2, 3], 1)]

    batch = gsm8k.build_batch("validation", rows, pad_id=0)

    assert batch.input_ids.tolist() == [[5, 6, 7, 8], [1, 2, 3, 0]]
    # HuggingFace convention: the token AT the position, masked over the prompt and over the padded tail.
    assert batch.labels.tolist() == [[IGNORE_INDEX, IGNORE_INDEX, 7, 8], [IGNORE_INDEX, 2, 3, IGNORE_INDEX]]
    assert batch.active_tokens == sum(row.answer_tokens for row in rows)


def test_the_shift_puts_each_answer_token_on_the_logit_that_predicts_it() -> None:
    """``shifted_labels()[r, s]`` is the token at ``s + 1``, which is what a logit at ``s`` predicts."""
    rows = [example("q1", " a1", [5, 6, 7, 8], 2)]

    shifted = gsm8k.build_batch("validation", rows, pad_id=0).shifted_labels()

    assert shifted.tolist() == [[IGNORE_INDEX, 7, 8, IGNORE_INDEX]]


def test_positions_are_row_local_and_padding_is_not_numbered() -> None:
    rows = [example("q1", " a1", [5, 6, 7, 8], 2), example("q2", " a2", [1, 2, 3], 1)]

    batch = gsm8k.build_batch("validation", rows, pad_id=0)

    assert batch.position_ids.tolist() == [[0, 1, 2, 3], [0, 1, 2, 0]]


def test_the_slice_is_the_largest_that_fits_one_forward() -> None:
    """Row count times padded width is what the forward holds, and the width is the longest row taken."""
    rows = [example(f"q{index}", " a", list(range(length)), 1) for index, length in enumerate([4, 6, 5, 40])]

    chosen, width = gsm8k.select_by_token_budget(rows, token_budget=17, minimum_rows=1)

    assert [row.length for row in chosen] == [4, 6]
    assert width == 6
    # Two rows occupy 2 x 6 slots. A third would occupy 3 x 6, which is over the budget even though the
    # row itself is shorter than the width, because it is the rectangle the forward holds.
    assert len(chosen) * width == 12
    assert 3 * width > 17


def test_a_slice_too_narrow_for_the_data_parallel_shards_is_refused() -> None:
    rows = [example("q0", " a", list(range(10)), 1), example("q1", " a", list(range(10)), 1)]

    with pytest.raises(ValueError, match="data-parallel"):
        gsm8k.select_by_token_budget(rows, token_budget=10, minimum_rows=2)


def test_an_empty_tokenizer_is_a_loud_failure() -> None:
    """The observed silent failure: no sidecars, one vocabulary entry, every string encodes to nothing."""
    with pytest.raises(ValueError, match="cannot represent"):
        gsm8k.assert_vocabulary_matches_model(1, 151936, "/models/reduced")


def test_a_tokenizer_wider_than_the_embedding_table_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot look up"):
        gsm8k.assert_vocabulary_matches_model(200000, 151936, "/models/reduced")


def test_a_tokenizer_inside_the_embedding_table_is_accepted() -> None:
    gsm8k.assert_vocabulary_matches_model(151669, 151936, "/models/reduced")


def test_the_padded_columns_fall_back_to_the_end_of_sequence_token() -> None:
    assert gsm8k.padding_token_id(SimpleNamespace(pad_token_id=7, eos_token_id=9)) == 7
    assert gsm8k.padding_token_id(SimpleNamespace(pad_token_id=None, eos_token_id=9)) == 9
    with pytest.raises(ValueError):
        gsm8k.padding_token_id(SimpleNamespace(pad_token_id=None, eos_token_id=None))


def test_a_shared_question_is_refused() -> None:
    shared = example("same question", " a", [1, 2, 3], 1)

    with pytest.raises(ValueError, match="both"):
        gsm8k.assert_disjoint([shared], [shared])


def test_the_validation_split_shares_no_question_with_the_training_split() -> None:
    """The two slices the check reads, from the dataset's own files, with nothing in common.

    A validation loss measured on rows the run trained on would agree between the two engines for a reason
    that has nothing to do with the checkpoint, so the disjointness is read off the data rather than
    inferred from the file names.
    """
    from arctic_platform.correctness.checks import inference_checkpoint_loss as check

    training = gsm8k.read_rows(check.TRAIN_SPLIT)
    held_out = gsm8k.read_rows(check.VALIDATION_SPLIT)

    assert training and held_out
    assert not {question for question, _ in training} & {question for question, _ in held_out}
