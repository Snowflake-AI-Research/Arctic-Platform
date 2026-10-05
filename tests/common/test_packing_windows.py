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

"""Packing-module unit tests: pure tensor plumbing, no ray/arctic_training/GPU."""

import pytest
import torch

from arctic_platform.common.packing import IGNORE_INDEX
from arctic_platform.common.packing import PackMetadata
from arctic_platform.common.packing import pack_microbatch
from arctic_platform.common.packing import pad_packed_microbatch
from arctic_platform.common.packing import token_budget_groups
from arctic_platform.common.packing import token_shard
from arctic_platform.common.packing import token_validity_mask
from arctic_platform.common.packing import token_window
from arctic_platform.common.packing import unpack_output
from arctic_platform.common.packing import window_cu_seqlens
from arctic_platform.common.packing import window_row_pieces


def _packed(row_lengths, pad_to=1):
    """Pack rows of the given valid lengths, right-padded to a common width first."""
    width = max(row_lengths)
    batch = {
        "input_ids": torch.zeros(len(row_lengths), width, dtype=torch.long),
        "position_ids": torch.zeros(len(row_lengths), width, dtype=torch.long),
        "labels": torch.full((len(row_lengths), width), IGNORE_INDEX, dtype=torch.long),
        "attention_mask": torch.zeros(len(row_lengths), width, dtype=torch.long),
    }
    token = 1
    for row, length in enumerate(row_lengths):
        batch["attention_mask"][row, :length] = 1
        batch["position_ids"][row, :length] = torch.arange(length)
        batch["labels"][row, :length] = torch.arange(length)
        batch["input_ids"][row, :length] = torch.arange(token, token + length)
        token += length
    packed, metadata = pack_microbatch(batch)
    return pad_packed_microbatch(packed, metadata, pad_to)


def test_token_budget_groups_place_long_rows_before_short_ones():
    """Descending placement, asserted against a grouping written out by hand rather than by the planner.

    Ascending order fills the first group with short rows, and the long rows then have nowhere to go but a group
    each. Lengths [6, 5, 4, 3] under a budget of 9 cost two calls placed longest-first (6+3 and 5+4) and three
    placed shortest-first (3+4, then 5, then 6), so the order is worth a model call here, and any test that
    derives its expectation from the planner cannot see the difference.
    """
    assert token_budget_groups([6, 5, 4, 3], 9) == [[0, 3], [1, 2]]

    # Same policy on lengths where both orders cost three calls: only the composition separates them.
    assert token_budget_groups([8, 7, 2, 2], 9) == [[0], [1, 2], [3]]

    # A row wider than the budget gets its own call rather than being dropped or split.
    assert token_budget_groups([12, 4, 4], 9) == [[0], [1, 2]]


def test_token_validity_reads_position_ids_when_the_request_ships_no_mask():
    """Positions alone must delimit a row's real tokens.

    A request large enough to need packing ships ``position_ids`` and no ``attention_mask``, because a 2D mask
    over a packed batch costs more to move than the positions it encodes. A row's real tokens are the leading
    run whose positions advance by one from the row's own origin; padding repeats or resets, so it falls out of
    that comparison. Deriving validity from ``input_ids`` instead would count padding as real, which sets every
    row's length to the full padded width and makes the token budget group by width rather than by content.
    """
    position_ids = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 0, 0, 0]])
    batch = {
        "input_ids": torch.ones(2, 5, dtype=torch.long),
        "position_ids": position_ids,
    }

    mask = token_validity_mask(batch)

    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
    # The row lengths the planner consumes: padding must not reach the budget.
    assert mask.sum(dim=1).tolist() == [3, 2]


def test_token_validity_stops_at_the_first_break_in_the_positions():
    """Validity is the leading run, so a coincidence after the break must not revive a pad token.

    Positions are only trustworthy up to the first discontinuity. Past it the row is padding, whose values are
    whatever the producer left there -- and a pad token whose value happens to equal ``p0 + i`` would otherwise be
    counted as real. That miscount is silent and it moves tokens: the packer bills the row a length one larger than
    its real prefix, so the extra column is copied into the call and the row's last real token is pushed out.
    """
    batch = {
        "input_ids": torch.ones(1, 5, dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 9, 4]]),
    }

    assert token_validity_mask(batch).tolist() == [[1, 1, 1, 0, 0]]


def test_packing_rejects_a_mask_whose_real_tokens_are_not_a_prefix():
    """Left padding has to be refused, because the packer reads lengths and then takes leading columns.

    ``pack_microbatch`` bills each row ``attention_mask.sum()`` tokens and copies that many columns from the row's
    start, so validity anywhere other than the front silently packs pad tokens and drops real ones -- an error that
    surfaces, if at all, as a loss that is merely wrong. Rows must be right-padded before they reach the packer.
    """
    batch = {
        "input_ids": torch.tensor([[0, 0, 11, 12, 13]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1]]),
        "position_ids": torch.tensor([[0, 0, 0, 1, 2]]),
    }

    with pytest.raises(ValueError, match="left-aligned"):
        pack_microbatch(batch)


def test_token_validity_prefers_a_supplied_mask_over_positions():
    """A request that does ship a 2D mask is authoritative: positions are the fallback, not an override."""
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        "position_ids": torch.tensor([[0, 1, 2, 3]]),
    }

    assert token_validity_mask(batch).tolist() == [[1, 1, 0, 0]]


def test_token_validity_rejects_a_batch_carrying_neither_signal():
    """Silently treating every token as real would corrupt every downstream length."""
    with pytest.raises(ValueError, match="attention_mask or 2D position_ids"):
        token_validity_mask({"input_ids": torch.ones(1, 4, dtype=torch.long)})


def test_token_budget_groups_are_ordered_and_complete():
    """Every rank derives the plan from its own shard, so the plan has to be a pure function of the lengths."""
    groups = token_budget_groups([3, 9, 1, 2, 5], 10)

    assert sorted(row for group in groups for row in group) == [0, 1, 2, 3, 4]
    assert groups == sorted(sorted(group) for group in groups)
    assert all(sum([3, 9, 1, 2, 5][row] for row in group) <= 10 for group in groups)


def test_splitting_to_a_call_count_keeps_the_rows_and_shrinks_the_widest_group():
    from arctic_platform.common.packing import split_groups_to_count

    lengths = [8, 8, 1]
    # The two-row group holds the most tokens, so it is the one that gives up a row, and it gives up a longest.
    assert split_groups_to_count([[0, 1], [2]], lengths, 3) == [[0], [1], [2]]
    assert split_groups_to_count([[0, 1, 2]], lengths, 2) == [[0], [1, 2]]
    # Already at the count: unchanged.
    assert split_groups_to_count([[0], [1, 2]], lengths, 2) == [[0], [1, 2]]

    # Two groups could each be split, so this is where the choice between them is visible. Splitting the wider
    # one lowers the request's widest call, which is what bounds peak activation memory; splitting the narrower
    # one leaves that call untouched and buys nothing.
    wide_and_narrow = [60, 40, 6, 4]
    assert split_groups_to_count([[0, 1], [2, 3]], wide_and_narrow, 3) == [[0], [1], [2, 3]]


def test_splitting_stops_when_no_group_holds_more_than_one_row():
    from arctic_platform.common.packing import split_groups_to_count

    # Three calls are asked for and two rows exist, so the result is short and the caller pads the remainder.
    assert split_groups_to_count([[0, 1]], [4, 4], 3) == [[0], [1]]


def test_splitting_refuses_to_lower_a_call_count():
    from arctic_platform.common.packing import split_groups_to_count

    with pytest.raises(ValueError, match="cannot reduce"):
        split_groups_to_count([[0], [1]], [4, 4], 1)


def test_window_cu_seqlens_are_the_row_boundaries_when_nothing_is_sharded():
    _packed_call, metadata = _packed([3, 2])

    boundaries = window_cu_seqlens(metadata, 1, 0)

    assert boundaries.tolist() == [0, 3, 5]
    assert boundaries.tolist() == metadata.cu_seqlens.tolist()


def test_window_cu_seqlens_clip_each_row_to_the_shard_that_holds_it():
    """Rows entirely outside a shard's window become empty segments, not missing ones.

    Loss reductions index per-rollout sums by segment, so every shard must report one segment per packed row
    even when it holds none of that row's tokens.
    """
    _packed_call, metadata = _packed([3, 3])

    assert window_cu_seqlens(metadata, 2, 0).tolist() == [0, 3, 3]
    assert window_cu_seqlens(metadata, 2, 1).tolist() == [0, 0, 3]


def test_window_cu_seqlens_split_a_row_that_straddles_the_shard_boundary():
    """A row cut mid-sequence contributes a suffix segment to one shard and a prefix to the next.

    Summing a row's segment over both shards must recover the row's whole token count, which is what makes a
    per-rollout reduction correct when the packed sequence does not divide along row boundaries.
    """
    _packed_call, metadata = _packed([4, 2])

    first = window_cu_seqlens(metadata, 2, 0).tolist()
    second = window_cu_seqlens(metadata, 2, 1).tolist()

    assert first == [0, 3, 3]
    assert second == [0, 1, 3]
    row_tokens = [
        (first[row + 1] - first[row]) + (second[row + 1] - second[row]) for row in range(metadata.batch_size)
    ]
    assert row_tokens == [4, 2]


def test_window_cu_seqlens_fold_tail_padding_into_the_last_row():
    """Padding added for shard divisibility must not become an extra segment.

    Losses that weight per rollout require exactly one segment per row, and they also check that the
    boundaries cover the window exactly -- so the pad tail belongs to the last row, whose ``IGNORE_INDEX``
    labels and zero loss weight keep it from contributing.
    """
    _packed_call, metadata = _packed([3], pad_to=2)

    assert metadata.padded_tokens == 4
    boundaries = window_cu_seqlens(metadata, 2, 1)

    assert boundaries.numel() == metadata.batch_size + 1
    assert boundaries.tolist() == [0, 2]


def test_window_cu_seqlens_cover_every_window_exactly():
    _packed_call, metadata = _packed([5, 1, 2], pad_to=4)

    for shard_index in range(4):
        boundaries = window_cu_seqlens(metadata, 4, shard_index)
        window_start, window_end = token_window(metadata.padded_tokens, 4, shard_index)

        assert boundaries.numel() == metadata.batch_size + 1
        assert int(boundaries[0]) == 0
        assert int(boundaries[-1]) == window_end - window_start
        assert (boundaries[1:] - boundaries[:-1] >= 0).all()


def test_window_row_pieces_agree_with_the_window_boundaries():
    """The token pieces handed back to the head must be the segments the loss reduced over.

    Lengths 4 and 2 pack to 6 tokens. Padding to a multiple of 4 makes 8, so the last
    shard's window includes a real pad tail and the adjustment below has to run.
    """
    packed_call, metadata = _packed([4, 2], pad_to=4)
    input_ids = packed_call["input_ids"]
    assert metadata.padded_tokens == 8

    saw_pad_tail = False
    for shard_index in range(2):
        window_start, window_end = token_window(metadata.padded_tokens, 2, shard_index)
        window = input_ids[:, window_start:window_end]
        pieces = window_row_pieces(window, metadata, 2, shard_index)
        boundaries = window_cu_seqlens(metadata, 2, shard_index).tolist()

        # window_cu_seqlens folds the pad tail into the last row. window_row_pieces
        # stops at the real row end, so that tail has to be subtracted back out.
        real_tokens = int(metadata.cu_seqlens[-1])
        pad_in_window = max(0, window_end - max(window_start, real_tokens))
        if pad_in_window:
            saw_pad_tail = True
        for row, piece in enumerate(pieces):
            segment = boundaries[row + 1] - boundaries[row]
            if row == metadata.batch_size - 1:
                segment -= pad_in_window
            assert piece.numel() == segment
    assert saw_pad_tail


def test_window_row_pieces_reject_a_value_with_no_token_axis():
    """A model output that is a scalar has to be named as one, because it is a value the RL worker really sees.

    The pipeline answers with every model output except the logits, so a forward that receives labels answers
    with a 0-dim loss beside its per-token tensors. Reading ``shape[0]`` of that value raises ``IndexError:
    tuple index out of range`` from inside the helper, which names neither the value nor the caller that sent
    it; the error has to say that the window carries no tokens.
    """
    _packed_call, metadata = _packed([4, 2], pad_to=2)

    with pytest.raises(ValueError, match="token axis"):
        window_row_pieces(torch.tensor(0.5), metadata, 2, 0)

    # Shard 0 of this padded call owns three tokens, all of them in the first row.
    assert [piece.numel() for piece in window_row_pieces(torch.zeros(3), metadata, 2, 0)] == [3, 0]


def _packed_with_context(valid_tokens: int, pad_to: int):
    """Pack one row whose per-token RL tensors live in a nested ``context``, as RL requests ship them."""
    batch = {
        "input_ids": torch.arange(1, valid_tokens + 1, dtype=torch.long).unsqueeze(0),
        "position_ids": torch.arange(valid_tokens, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.ones(1, valid_tokens, dtype=torch.long),
        "temperature": torch.ones(1, valid_tokens, dtype=torch.float32),
        "context": {
            "loss_mask": torch.ones(1, valid_tokens, dtype=torch.float32),
            "advantages": torch.ones(1, valid_tokens, dtype=torch.float32),
            "labels": torch.arange(valid_tokens, dtype=torch.long).unsqueeze(0),
        },
    }
    packed, metadata = pack_microbatch(batch)
    return pad_packed_microbatch(packed, metadata, pad_to)


def test_divisibility_padding_reaches_tensors_nested_in_context():
    """RL ships per-token tensors inside ``context``; padding only the top level leaves them a different width.

    A leaf left at the unpadded width no longer matches the width ``token_shard`` slices, so it would travel to
    the worker whole while its siblings arrive sharded -- and the worker flattens ``context`` over the top level,
    so the unsharded copy wins.
    """
    packed, metadata = _packed_with_context(97, 2)

    assert metadata.padded_tokens == 98
    assert int(packed["input_ids"].shape[1]) == 98
    for key, value in packed["context"].items():
        assert int(value.shape[1]) == 98, f"context[{key!r}] was not padded"

    assert packed["context"]["loss_mask"][0, -1].item() == 0.0
    assert packed["context"]["labels"][0, -1].item() == IGNORE_INDEX
    assert packed["temperature"][0, -1].item() == 1.0


def test_token_shard_slices_context_leaves_to_the_same_width_as_model_inputs():
    packed, metadata = _packed_with_context(97, 2)

    window = token_shard(packed, metadata, 2, 1)

    assert int(window["input_ids"].shape[1]) == 49
    for key, value in window["context"].items():
        assert int(value.shape[1]) == 49, f"context[{key!r}] was not sharded"


def test_token_shard_rejects_a_token_tensor_that_escaped_the_divisibility_padding():
    packed, metadata = _packed_with_context(97, 2)
    packed["context"]["loss_mask"] = torch.ones(1, 97, dtype=torch.float32)

    with pytest.raises(ValueError, match="unpadded token width"):
        token_shard(packed, metadata, 2, 0)


def _row_by_row_pack(value, row_lengths):
    """Reference pack: copy each row's leading valid tokens, one row at a time."""
    packed = torch.empty((sum(row_lengths), *value.shape[2:]), dtype=value.dtype)
    offset = 0
    for row, length in enumerate(row_lengths):
        packed[offset : offset + length] = value[row, :length]
        offset += length
    return packed.unsqueeze(0)


@pytest.mark.parametrize("row_lengths", [(3, 2), (1, 7, 4), (5, 0, 5), (8,), (2, 2, 2, 2)])
def test_packing_places_the_same_tokens_as_a_row_by_row_copy(row_lengths):
    width = max(row_lengths)
    rows = len(row_lengths)
    batch = {
        "input_ids": torch.arange(rows * width, dtype=torch.long).reshape(rows, width),
        "position_ids": torch.zeros(rows, width, dtype=torch.long),
        "attention_mask": torch.zeros(rows, width, dtype=torch.long),
        "context": {
            # A trailing dimension and a bool leaf: packing addresses tokens, whatever rides on them.
            "hidden": torch.randn(rows, width, 4),
            "loss_mask": torch.randint(0, 2, (rows, width), dtype=torch.bool),
        },
    }
    for row, length in enumerate(row_lengths):
        batch["attention_mask"][row, :length] = 1
        batch["position_ids"][row, :length] = torch.arange(length)

    packed, metadata = pack_microbatch(batch)

    assert metadata.padded_tokens == sum(row_lengths)
    assert torch.equal(packed["input_ids"], _row_by_row_pack(batch["input_ids"], row_lengths))
    assert torch.equal(packed["position_ids"], _row_by_row_pack(batch["position_ids"], row_lengths))
    assert torch.equal(packed["context"]["hidden"], _row_by_row_pack(batch["context"]["hidden"], row_lengths))
    assert torch.equal(packed["context"]["loss_mask"], _row_by_row_pack(batch["context"]["loss_mask"], row_lengths))


@pytest.mark.parametrize("row_lengths", [(3, 2), (5, 0, 5), (1, 7, 4)])
def test_unpacking_returns_every_token_to_the_row_it_came_from(row_lengths):
    width = max(row_lengths)
    rows = len(row_lengths)
    hidden = torch.randn(rows, width, 3)
    batch = {
        "input_ids": torch.zeros(rows, width, dtype=torch.long),
        "position_ids": torch.zeros(rows, width, dtype=torch.long),
        "attention_mask": torch.zeros(rows, width, dtype=torch.long),
        "hidden": hidden,
    }
    for row, length in enumerate(row_lengths):
        batch["attention_mask"][row, :length] = 1

    packed, metadata = pack_microbatch(batch)
    # Pad the call the way a sequence split requires: the tail tokens belong to no row and must not land in one.
    packed, metadata = pad_packed_microbatch(packed, metadata, 4)
    restored = unpack_output(packed["hidden"], metadata)

    assert restored.shape == hidden.shape
    for row, length in enumerate(row_lengths):
        assert torch.equal(restored[row, :length], hidden[row, :length])
        assert torch.equal(restored[row, length:], torch.zeros(width - length, 3))


def test_unpacking_rejects_metadata_whose_boundaries_contradict_its_row_lengths():
    metadata = PackMetadata(
        batch_size=2,
        sequence_length=4,
        valid_lengths=torch.tensor([3, 1], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
        padded_tokens=4,
    )

    with pytest.raises(ValueError, match="disagrees with itself"):
        unpack_output(torch.zeros(1, 4), metadata)


def test_model_packing_reexports_the_shared_definitions():
    """The model stack and this package must share one IGNORE_INDEX and one position decoder."""
    from arctic_platform.common.packing import IGNORE_INDEX as shared_ignore
    from arctic_platform.common.packing import cu_seqlens_from_position_ids as shared_cu
    from arctic_platform.model.implementations.gpu.packing import IGNORE_INDEX as model_ignore
    from arctic_platform.model.implementations.gpu.packing import cu_seqlens_from_position_ids as model_cu

    assert model_ignore is shared_ignore
    assert model_cu is shared_cu
    assert shared_ignore == -100
