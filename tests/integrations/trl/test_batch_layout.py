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

"""CPU tests for packed ``[1, T]`` <-> padded ``[B, S]`` layout helpers."""

from __future__ import annotations

import pytest
import torch

# client.py imports `trl.experimental.api` at module load; skip cleanly on a minimal image.
pytest.importorskip("trl.experimental.api")

from arctic_platform.integrations.trl import client as C  # noqa: E402
from arctic_platform.testing_utils import torch_assert_equal  # noqa: E402

# Two packed sequences of real-token length 3 and 2 -> packed row of length T=5.
PACKED_IDS = torch.tensor([[10, 11, 12, 30, 31]], dtype=torch.long)
PACKED_POS = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)
PACKED_MASK = torch.tensor([[0, 0, 1, 0, 1]], dtype=torch.long)  # completion_mask (unused by unpack, but real-shaped)
SEQ_LENS = torch.tensor([3, 2], dtype=torch.long)


class TestSegmentLengths:
    def test_lengths_from_position_resets(self):
        torch_assert_equal(C._segment_lengths(PACKED_POS), SEQ_LENS)

    def test_single_sequence(self):
        torch_assert_equal(C._segment_lengths(torch.tensor([[0, 1, 2, 3]])), torch.tensor([4]))

    def test_three_uneven_sequences(self):
        pos = torch.tensor([[0, 1, 0, 1, 2, 3, 0]])  # lengths 2, 4, 1
        torch_assert_equal(C._segment_lengths(pos), torch.tensor([2, 4, 1]))

    def test_missing_leading_zero_is_tolerated(self):
        # Missing leading 0: still inject a start at index 0.
        pos = torch.tensor([[1, 2, 0, 1]])  # -> starts {0 (injected), 2} -> lengths 2, 2
        torch_assert_equal(C._segment_lengths(pos), torch.tensor([2, 2]))


class TestUnpackToPaddedRows:
    def test_shapes_and_values(self):
        batch = C._unpack_to_padded_rows(PACKED_IDS, PACKED_POS, PACKED_MASK, SEQ_LENS)

        torch_assert_equal(batch["input_ids"], torch.tensor([[10, 11, 12], [30, 31, 0]], dtype=torch.long))
        torch_assert_equal(batch["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long))
        # position_ids are a per-row arange over real tokens, 0 on pad.
        torch_assert_equal(batch["position_ids"], torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long))
        # Zero-width prompts: the whole row is treated as response (client masks prompts itself).
        assert batch["prompts"].shape == (2, 0)

    def test_pad_slot_is_zero_id(self):
        batch = C._unpack_to_padded_rows(PACKED_IDS, PACKED_POS, PACKED_MASK, SEQ_LENS)
        # The [1,2] slot is padding for the length-2 second sequence.
        assert batch["input_ids"][1, 2].item() == 0
        assert batch["attention_mask"][1, 2].item() == 0


class TestUnpackRepackRoundTrip:
    @pytest.mark.parametrize(
        "row,lens",
        [
            (torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]), torch.tensor([3, 2])),
            (torch.tensor([[0.5, -1.0, 2.0, 3.0, -4.0, 5.0]]), torch.tensor([2, 1, 3])),
            (torch.tensor([[7.0, 8.0, 9.0]]), torch.tensor([3])),  # single sequence
        ],
    )
    def test_repack_is_left_inverse_of_unpack(self, row, lens):
        # _repack_to_row(_unpack_to_padded(row)) must return the original packed row exactly (real
        # entries only; the padded scratch space in between never leaks back).
        padded = C._unpack_to_padded(row, lens)
        assert padded.shape == (int(lens.numel()), int(lens.max().item()))
        torch_assert_equal(C._repack_to_row(padded, lens), row)

    def test_unpack_zeros_the_pad(self):
        padded = C._unpack_to_padded(torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([2, 1]))
        # row0 real len 2 -> [1,2]; row1 real len 1 -> [3, 0(pad)]
        torch_assert_equal(padded, torch.tensor([[1.0, 2.0], [3.0, 0.0]]))

    def test_empty_seq_lens(self):
        assert C._repack_to_row(torch.zeros((0, 0)), torch.tensor([], dtype=torch.long)).shape == (1, 0)


class TestShiftForTrl:
    def test_drops_trailing_slot(self):
        # roll(-1) frame: entry j = log p(token j+1). TRL consumes log_probs[k] against
        # old_log_probs[:,1:][k] (token k+1), so the correct alignment drops the LAST slot (the
        # per-row roll-around), not the first.
        row = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        torch_assert_equal(C._shift_for_trl(row), torch.tensor([[1.0, 2.0, 3.0]]))

    def test_unshift_restores_shape_with_trailing_zero(self):
        row = torch.tensor([[1.0, 2.0, 3.0]])
        restored = C._unshift_from_trl(C._shift_for_trl(row))
        # Round-trips shape [1, T]; the dropped trailing slot comes back as 0 (never a valid
        # completion position), so it equals the original with its last entry zeroed.
        assert restored.shape == row.shape
        torch_assert_equal(restored, torch.tensor([[1.0, 2.0, 0.0]]))

    def test_shift_unshift_preserves_leading_entries(self):
        row = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5]])
        restored = C._unshift_from_trl(C._shift_for_trl(row))
        torch_assert_equal(restored[:, :-1], row[:, :-1])
        assert restored[0, -1].item() == 0.0


class TestPadRows:
    def test_shapes_values_and_lengths(self):
        batch, lens = C._pad_rows([[10, 11, 12], [30, 31]], pad_token_id=0)
        assert lens == [3, 2]
        torch_assert_equal(batch["input_ids"], torch.tensor([[10, 11, 12], [30, 31, 0]], dtype=torch.long))
        torch_assert_equal(batch["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long))
        torch_assert_equal(batch["position_ids"], torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long))
        assert batch["prompts"].shape == (2, 0)

    def test_nonzero_pad_token_lands_in_pad_slots_only(self):
        batch, _ = C._pad_rows([[10, 11, 12], [30, 31]], pad_token_id=99)
        # Only the [1,2] pad slot gets the sentinel; real tokens are untouched.
        assert batch["input_ids"][1, 2].item() == 99
        assert batch["input_ids"][0].tolist() == [10, 11, 12]
        assert batch["attention_mask"][1, 2].item() == 0

    def test_equal_length_rows(self):
        batch, lens = C._pad_rows([[1, 2], [3, 4]], pad_token_id=0)
        assert lens == [2, 2]
        assert batch["input_ids"].shape == (2, 2)
        torch_assert_equal(batch["attention_mask"], torch.ones((2, 2), dtype=torch.long))
