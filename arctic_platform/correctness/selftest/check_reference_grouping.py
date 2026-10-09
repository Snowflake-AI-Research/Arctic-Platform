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

"""Selftests for diagnostic-only reference row grouping."""

from __future__ import annotations

import pytest
import torch

from arctic_platform.correctness.reference.hf_single_gpu import fixed_row_groups
from arctic_platform.correctness.reference.hf_single_gpu import mixer_boundaries
from arctic_platform.correctness.reference.hf_single_gpu import mixer_call_inputs
from arctic_platform.correctness.reference.hf_single_gpu import reference_row_groups


def test_mixer_packing_preserves_one_row_reference_default() -> None:
    groups = reference_row_groups([8192] * 4, 8192, 65536, mixer_packing=True)

    assert groups == [[0], [1], [2], [3]]


def test_mixer_packing_can_group_rows_for_diagnostics() -> None:
    groups = reference_row_groups(
        [8192] * 10,
        8192,
        65536,
        mixer_packing=True,
        mixer_packing_group_rows=4,
    )

    assert groups == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_fixed_row_groups_rejects_non_positive_group_size() -> None:
    with pytest.raises(ValueError, match="group_rows"):
        fixed_row_groups([1, 2, 3], 0)


def test_mixer_boundaries_describe_grouped_row_edges() -> None:
    boundaries = mixer_boundaries(row_length=4, rows=3, device="cpu")

    assert torch.equal(boundaries["cu_seq_lens_q"], torch.tensor([0, 4, 8, 12], dtype=torch.int32))
    assert torch.equal(
        boundaries["seq_idx"],
        torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [2, 2, 2, 2],
            ],
            dtype=torch.int32,
        ),
    )


def test_mixer_boundaries_can_flatten_grouped_rows_for_varlen_decoder() -> None:
    boundaries = mixer_boundaries(row_length=4, rows=3, device="cpu", flattened=True)

    assert torch.equal(boundaries["cu_seq_lens_q"], torch.tensor([0, 4, 8, 12], dtype=torch.int32))
    assert torch.equal(
        boundaries["seq_idx"],
        torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]], dtype=torch.int32),
    )


def test_mixer_call_inputs_flatten_only_multirow_varlen_calls() -> None:
    input_ids = torch.arange(12).reshape(3, 4)
    labels = input_ids + 100

    packed_ids, packed_labels, mixer_kwargs = mixer_call_inputs(input_ids, labels, "cpu")

    assert torch.equal(packed_ids, torch.arange(12).reshape(1, 12))
    assert torch.equal(packed_labels, torch.arange(100, 112).reshape(1, 12))
    assert torch.equal(mixer_kwargs["seq_idx"], torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]]))

    one_ids, one_labels, one_kwargs = mixer_call_inputs(input_ids[:1], labels[:1], "cpu")

    assert one_ids.shape == (1, 4)
    assert one_labels.shape == (1, 4)
    assert one_kwargs["seq_idx"].shape == (1, 4)
