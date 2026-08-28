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

"""Golden-file snapshot for the Arctic wire payload.

``_prepare_padded_arctic_batch_dict`` converts verl's nested-jagged
input format into the dense padded layout Arctic expects on the wire.
It sits on the critical path between the two frameworks so a regression
here is silent and expensive to find via E2E. This test pins the
shape/values for a small hand-computed input; any wire-format change
must intentionally update these numbers.
"""

from __future__ import annotations

import importlib

import pytest
import torch

# The adapter module pulls in `omegaconf` at import time (used by
# `_create_ds_config`). If it isn't installed (e.g. running on a
# minimal CPU box without the [verl] / [rl] extras), skip the entire
# module -- the helper we exercise is behind that import.
pytest.importorskip("omegaconf")
pytest.importorskip("transformers")


PAD = 0


def _make_batch() -> dict:
    """Two jagged sequences: (prompt_len, response_len) = (3, 2) and (2, 3).

    Uses nested/jagged tensors -- the primary code path taken by
    ``_prepare_padded_arctic_batch_dict``. The rectangular (non-nested)
    fallback branch is exercised only in production by upstream verl
    workers that already flattened the batch and is intentionally not
    covered here.
    """
    seq0_prompt = [10, 11, 12]
    seq0_response = [30, 31]
    seq1_prompt = [20, 21]
    seq1_response = [40, 41, 42]

    input_ids = torch.nested.nested_tensor(
        [
            torch.tensor(seq0_prompt + seq0_response, dtype=torch.long),
            torch.tensor(seq1_prompt + seq1_response, dtype=torch.long),
        ],
        layout=torch.jagged,
    )
    prompts = torch.nested.nested_tensor(
        [
            torch.tensor(seq0_prompt, dtype=torch.long),
            torch.tensor(seq1_prompt, dtype=torch.long),
        ],
        layout=torch.jagged,
    )
    responses = torch.nested.nested_tensor(
        [
            torch.tensor(seq0_response, dtype=torch.long),
            torch.tensor(seq1_response, dtype=torch.long),
        ],
        layout=torch.jagged,
    )
    # attention_mask is rectangular (padded to max full seq len); the
    # helper only reads it on the non-nested branch, but downstream
    # code still expects it in the batch dict.
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],  # prompt 3 + response 2, right-pad 1
            [1, 1, 1, 1, 1, 0],  # prompt 2 + response 3, right-pad 1
        ],
        dtype=torch.long,
    )
    position_ids = torch.nested.nested_tensor(
        [
            torch.arange(len(seq0_prompt) + len(seq0_response), dtype=torch.long),
            torch.arange(len(seq1_prompt) + len(seq1_response), dtype=torch.long),
        ],
        layout=torch.jagged,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompts": prompts,
        "responses": responses,
        "position_ids": position_ids,
    }


def _adapter(verl_stub):
    """Import the adapter under the shared verl / arctic_platform.rl stubs."""
    return importlib.import_module("arctic_platform.integrations.verl.adapter")


def test_prepare_padded_arctic_batch_dict_shape(verl_stub) -> None:
    adapter = _adapter(verl_stub)

    batch_dict, max_prompt_len, max_response_len = adapter._prepare_padded_arctic_batch_dict(
        _make_batch(), pad_token_id=PAD, drop_position_ids=True
    )

    # Nested (jagged) inputs -> max_prompt_len / max_response_len come
    # from per-row lengths (max prompt=3, max response=3).
    assert max_prompt_len == 3
    assert max_response_len == 3
    assert batch_dict["input_ids"].shape == (2, 6)
    assert "attention_mask" in batch_dict
    assert "prompts" in batch_dict
    # drop_position_ids=True -> position_ids not included in the payload.
    assert "position_ids" not in batch_dict


def test_prepare_padded_arctic_batch_dict_values(verl_stub) -> None:
    """Pin the exact wire bytes for a hand-computed input.

    A change here means the on-wire format changed and every deployed
    Arctic server has to be updated in lockstep -- so this test is a
    tripwire, not a spec.
    """
    adapter = _adapter(verl_stub)

    batch_dict, _, _ = adapter._prepare_padded_arctic_batch_dict(
        _make_batch(), pad_token_id=PAD, drop_position_ids=True
    )

    # For each row: left-pad prompt to 3, right-pad response to 3.
    # seq 0: prompt=[10,11,12] (no pad), response=[30,31] (right-pad 1)
    # seq 1: prompt=[20,21]    (left-pad 1), response=[40,41,42] (no pad)
    expected_input_ids = torch.tensor(
        [
            [10, 11, 12, 30, 31, PAD],
            [PAD, 20, 21, 40, 41, 42],
        ],
        dtype=torch.long,
    )
    assert torch.equal(
        batch_dict["input_ids"], expected_input_ids
    ), f"Wire format regressed:\n  got={batch_dict['input_ids'].tolist()}\n  expected={expected_input_ids.tolist()}"


def test_prepare_padded_arctic_batch_dict_keeps_position_ids(verl_stub) -> None:
    """drop_position_ids=False -> position_ids must round-trip in the payload."""
    adapter = _adapter(verl_stub)

    batch_dict, _, _ = adapter._prepare_padded_arctic_batch_dict(
        _make_batch(), pad_token_id=PAD, drop_position_ids=False
    )

    assert "position_ids" in batch_dict
    assert batch_dict["position_ids"].shape == batch_dict["input_ids"].shape


@pytest.mark.parametrize("drop_position_ids", [True, False])
def test_prepare_padded_arctic_batch_dict_pad_token_used(verl_stub, drop_position_ids: bool) -> None:
    """Non-zero pad tokens land in the padded slots, not the payload slots."""
    adapter = _adapter(verl_stub)

    non_zero_pad = 99
    batch_dict, _, _ = adapter._prepare_padded_arctic_batch_dict(
        _make_batch(), pad_token_id=non_zero_pad, drop_position_ids=drop_position_ids
    )

    # Row 0: response is length 2, padded to 3 on the right with the
    # sentinel; row 1: prompt is length 2, padded to 3 on the left.
    assert batch_dict["input_ids"][0, -1].item() == non_zero_pad
    assert batch_dict["input_ids"][1, 0].item() == non_zero_pad
