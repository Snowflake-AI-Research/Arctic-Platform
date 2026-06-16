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

"""CPU unit test for the flash-attention sequence packer (``processors/packing.py``).

``pack_sequences`` flattens a padded ``[B, S]`` tensor dict into a single ``[1, T]`` row with ``cu_seqlens`` /
``position_ids``; ``unpack_sequences`` reverses it; ``pad_packed_for_model`` pads the packed row to a 256-token page
boundary for the model forward. The live DeepSpeed worker runs with ``pack=False``, so these are exercised directly
on CPU via a pack -> unpack round-trip (no GPU, no dist)::

    pytest tests/rl/test_packing.py
"""

from __future__ import annotations

import torch

from arctic_platform.rl.processors.packing import N_TOKENS_PER_PAGE
from arctic_platform.rl.processors.packing import pack_sequences
from arctic_platform.rl.processors.packing import pad_packed_for_model
from arctic_platform.rl.processors.packing import unpack_sequences
from arctic_platform.testing_utils import TestCasePlus

seq_len = 6
real_token_counts = [6, 4, 2]  # per-row real tokens; total T = 12


def _make_padded_dict() -> dict:
    gen = torch.Generator().manual_seed(0)
    batch_size = len(real_token_counts)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for row, count in enumerate(real_token_counts):
        attention_mask[row, :count] = 1
    # Zero the padded region so an unpack (which pad-fills with 0) reconstructs the input exactly.
    input_ids = torch.randint(1, 100, (batch_size, seq_len), generator=gen) * attention_mask
    extra = torch.randn(batch_size, seq_len, 2, generator=gen) * attention_mask.unsqueeze(-1)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "extra": extra}


class TestPackSequences(TestCasePlus):
    def test_pack_shapes_and_metadata(self):
        data = _make_padded_dict()
        total_tokens = sum(real_token_counts)
        packed = pack_sequences(data)

        self.assertEqual(packed["input_ids"].shape, (1, total_tokens))
        self.assertEqual(packed["extra"].shape, (1, total_tokens, 2))
        self.assertEqual(packed["cu_seqlens"].tolist(), [0, 6, 10, 12])
        # position_ids restart at 0 within each packed sequence.
        expected_positions = torch.cat([torch.arange(n) for n in real_token_counts])
        self.assertTrue(torch.equal(packed["position_ids"][0], expected_positions))

    def test_round_trip_reconstructs_padded_tensors(self):
        data = _make_padded_dict()
        packed = pack_sequences(data)
        meta = packed["_pack_meta"]
        for key in ("input_ids", "extra"):
            unpacked = unpack_sequences(packed[key], meta)
            self.assertTrue(torch.equal(unpacked, data[key]), f"{key} did not round-trip through pack/unpack")

    def test_pad_packed_for_model_aligns_to_page(self):
        data = _make_padded_dict()
        packed = pack_sequences(data)
        model_kwargs, pad_length = pad_packed_for_model(packed)

        total_tokens = sum(real_token_counts)
        padded_len = model_kwargs["input_ids"].shape[1]
        self.assertEqual(padded_len % N_TOKENS_PER_PAGE, 0, "packed row not aligned to a page boundary")
        self.assertEqual(pad_length, padded_len - total_tokens)
        self.assertEqual(model_kwargs["max_length_q"], max(real_token_counts))
        self.assertTrue(torch.equal(model_kwargs["cu_seq_lens_q"], packed["cu_seqlens"]))
