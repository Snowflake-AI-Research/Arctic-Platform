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

"""CPU unit test for the token-budget microbatch splitter (``processors/microbatch.py``).

``split_padded_tensor_dict_into_mb_list`` first-fit-decreasing-packs padded rows into microbatches under a token
budget. The DeepSpeed worker microbatches via ``gradient_accumulation_steps`` and calls ``run_pipeline(pack=False)``,
so this packer is never reached over the live GPU path -- it is covered here directly on CPU (no GPU, no dist:
``dist.is_initialized()`` is False so the cross-rank count sync is skipped)::

    pytest tests/rl/test_microbatch.py
"""

from __future__ import annotations

import torch

from arctic_platform.rl.processors.microbatch import MicroBatchSpec
from arctic_platform.rl.processors.microbatch import split_padded_tensor_dict_into_mb_list
from arctic_platform.testing_utils import TestCasePlus

seq_len = 8
real_token_counts = [8, 7, 6, 5, 4, 3]  # per-row real tokens; max (8) <= capacity so FFD never overflows
max_tokens_per_mb = 10  # forces several bins (total 33 tokens / 10)


def _make_padded_dict() -> dict:
    gen = torch.Generator().manual_seed(0)
    batch_size = len(real_token_counts)
    input_ids = torch.randint(1, 100, (batch_size, seq_len), generator=gen)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for row, count in enumerate(real_token_counts):
        attention_mask[row, :count] = 1
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0) * attention_mask
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "row_id": list(range(batch_size)),  # non-tensor -> not split, replicated into every microbatch
    }


class TestSplitPaddedTensorDictIntoMbList(TestCasePlus):
    def test_splits_and_round_trips(self):
        data = _make_padded_dict()
        batch_size = data["input_ids"].shape[0]
        mb_list = split_padded_tensor_dict_into_mb_list(data, MicroBatchSpec(max_tokens_per_mb=max_tokens_per_mb))

        self.assertGreater(len(mb_list), 1, "token budget should force more than one microbatch")

        # forward_indices is a permutation of the rows; backward_indices is its exact inverse.
        forward = torch.tensor(mb_list.forward_indices)
        backward = torch.tensor(mb_list.backward_indices)
        self.assertEqual(sorted(forward.tolist()), list(range(batch_size)), "forward_indices not a permutation")
        self.assertEqual(forward[backward].tolist(), list(range(batch_size)), "backward_indices is not the inverse")

        # No microbatch exceeds the token budget, and the per-group token counts cover the batch exactly.
        self.assertLessEqual(max(mb_list.group_lens), max_tokens_per_mb, "a microbatch exceeded the token budget")
        self.assertEqual(sum(mb_list.group_lens), int(data["attention_mask"].sum()), "group_lens lost tokens")

        # Concatenating the split tensors back and undoing the reorder reproduces the original padded tensors.
        for key in ("input_ids", "attention_mask", "position_ids"):
            packed = torch.cat([mb[key] for mb in mb_list.mbs], dim=0)
            self.assertTrue(torch.equal(packed[backward], data[key]), f"{key} did not round-trip through split")

        # Non-tensor fields ride along unsplit in every microbatch.
        for mb in mb_list.mbs:
            self.assertEqual(mb["row_id"], data["row_id"], "non-tensor field was altered")
