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
"""``split_dict`` must return one shard per DP rank for any batch size ``B >= n``."""

from __future__ import annotations

import torch

from arctic_platform.common.utils.batch import split_dict
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import torch_assert_equal


class TestSplitDictRemainder(TestCasePlus):
    def test_tensor_split_returns_one_shard_per_rank(self):
        # torch.chunk(B, n) can return fewer than n tensors (B=6 n=4 → 3 chunks).
        for batch_size, num_chunks in ((6, 4), (5, 4), (9, 4), (13, 8), (4, 4), (7, 4)):
            ids = torch.arange(batch_size * 3).view(batch_size, 3)
            shards = split_dict({"input_ids": ids}, num_chunks)
            self.assertEqual(len(shards), num_chunks, msg=f"B={batch_size} n={num_chunks}")
            rows = [int(s["input_ids"].shape[0]) for s in shards]
            self.assertEqual(sum(rows), batch_size, msg=f"B={batch_size} n={num_chunks} rows={rows}")
            torch_assert_equal(torch.cat([s["input_ids"] for s in shards], dim=0), ids)

    def test_batch_smaller_than_ranks_is_rejected(self):
        ids = torch.arange(6).view(3, 2)
        with self.assertRaises(ValueError):
            split_dict({"input_ids": ids}, 4)
