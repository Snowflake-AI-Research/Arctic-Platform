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
"""``reorg_global_batch`` always returns ``(batch, indices)``, including when it cannot load-balance."""

from __future__ import annotations

import torch

from arctic_platform.rl.zorro_train.seqlen_balancing import reorg_global_batch
from arctic_platform.testing_utils import TestCasePlus


class TestReorgGlobalBatchArity(TestCasePlus):
    def _ids(self, batch_size: int, seq_len: int = 4) -> torch.Tensor:
        return torch.arange(batch_size * seq_len).view(batch_size, seq_len)

    def test_small_batch_returns_batch_and_indices(self):
        for batch_size, world_size in ((4, 4), (2, 4)):
            batch = {
                "input_ids": self._ids(batch_size),
                "attention_mask": torch.ones(batch_size, 4, dtype=torch.long),
                "position_ids": torch.arange(4).expand(batch_size, 4),
            }
            out = reorg_global_batch(
                batch,
                response_length=2,
                world_size=world_size,
                max_token_len=16,
                max_group_length_threshold=2,
            )
            self.assertIsInstance(out, tuple, msg=f"B={batch_size} W={world_size}")
            self.assertEqual(len(out), 2, msg=f"B={batch_size} W={world_size}")
            got, indices = out
            self.assertIs(got, batch)
            self.assertIsNone(indices)

    def test_two_key_dict_does_not_unpack_as_items(self):
        batch = {
            "input_ids": self._ids(4),
            "attention_mask": torch.ones(4, 4, dtype=torch.long),
        }
        got, indices = reorg_global_batch(
            batch,
            response_length=2,
            world_size=4,
            max_token_len=16,
            max_group_length_threshold=2,
        )
        self.assertIs(got, batch)
        self.assertIsNone(indices)
        self.assertTrue(torch.is_tensor(got["input_ids"]))
