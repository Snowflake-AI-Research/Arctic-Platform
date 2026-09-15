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

from arctic_platform.common.utils.batch import _split_batch
from arctic_platform.common.utils.batch import dp_sp_world_size
from arctic_platform.common.utils.batch import sp_size_from_job_config
from arctic_platform.common.utils.batch import split_dict
from arctic_platform.common.utils.batch import unpack_batch
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

    def test_rollout_is_weights_in_batch_is_dp_sharded(self):
        weights = torch.arange(4, dtype=torch.float32)
        envelope = {
            "batch": {
                "input_ids": torch.arange(8).view(4, 2),
                "attention_mask": torch.ones(4, 2, dtype=torch.long),
                "rollout_is_weights": weights,
            },
            "meta": {},
            "processing": {"loss_fn": "verl_grpo"},
        }
        shards, _ = _split_batch(envelope, num_workers=2)
        self.assertEqual(shards[0]["batch"]["rollout_is_weights"].tolist(), [0.0, 1.0])
        self.assertEqual(shards[1]["batch"]["rollout_is_weights"].tolist(), [2.0, 3.0])

    def test_batch_dim_keys_in_meta_are_promoted_and_sharded(self):
        weights = torch.arange(4, dtype=torch.float32)
        advantages = torch.arange(8, dtype=torch.float32).view(4, 2)
        envelope = {
            "batch": {
                "input_ids": torch.arange(8).view(4, 2),
                "attention_mask": torch.ones(4, 2, dtype=torch.long),
            },
            "meta": {"rollout_is_weights": weights, "advantages": advantages, "dp_size": 2},
            "processing": {"loss_fn": "ap_grpo"},
        }
        shards, _ = _split_batch(envelope, num_workers=2)
        self.assertNotIn("rollout_is_weights", shards[0]["meta"])
        self.assertNotIn("advantages", shards[0]["meta"])
        self.assertEqual(shards[0]["meta"]["dp_size"], 2)
        self.assertEqual(shards[0]["batch"]["rollout_is_weights"].tolist(), [0.0, 1.0])
        self.assertEqual(shards[1]["batch"]["rollout_is_weights"].tolist(), [2.0, 3.0])
        self.assertEqual(shards[0]["batch"]["advantages"].tolist(), [[0.0, 1.0], [2.0, 3.0]])

    def test_cortex_context_batch_dim_keys_land_in_batch(self):
        advantages = torch.arange(8, dtype=torch.float32).view(4, 2)
        loss_mask = torch.ones(4, 2, dtype=torch.bool)
        envelope = {
            "kwargs": {
                "input_ids": torch.arange(8).view(4, 2),
                "attention_mask": torch.ones(4, 2, dtype=torch.long),
            },
            "context": {
                "advantages": advantages,
                "loss_mask": loss_mask,
                "prompt_group_ids": torch.tensor([7, 7, 8, 8]),
                "max_prompt_len": 3,
            },
            "processing": {"loss_fn": "ap_grpo"},
        }
        _, batch_data, meta_data, _ = unpack_batch(envelope)
        self.assertIn("advantages", batch_data)
        self.assertIn("loss_mask", batch_data)
        self.assertIn("prompt_group_ids", batch_data)
        self.assertEqual(meta_data, {"max_prompt_len": 3})
        shards, _ = _split_batch(envelope, num_workers=2)
        self.assertEqual(shards[0]["batch"]["prompt_group_ids"].tolist(), [7, 7])
        self.assertEqual(shards[1]["batch"]["prompt_group_ids"].tolist(), [8, 8])
        self.assertEqual(shards[0]["meta"], {"max_prompt_len": 3, "dp_size": 2})


class TestDpSizeDividesBySp(TestCasePlus):
    def _envelope(self):
        return {
            "batch": {
                "input_ids": torch.arange(16).view(8, 2),
                "attention_mask": torch.ones(8, 2, dtype=torch.long),
            },
            "meta": {},
            "processing": {"loss_fn": "ap_grpo"},
        }

    def test_dp_sp_world_size(self):
        self.assertEqual(dp_sp_world_size(8, 1), 8)
        self.assertEqual(dp_sp_world_size(8, 2), 4)
        with self.assertRaises(ValueError):
            dp_sp_world_size(8, 3)

    def test_split_stamps_world_over_sp(self):
        shards, _ = _split_batch(self._envelope(), num_workers=8, sp_size=2)
        self.assertEqual(len(shards), 8)
        self.assertEqual(shards[0]["meta"]["dp_size"], 4)
        self.assertEqual(shards[7]["meta"]["dp_size"], 4)

    def test_split_default_sp_is_world(self):
        shards, _ = _split_batch(self._envelope(), num_workers=8)
        self.assertEqual(shards[0]["meta"]["dp_size"], 8)

    def test_sp_size_from_verl_ds_config(self):
        self.assertEqual(sp_size_from_job_config({"ds_config": {"sequence_parallel_size": 2}}), 2)

    def test_sp_size_from_training_config(self):
        self.assertEqual(sp_size_from_job_config({"training_config": {"sp_size": 4}}), 4)

    def test_sp_size_unset_is_one(self):
        self.assertEqual(sp_size_from_job_config({}), 1)
        self.assertEqual(sp_size_from_job_config(None), 1)

    def test_sp_size_conflict_raises(self):
        with self.assertRaises(ValueError):
            sp_size_from_job_config(
                {
                    "training_config": {"sp_size": 2},
                    "ds_config": {"sequence_parallel_size": 4},
                }
            )
