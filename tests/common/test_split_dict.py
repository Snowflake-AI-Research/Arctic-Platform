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
from arctic_platform.common.utils.batch import split_dict
from arctic_platform.common.utils.batch import unpack_batch
from arctic_platform.common.utils.server_models import JobConfig
from arctic_platform.common.utils.server_models import sp_size_from_job_config
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
        teacher = torch.arange(8, dtype=torch.float32).view(4, 2)
        observation_counts = torch.arange(1, 5, dtype=torch.float32)
        envelope = {
            "batch": {
                "input_ids": torch.arange(8).view(4, 2),
                "attention_mask": torch.ones(4, 2, dtype=torch.long),
            },
            "meta": {
                "rollout_is_weights": weights,
                "advantages": advantages,
                "teacher_log_probs_shifted": teacher,
                "echo_observation_token_counts": observation_counts,
                "dp_size": 2,
            },
            "processing": {"loss_fn": "ap_grpo"},
        }
        shards, _ = _split_batch(envelope, num_workers=2)
        self.assertNotIn("rollout_is_weights", shards[0]["meta"])
        self.assertNotIn("advantages", shards[0]["meta"])
        self.assertNotIn("teacher_log_probs_shifted", shards[0]["meta"])
        self.assertNotIn("echo_observation_token_counts", shards[0]["meta"])
        self.assertEqual(shards[0]["meta"]["dp_size"], 2)
        self.assertEqual(shards[0]["batch"]["rollout_is_weights"].tolist(), [0.0, 1.0])
        self.assertEqual(shards[1]["batch"]["rollout_is_weights"].tolist(), [2.0, 3.0])
        self.assertEqual(shards[0]["batch"]["advantages"].tolist(), [[0.0, 1.0], [2.0, 3.0]])
        self.assertEqual(shards[0]["batch"]["teacher_log_probs_shifted"].tolist(), [[0.0, 1.0], [2.0, 3.0]])
        self.assertEqual(shards[1]["batch"]["echo_observation_token_counts"].tolist(), [3.0, 4.0])

    def test_none_rollout_is_weights_stays_in_meta(self):
        """SkyRL overlay sends rollout_is_weights=None; promoting it 500s the worker."""
        envelope = {
            "batch": {
                "input_ids": torch.arange(8).view(4, 2),
                "attention_mask": torch.ones(4, 2, dtype=torch.long),
            },
            "meta": {"rollout_is_weights": None, "dp_size": 1, "temperature": 1.0},
            "processing": {"post": ["compute_logprobs"], "loss_fn": None},
        }
        _, batch_data, meta_data, _ = unpack_batch(envelope)
        self.assertNotIn("rollout_is_weights", batch_data)
        self.assertIsNone(meta_data["rollout_is_weights"])
        shards, _ = _split_batch(envelope, num_workers=2)
        self.assertNotIn("rollout_is_weights", shards[0]["batch"])
        self.assertIsNone(shards[0]["meta"]["rollout_is_weights"])
        for k, v in shards[0]["batch"].items():
            getattr(v, "shape")

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

    def test_split_rejects_sp_greater_than_one(self):
        with self.assertRaises(ValueError) as ctx:
            _split_batch(self._envelope(), num_workers=8, sp_size=2)
        self.assertIn("sequence-parallel data-plane is not implemented", str(ctx.exception))

    def test_split_rejects_sp_that_does_not_divide_workers(self):
        with self.assertRaises(ValueError):
            _split_batch(self._envelope(), num_workers=8, sp_size=3)

    def test_cortex_context_return_fwd_batch_lands_on_shard_meta(self):
        envelope = {
            "kwargs": {
                "input_ids": torch.arange(8).view(4, 2),
                "attention_mask": torch.ones(4, 2, dtype=torch.long),
            },
            "context": {"return_fwd_batch": True},
            "processing": {"loss_fn": "grpo"},
        }
        shards, _ = _split_batch(envelope, num_workers=2)
        self.assertTrue(shards[0]["meta"].get("return_fwd_batch"))
        self.assertTrue(shards[1]["meta"].get("return_fwd_batch"))

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

    def test_job_config_exposes_sp_size(self):
        job_config = JobConfig(model_name="m", ds_config={"sequence_parallel_size": 2})
        self.assertEqual(job_config.sp_size, 2)
        self.assertEqual(sp_size_from_job_config(job_config), 2)
        self.assertEqual(JobConfig(model_name="m").sp_size, 1)

    def test_job_config_validates_sp_size_at_construction(self):
        with self.assertRaises(ValueError):
            JobConfig(model_name="m", training_config={"sp_size": 0})
        with self.assertRaises(ValueError):
            JobConfig(
                model_name="m",
                training_config={"sp_size": 2},
                ds_config={"sequence_parallel_size": 4},
            )

    def test_job_config_dump_keeps_client_wire_shape(self):
        # model_dump() is forwarded verbatim to workers; sp_size is derived, not a field.
        dumped = JobConfig(model_name="m", ds_config={"sequence_parallel_size": 2}).model_dump()
        self.assertNotIn("sp_size", dumped)
        self.assertEqual(sp_size_from_job_config(dumped), 2)

    def test_log_prob_job_ignores_training_ds_config_when_log_prob_declares_sp(self):
        job_config = JobConfig(
            model_name="m",
            job_type="log_prob",
            ds_config={"sequence_parallel_size": 2},
            log_prob_config={"sequence_parallel_size": 1},
        )
        self.assertEqual(job_config.sp_size, 1)

    def test_log_prob_job_falls_back_to_ds_config_when_log_prob_omits_sp(self):
        job_config = JobConfig(
            model_name="m",
            job_type="log_prob",
            ds_config={"sequence_parallel_size": 2},
        )
        self.assertEqual(job_config.sp_size, 2)

    def test_training_job_ignores_log_prob_config_sp(self):
        job_config = JobConfig(
            model_name="m",
            job_type="training",
            ds_config={"sequence_parallel_size": 2},
            log_prob_config={"sequence_parallel_size": 4},
        )
        self.assertEqual(job_config.sp_size, 2)

    def test_parallelism_degree_rejects_bool_and_float(self):
        from arctic_platform.common.utils.server_models import resolve_parallelism_degree

        with self.assertRaises(ValueError):
            resolve_parallelism_degree(True, "sp_size")
        with self.assertRaises(ValueError):
            resolve_parallelism_degree(1.5, "sp_size")
        with self.assertRaises(ValueError):
            resolve_parallelism_degree("2", "sp_size")
        self.assertEqual(resolve_parallelism_degree(2, "sp_size"), 2)
