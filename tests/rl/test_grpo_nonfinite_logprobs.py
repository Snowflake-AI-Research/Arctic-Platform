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

"""A missing trainer log-prob is counted at policy positions and excluded from policy math."""

import math

import torch

from arctic_platform.rl.processors import resolve_loss
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal


def run_loss(logprobs, mask, config, *, name="grpo", **context):
    values = torch.tensor(logprobs, requires_grad=True)
    batch = {
        "input_ids": torch.arange(3),
        "cu_seqlens": torch.tensor([0, 3]),
        "old_log_probs_shifted": torch.tensor([-1.0, -0.7, -2.0]),
        "advantages": torch.tensor([1.0, 2.0, -1.0]),
        "loss_mask": torch.tensor(mask),
        **context,
    }
    loss, metrics = resolve_loss(name).loss({"logprobs": values}, batch, {}, config, "cpu")
    loss.backward()
    return loss.detach(), values.grad, metrics


class TestGrpoNonfiniteLogprobs(TestCasePlus):
    def test_policy_metrics_and_gradient_match_an_explicit_mask(self):
        cispo = {"use_cispo_loss": True, "is_weight_clip_max": 5.0, "ratio_mask_bounds_pos": [0.2, 5.0]}
        global_scale = {**cispo, "batch_num_tokens": 2, "dp_size": 1}
        reference_loss, reference_grad, reference = run_loss([-0.9, -0.6, -1.9], [1, 0, 0], global_scale)
        self.assertAlmostEqual(reference["importance_weight"], math.exp(0.1), places=6)
        self.assertNotEqual(reference_grad[0].item(), 0.0)
        for config in (cispo, global_scale):
            for bad in (float("nan"), float("inf"), -float("inf")):
                # One missing policy value and one missing padding value.
                loss, grad, metrics = run_loss([-0.9, bad, bad], [1, 1, 0], config)
                torch_assert_equal(loss, reference_loss)
                torch_assert_equal(grad, reference_grad)
                for key in (
                    "importance_weight",
                    "approx_kl",
                    "entropy",
                    "grpo_stats_token_count",
                    "cispo_trainable_token_count",
                    "ratio_trainable_token_count",
                ):
                    self.assertEqual(metrics[key], reference[key], key)
                self.assertEqual(metrics["grpo_nonfinite_logprob_token_count"], 1)
        self.assertEqual(reference["grpo_nonfinite_logprob_token_count"], 0)

    def test_missing_policy_shard_and_padding_have_distinct_counts(self):
        for mask, count in (([1, 1, 0], 2), ([0, 0, 0], 0)):
            loss, grad, metrics = run_loss([float("nan")] * 3, mask, {})
            self.assertEqual(loss.item(), 0.0)
            torch_assert_equal(grad, torch.zeros_like(grad))
            self.assertEqual(metrics["grpo_stats_token_count"], 0)
            self.assertEqual(metrics["grpo_nonfinite_logprob_token_count"], count)

    def test_packed_split_preserves_loss_and_gradient(self):
        values = [-0.9, float("nan"), -1.9]
        config = {"use_cispo_loss": True, "is_weight_clip_max": 5.0}
        masks = ([1, 1, 0], [0, 0, 1])
        batches = [{"input_ids": torch.arange(3), "loss_mask": torch.tensor(mask)} for mask in masks]
        reduction = resolve_loss("grpo").packed_reduction_callback(batches, config, "grpo")
        combined_loss, combined_grad, combined = run_loss(values, [1, 1, 1], config)
        parts = [run_loss(values, mask, config) for mask in masks]
        split_loss = sum(w * loss for w, (loss, _, _) in zip(reduction.loss_scales, parts))
        split_grad = sum(w * grad for w, (_, grad, _) in zip(reduction.loss_scales, parts))
        # Summing two weighted float32 losses changes the reduction order.
        torch_assert_close(split_loss, combined_loss, rtol=0, atol=1e-7)
        torch_assert_close(split_grad, combined_grad, rtol=0, atol=1e-7)
        self.assertEqual(sum(metrics["grpo_nonfinite_logprob_token_count"] for _, _, metrics in parts), 1)
        self.assertEqual(combined["grpo_nonfinite_logprob_token_count"], 1)

    def test_extreme_old_logprob_at_excluded_position(self):
        for config in ({}, {"use_sapo_loss": True}):
            loss, grad, _ = run_loss(
                [-0.9, float("nan"), -1.9],
                [1, 1, 0],
                config,
                old_log_probs_shifted=torch.tensor([-1.0, -1000.0, -2.0]),
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(grad).all())
            self.assertEqual(grad[1].item(), 0.0)

    def test_request_mask_is_validated_before_exclusion(self):
        values = [-0.9, float("nan"), -1.9]
        for mask, error in (([1], "shape"), ([1, 0.5, 0], "binary")):
            with self.assertRaisesRegex(ValueError, error):
                run_loss(values, mask, {})
        with self.assertRaisesRegex(ValueError, "ignore_index"):
            run_loss(values, [1, 1, 0], {}, labels=torch.tensor([1, -100, -100]))
        with self.assertRaisesRegex(ValueError, "overlaps loss_mask"):
            run_loss(
                values,
                [1, 1, 0],
                {"aux_ce_weight": 0.5, "echo_global_num_sequences": 1},
                name="grpo_echo_v1",
                sft_mask=torch.tensor([False, True, False]),
                echo_observation_mask=torch.tensor([False, True, False]),
            )
