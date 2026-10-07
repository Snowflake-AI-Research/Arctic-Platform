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

"""Request-global CISPO signed mass, including the single-call preflight."""

import torch

from arctic_platform.rl.processors.functional import RatioMasks
from arctic_platform.rl.processors.grpo import _grpo_model_call_count_callback
from arctic_platform.rl.processors.grpo import grpo_loss
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import torch_assert_close


class TestRatioRebalance(TestCasePlus):
    def run_loss(self, advantages, delta, config, **context):
        values = torch.full_like(advantages, -2.0, requires_grad=True)
        loss, metrics = grpo_loss(
            {"logprobs": values},
            dict(
                old_log_probs_shifted=values.detach() - delta,
                advantages=advantages,
                loss_mask=torch.ones_like(values, dtype=torch.bool),
                **context,
            ),
            {},
            dict(use_cispo_loss=True, is_weight_clip_max=5.0, **config),
            "cpu",
        )
        loss.backward()
        return loss.detach(), values.grad, metrics

    def test_restore_each_sign_with_original_weights(self):
        advantages = torch.tensor([[1.0, 3.0, -2.0, -2.0], [2.0, 2.0, -1.0, -3.0]])
        delta = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        for mode in ("token-mean", "prompt-mean"):
            context = {} if mode == "token-mean" else dict(sequence_loss_weights=torch.tensor([0.25, 0.75]))
            config = dict(
                ratio_mask_rebalance=True,
                ratio_mask_bounds_pos=[0.9, 1.1],
                ratio_mask_bounds_neg=[0.9, 1.1],
                loss_agg_mode=mode,
                batch_num_tokens=8,
                dp_size=1,
            )
            _, grad, stats = self.run_loss(advantages, delta, config, **context)
            for sign in ("pos", "neg"):
                before = stats[f"ratio_mask_{sign}_pre_mass_sum"]
                after = stats[f"ratio_rebalance_{sign}_post_mass_sum"]
                self.assertAlmostEqual(before, after, places=6)
                self.assertAlmostEqual(after, 1.0, places=6)
            self.assertAlmostEqual(float(grad.clamp(max=0).abs().sum()), 1.0, places=6)
            self.assertAlmostEqual(float(grad.clamp(min=0).sum()), 1.0, places=6)
            self.assertEqual(stats["ratio_mask_rebalance"], 1.0)

    def test_unmasked_exact_noop_and_default_off(self):
        advantages = torch.tensor([[1.0, 3.0, -2.0, -2.0]])
        delta = torch.tensor([[0.1, -0.1, 2.0, 0.0]])
        base = self.run_loss(advantages, delta, {})
        for config in (dict(ratio_mask_rebalance=False), dict(ratio_mask_rebalance=True)):
            loss, grad, _ = self.run_loss(advantages, delta, config)
            torch_assert_close(loss, base[0], atol=0, rtol=0)
            torch_assert_close(grad, base[1], atol=0, rtol=0)
        self.assertIsNone(RatioMasks.from_config(dict(ratio_mask_rebalance=False)))

    def test_missing_sign_is_finite_and_reported(self):
        _, grad, stats = self.run_loss(
            torch.tensor([[1.0, 3.0, -2.0, -2.0]]),
            torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            dict(ratio_mask_rebalance=True, ratio_mask_bounds_pos=[0.9, 1.1]),
        )
        self.assertTrue(torch.isfinite(grad).all())
        self.assertEqual(stats["ratio_mask_pos_kept_mass_sum"], 0.0)
        self.assertEqual(stats["ratio_rebalance_pos_post_mass_sum"], 0.0)
        self.assertEqual(stats["ratio_rebalance_pos_unrestored_mass_sum"], 1.0)
        self.assertEqual(stats["ratio_rebalance_neg_post_mass_sum"], 1.0)

    def test_request_preflight_and_typed_flag(self):
        for counts in ([2, 2], [1, 2], [None], []):
            with self.assertRaisesRegex(ValueError, "ratio_mask_rebalance.*one synchronized"):
                _grpo_model_call_count_callback(counts, dict(ratio_mask_rebalance=True))
        _grpo_model_call_count_callback([1, 1], dict(ratio_mask_rebalance=True))
        _grpo_model_call_count_callback([2, 2], dict(ratio_mask_rebalance=False))
        for bad in (None, 1, "true"):
            with self.assertRaisesRegex(ValueError, "ratio_mask_rebalance"):
                RatioMasks.from_config(dict(ratio_mask_rebalance=bad))
