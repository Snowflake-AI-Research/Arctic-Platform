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

from arctic_platform.common.utils.batch import combine_metric_microbatches, combine_metric_shards

from arctic_platform.rl.processors.grpo import grpo_loss
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import torch_assert_close


class TestRatioMass(TestCasePlus):
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

    def test_microbatches_preserve_mass_and_loss(self):
        advantages = torch.tensor([[1.0, 3.0, -2.0, -2.0], [2.0, 2.0, -1.0, -3.0]])
        delta = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        for mode in ("token-mean", "prompt-mean"):
            for masks in (
                {"ratio_stats": True},
                {"ratio_mask_bounds_pos": [0.9, 1.1], "ratio_mask_bounds_neg": [0.9, 1.1]},
            ):
                config = dict(loss_agg_mode=mode, batch_num_tokens=8, dp_size=1, **masks)
                context = {} if mode == "token-mean" else dict(sequence_loss_weights=torch.tensor([0.25, 0.75]))
                loss, grad, whole = self.run_loss(advantages, delta, config, **context)
                parts = []
                for i in range(2):
                    row_context = {k: v[i : i + 1] for k, v in context.items()}
                    parts.append(self.run_loss(advantages[i : i + 1], delta[i : i + 1], config, **row_context))
                merged = combine_metric_shards([combine_metric_microbatches([p[2] for p in parts])])
                torch_assert_close(sum(p[0] for p in parts), loss)
                torch_assert_close(torch.cat([p[1] for p in parts]), grad)
                expected_kept = (
                    (1.0, 1.0)
                    if "ratio_stats" in masks
                    else ((0.625, 0.875) if mode == "token-mean" else (0.8125, 0.8125))
                )
                for index, sign in enumerate(("pos", "neg")):
                    for stage, expected in (("pre", 1.0), ("kept", expected_kept[index])):
                        key = f"ratio_mask_{sign}_{stage}_mass_sum"
                        self.assertAlmostEqual(whole[key], expected)
                        self.assertAlmostEqual(merged[key], expected)
                if "ratio_stats" in masks:
                    base = self.run_loss(
                        advantages, delta, {k: v for k, v in config.items() if k != "ratio_stats"}, **context
                    )
                    torch_assert_close(loss, base[0], atol=0, rtol=0)
                    torch_assert_close(grad, base[1], atol=0, rtol=0)
