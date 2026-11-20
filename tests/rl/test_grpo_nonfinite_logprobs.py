"""GRPO with a non-finite trainer log-prob at a trainable position: the token gets no gradient, stays out of the policy
statistics, and is counted in ``grpo_nonfinite_logprob_token_count``."""

from __future__ import annotations

import math
import unittest

import torch

from arctic_platform.rl.processors import resolve_loss

CONFIG = {
    "use_cispo_loss": True,
    "eps_clip": 0.2,
    "is_weight_clip_max": 5.0,
    "loss_agg_mode": "token-mean",
    "prox_logp_method": "recompute",
    "importance_sampling_level": "token",
    "current_version": 1,
    "ratio_mask_bounds_pos": [0.2, 5.0],
    "ratio_mask_bounds_neg": [0.2, 5.0],
    "batch_num_tokens": 5.0,
    "dp_size": 1,
}
OLD = [-1.0, -0.5, -2.0, -0.7, -1.2, 0.0]
ADV = [1.0, -1.0, 0.5, 2.0, -0.3, 0.0]
MASK = [True, True, True, True, True, False]


def _run(bad: dict[int, float]):
    old = torch.tensor(OLD)
    context = dict(
        old_log_probs_shifted=old,
        advantages=torch.tensor(ADV),
        loss_mask=torch.tensor(MASK),
        input_ids=torch.arange(6),
        cu_seqlens=torch.tensor([0, 6]),
        packed_loss_scale_factor=6,
    )
    logprobs = old + 0.1
    for position, value in bad.items():
        logprobs[position] = value
    logprobs.requires_grad_(True)
    loss, metrics = resolve_loss("grpo").loss({"logprobs": logprobs}, {}, context, dict(CONFIG), "cpu")
    loss.backward()
    return loss, logprobs.grad, metrics


class TestGrpoNonFiniteLogprobs(unittest.TestCase):
    def test_all_finite_counts_zero(self):
        _, _, metrics = _run({})
        self.assertEqual(float(metrics["grpo_nonfinite_logprob_token_count"]), 0.0)

    def test_nonfinite_token_is_counted_and_left_out(self):
        finite_loss, finite_grad, finite_metrics = _run({})
        for value in (float("nan"), float("-inf"), float("inf")):
            loss, grad, metrics = _run({3: value})
            self.assertTrue(torch.isfinite(loss), value)
            self.assertEqual(float(grad[3]), 0.0, value)
            keep = [0, 1, 2, 4]
            self.assertTrue(torch.equal(grad[keep], finite_grad[keep]), value)
            # The policy statistics cover the four finite trainable tokens only, each with ratio exp(0.1).
            self.assertEqual(float(metrics["grpo_stats_token_count"]), 4.0, value)
            self.assertAlmostEqual(float(metrics["importance_weight"]), math.exp(0.1), places=6, msg=value)
            self.assertAlmostEqual(float(metrics["approx_kl"]), float(finite_metrics["approx_kl"]), places=6, msg=value)
            self.assertEqual(float(metrics["grpo_nonfinite_logprob_token_count"]), 1.0, value)


if __name__ == "__main__":
    unittest.main()
