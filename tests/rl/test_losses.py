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

"""CPU unit tests for the RL loss math (``processors/functional.py`` and ``processors/grpo.py``).

The GPU update-actor tests (test_train_engine / test_e2e) only ever run one default loss config, so the many
config-driven branches -- aggregation modes, KL estimators, importance-sampling levels, dual clip, SAPO, proximal-
logp methods, M2PO masking, entropy/KL auxiliary terms -- never execute. These are pure tensor functions, so they
are exercised here directly on CPU with tiny deterministic tensors (no GPU, no Ray)::

    pytest tests/rl/test_losses.py

Collectives are avoided (``masked_normalization`` is called with ``all_reduce=False``): conftest leaves a single-rank
NCCL group initialized, and an all-reduce of CPU tensors over NCCL would error.
"""

from __future__ import annotations

import torch

from arctic_platform.rl.processors.functional import _compute_sequence_level_ratio_and_advantages
from arctic_platform.rl.processors.functional import agg_loss
from arctic_platform.rl.processors.functional import kl_penalty
from arctic_platform.rl.processors.functional import masked_normalization
from arctic_platform.rl.processors.functional import ppo_actor_loss_fn
from arctic_platform.rl.processors.functional import sapo_loss_fn
from arctic_platform.rl.processors.grpo import PROX_APPROX_METHOD_LINEAR
from arctic_platform.rl.processors.grpo import PROX_APPROX_METHOD_LOGLINEAR
from arctic_platform.rl.processors.grpo import PROX_APPROX_METHOD_ROLLOUT
from arctic_platform.rl.processors.grpo import PROX_LOGP_METHOD_METRICS
from arctic_platform.rl.processors.grpo import PROX_LOGP_METHOD_RECOMPUTE
from arctic_platform.rl.processors.grpo import _apply_m2po_masking
from arctic_platform.rl.processors.grpo import _resolve_proximal_logp
from arctic_platform.rl.processors.grpo import compute_prox_logp_approximations
from arctic_platform.rl.processors.grpo import grpo_loss
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import torch_assert_close


class TestAggLoss(TestCasePlus):
    def test_token_mean_and_dp_scaling(self):
        loss_mat = torch.ones(2, 4)
        mask = torch.ones(2, 4, dtype=torch.bool)
        self.assertAlmostEqual(agg_loss(loss_mat, mask).item(), 1.0, places=5)
        # token-mean multiplies by dp_size (the caller divides by the global token count fed as batch_num_tokens).
        self.assertAlmostEqual(agg_loss(loss_mat, mask, dp_size=2).item(), 2.0, places=5)

    def test_token_mean_respects_mask(self):
        loss_mat = torch.tensor([[2.0, 4.0, 100.0, 100.0]])
        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        self.assertAlmostEqual(agg_loss(loss_mat, mask).item(), 3.0, places=5)

    def test_all_sequence_modes_finite(self):
        loss_mat = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        for mode in ("seq-mean-token-sum", "seq-mean-token-sum-norm", "seq-mean-token-mean"):
            loss = agg_loss(loss_mat, mask, loss_agg_mode=mode)
            self.assertEqual(loss.ndim, 0, mode)
            self.assertTrue(torch.isfinite(loss), mode)

    def test_seq_mean_token_mean_value(self):
        loss_mat = torch.tensor([[1.0, 3.0, 0.0], [2.0, 0.0, 0.0]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        # per-seq token means: (1+3)/2=2 and 2/1=2 -> mean across the two seqs = 2.
        self.assertAlmostEqual(agg_loss(loss_mat, mask, loss_agg_mode="seq-mean-token-mean").item(), 2.0, places=5)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            agg_loss(torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool), loss_agg_mode="bogus")


class TestKlPenalty(TestCasePlus):
    def test_methods_match_formulas(self):
        logp = torch.tensor([-1.0, -2.0, -0.5])
        ref = torch.tensor([-1.5, -1.0, -0.5])
        torch_assert_close(kl_penalty(logp, ref, "k1"), logp - ref, rtol=0, atol=1e-6, msg="k1")
        torch_assert_close(kl_penalty(logp, ref, "kl"), logp - ref, rtol=0, atol=1e-6, msg="kl")
        torch_assert_close(kl_penalty(logp, ref, "abs"), (logp - ref).abs(), rtol=0, atol=1e-6, msg="abs")
        torch_assert_close(kl_penalty(logp, ref, "k2"), 0.5 * (logp - ref).square(), rtol=0, atol=1e-6, msg="k2")
        d = ref - logp
        torch_assert_close(kl_penalty(logp, ref, "k3"), d.exp() - d - 1, rtol=0, atol=1e-5, msg="k3")

    def test_identical_policies_give_zero_kl(self):
        logp = torch.tensor([-1.0, -2.0, -3.0])
        for method in ("k1", "abs", "k2", "k3", "low_var_kl"):
            kl = kl_penalty(logp, logp.clone(), method)
            torch_assert_close(kl, torch.zeros_like(kl), rtol=0, atol=1e-6, msg=method)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            kl_penalty(torch.zeros(2), torch.zeros(2), "bogus")


class TestPpoActorLoss(TestCasePlus):
    def test_on_policy_ratio_is_one(self):
        logp = torch.full((2, 3), -1.0)
        adv = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        mask = torch.ones(2, 3, dtype=torch.bool)
        loss, stat = ppo_actor_loss_fn(logp, logp.clone(), logp.clone(), adv, 0.2, mask)
        # ratio == 1 everywhere -> token-mean loss is -mean(advantages).
        self.assertAlmostEqual(loss.item(), -adv.mean().item(), places=5)
        torch_assert_close(stat["importance_weight"], torch.ones_like(adv), rtol=0, atol=1e-5)

    def test_positive_advantage_high_ratio_is_clipped(self):
        mask = torch.ones(1, 3, dtype=torch.bool)
        proximal = torch.full((1, 3), -1.0)
        logp = proximal + 1.0  # ratio = e >> 1 + eps_clip
        adv = torch.ones(1, 3)
        _, stat = ppo_actor_loss_fn(logp, proximal, proximal.clone(), adv, 0.2, mask)
        self.assertTrue(stat["clip_mask"].all(), "high ratio with positive advantage should clip")

    def test_dual_clip_engages_for_negative_advantage(self):
        mask = torch.ones(1, 3, dtype=torch.bool)
        proximal = torch.full((1, 3), -1.0)
        logp = proximal + 2.0  # large ratio
        adv = torch.full((1, 3), -1.0)
        _, stat = ppo_actor_loss_fn(logp, proximal, proximal.clone(), adv, 0.2, mask, c_clip=3.0)
        self.assertTrue(stat["dual_clip_mask"].any(), "dual clip should engage for large negative-advantage ratio")

    def test_behav_imp_weight_cap_masks_tokens(self):
        mask = torch.ones(1, 3, dtype=torch.bool)
        logp = torch.full((1, 3), -1.0)
        proximal = logp.clone()
        old = proximal - 5.0  # behav_imp_weight = exp(proximal - old) = e^5, far above the cap
        _, stat = ppo_actor_loss_fn(logp, proximal, old, torch.ones(1, 3), 0.2, mask, behav_imp_weight_cap=1.5)
        self.assertEqual(int(stat["behave_mask"].sum()), 0, "all tokens should be capped out")

    def test_sequence_importance_sampling_2d(self):
        mask = torch.ones(2, 3, dtype=torch.bool)
        logp = torch.randn(2, 3)
        loss, stat = ppo_actor_loss_fn(
            logp, logp.clone(), logp.clone(), torch.randn(2, 3), 0.2, mask, importance_sampling_level="sequence"
        )
        self.assertTrue(torch.isfinite(loss))

    def test_invalid_importance_sampling_level_raises(self):
        mask = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaises(ValueError):
            ppo_actor_loss_fn(
                torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), 0.2, mask,
                importance_sampling_level="bogus",
            )


class TestSapoLoss(TestCasePlus):
    def test_basic_finite(self):
        mask = torch.ones(2, 3, dtype=torch.bool)
        logp = torch.randn(2, 3)
        loss, stat = sapo_loss_fn(logp, logp.clone(), torch.randn(2, 3), 1.0, 1.05, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("sapo_soft_gate", stat)

    def test_nonpositive_temperature_raises(self):
        mask = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaises(ValueError):
            sapo_loss_fn(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), 0.0, 1.0, mask)


class TestSequenceRatioAndAdvantages(TestCasePlus):
    def test_2d_path_broadcasts_per_sequence(self):
        log_ratio = torch.zeros(2, 3)  # ratio == 1
        adv = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        mask = torch.ones(2, 3, dtype=torch.bool)
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(log_ratio, adv, mask, None)
        torch_assert_close(ratio, torch.ones_like(ratio), rtol=0, atol=1e-6)
        torch_assert_close(advantages, adv, rtol=0, atol=1e-6)

    def test_1d_packed_path(self):
        log_ratio = torch.zeros(5)
        adv = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0])
        mask = torch.ones(5, dtype=torch.bool)
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(log_ratio, adv, mask, cu_seqlens)
        torch_assert_close(ratio, torch.ones_like(ratio), rtol=0, atol=1e-6)
        torch_assert_close(advantages, adv, rtol=0, atol=1e-6)

    def test_1d_requires_cu_seqlens(self):
        with self.assertRaises(ValueError):
            _compute_sequence_level_ratio_and_advantages(
                torch.zeros(4), torch.zeros(4), torch.ones(4, dtype=torch.bool), None
            )


class TestMaskedNormalization(TestCasePlus):
    def test_no_mask_zero_mean_unit_std(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = masked_normalization(x, all_reduce=False)
        self.assertAlmostEqual(float(out.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(out.std(unbiased=False)), 1.0, places=3)

    def test_with_mask_ignores_padded_entries(self):
        x = torch.tensor([[1.0, 3.0, 999.0, 999.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        out = masked_normalization(x, mask, dim=[0, 1], all_reduce=False)
        self.assertTrue(torch.isfinite(out[mask.bool()]).all())


class TestProximalLogp(TestCasePlus):
    def test_compute_approximations_all_methods(self):
        old = torch.tensor([-1.0, -2.0])
        logp = torch.tensor([-0.5, -2.5])
        versions = torch.tensor([0, 1])
        approx = compute_prox_logp_approximations(old, logp, versions, current_version=3)
        for key in (PROX_APPROX_METHOD_LOGLINEAR, PROX_APPROX_METHOD_LINEAR, PROX_APPROX_METHOD_ROLLOUT):
            self.assertIn(key, approx)
        torch_assert_close(approx[PROX_APPROX_METHOD_ROLLOUT], old, rtol=0, atol=1e-6)

    def test_resolve_recompute_returns_old_logp(self):
        old = torch.tensor([-1.0, -2.0])
        out = _resolve_proximal_logp(None, PROX_LOGP_METHOD_RECOMPUTE, old, old.clone(), None, None)
        torch_assert_close(out, old, rtol=0, atol=1e-6)

    def test_resolve_passthrough_when_provided(self):
        old = torch.tensor([-1.0, -2.0])
        prox = torch.tensor([-0.7, -1.7])
        out = _resolve_proximal_logp(prox, PROX_LOGP_METHOD_RECOMPUTE, old, old.clone(), None, None)
        torch_assert_close(out, prox, rtol=0, atol=1e-6)

    def test_resolve_raises_when_prox_missing_and_forward_required(self):
        old = torch.tensor([-1.0, -2.0])
        with self.assertRaises(ValueError):
            _resolve_proximal_logp(None, PROX_LOGP_METHOD_METRICS, old, old.clone(), None, None)


class TestM2poMasking(TestCasePlus):
    def test_masks_high_delta_tokens(self):
        old = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        prox = torch.tensor([[0.0, 0.0, 5.0, 0.0]])  # one token has a large squared delta
        loss_mask = torch.ones(1, 4, dtype=torch.bool)
        out = _apply_m2po_masking(old, prox, loss_mask, m2_threshold=0.5)
        self.assertFalse(bool(out[0, 2]), "the high-delta token should be masked out")
        self.assertTrue(out.sum() >= 1, "M2PO must keep at least one token")


class TestGrpoLoss(TestCasePlus):
    """Integration of the registered ``grpo_loss`` entry point across its optional config branches."""

    def _context(self, batch_size=2, seq_len=3, **extra):
        gen = torch.Generator().manual_seed(0)
        ctx = {
            "old_log_probs_shifted": torch.randn(batch_size, seq_len, generator=gen),
            "advantages": torch.randn(batch_size, seq_len, generator=gen),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
        ctx.update(extra)
        return ctx

    def _outputs(self, batch_size=2, seq_len=3):
        return {"logprobs": torch.randn(batch_size, seq_len, generator=torch.Generator().manual_seed(1))}

    def test_default_config_returns_scalar_and_metrics(self):
        loss, metrics = grpo_loss(self._outputs(), self._context(), {}, "cpu")
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        for key in ("approx_kl", "importance_weight", "clip_ratio", "entropy", "loss"):
            self.assertIn(key, metrics)

    def test_all_loss_agg_modes(self):
        for mode in ("token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm", "seq-mean-token-mean"):
            loss, _ = grpo_loss(self._outputs(), self._context(), {"loss_agg_mode": mode}, "cpu")
            self.assertTrue(torch.isfinite(loss), mode)

    def test_entropy_bonus_changes_loss(self):
        outputs, context = self._outputs(), self._context()
        base, _ = grpo_loss(outputs, context, {}, "cpu")
        bonus, _ = grpo_loss(outputs, context, {"entropy_coeff": 0.1}, "cpu")
        self.assertNotAlmostEqual(base.item(), bonus.item(), places=6)

    def test_kl_loss_branch(self):
        context = self._context(ref_log_probs_shifted=torch.randn(2, 3, generator=torch.Generator().manual_seed(2)))
        loss, _ = grpo_loss(self._outputs(), context, {"use_kl_loss": True, "kl_loss_coef": 0.1}, "cpu")
        self.assertTrue(torch.isfinite(loss))

    def test_kl_loss_without_reference_raises(self):
        with self.assertRaises(ValueError):
            grpo_loss(self._outputs(), self._context(), {"use_kl_loss": True}, "cpu")

    def test_sapo_branch(self):
        loss, _ = grpo_loss(self._outputs(), self._context(), {"use_sapo_loss": True}, "cpu")
        self.assertTrue(torch.isfinite(loss))

    def test_sapo_with_decoupled_raises(self):
        with self.assertRaises(ValueError):
            grpo_loss(self._outputs(), self._context(), {"use_sapo_loss": True, "use_decoupled_loss": True}, "cpu")

    def test_dual_clip_and_behav_cap_config(self):
        config = {"c_clip": 3.0, "behav_imp_weight_cap": 2.0, "eps_clip_higher": 0.3}
        loss, _ = grpo_loss(self._outputs(), self._context(), config, "cpu")
        self.assertTrue(torch.isfinite(loss))

    def test_sequence_importance_sampling_config(self):
        loss, _ = grpo_loss(self._outputs(), self._context(), {"importance_sampling_level": "sequence"}, "cpu")
        self.assertTrue(torch.isfinite(loss))

    def test_m2po_masking_config(self):
        loss, _ = grpo_loss(self._outputs(), self._context(), {"m2_threshold": 0.5}, "cpu")
        self.assertTrue(torch.isfinite(loss))

    def test_logits_fallback_path(self):
        # No precomputed logprobs: grpo_loss derives them from logits + context input_ids (roll(-1) labels).
        batch_size, seq_len, vocab = 2, 3, 7
        outputs = {"logits": torch.randn(batch_size, seq_len, vocab, generator=torch.Generator().manual_seed(3))}
        context = self._context(batch_size, seq_len, input_ids=torch.randint(0, vocab, (batch_size, seq_len)))
        loss, _ = grpo_loss(outputs, context, {}, "cpu")
        self.assertTrue(torch.isfinite(loss))
