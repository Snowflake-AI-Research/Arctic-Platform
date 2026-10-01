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

import math
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from arctic_platform.common.registry import LOSS_FNS
from arctic_platform.common.utils.batch import combine_metric_shards
from arctic_platform.common.utils.batch import promote_batch_dim_to_batch
from arctic_platform.rl.processors import resolve_loss
from arctic_platform.rl.processors.functional import _compute_sequence_level_ratio_and_advantages
from arctic_platform.rl.processors.functional import _resolve_dp_size
from arctic_platform.rl.processors.functional import agg_loss
from arctic_platform.rl.processors.functional import echo_env_prediction_loss_fn
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
from arctic_platform.rl.processors.grpo import _grpo_packed_loss_reduction
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
        self.assertAlmostEqual(agg_loss(loss_mat, mask, dp_size=1).item(), 1.0, places=5)
        # token-mean multiplies by dp_size (the caller divides by the global token count fed as batch_num_tokens).
        self.assertAlmostEqual(agg_loss(loss_mat, mask, dp_size=2).item(), 2.0, places=5)

    def test_explicit_denominators_must_be_safe_numeric_counts(self):
        values = torch.ones((1, 1))
        mask = torch.ones_like(values, dtype=torch.bool)
        for invalid in (True, -1, float("inf"), float("nan"), "1"):
            with (
                self.subTest(batch_num_tokens=invalid),
                self.assertRaisesRegex(ValueError, "batch_num_tokens must be a finite non-negative number"),
            ):
                agg_loss(values, mask, batch_num_tokens=invalid, dp_size=1)
        for invalid in (True, -1, 1.5, float("inf"), "1"):
            with (
                self.subTest(global_batch_size=invalid),
                self.assertRaisesRegex(ValueError, "global_batch_size must be a non-negative integer"),
            ):
                agg_loss(
                    values,
                    mask,
                    loss_agg_mode="seq-mean-token-sum",
                    global_batch_size=invalid,
                    dp_size=1,
                )

    def test_dp_size_rejects_non_positive(self):
        self.assertEqual(_resolve_dp_size(None, None), 1)
        with self.assertRaises(ValueError):
            _resolve_dp_size(0, None)
        with self.assertRaises(ValueError):
            _resolve_dp_size(None, batch_num_tokens=8)

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
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                0.2,
                mask,
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
        loss, metrics = grpo_loss(self._outputs(), self._context(), {}, {}, "cpu")
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        for key in ("approx_kl", "importance_weight", "clip_ratio", "entropy"):
            self.assertIn(key, metrics)
        self.assertNotIn("loss", metrics)

    def test_all_loss_agg_modes(self):
        for mode in ("token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm", "seq-mean-token-mean"):
            loss, _ = grpo_loss(self._outputs(), self._context(), {}, {"loss_agg_mode": mode}, "cpu")
            self.assertTrue(torch.isfinite(loss), mode)

    def test_entropy_bonus_changes_loss(self):
        outputs, context = self._outputs(), self._context()
        base, _ = grpo_loss(outputs, context, {}, {}, "cpu")
        bonus, _ = grpo_loss(outputs, context, {}, {"entropy_coeff": 0.1}, "cpu")
        self.assertNotAlmostEqual(base.item(), bonus.item(), places=6)

    def test_kl_loss_branch(self):
        context = self._context(ref_log_probs_shifted=torch.randn(2, 3, generator=torch.Generator().manual_seed(2)))
        loss, _ = grpo_loss(self._outputs(), context, {}, {"use_kl_loss": True, "kl_loss_coef": 0.1}, "cpu")
        self.assertTrue(torch.isfinite(loss))

    def test_kl_loss_without_reference_raises(self):
        with self.assertRaises(ValueError):
            grpo_loss(self._outputs(), self._context(), {}, {"use_kl_loss": True}, "cpu")

    def test_sapo_branch(self):
        loss, _ = grpo_loss(self._outputs(), self._context(), {}, {"use_sapo_loss": True}, "cpu")
        self.assertTrue(torch.isfinite(loss))

    def test_sapo_with_decoupled_raises(self):
        with self.assertRaises(ValueError):
            grpo_loss(self._outputs(), self._context(), {}, {"use_sapo_loss": True, "use_decoupled_loss": True}, "cpu")

    def test_dual_clip_and_behav_cap_config(self):
        config = {"c_clip": 3.0, "behav_imp_weight_cap": 2.0, "eps_clip_higher": 0.3}
        loss, _ = grpo_loss(self._outputs(), self._context(), {}, config, "cpu")
        self.assertTrue(torch.isfinite(loss))

    def test_sequence_importance_sampling_config(self):
        loss, _ = grpo_loss(self._outputs(), self._context(), {}, {"importance_sampling_level": "sequence"}, "cpu")
        self.assertTrue(torch.isfinite(loss))

    def test_m2po_masking_config(self):
        loss, _ = grpo_loss(self._outputs(), self._context(), {}, {"m2_threshold": 0.5}, "cpu")
        self.assertTrue(torch.isfinite(loss))

    def test_logits_fallback_path(self):
        # No precomputed logprobs: grpo_loss derives them from logits + context input_ids (roll(-1) labels).
        batch_size, seq_len, vocab = 2, 3, 7
        outputs = {"logits": torch.randn(batch_size, seq_len, vocab, generator=torch.Generator().manual_seed(3))}
        context = self._context(batch_size, seq_len, input_ids=torch.randint(0, vocab, (batch_size, seq_len)))
        loss, _ = grpo_loss(outputs, context, {}, {}, "cpu")
        self.assertTrue(torch.isfinite(loss))


def _sp_loss_worker(rank: int, init_file: str):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", world_size=2, rank=rank)
    try:
        values = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        context = {
            "old_log_probs_shifted": torch.tensor([[-1.0, -1.0]]),
            "advantages": torch.ones_like(values),
            "loss_mask": torch.tensor([[rank == 1, rank == 1]]),
        }
        config = {
            "use_cispo_loss": True,
            "is_weight_clip_max": 10.0,
            "importance_sampling_level": "sequence",
            "batch_num_tokens": 2,
            "dp_size": 1,
        }
        with (
            patch("arctic_platform.rl.processors.grpo._get_sequence_parallel_group", return_value=dist.group.WORLD),
            patch(
                "arctic_platform.rl.processors.functional._get_sequence_parallel_group", return_value=dist.group.WORLD
            ),
        ):
            loss, _ = grpo_loss({"logprobs": values}, context, {}, config, "cpu")
            loss.backward()
        if rank == 0:
            torch_assert_close(values.grad, torch.zeros_like(values), rtol=0, atol=1e-6)
            assert loss.item() == 0.0
        else:
            assert torch.isfinite(loss)
            assert torch.isfinite(values.grad).all()
            torch_assert_close(values.grad, torch.full_like(values, -math.exp(-0.5) / 2), rtol=0, atol=1e-6)

        with (
            patch("arctic_platform.rl.processors.grpo._get_sequence_parallel_group", return_value=dist.group.WORLD),
            patch(
                "arctic_platform.rl.processors.functional._get_sequence_parallel_group", return_value=dist.group.WORLD
            ),
        ):
            ratio_values = torch.tensor([[-0.5 if rank == 0 else -1.5]], requires_grad=True)
            ratio_context = {
                "old_log_probs_shifted": torch.tensor([[-1.0]]),
                "advantages": torch.ones_like(ratio_values),
                "loss_mask": torch.ones_like(ratio_values, dtype=torch.bool),
            }
            ratio_loss, ratio_metrics = grpo_loss(
                {"logprobs": ratio_values},
                ratio_context,
                {},
                {
                    "use_cispo_loss": True,
                    "is_weight_clip_max": 10.0,
                    "batch_num_tokens": 2,
                    "dp_size": 1,
                    "seq_mask_bounds_pos": [-0.1, 0.1],
                },
                "cpu",
            )
            ratio_loss.backward()
            echo_values = torch.tensor([[-1.0]], requires_grad=True)
            echo_loss, echo_metrics = echo_env_prediction_loss_fn(
                echo_values,
                torch.tensor([[rank == 0]]),
                torch.tensor([[True]]),
                torch.zeros((1, 1), dtype=torch.bool),
                global_num_echo_sequences=1,
                observation_token_counts=torch.tensor([4]),
            )
            echo_loss.backward()
            prompt_values = torch.tensor([[1.0 if rank == 0 else 3.0]], requires_grad=True)
            prompt_loss = agg_loss(
                prompt_values,
                torch.ones_like(prompt_values, dtype=torch.bool),
                loss_agg_mode="prompt-mean",
                sequence_loss_weights=torch.tensor([0.5]),
            )
            prompt_loss.backward()
            sequence_values = torch.tensor(
                [[1.0, 10.0] if rank == 0 else [3.0, 6.0]],
                requires_grad=True,
            )
            sequence_mask = torch.tensor([[True, False] if rank == 0 else [True, True]])
            sequence_loss = agg_loss(
                sequence_values,
                sequence_mask,
                loss_agg_mode="seq-mean-token-mean",
                cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32),
            )
            sequence_loss.backward()
            ppo_values = torch.tensor([[-0.95 if rank == 0 else -1.05]], requires_grad=True)
            ppo_loss, _ = grpo_loss(
                {"logprobs": ppo_values},
                {
                    "old_log_probs_shifted": torch.tensor([[-1.0]]),
                    "advantages": torch.ones_like(ppo_values),
                    "loss_mask": torch.ones_like(ppo_values, dtype=torch.bool),
                },
                {},
                {"importance_sampling_level": "sequence", "batch_num_tokens": 2, "dp_size": 1},
                "cpu",
            )
            ppo_loss.backward()

        self_expected = torch.full_like(ratio_values, -math.exp(0.5 if rank == 0 else -0.5) / 2)
        torch_assert_close(ratio_values.grad, self_expected, rtol=0, atol=1e-6)
        assert ratio_metrics["seq_mask_pos_drop_count"] == 0.0
        assert ratio_metrics["seq_mask_pos_sequence_count"] == 0.0
        assert ratio_metrics["ratio_trainable_token_count"] == (2.0 if rank == 0 else 0.0)
        assert sum(value for key, value in ratio_metrics.items() if key.startswith("seq_stat_bin_")) == (
            1.0 if rank == 0 else 0.0
        )
        assert echo_metrics["num_real_sequences"] == (1 if rank == 0 else 0)
        torch_assert_close(echo_values.grad, torch.tensor([[-0.25 if rank == 0 else 0.0]]), rtol=0, atol=1e-6)
        assert echo_loss.item() == (0.25 if rank == 0 else 0.0)
        torch_assert_close(prompt_values.grad, torch.tensor([[0.25]]), rtol=0, atol=1e-6)
        assert prompt_loss.item() == (0.25 if rank == 0 else 0.75)
        torch_assert_close(
            sequence_values.grad,
            torch.tensor([[0.25, 0.0] if rank == 0 else [0.25, 0.5]]),
            rtol=0,
            atol=1e-6,
        )
        assert sequence_loss.item() == (0.25 if rank == 0 else 3.75)
        torch_assert_close(ppo_values.grad, torch.tensor([[-0.5]]), rtol=0, atol=1e-6)

        try:
            agg_loss(
                torch.ones((1, 1)),
                torch.tensor([[rank == 1]]),
                batch_num_tokens=0,
            )
        except ValueError as error:
            assert "batch_num_tokens=0" in str(error)
        else:
            raise AssertionError("every rank must reject a zero denominator when any rank has policy tokens")

        zero_values = torch.zeros((1, 1), requires_grad=True)
        zero_context = {
            "input_ids": torch.ones((1, 1), dtype=torch.long),
            "old_log_probs_shifted": torch.zeros_like(zero_values),
            "advantages": torch.ones_like(zero_values),
            "loss_mask": torch.tensor([[rank == 1]]),
        }
        with patch("arctic_platform.rl.processors.grpo._get_sequence_parallel_group", return_value=None):
            try:
                grpo_loss(
                    {"logprobs": zero_values},
                    zero_context,
                    {},
                    {"batch_num_tokens": 0, "dp_size": 1},
                    "cpu",
                )
            except ValueError as error:
                assert "batch_num_tokens=0" in str(error)
            else:
                raise AssertionError("empty GRPO shards must join zero-denominator validation")

        mixed_microbatch = {
            **zero_context,
            "loss_mask": torch.ones((1, 1), dtype=torch.bool) if rank == 0 else torch.zeros((1, 1), dtype=torch.bool),
            "nll_mask": torch.ones((1, 1), dtype=torch.bool),
        }
        with patch("arctic_platform.rl.processors.grpo._get_sequence_parallel_group", return_value=dist.group.WORLD):
            try:
                resolve_loss("ap_grpo_mixed_v1").packed_reduction_callback(
                    [mixed_microbatch],
                    {"use_cispo_loss": True, "is_weight_clip_max": 2.0},
                    "ap_grpo_mixed_v1",
                )
            except ValueError as error:
                assert "nll_mask must be a subset" in str(error) or "another sequence-parallel rank" in str(error)
            else:
                raise AssertionError("every SP rank must reject shard-local mixed validation failures")

            echo_context = {
                "input_ids": torch.ones((1, 1), dtype=torch.long),
                "loss_mask": torch.tensor([[rank == 1]]),
                "sft_mask": torch.zeros((1, 1), dtype=torch.bool),
                "echo_observation_mask": torch.ones((1, 1), dtype=torch.bool),
            }
            try:
                resolve_loss("ap_grpo_echo_v1").validation_callback(
                    echo_context,
                    {"aux_ce_weight": 0.5, "echo_global_num_sequences": 1},
                )
            except ValueError as error:
                assert "echo_observation_mask overlaps" in str(error) or "another sequence-parallel rank" in str(error)
            else:
                raise AssertionError("every SP rank must reject shard-local ECHO validation failures")
    finally:
        dist.destroy_process_group()


class TestMigratedGrpo(TestCasePlus):
    def test_whole_request_callbacks_reject_cross_window_contract_errors(self):
        request = {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "processing": {
                "loss_fn": "ap_grpo_mixed_v1",
                "config": {"use_cispo_loss": True, "is_weight_clip_max": 2.0},
            },
            "context": {
                "loss_mask": torch.tensor([[True, False]]),
                "nll_mask": torch.tensor([[False, True]]),
            },
        }
        with self.assertRaisesRegex(ValueError, "nll_mask must be a subset"):
            resolve_loss("ap_grpo_mixed_v1").batching_callback(request)

        request["processing"] = {
            "loss_fn": "ap_grpo_echo_v1",
            "config": {"aux_ce_weight": 0.5, "echo_global_num_sequences": 1},
        }
        request["context"] = {
            "loss_mask": torch.tensor([[True, False]]),
            "sft_mask": torch.tensor([[False, True]]),
            "echo_observation_mask": torch.tensor([[True, True]]),
        }
        with self.assertRaisesRegex(ValueError, "echo_observation_mask overlaps loss_mask"):
            resolve_loss("ap_grpo_echo_v1").batching_callback(request)

    def test_ratio_masks_keep_original_normalizer_and_penalty_gradient(self):
        values = torch.tensor([[-1.0, -1.0, -1.0]], requires_grad=True)
        context = {
            "old_log_probs_shifted": torch.tensor([[-1.0, -2.0, -1.0]]),
            "advantages": torch.tensor([[1.0, 1.0, -1.0]]),
            "loss_mask": torch.ones_like(values, dtype=torch.bool),
        }
        loss, metrics = grpo_loss(
            {"logprobs": values},
            context,
            {},
            {
                "use_cispo_loss": True,
                "is_weight_clip_max": 10.0,
                "ratio_mask_bounds_pos": [0.5, 1.5],
                "log_ratio_sq_coef": 0.5,
            },
            "cpu",
        )
        self.assertAlmostEqual(loss.item(), 1 / 6, places=6)
        self.assertEqual(metrics["ratio_mask_pos_high_drop_count"], 1.0)
        self.assertEqual(metrics["ratio_mask_dropped_token_count"], 1.0)
        self.assertEqual(metrics["ratio_trainable_token_count"], 3.0)
        self.assertEqual(metrics["log_ratio_sq_sum"], 1.0)
        self.assertEqual(sum(value for key, value in metrics.items() if key.startswith("ratio_joint_")), 3.0)
        loss.backward()
        torch_assert_close(values.grad, torch.tensor([[-1 / 3, 1 / 3, 1 / 3]]), rtol=0, atol=1e-6)

    def test_packed_ratio_sequence_gate_uses_per_sequence_totals(self):
        values = torch.tensor([-0.5, -1.0, -1.0], requires_grad=True)
        context = {
            "old_log_probs_shifted": torch.full_like(values, -1.0),
            "advantages": torch.ones_like(values),
            "loss_mask": torch.ones_like(values, dtype=torch.bool),
            "cu_seqlens": torch.tensor([0, 2, 3], dtype=torch.int32),
        }
        loss, metrics = grpo_loss(
            {"logprobs": values},
            context,
            {},
            {
                "use_cispo_loss": True,
                "is_weight_clip_max": 10.0,
                "batch_num_tokens": 3,
                "dp_size": 1,
                "seq_mask_bounds_pos": [-0.1, 0.1],
            },
            "cpu",
        )
        self.assertEqual(metrics["seq_mask_pos_sequence_count"], 1.0)
        self.assertEqual(metrics["seq_mask_pos_drop_count"], 2.0)
        self.assertEqual(metrics["ratio_mask_dropped_token_count"], 2.0)
        self.assertAlmostEqual(loss.item(), 1 / 3, places=6)
        loss.backward()
        torch_assert_close(values.grad, torch.tensor([0.0, 0.0, -1 / 3]), rtol=0, atol=1e-6)

    def test_packed_sequence_aggregation_uses_sequence_boundaries(self):
        values = torch.tensor([[1.0, 3.0, 9.0]])
        mask = torch.ones_like(values, dtype=torch.bool)
        cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)
        expected = {
            "seq-mean-token-sum": 6.5,
            "seq-mean-token-sum-norm": 3.25,
            "seq-mean-token-mean": 5.5,
        }
        for mode, expected_loss in expected.items():
            with self.subTest(mode=mode):
                loss = agg_loss(values, mask, loss_agg_mode=mode, cu_seqlens=cu_seqlens)
                self.assertEqual(loss.item(), expected_loss)

    def test_ratio_stats_only_keeps_baseline_loss_and_empty_metric_keys(self):
        context = {
            "old_log_probs_shifted": torch.tensor([[-1.0]]),
            "advantages": torch.ones((1, 1)),
            "loss_mask": torch.ones((1, 1), dtype=torch.bool),
        }
        outputs = {"logprobs": torch.tensor([[-1.0]])}
        base = {"use_cispo_loss": True, "is_weight_clip_max": 5.0}
        baseline, baseline_metrics = grpo_loss(outputs, context, {}, base, "cpu")
        tracked, tracked_metrics = grpo_loss(outputs, context, {}, {**base, "ratio_stats": True}, "cpu")
        self.assertEqual(baseline.item(), tracked.item())
        self.assertFalse(any(key.startswith("ratio_") for key in baseline_metrics))
        self.assertEqual(tracked_metrics["ratio_trainable_token_count"], 1.0)
        empty = {**context, "loss_mask": torch.zeros((1, 1), dtype=torch.bool)}
        _, empty_metrics = grpo_loss(outputs, empty, {}, {**base, "ratio_stats": True}, "cpu")
        self.assertEqual(set(empty_metrics), set(tracked_metrics))
        self.assertEqual(empty_metrics["ratio_trainable_token_count"], 0.0)

    def test_ratio_contract_rejects_bad_config_and_mixed_nll(self):
        values = torch.tensor([[-1.0]])
        context = {
            "old_log_probs_shifted": values,
            "advantages": torch.ones_like(values),
            "loss_mask": torch.ones_like(values, dtype=torch.bool),
        }
        outputs = {"logprobs": values}
        for invalid in ([0.5, 0.5], [True, 2], [-1, 2]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "ratio_mask_bounds_pos"):
                grpo_loss(outputs, context, {}, {"use_cispo_loss": True, "ratio_mask_bounds_pos": invalid}, "cpu")
        with self.assertRaisesRegex(ValueError, "use_cispo_loss"):
            grpo_loss(outputs, context, {}, {"ratio_stats": True}, "cpu")
        with self.assertRaisesRegex(ValueError, "finite positive"):
            grpo_loss(
                outputs,
                context,
                {},
                {"use_cispo_loss": True, "is_weight_clip_max": 2.0, "ratio_m2_threshold": 0.0},
                "cpu",
            )
        with (
            patch("arctic_platform.rl.processors.grpo._get_sequence_parallel_group", return_value=object()),
            self.assertRaisesRegex(ValueError, "does not support sequence parallelism"),
        ):
            grpo_loss(
                outputs,
                context,
                {},
                {"use_cispo_loss": True, "is_weight_clip_max": 2.0, "ratio_m2_threshold": 0.1},
                "cpu",
            )
        with self.assertRaisesRegex(ValueError, "ratio-mask keys"):
            LOSS_FNS["ap_grpo_mixed_v1"](
                outputs,
                {**context, "nll_mask": context["loss_mask"]},
                {},
                {"use_cispo_loss": True, "is_weight_clip_max": 2.0, "log_ratio_sq_coef": 0.1},
                "cpu",
            )
        for invalid_clip in (True, "2"):
            with self.subTest(invalid_clip=invalid_clip), self.assertRaisesRegex(ValueError, "finite positive"):
                LOSS_FNS["ap_grpo_mixed_v1"](
                    outputs,
                    {**context, "nll_mask": context["loss_mask"]},
                    {},
                    {"use_cispo_loss": True, "is_weight_clip_max": invalid_clip},
                    "cpu",
                )
        for invalid_entropy in (None, False, "0", float("nan"), 0.1):
            with self.subTest(invalid_entropy=invalid_entropy), self.assertRaisesRegex(ValueError, "does not support"):
                LOSS_FNS["ap_grpo_mixed_v1"](
                    outputs,
                    {**context, "nll_mask": context["loss_mask"]},
                    {},
                    {
                        "use_cispo_loss": True,
                        "is_weight_clip_max": 2.0,
                        "entropy_coeff": invalid_entropy,
                    },
                    "cpu",
                )
        with self.assertRaisesRegex(ValueError, "requires CISPO"):
            LOSS_FNS["ap_grpo_mixed_v1"](
                outputs,
                {**context, "nll_mask": context["loss_mask"]},
                {},
                {"use_cispo_loss": "true", "is_weight_clip_max": 2.0},
                "cpu",
            )

    def test_packed_reduction_leaves_model_call_validation_to_request_boundaries(self):
        microbatch = {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "loss_mask": torch.ones((1, 2), dtype=torch.bool),
        }
        reduction = _grpo_packed_loss_reduction(
            [microbatch, microbatch],
            {"use_cispo_loss": True, "ratio_m2_threshold": 0.1},
            "ap_grpo",
        )
        self.assertEqual(len(reduction.loss_scales), 2)

    def test_new_columns_move_from_meta_to_shardable_batch(self):
        meta = {
            "nll_mask": torch.tensor([[1, 0], [0, 1]]),
            "teacher_log_probs_shifted": torch.zeros((2, 2)),
            "echo_observation_token_counts": torch.tensor([2, 3]),
        }
        batch, remainder = promote_batch_dim_to_batch({"input_ids": torch.zeros((2, 2))}, meta)
        self.assertEqual(remainder, {})
        self.assertEqual(set(batch) - {"input_ids"}, set(meta))

    def test_sequence_parallel_empty_policy_ratio_masks_and_echo(self):
        mp.spawn(_sp_loss_worker, args=(self.get_auto_remove_tmp_dir_str() + "/group",), nprocs=2)

    def test_prompt_mean_compensates_for_dp_averaging(self):
        values = torch.tensor([[2.0]], requires_grad=True)
        result = agg_loss(
            values,
            torch.ones_like(values, dtype=torch.bool),
            loss_agg_mode="prompt-mean",
            sequence_loss_weights=torch.tensor([1.0]),
            dp_size=4,
        )
        self.assertEqual(result.item(), 8.0)
        result.backward()
        torch_assert_close(values.grad, torch.full_like(values, 4.0))

    def test_explicit_zero_global_denominator_rejects_policy_tokens(self):
        values = torch.tensor([[1.0]])
        mask = torch.ones_like(values, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "batch_num_tokens=0"):
            agg_loss(values, mask, dp_size=1, batch_num_tokens=0)
        with self.assertRaisesRegex(ValueError, "global_batch_size=0"):
            agg_loss(values, mask, loss_agg_mode="seq-mean-token-sum", dp_size=1, global_batch_size=0)
        with self.assertRaisesRegex(ValueError, "global_batch_size=0"):
            agg_loss(
                values,
                mask,
                loss_agg_mode="prompt-mean",
                prompt_group_ids=torch.tensor([0]),
                global_batch_size=0,
            )

    def test_echo_uses_explicit_full_observation_denominator(self):
        values = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        observations = torch.tensor([[True, True]])
        loss, metrics = echo_env_prediction_loss_fn(
            values,
            mask,
            observations,
            torch.zeros_like(mask),
            global_num_echo_sequences=1,
            observation_token_counts=torch.tensor([4]),
        )
        self.assertEqual(loss.item(), 0.25)
        self.assertEqual(metrics["observation_token_count"].item(), 2)
        loss.backward()
        torch_assert_close(values.grad, torch.tensor([[-0.25, 0.0]]))

        context = {
            "old_log_probs_shifted": values.detach(),
            "advantages": torch.zeros_like(values),
            "loss_mask": torch.zeros_like(mask),
            "sft_mask": mask,
            "echo_observation_mask": observations,
        }
        config = {"aux_ce_weight": 1.0, "echo_global_num_sequences": 1}
        _, local_metrics = LOSS_FNS["ap_grpo_echo_v1"]({"logprobs": values.detach()}, context, {}, config, "cpu")
        _, full_metrics = LOSS_FNS["ap_grpo_echo_v1"](
            {"logprobs": values.detach()},
            {**context, "echo_observation_token_counts": torch.tensor([4])},
            {},
            config,
            "cpu",
        )
        self.assertNotIn("echo_full_observation_denominator", local_metrics)
        self.assertEqual(full_metrics["echo_full_observation_denominator"], 1.0)

        with self.assertRaisesRegex(ValueError, "observation_token_counts"):
            echo_env_prediction_loss_fn(
                values,
                mask,
                observations,
                torch.zeros_like(mask),
                global_num_echo_sequences=1,
                observation_token_counts=torch.tensor([1]),
            )
        with self.assertRaisesRegex(ValueError, "integer counts"):
            echo_env_prediction_loss_fn(
                values,
                mask,
                observations,
                torch.zeros_like(mask),
                global_num_echo_sequences=1,
                observation_token_counts=torch.tensor([2.5]),
            )
        with self.assertRaisesRegex(ValueError, "one full observation count per sequence"):
            echo_env_prediction_loss_fn(
                values,
                mask,
                observations,
                torch.zeros_like(mask),
                global_num_echo_sequences=1,
                observation_token_counts=torch.tensor([[4]]),
            )

    def test_echo_bfloat16_keeps_257_token_denominator_in_float32(self):
        values = torch.full((1, 257), -1.0, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.ones_like(values, dtype=torch.bool)
        loss, _ = echo_env_prediction_loss_fn(
            values,
            mask,
            mask,
            torch.zeros_like(mask),
            global_num_echo_sequences=1,
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.item(), 1.0)
        loss.backward()
        torch_assert_close(values.grad, torch.full_like(values, -1 / 257), rtol=0, atol=1e-5)

    def test_teacher_term_changes_only_scored_policy_tokens(self):
        values = torch.tensor([[-1.0, -1.0]], requires_grad=True)
        context = {
            "old_log_probs_shifted": torch.tensor([[-1.0, -1.0]]),
            "advantages": torch.zeros_like(values),
            "loss_mask": torch.ones_like(values, dtype=torch.bool),
            "teacher_log_probs_shifted": torch.tensor([[0.0, float("nan")]]),
        }
        loss, metrics = grpo_loss(
            {"logprobs": values},
            context,
            {},
            {"teacher_tau": 1.0, "teacher_clip": 2.0, "use_cispo_loss": True, "is_weight_clip_max": 5.0},
            "cpu",
        )
        self.assertEqual(metrics["teacher_term_token_count"], 1.0)
        self.assertEqual(metrics["teacher_clipped_log_ratio_sum"], 1.0)
        loss.backward()
        torch_assert_close(values.grad, torch.tensor([[-0.5, 0.0]]))

    def test_teacher_term_rejects_broadcastable_or_non_floating_inputs(self):
        values = torch.full((2, 2), -1.0)
        base_context = {
            "old_log_probs_shifted": values,
            "advantages": torch.zeros_like(values),
            "loss_mask": torch.ones_like(values, dtype=torch.bool),
        }
        config = {"teacher_tau": 1.0, "teacher_clip": 2.0}
        with self.assertRaisesRegex(ValueError, "exactly match"):
            grpo_loss(
                {"logprobs": values},
                {**base_context, "teacher_log_probs_shifted": torch.zeros((1, 2))},
                {},
                config,
                "cpu",
            )
        for invalid_tau in (False, "0", "1"):
            with self.subTest(teacher_tau=invalid_tau), self.assertRaisesRegex(ValueError, "teacher_tau"):
                grpo_loss(
                    {"logprobs": values},
                    {**base_context, "teacher_log_probs_shifted": torch.zeros_like(values)},
                    {},
                    {"teacher_tau": invalid_tau, "teacher_clip": 2.0},
                    "cpu",
                )
        with self.assertRaisesRegex(ValueError, "floating-point tensor"):
            grpo_loss(
                {"logprobs": values},
                {**base_context, "teacher_log_probs_shifted": torch.zeros((2, 2), dtype=torch.int64)},
                {},
                config,
                "cpu",
            )

    def test_empty_teacher_shard_preserves_metric_contract(self):
        values = torch.tensor([[-1.0]], requires_grad=True)
        context = {
            "old_log_probs_shifted": torch.tensor([[-1.0]]),
            "advantages": torch.zeros_like(values),
            "loss_mask": torch.zeros_like(values, dtype=torch.bool),
            "teacher_log_probs_shifted": torch.tensor([[0.0]]),
        }
        loss, metrics = grpo_loss(
            {"logprobs": values},
            context,
            {},
            {"teacher_tau": 1.0, "teacher_clip": 2.0},
            "cpu",
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["teacher_tau"], 1.0)
        self.assertEqual(metrics["teacher_term_token_count"], 0.0)
        self.assertEqual(metrics["teacher_log_ratio_sum"], 0.0)
        self.assertEqual(metrics["teacher_clipped_log_ratio_sum"], 0.0)
        loss.backward()
        torch_assert_close(values.grad, torch.zeros_like(values))

    def test_mixed_loss_and_worker_metrics_exclude_nll_from_rl_means(self):
        values = torch.tensor([[-2.0, -3.0]], requires_grad=True)
        context = {
            "old_log_probs_shifted": torch.tensor([[-2.0, float("nan")]]),
            "advantages": torch.tensor([[1.0, float("nan")]]),
            "loss_mask": torch.ones_like(values, dtype=torch.bool),
            "nll_mask": torch.tensor([[False, True]]),
        }
        loss, metrics = LOSS_FNS["ap_grpo_mixed_v1"](
            {"logprobs": values},
            context,
            {},
            {"use_cispo_loss": True, "is_weight_clip_max": 5.0},
            "cpu",
        )
        self.assertEqual(loss.item(), 2.5)
        self.assertEqual(metrics["nll_sum"], 3.0)
        self.assertEqual(metrics["grpo_stats_token_count"], 1.0)
        loss.backward()
        torch_assert_close(values.grad, torch.tensor([[-0.5, -0.5]]))
        merged = combine_metric_shards([metrics, {**metrics, "grpo_importance_weight_sum": 3.0}])
        resolve_loss("ap_grpo_mixed_v1").metrics_callback([], merged)
        self.assertEqual(merged["importance_weight"], 2.0)

    def test_mixed_rejects_nonfinite_logprobs_on_nll_tokens(self):
        values = torch.tensor([[-2.0, float("nan")]], requires_grad=True)
        context = {
            "old_log_probs_shifted": torch.tensor([[-2.0, -3.0]]),
            "advantages": torch.ones_like(values),
            "loss_mask": torch.ones_like(values, dtype=torch.bool),
            "nll_mask": torch.tensor([[False, True]]),
        }
        with self.assertRaisesRegex(ValueError, "finite at every active nll_mask"):
            LOSS_FNS["ap_grpo_mixed_v1"](
                {"logprobs": values},
                context,
                {},
                {"use_cispo_loss": True, "is_weight_clip_max": 5.0},
                "cpu",
            )

    def test_mixed_rejects_non_subset_and_plain_loss_refuses_nll(self):
        context = {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "old_log_probs_shifted": torch.tensor([[-1.0, -1.0]]),
            "advantages": torch.ones((1, 2)),
            "loss_mask": torch.tensor([[True, False]]),
            "nll_mask": torch.tensor([[False, True]]),
        }
        outputs = {"logprobs": torch.tensor([[-1.0, -1.0]])}
        with self.assertRaisesRegex(ValueError, "subset"):
            LOSS_FNS["ap_grpo_mixed_v1"](
                outputs, context, {}, {"use_cispo_loss": True, "is_weight_clip_max": 5.0}, "cpu"
            )
        with self.assertRaisesRegex(ValueError, "nll_mask requires"):
            grpo_loss(outputs, context, {}, {}, "cpu")
        with self.assertRaisesRegex(ValueError, "nll_mask requires"):
            LOSS_FNS["grpo"](outputs, context, {}, {}, "cpu")
        with self.assertRaisesRegex(ValueError, "loss_fn='grpo_mixed_v1'"):
            _grpo_packed_loss_reduction([context], {}, "grpo")

    def test_mixed_rejects_behavioral_importance_sampling_inputs(self):
        values = torch.tensor([[-1.0, -1.0]])
        context = {
            "old_log_probs_shifted": values,
            "advantages": torch.ones_like(values),
            "loss_mask": torch.ones_like(values, dtype=torch.bool),
            "nll_mask": torch.tensor([[False, True]]),
        }
        config = {"use_cispo_loss": True, "is_weight_clip_max": 5.0}
        for key, value in (
            ("behav_imp_weight_cap", 1.5),
            ("current_version", 2),
            ("prox_logp_method", "loglinear"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "decoupled"):
                LOSS_FNS["ap_grpo_mixed_v1"](
                    {"logprobs": values},
                    context,
                    {},
                    {**config, key: value},
                    "cpu",
                )
        with self.assertRaisesRegex(ValueError, "decoupled"):
            LOSS_FNS["ap_grpo_mixed_v1"](
                {"logprobs": values},
                {**context, "prox_logp_shifted": values + 1.0},
                {},
                config,
                "cpu",
            )

    def test_unprefixed_echo_accepts_ratio_telemetry(self):
        values = torch.tensor([[-1.0, -1.0]])
        context = {
            "old_log_probs_shifted": values,
            "advantages": torch.ones_like(values),
            "loss_mask": torch.tensor([[True, False]]),
            "sft_mask": torch.tensor([[False, True]]),
            "echo_observation_mask": torch.tensor([[False, True]]),
        }
        loss, metrics = LOSS_FNS["grpo_echo_v1"](
            {"logprobs": values},
            context,
            {},
            {
                "use_cispo_loss": True,
                "is_weight_clip_max": 5.0,
                "ratio_stats": True,
                "aux_ce_weight": 0.0,
                "echo_global_num_sequences": 1,
            },
            "cpu",
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["ratio_trainable_token_count"], 1.0)
        self.assertEqual(metrics["echo_contract_version"], 1.0)
