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

"""Signed advantage mass reported by the CISPO ratio masks (``ratio_mask_{pos,neg}_{pre,kept}_mass_sum``).

Every test runs the real ``grpo_loss``. The shared request is five rows of four tokens whose advantages, row weights
and denominators are binary fractions, so the independent reference below (plain tensor arithmetic, no ``agg_loss``)
matches the loss exactly and every comparison is an equality unless a comment says otherwise. The last row has no
policy token: it stands for a worker or microbatch with nothing to train.
"""

from __future__ import annotations

import itertools
import math

import torch

from arctic_platform.common.utils.batch import combine_metric_microbatches
from arctic_platform.common.utils.batch import combine_metric_shards
from arctic_platform.rl.processors.grpo import grpo_loss
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import torch_assert_close

STAGES_AND_SIGNS = tuple(itertools.product(("pre", "kept"), ("pos", "neg")))
MASS_KEYS = tuple(f"ratio_mask_{sign}_{stage}_mass_sum" for stage, sign in STAGES_AND_SIGNS)
K3_KEY = "ratio_mask_kept_k3_sum"
LOG_RATIO_CLAMP = (
    20.0  # the bound on the log ratio in the reported k3 sum (dss-platform's trainable_logprob_lowvar_all)
)
# Everything the ratio controls add on top of the pre-existing metrics: the four masses and the kept-token k3 sum.
NEW_KEYS = (*MASS_KEYS, K3_KEY)
ROW_TOKENS = 4
ADVANTAGES = torch.tensor(
    [
        [1.0, 3.0, -2.0, -1.0],  # sequence mean +0.25
        [2.0, 2.0, -1.0, -4.0],  # -0.25
        [4.0, -2.0, 2.0, 0.0],  # +1.0
        [-3.0, 1.0, -2.0, -1.0],  # -1.25
        [0.0, 0.0, 0.0, 0.0],
    ]
)
# Two tokens of every policy row have a log-ratio of +1 and -1, far outside the [0.9, 1.1] band below, so the ratio gate
# drops them. They cancel within the row, so the sequence-level ratio of every row is exactly 1 as well.
DELTA = torch.tensor([[0, 1, -1, 0], [0, -1, 0, 1], [1, -1, 0, 0], [0, 0, 1, -1], [0, 0, 0, 0]], dtype=torch.float32)
LOSS_MASK = torch.tensor([[True] * 4] * 4 + [[False] * 4])
RATIO_GATES = {"ratio_mask_bounds_pos": [0.9, 1.1], "ratio_mask_bounds_neg": [0.9, 1.1]}
KEPT = LOSS_MASK & (DELTA == 0)
ROW_WEIGHTS = torch.tensor([0.25, 0.5, 0.125, 0.125, 0.5])
PROMPT_IDS = torch.tensor([0, 0, 1, 1, 2])
# Prompt 0 owns rows 0 and 2, so it straddles the first two microbatches below. As DSS does, every row carries the
# step-global token count of its prompt (8, 4, 4 and 0 tokens).
CUT_PROMPT_IDS = torch.tensor([0, 1, 0, 2, 3])
CUT_PROMPT_TOKEN_COUNTS = torch.tensor([8.0, 4.0, 8.0, 4.0, 0.0])
ROWS_PER_MICROBATCH = ([0, 1], [2, 3], [4])
ALL_ROWS = [0, 1, 2, 3, 4]
ROW_CONTEXT_KEYS = ("sequence_loss_weights", "prompt_group_ids", "prompt_token_counts")
LEVELS = ("token", "sequence")
LAYOUTS = ("padded", "packed")

# mode -> (config, context, reference normalisation of a [row, token] tensor of per-token terms). The reference restates
# what each mode divides by, for the step-global denominators the request carries.
MODES = {
    "token-mean": (dict(batch_num_tokens=16), {}, lambda x: x.sum() / 16),
    "seq-mean-token-sum": (dict(global_batch_size=4), {}, lambda x: x.sum() / 4),
    "seq-mean-token-sum-norm": (dict(global_batch_size=4), dict(packed_loss_scale_factor=8), lambda x: x.sum() / 32),
    "seq-mean-token-mean": (dict(global_batch_size=4), {}, lambda x: (x.sum(dim=1) / ROW_TOKENS).sum() / 4),
    "prompt-mean weighted": (
        {},
        dict(sequence_loss_weights=ROW_WEIGHTS),
        lambda x: (ROW_WEIGHTS.to(x.dtype) * x.sum(dim=1) / ROW_TOKENS).sum(),
    ),
    "prompt-mean grouped": (
        dict(global_batch_size=2),
        dict(prompt_group_ids=PROMPT_IDS),
        lambda x: (x[:2].sum() / (2 * ROW_TOKENS) + x[2:4].sum() / (2 * ROW_TOKENS)) / 2,
    ),
    # The last prompt (row 4) has no tokens, so it adds 0 to the sum but still counts in global_batch_size.
    "prompt-mean grouped, cut prompts": (
        dict(global_batch_size=4),
        dict(prompt_group_ids=CUT_PROMPT_IDS, prompt_token_counts=CUT_PROMPT_TOKEN_COUNTS),
        lambda x: (x[[0, 2]].sum() / 8 + x[1].sum() / 4 + x[3].sum() / 4) / 4,
    ),
}


def k3(log_ratio: float) -> float:
    """The per-token k3 that dss-platform's trainable_logprob_lowvar_all averages: exp(d) - 1 - d for d clamped to +-20."""
    clamped = max(-LOG_RATIO_CLAMP, min(LOG_RATIO_CLAMP, log_ratio))
    return math.expm1(clamped) - clamped


def effective_advantages(level: str) -> torch.Tensor:
    """The advantages the loss uses: the token-level ones, or each sequence's mean under sequence-level IS."""
    if level == "token":
        return ADVANTAGES
    return ADVANTAGES.mean(dim=1, keepdim=True).expand_as(ADVANTAGES)


def reference_mass(mode: str, level: str, kept: torch.Tensor) -> dict[str, float]:
    normalise = MODES[mode][2]
    advantages = effective_advantages(level).double()
    masses = {}
    for stage, sign in STAGES_AND_SIGNS:
        magnitude = advantages.clamp(min=0) if sign == "pos" else (-advantages).clamp(min=0)
        trained = kept if stage == "kept" else LOSS_MASK
        masses[f"ratio_mask_{sign}_{stage}_mass_sum"] = normalise(magnitude * trained).item()
    return masses


class TestRatioMass(TestCasePlus):
    def run_loss(self, advantages, delta, config, **context):
        values = torch.full_like(advantages, -2.0, requires_grad=True)
        context.setdefault("loss_mask", torch.ones_like(values, dtype=torch.bool))
        loss, metrics = grpo_loss(
            {"logprobs": values},
            dict(old_log_probs_shifted=values.detach() - delta, advantages=advantages, **context),
            {},
            dict(use_cispo_loss=True, is_weight_clip_max=5.0, **config),
            "cpu",
        )
        loss.backward()
        return loss.detach(), values.grad, metrics

    def run_rows(self, mode, level, layout, rows, controls=None, dp_size=1, without=()):
        """The given rows of the shared request as one model call; returns (loss, grad, metrics)."""
        mode_config, mode_context, _ = MODES[mode]
        mode_context = {key: value for key, value in mode_context.items() if key not in without}
        config = dict(
            loss_agg_mode=mode.split(" ")[0], importance_sampling_level=level, dp_size=dp_size, **mode_config
        )
        config.update(controls or {})
        context = {key: value[rows] if key in ROW_CONTEXT_KEYS else value for key, value in mode_context.items()}
        advantages, delta, mask = ADVANTAGES[rows], DELTA[rows], LOSS_MASK[rows]
        if layout == "packed":
            advantages, delta, mask = advantages.reshape(-1), delta.reshape(-1), mask.reshape(-1)
            context["cu_seqlens"] = torch.arange(0, (len(rows) + 1) * ROW_TOKENS, ROW_TOKENS, dtype=torch.int32)
        return self.run_loss(advantages, delta, config, loss_mask=mask, **context)

    def test_statistics_leave_loss_gradient_and_existing_metrics_unchanged(self):
        for mode, level, layout, dp_size in itertools.product(MODES, LEVELS, LAYOUTS, (1, 4)):
            with self.subTest(mode=mode, level=level, layout=layout, dp_size=dp_size):
                # With the gates on, a kept token has ratio 1 and logprob -2, so its policy term is exactly
                # 2 * A and its gradient -A. The reference differentiates the independent normalisation.
                loss, grad, gated_metrics = self.run_rows(mode, level, layout, ALL_ROWS, RATIO_GATES, dp_size)
                values = torch.full_like(ADVANTAGES, -2.0, requires_grad=True)
                expected = dp_size * MODES[mode][2](KEPT * (-effective_advantages(level)) * values)
                expected.backward()
                torch_assert_close(loss, expected.detach(), rtol=0, atol=0)
                torch_assert_close(grad.reshape(ADVANTAGES.shape), values.grad, rtol=0, atol=0)
                # The loss carries the DP compensation; the mass does not.
                self.assertEqual({key: gated_metrics[key] for key in MASS_KEYS}, reference_mass(mode, level, KEPT))

                # With no gate only the telemetry switch differs, so the same code computes the objective.
                off_loss, off_grad, off_metrics = self.run_rows(mode, level, layout, ALL_ROWS, dp_size=dp_size)
                on_loss, on_grad, on_metrics = self.run_rows(
                    mode, level, layout, ALL_ROWS, {"ratio_stats": True}, dp_size
                )
                torch_assert_close(on_loss, off_loss, rtol=0, atol=0)
                torch_assert_close(on_grad, off_grad, rtol=0, atol=0)
                self.assertEqual({key: on_metrics[key] for key in off_metrics}, off_metrics)
                self.assertFalse(set(NEW_KEYS) & set(off_metrics))
                self.assertTrue(set(NEW_KEYS) <= set(on_metrics))

    def test_mass_matches_reference_and_is_additive_over_microbatches(self):
        for mode, level, layout in itertools.product(MODES, LEVELS, LAYOUTS):
            for controls, kept in ((RATIO_GATES, KEPT), ({"ratio_stats": True}, LOSS_MASK)):
                with self.subTest(mode=mode, level=level, layout=layout, controls=sorted(controls)):
                    expected = reference_mass(mode, level, kept)
                    whole_loss, _, whole = self.run_rows(mode, level, layout, ALL_ROWS, controls)
                    # The last microbatch has nothing to train, and in the cut-prompts mode the first prompt straddles
                    # two microbatches. (The engine rejects several microbatches for seq-mean-token-sum-norm; with the
                    # explicit scale used here the sums still split.)
                    parts = [self.run_rows(mode, level, layout, rows, controls) for rows in ROWS_PER_MICROBATCH]
                    merged = combine_metric_shards([combine_metric_microbatches([part[2] for part in parts])])
                    for key in MASS_KEYS:
                        self.assertEqual(whole[key], expected[key], msg=key)
                        self.assertEqual(merged[key], expected[key], msg=key)
                    # k3 = exp(d) - 1 - d over the kept tokens. Every policy row has one +1 and one -1 log-ratio token:
                    # with the gates both are dropped, with ratio_stats alone both are kept. It is a plain fp32 sum of
                    # at most 8 terms below 0.72 (ulp ~5e-7 at 4), so 5e-6 absorbs the summation order of the split runs,
                    # while a missing or doubled token moves it by >= 0.3.
                    expected_k3 = 4 * (math.e - 2 + 1 / math.e) if controls == {"ratio_stats": True} else 0.0
                    for metrics in (whole, merged):
                        torch_assert_close(metrics["ratio_mask_kept_k3_sum"], expected_k3, rtol=0, atol=5e-6)
                    if controls is RATIO_GATES:
                        # Exact only with the gates on: every kept token then has ratio 1 (see the test above).
                        torch_assert_close(torch.stack([part[0] for part in parts]).sum(), whole_loss, rtol=0, atol=0)

    def test_cut_prompts_are_additive_only_with_global_prompt_token_counts(self):
        mode = "prompt-mean grouped, cut prompts"
        expected = reference_mass(mode, "token", KEPT)
        for layout in LAYOUTS:
            for without, additive in (((), True), (("prompt_token_counts",), False)):
                with self.subTest(layout=layout, without=without):
                    whole = self.run_rows(mode, "token", layout, ALL_ROWS, RATIO_GATES, without=without)[2]
                    parts = [
                        self.run_rows(mode, "token", layout, rows, RATIO_GATES, without=without)[2]
                        for rows in ROWS_PER_MICROBATCH
                    ]
                    merged = combine_metric_microbatches(parts)
                    # Unsplit, every prompt is whole in the call and counts itself.
                    self.assertEqual({key: whole[key] for key in MASS_KEYS}, expected)
                    # Split, a prompt's rows are normalised by the tokens of the rows in each call unless the global
                    # count says otherwise; the loss behaves the same way, so the mass follows it.
                    self.assertEqual({key: merged[key] for key in MASS_KEYS} == expected, additive)

    def test_seq_mean_token_sum_norm_is_additive_only_with_an_explicit_scale(self):
        # Two packed sequences of 2 and 4 tokens, advantage +1. Unsplit, the default scale is the longest segment (4).
        config = dict(loss_agg_mode="seq-mean-token-sum-norm", global_batch_size=2, dp_size=1, ratio_stats=True)

        def pos_pre(*segment_lengths, **context):
            boundaries = torch.tensor([0, *itertools.accumulate(segment_lengths)], dtype=torch.int32)
            tokens = int(boundaries[-1])
            metrics = self.run_loss(torch.ones(tokens), torch.zeros(tokens), config, cu_seqlens=boundaries, **context)[
                2
            ]
            return metrics["ratio_mask_pos_pre_mass_sum"]

        for context, split in (({}, 1.0), ({"packed_loss_scale_factor": 4}, 0.75)):
            with self.subTest(context=context):
                self.assertEqual(pos_pre(2, 4, **context), 0.75)
                # Without the scale each call divides by its own longest segment: 2 / 2 / 2 + 4 / 2 / 4 = 1, not 3 / 4.
                self.assertEqual(pos_pre(2, **context) + pos_pre(4, **context), split)

    def m2_outlier_rows(self):
        """Row A has one outlier (log-ratio offset 2, advantage +2), row B none; both have four policy tokens."""
        return {
            "A": (torch.tensor([[2.0, 1.0, 1.0, 1.0]]), torch.tensor([[2.0, 0.0, 0.0, 0.0]])),
            "B": (torch.ones(1, 4), torch.zeros(1, 4)),
        }

    def test_m2po_kept_mass_is_what_each_workers_own_ranking_kept(self):
        # ratio_m2_threshold ranks the tokens of one model call. Alone, row A's outlier is dropped, but among the eight
        # tokens of both rows the average stays under the threshold.
        rows = self.m2_outlier_rows()
        config = dict(batch_num_tokens=8, dp_size=1, ratio_m2_threshold=0.75)
        whole = self.run_loss(
            torch.cat([rows["A"][0], rows["B"][0]]), torch.cat([rows["A"][1], rows["B"][1]]), config
        )[2]
        parts = [self.run_loss(*rows[name], config)[2] for name in rows]
        merged = combine_metric_microbatches(parts)
        self.assertEqual((whole["ratio_m2_drop_count"], merged["ratio_m2_drop_count"]), (0.0, 1.0))
        # pre does not depend on the placement; kept loses the dropped token's 2 / 8 only when A is ranked alone.
        self.assertEqual(whole["ratio_mask_pos_pre_mass_sum"], merged["ratio_mask_pos_pre_mass_sum"])
        self.assertEqual(
            (whole["ratio_mask_pos_kept_mass_sum"], merged["ratio_mask_pos_kept_mass_sum"]), (1.125, 0.875)
        )

    def test_m2po_kept_k3_sum_depends_on_row_placement_like_the_kept_mass(self):
        # Same rows as above: the outlier token (log-ratio 2) is in the kept k3 sum when both rows are ranked together
        # and out of it when row A is ranked alone. The token counts do not move: ratio_m2_threshold is a drop.
        rows = self.m2_outlier_rows()
        config = dict(batch_num_tokens=8, dp_size=1, ratio_m2_threshold=0.75)
        whole = self.run_loss(
            torch.cat([rows["A"][0], rows["B"][0]]), torch.cat([rows["A"][1], rows["B"][1]]), config
        )[2]
        merged = combine_metric_microbatches([self.run_loss(*rows[name], config)[2] for name in rows])
        # fp32 expm1 against a float64 reference: ~1e-7 relative; the outlier's k3 is 4.39 and every other token has 0.
        torch_assert_close(whole[K3_KEY], k3(2.0), rtol=0, atol=1e-5)
        self.assertEqual(merged[K3_KEY], 0.0)
        for key in ("ratio_trainable_token_count", "ratio_mask_dropped_token_count"):
            self.assertEqual(
                (whole[key], merged[key]), (8.0, 8.0) if key.endswith("trainable_token_count") else (0.0, 1.0)
            )

    def test_legacy_m2_threshold_makes_pre_kept_k3_and_the_counts_depend_on_row_placement(self):
        # The legacy m2_threshold ranks one call's tokens and shrinks the loss mask before the ratio controls run, so
        # the outlier token leaves pre, kept, the k3 sum and ratio_trainable_token_count when row A is ranked alone.
        rows = self.m2_outlier_rows()
        config = dict(batch_num_tokens=8, dp_size=1, ratio_stats=True, m2_threshold=0.75)

        def run(*tensors):
            return self.run_loss(*tensors, config, prox_logp_shifted=torch.full_like(tensors[0], -2.0))[2]

        whole = run(torch.cat([rows["A"][0], rows["B"][0]]), torch.cat([rows["A"][1], rows["B"][1]]))
        merged = combine_metric_microbatches([run(*rows[name]) for name in rows])
        self.assertEqual(
            [whole[key] for key in ("ratio_trainable_token_count", "ratio_mask_pos_pre_mass_sum")], [8.0, 1.125]
        )
        self.assertEqual(
            [merged[key] for key in ("ratio_trainable_token_count", "ratio_mask_pos_pre_mass_sum")], [7.0, 0.875]
        )
        self.assertEqual(
            (whole["ratio_mask_pos_kept_mass_sum"], merged["ratio_mask_pos_kept_mass_sum"]), (1.125, 0.875)
        )
        torch_assert_close(whole[K3_KEY], k3(2.0), rtol=0, atol=1e-5)  # fp32 expm1, see the ratio_m2_threshold test
        self.assertEqual(merged[K3_KEY], 0.0)

    def test_legacy_m2_threshold_tokens_are_in_neither_pre_nor_kept(self):
        # Advantages +1 +2 -1 -2; the third token is the M2PO outlier (|old - proximal| = 2) for both mechanisms.
        # The legacy m2_threshold shrinks the loss mask before the ratio controls, so that token leaves pre as well;
        # ratio_m2_threshold is one of the drops, so its token stays in pre and leaves kept.
        advantages = torch.tensor([[1.0, 2.0, -1.0, -2.0]])
        delta = torch.tensor([[0.0, 0.0, 2.0, 0.0]])
        proximal = dict(prox_logp_shifted=torch.full_like(advantages, -2.0))
        cases = {
            "no M2PO": ({}, [0.75, 0.75, 0.75, 0.75]),
            "m2_threshold": (dict(m2_threshold=0.5), [0.75, 0.5, 0.75, 0.5]),
            "ratio_m2_threshold": (dict(ratio_m2_threshold=0.5), [0.75, 0.75, 0.75, 0.5]),
        }
        for name, (gates, masses) in cases.items():
            with self.subTest(name):
                config = dict(batch_num_tokens=4, dp_size=1, ratio_stats=True, **gates)
                metrics = self.run_loss(advantages, delta, config, **proximal)[2]
                self.assertEqual([metrics[key] for key in MASS_KEYS], masses)

    def test_teacher_term_is_part_of_the_advantage_in_the_mass(self):
        # The teacher is 1 nat above the policy, so with tau 0.5 every advantage moves by +0.5.
        advantages = torch.tensor([[2.0, 1.0, -2.0, -4.0]])
        delta = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        teacher = dict(teacher_tau=0.5, teacher_clip=2.0)
        context = dict(teacher_log_probs_shifted=torch.full_like(advantages, -1.0))
        config = dict(batch_num_tokens=4, dp_size=1, ratio_mask_bounds_pos=[0.9, 1.1])
        without = self.run_loss(advantages, delta, config)[2]
        with_teacher = self.run_loss(advantages, delta, {**config, **teacher}, **context)[2]
        # Effective advantages +2.5 +1.5 -1.5 -3.5: pre is 4/4 and 5/4; the ratio gate drops the first token.
        self.assertEqual([without[key] for key in MASS_KEYS], [0.75, 1.5, 0.25, 1.5])
        self.assertEqual([with_teacher[key] for key in MASS_KEYS], [1.0, 1.25, 0.375, 1.25])

    def test_importance_and_behavior_weights_do_not_scale_the_mass(self):
        advantages = torch.tensor([[2.0, 1.0, -2.0, -4.0]])
        delta = torch.zeros_like(advantages)
        # Token 0 has a behavior weight exp(1) above the cap of 1.5, so the loss zeroes it. rollout_is_weights triples it.
        proximal = dict(prox_logp_shifted=torch.tensor([[-1.0, -2.0, -2.0, -2.0]]))
        masses = [0.75, 1.5, 0.75, 1.5]
        config = dict(batch_num_tokens=4, dp_size=1, ratio_stats=True)
        base_loss, _, base_metrics = self.run_loss(advantages, delta, config, **proximal)
        for name, extra_config, context in (
            ("rollout_is_weights", {}, dict(rollout_is_weights=torch.full_like(advantages, 3.0))),
            ("behav_imp_weight_cap", dict(behav_imp_weight_cap=1.5), {}),
        ):
            with self.subTest(name):
                loss, _, metrics = self.run_loss(advantages, delta, {**config, **extra_config}, **proximal, **context)
                self.assertNotEqual(loss.item(), base_loss.item(), msg="the weight must change the objective")
                self.assertEqual([metrics[key] for key in MASS_KEYS], masses)
        self.assertEqual([base_metrics[key] for key in MASS_KEYS], masses)

    def test_sequence_level_mass_uses_effective_advantages(self):
        # One row whose token-level advantages are +3, -1, +1, -1 (mean +0.5). Token level: pos 4, neg 2. Sequence
        # level: the loss sees +0.5 on every token, so pos is 4 * 0.5 and neg is 0. The ratio gate drops the second
        # token, whose token-level advantage is negative; its effective advantage is positive, so "pos" loses it.
        advantages = torch.tensor([[3.0, -1.0, 1.0, -1.0]])
        delta = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        expected = {
            "token": dict(pos_pre=1.0, neg_pre=0.5, pos_kept=1.0, neg_kept=0.25),
            "sequence": dict(pos_pre=0.5, neg_pre=0.0, pos_kept=0.375, neg_kept=0.0),
        }
        for level, masses in expected.items():
            with self.subTest(level=level):
                config = dict(batch_num_tokens=4, dp_size=1, importance_sampling_level=level, **RATIO_GATES)
                metrics = self.run_loss(advantages, delta, config)[2]
                for name, value in masses.items():
                    sign, stage = name.split("_")
                    self.assertEqual(metrics[f"ratio_mask_{sign}_{stage}_mass_sum"], value, msg=name)

    def test_kept_mass_excludes_tokens_dropped_by_every_active_gate(self):
        # Token-mean over one row with A = +2, +1, -2, -4, so pre is pos 3/4 and neg 6/4. delta is each token's
        # log-ratio offset (old = logprob - delta); the logprob is -2 everywhere.
        advantages = torch.tensor([[2.0, 1.0, -2.0, -4.0]])
        far = 0.5
        cases = {
            "ratio band, high side, positive token": (dict(ratio_mask_bounds_pos=[0.9, 1.1]), [1, 0, 0, 0], 0.25, 1.5),
            "ratio band, low side, negative token": (dict(ratio_mask_bounds_neg=[0.9, 1.1]), [0, 0, 0, -1], 0.75, 0.5),
            "probability gap, positive token": (dict(prob_diff_mask_max_pos=0.05), [1, 0, 0, 0], 0.25, 1.5),
            "probability gap, negative token": (dict(prob_diff_mask_max_neg=0.05), [0, 0, -1, 0], 0.75, 1.0),
            "sequence gate, positive tokens": (dict(seq_mask_bounds_pos=[-0.1, 0.1]), [far] * 4, 0.0, 1.5),
            "sequence gate, negative tokens": (dict(seq_mask_bounds_neg=[-0.1, 0.1]), [far] * 4, 0.75, 0.0),
            "M2PO": (dict(ratio_m2_threshold=0.5), [0, 2, 0, 0], 0.5, 1.5),
            # Both positive tokens are outside the ratio band and the probability gap, and the second is also the
            # M2PO outlier. Subtracting a token once per gate would take kept below zero.
            "overlapping gates": (
                dict(ratio_mask_bounds_pos=[0.9, 1.1], prob_diff_mask_max_pos=0.05, ratio_m2_threshold=0.5),
                [1, 2, 0, 0],
                0.0,
                1.5,
            ),
        }
        for name, (gates, delta, pos_kept, neg_kept) in cases.items():
            with self.subTest(name):
                config = dict(batch_num_tokens=4, dp_size=1, **gates)
                metrics = self.run_loss(advantages, torch.tensor([delta], dtype=torch.float32), config)[2]
                self.assertEqual(
                    {key: metrics[key] for key in MASS_KEYS},
                    {
                        "ratio_mask_pos_pre_mass_sum": 0.75,
                        "ratio_mask_neg_pre_mass_sum": 1.5,
                        "ratio_mask_pos_kept_mass_sum": pos_kept,
                        "ratio_mask_neg_kept_mass_sum": neg_kept,
                    },
                )

    def test_without_step_global_denominators_mass_is_a_per_call_mean(self):
        # The sums carry the loss's denominators (docs/rl.md). With none supplied, each call divides by its own token,
        # sequence or prompt count, exactly as the loss does, so the sum over calls is not the unsplit value. (The
        # grouped prompt count is only all-reduced when dp_size > 1 in a multi-rank group; here dp_size is 1, and under
        # pytest torch.distributed is a 1-rank group anyway.) Each of the two rows has positive |A| mass 4 over
        # 4 tokens, and the mode's unsplit value is X; every call alone also gets X.
        for mode, unsplit, per_row_context in (
            ("token-mean", 1.0, {}),
            ("seq-mean-token-sum", 4.0, {}),
            ("seq-mean-token-mean", 1.0, {}),
            ("prompt-mean", 1.0, dict(prompt_group_ids=torch.tensor([0, 1]))),
        ):
            with self.subTest(mode=mode):
                config = dict(loss_agg_mode=mode, dp_size=1, ratio_stats=True)

                def run(rows):
                    context = {key: value[rows] for key, value in per_row_context.items()}
                    return self.run_loss(ADVANTAGES[rows], DELTA[rows], config, **context)[2]

                whole = run(slice(0, 2))
                parts = [run(slice(i, i + 1)) for i in range(2)]
                self.assertEqual(whole["ratio_mask_pos_pre_mass_sum"], unsplit)
                self.assertEqual([part["ratio_mask_pos_pre_mass_sum"] for part in parts], [unsplit, unsplit])
                self.assertEqual(combine_metric_microbatches(parts)["ratio_mask_pos_pre_mass_sum"], 2 * unsplit)

    def test_kept_token_mean_k3_follows_from_the_reported_sum_and_counts(self):
        # Log-ratios +0.5, -0.5, 0 and 0 (the logprob is -2 and old = logprob - delta, so they are exact in fp32).
        # The positive-side band [0.5, 1.3] drops the first token (ratio e^0.5 = 1.65) and keeps the second (0.61).
        advantages = torch.ones(1, 4)
        delta = torch.tensor([[0.5, -0.5, 0.0, 0.0]])
        config = dict(batch_num_tokens=4, dp_size=1, ratio_mask_bounds_pos=[0.5, 1.3])
        metrics = self.run_loss(advantages, delta, config)[2]
        kept_k3 = math.expm1(-0.5) + 0.5
        # fp32 expm1 against a float64 reference: ~1e-8 relative; a wrong token moves the sum by >= 0.1.
        torch_assert_close(metrics[K3_KEY], kept_k3, rtol=0, atol=1e-6)
        kept_tokens = metrics["ratio_trainable_token_count"] - metrics["ratio_mask_dropped_token_count"]
        self.assertEqual(kept_tokens, 3.0)
        torch_assert_close(metrics[K3_KEY] / kept_tokens, kept_k3 / 3, rtol=0, atol=1e-6)

    def test_mean_k3_gate_selects_the_tokens_the_reported_sum_adds_up(self):
        # Log-ratios +0.5, -0.5, +0.5, -0.5: the sequence mean log ratio is 0, but the mean k3 is ~0.128, above the
        # bound 0.1. Only seq_mask_stat="mean_k3" gates the positive tokens (A = +2, +1) out, and the reported sum then
        # covers just the tokens that were kept.
        advantages = torch.tensor([[2.0, 1.0, -2.0, -4.0]])
        delta = torch.tensor([[0.5, -0.5, 0.5, -0.5]])
        pair = k3(0.5) + k3(-0.5)
        # mean_log_ratio keeps all four tokens; mean_k3 drops the two positive ones and keeps the negative pair.
        for stat, pos_kept, kept_k3 in (("mean_log_ratio", 0.75, 2 * pair), ("mean_k3", 0.0, pair)):
            with self.subTest(seq_mask_stat=stat):
                config = dict(batch_num_tokens=4, dp_size=1, seq_mask_bounds_pos=[-0.1, 0.1], seq_mask_stat=stat)
                metrics = self.run_loss(advantages, delta, config)[2]
                self.assertEqual(metrics["ratio_mask_pos_kept_mass_sum"], pos_kept)
                # fp32 expm1 against a float64 reference: ~1e-8 relative; a wrong token moves the sum by >= 0.1.
                torch_assert_close(metrics[K3_KEY], kept_k3, rtol=0, atol=1e-6)

    def test_mean_k3_gate_uses_the_unclamped_k3_while_the_reported_sum_clamps(self):
        # One token with log-ratio 25 in a row of four. Unclamped, the row's mean k3 is exp(25)/4 ~ 1.8e10, above the
        # bound 1e9, so the mean_k3 gate drops the row's positive tokens; clamped at 20 it would be ~1.2e8 and drop
        # nothing. The gate must keep deciding on the unclamped value (which tokens are dropped does not change), while
        # the reported sum of the tokens that are kept is clamped.
        advantages = torch.tensor([[2.0, 1.0, -2.0, -4.0]])
        delta = torch.tensor([[25.0, 0.0, 0.0, 0.0]])
        for stat, pos_kept, kept_k3 in (("mean_log_ratio", 0.75, k3(25.0)), ("mean_k3", 0.0, 0.0)):
            with self.subTest(seq_mask_stat=stat):
                config = dict(batch_num_tokens=4, dp_size=1, seq_mask_bounds_pos=[0.0, 1e9], seq_mask_stat=stat)
                metrics = self.run_loss(advantages, delta, config)[2]
                self.assertEqual(metrics["ratio_mask_pos_kept_mass_sum"], pos_kept)
                # fp32 expm1(20) = 4.85e8 has ulp 32, ~7e-8 relative; 1e-6 is ~16 ulp, an unclamped value is 150x off.
                torch_assert_close(metrics[K3_KEY], kept_k3, rtol=1e-6, atol=0)

    def test_kept_k3_sum_clamps_the_log_ratio_and_stays_finite(self):
        # Finite log-probs can still differ by a lot (trainer -2, sampler -2 - delta). The reported sum clamps the log
        # ratio to +-20 as trainable_logprob_lowvar_all does; unclamped, exp(89) already overflows fp32.
        for delta in (0.5, 20.0, 25.0, -25.0, 40.0, 88.0, 89.0, 100.0, -100.0, 1000.0):
            with self.subTest(delta=delta):
                metrics = self.run_loss(
                    torch.ones(1, 1), torch.tensor([[delta]]), dict(batch_num_tokens=1, dp_size=1, ratio_stats=True)
                )[2]
                self.assertTrue(math.isfinite(metrics[K3_KEY]))
                # fp32 expm1 at 4.85e8 has ulp 32 (~7e-8 relative); 1e-6 is ~16 ulp. Unclamped, 25 is 150x off.
                torch_assert_close(metrics[K3_KEY], k3(delta), rtol=1e-6, atol=0)

    def test_non_finite_trainer_logprob_is_sanitised_to_zero_and_still_counted_in_the_k3_population(self):
        # docs/rl.md: a trainer log-prob that is not finite is replaced by 0 before the loss, its token still counts in
        # ratio_trainable_token_count and in the kept population, and its log ratio is then minus the sampler
        # log-prob. (dss-platform's trainable_logprob_lowvar_all leaves such a token out of its count and sum.)
        sampler = torch.tensor([[-3.0, -2.0, -2.0, -2.0]])
        for value in (float("-inf"), float("nan"), float("inf")):
            with self.subTest(trainer_logprob=value):
                trainer = torch.tensor([[value, -2.0, -2.0, -2.0]], requires_grad=True)
                context = dict(
                    old_log_probs_shifted=sampler,
                    advantages=torch.ones(1, 4),
                    loss_mask=torch.ones(1, 4, dtype=torch.bool),
                )
                config = dict(
                    use_cispo_loss=True, is_weight_clip_max=5.0, ratio_stats=True, batch_num_tokens=4, dp_size=1
                )
                loss, metrics = grpo_loss({"logprobs": trainer}, context, {}, config, "cpu")
                loss.backward()
                self.assertTrue(torch.isfinite(loss) and torch.isfinite(trainer.grad).all())
                self.assertEqual(metrics["ratio_trainable_token_count"], 4.0)
                self.assertEqual(metrics["ratio_mask_dropped_token_count"], 0.0)
                self.assertTrue(math.isfinite(metrics[K3_KEY]))
                # The other three tokens have a log ratio of 0 and add nothing: the sum is k3(0 - (-3)) = e^3 - 4 = 16.09.
                # fp32 expm1 against a float64 reference: ~1e-7 relative.
                torch_assert_close(metrics[K3_KEY], k3(0.0 - sampler[0, 0].item()), rtol=1e-6, atol=0)

    def test_kept_k3_sum_with_an_outlier_among_ordinary_tokens(self):
        ordinary = [0.1, -0.1, 0.05, -0.05, 0.0, 0.2, -0.2]
        ordinary_k3 = sum(k3(value) for value in ordinary)
        gates = dict(ratio_mask_bounds_pos=[0.5, 2.0])  # keeps the ordinary tokens, drops any outlier (A > 0 for all)
        for outlier, kept_sum in (
            (-100.0, k3(-100.0) + ordinary_k3),  # 19 + 0.053: the ordinary tokens are still visible next to it
            (100.0, k3(100.0) + ordinary_k3),
        ):
            delta = torch.tensor([[outlier, *ordinary]])
            base = dict(batch_num_tokens=8, dp_size=1, ratio_stats=True)
            with self.subTest(outlier=outlier, gates="off"):
                metrics = self.run_loss(torch.ones(1, 8), delta, base)[2]
                self.assertTrue(math.isfinite(metrics[K3_KEY]))
                # fp32 sums: ulp 2e-6 at 19, 32 at 4.85e8; rtol 1e-6 still resolves the ordinary 0.053 only for 19.
                torch_assert_close(metrics[K3_KEY], kept_sum, rtol=1e-6, atol=0)
            with self.subTest(outlier=outlier, gates="drop the outlier"):
                metrics = self.run_loss(torch.ones(1, 8), delta, {**base, **gates})[2]
                self.assertEqual(metrics["ratio_mask_dropped_token_count"], 1.0)
                # Only the seven ordinary tokens remain; their fp32 sum differs from float64 by ~1e-8.
                torch_assert_close(metrics[K3_KEY], ordinary_k3, rtol=0, atol=1e-6)

    def test_metrics_appear_only_when_ratio_controls_or_stats_are_active(self):
        advantages = torch.tensor([[1.0, -1.0]])
        delta = torch.zeros_like(advantages)
        inactive = ({}, {"ratio_stats": False})
        active = (
            {"ratio_stats": True},
            {"ratio_mask_bounds_pos": [0.9, 1.1]},
            {"prob_diff_mask_max_neg": 0.1},
            {"seq_mask_bounds_pos": [-0.1, 0.1]},
            {"ratio_m2_threshold": 0.5},
            {"log_ratio_sq_coef": 0.1},
        )
        masks = (torch.ones_like(advantages, dtype=torch.bool), torch.zeros_like(advantages, dtype=torch.bool))
        for config, mask in itertools.product(inactive, masks):
            with self.subTest(config=config, policy_tokens=int(mask.sum())):
                metrics = self.run_loss(advantages, delta, config, loss_mask=mask)[2]
                self.assertFalse(any(key.startswith("ratio_") for key in metrics))
        for config, mask in itertools.product(active, masks):
            with self.subTest(config=config, policy_tokens=int(mask.sum())):
                metrics = self.run_loss(advantages, delta, config, loss_mask=mask)[2]
                self.assertTrue(set(NEW_KEYS) <= set(metrics))
                if not mask.any():
                    self.assertEqual([metrics[key] for key in NEW_KEYS], [0.0] * 5)

    def test_statistics_do_not_hide_denominator_and_mode_errors(self):
        advantages = torch.tensor([[1.0, -1.0]])
        delta = torch.zeros_like(advantages)
        failures = (
            ("batch_num_tokens=0", dict(loss_agg_mode="token-mean", batch_num_tokens=0), {}),
            ("global_batch_size=0", dict(loss_agg_mode="seq-mean-token-sum", global_batch_size=0), {}),
            (
                "global_batch_size=0",
                dict(loss_agg_mode="prompt-mean", global_batch_size=0),
                dict(prompt_group_ids=torch.tensor([0])),
            ),
            ("prompt_group_ids", dict(loss_agg_mode="prompt-mean"), {}),
            ("loss_agg_mode", dict(loss_agg_mode="mean-of-means"), {}),
        )
        for (message, config, context), controls in itertools.product(
            failures, ({}, {"ratio_stats": True}, RATIO_GATES)
        ):
            with self.subTest(message=message, controls=sorted(controls)), self.assertRaisesRegex(ValueError, message):
                self.run_loss(advantages, delta, dict(dp_size=1, **config, **controls), **context)

    def test_empty_step_with_explicit_zero_denominators_reports_zero_mass(self):
        advantages = torch.tensor([[1.0, -1.0]])
        empty = torch.zeros_like(advantages, dtype=torch.bool)
        for mode, denominators in (
            ("token-mean", dict(batch_num_tokens=0)),
            ("seq-mean-token-sum", dict(global_batch_size=0)),
        ):
            with self.subTest(mode=mode):
                config = dict(loss_agg_mode=mode, dp_size=1, ratio_stats=True, **denominators)
                loss, grad, metrics = self.run_loss(advantages, torch.zeros_like(advantages), config, loss_mask=empty)
                self.assertEqual(loss.item(), 0.0)
                self.assertEqual(grad.abs().sum().item(), 0.0)
                self.assertEqual([metrics[key] for key in NEW_KEYS], [0.0] * 5)
