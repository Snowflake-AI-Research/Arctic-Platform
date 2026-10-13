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

"""``ratio_mask_rebalance``: restoring each sign's masked advantage mass, and the one-model-call-per-worker guard.

Every test runs the real ``grpo_loss`` (or the real loss object's callbacks) on the five-row request of
``test_ratio_mass.py``, whose advantages, weights and denominators are binary fractions, so its independent reference
(no ``agg_loss``) gives the pre and kept masses exactly. The scale a sign gets is ``pre / kept`` of those references.
"""

from __future__ import annotations

import itertools
import re

import torch
from test_ratio_mass import ADVANTAGES
from test_ratio_mass import ALL_ROWS
from test_ratio_mass import DELTA
from test_ratio_mass import K3_KEY
from test_ratio_mass import KEPT
from test_ratio_mass import LAYOUTS
from test_ratio_mass import LEVELS
from test_ratio_mass import LOSS_MASK
from test_ratio_mass import MASS_KEYS
from test_ratio_mass import MODES
from test_ratio_mass import RATIO_GATES
from test_ratio_mass import ROW_CONTEXT_KEYS
from test_ratio_mass import ROW_TOKENS
from test_ratio_mass import reference_mass

from arctic_platform.common.utils.batch import combine_metric_microbatches
from arctic_platform.common.utils.batch import combine_metric_shards
from arctic_platform.rl.processors import resolve_loss
from arctic_platform.rl.processors.functional import RatioMasks
from arctic_platform.rl.processors.grpo import grpo_loss
from arctic_platform.rl.processors.pipeline import run_pipeline
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import torch_assert_close

FLAG = "ratio_mask_rebalance"
REBALANCE_KEYS = tuple(
    f"ratio_rebalance_{sign}_{stage}_mass_sum" for sign in ("pos", "neg") for stage in ("post", "unrestored")
)
# Dropping every positive-side (or negative-side) token: a ratio band no token reaches, because every ratio is <= e.
NO_POSITIVE_SURVIVOR = {"ratio_mask_bounds_pos": [5.0, 6.0]}
NO_NEGATIVE_SURVIVOR = {"ratio_mask_bounds_neg": [5.0, 6.0]}
# Bounds so wide that no token of the request is dropped.
DROP_NOTHING = {"ratio_mask_bounds_pos": [1e-6, 1e6], "ratio_mask_bounds_neg": [1e-6, 1e6]}
# Other gates, each with the request's own delta and the exact mask of tokens that must survive.
# Probability gap: a positive log ratio (old log-prob below the trainer's) gives a positive gap exp(lp) - exp(old), which
# only a positive-side token (A >= 0) can exceed prob_diff_mask_max_pos; a negative log ratio does the same to negative-side
# tokens. |log ratio| = 1 gives gaps of 0.086 and 0.23, both above the 0.05 limit.
PROB_GAP = {"prob_diff_mask_max_pos": 0.05, "prob_diff_mask_max_neg": 0.05}
KEPT_PROB_GAP = LOSS_MASK & ~((DELTA > 0) & (ADVANTAGES >= 0)) & ~((DELTA < 0) & (ADVANTAGES < 0))
# Sequence gate: rows 0 and 3 have a mean log ratio (0.3, 0.25) above the 0.1 bound and lose every token; row 2 (0.05)
# stays inside it. Their mean k3 (0.050, 0.074) is above 0.03 and row 2's (0.005) below, so mean_k3 drops the same rows.
SEQUENCE_DELTA = torch.tensor(
    [[0.3, 0.3, 0.3, 0.3], [0, 0, 0, 0], [0.2, 0, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0, 0]], dtype=torch.float32
)
SEQUENCE_GATES = {"seq_mask_bounds_pos": [-0.1, 0.1], "seq_mask_bounds_neg": [-0.1, 0.1]}
SEQUENCE_K3_GATES = {
    **SEQUENCE_GATES,
    "seq_mask_stat": "mean_k3",
    "seq_mask_bounds_pos": [-1.0, 0.03],
    "seq_mask_bounds_neg": [-1.0, 0.03],
}
KEPT_SEQUENCE = LOSS_MASK & torch.tensor([[False], [True], [True], [False], [True]])
# M2PO over the 16 policy tokens: squared log ratios 4 (row 0, token 1), 1 (row 0, token 2), 0.25 and thirteen zeros.
# With the threshold 0.05 the running average of what is left first falls under it after the top two are removed.
M2_DELTA = torch.zeros(5, 4)
M2_DELTA[0, 1], M2_DELTA[0, 2], M2_DELTA[2, 0] = 2.0, -1.0, 0.5
M2_THRESHOLD = {"ratio_m2_threshold": 0.05}
KEPT_M2 = LOSS_MASK.clone()
KEPT_M2[0, 1] = KEPT_M2[0, 2] = False
# Teacher term: the teacher is 1 nat above the policy at every token, so with tau 2 every advantage gains 2 (the clip is
# not reached), which turns the -1 advantages positive.
TEACHER = {"teacher_tau": 2.0, "teacher_clip": 2.0}
# DSS hands the model-call-count callback one count per worker shard, all equal to the request-wide number of calls
# (dss-platform c37cd1d, sp/data_plane.py: ``shard[MODEL_CALL_COUNT_KEY] = request_model_call_count``).
WORLD_SIZE = 4


def masses_for(
    mode: str, level: str, kept: torch.Tensor, token_advantages: torch.Tensor = ADVANTAGES
) -> dict[str, float]:
    """``reference_mass`` for other token-level advantages (e.g. with the teacher term added), same exact references."""
    if token_advantages is ADVANTAGES:
        return reference_mass(mode, level, kept)
    normalise = MODES[mode][2]
    effective = token_advantages if level == "token" else token_advantages.mean(dim=1, keepdim=True)
    effective = effective.expand_as(token_advantages).double()
    masses = {}
    for stage, sign in itertools.product(("pre", "kept"), ("pos", "neg")):
        magnitude = effective.clamp(min=0) if sign == "pos" else (-effective).clamp(min=0)
        masses[f"ratio_mask_{sign}_{stage}_mass_sum"] = normalise(
            magnitude * (kept if stage == "kept" else LOSS_MASK)
        ).item()
    return masses


def sign_scales(mode, level, kept, token_advantages=ADVANTAGES) -> dict[str, float]:
    """The independent expectation: pre / kept per sign from the exact reference masses; 1 for a sign nothing survives."""
    masses = masses_for(mode, level, kept, token_advantages)
    return {
        sign: (
            masses[f"ratio_mask_{sign}_pre_mass_sum"] / masses[f"ratio_mask_{sign}_kept_mass_sum"]
            if masses[f"ratio_mask_{sign}_kept_mass_sum"] > 0
            else 1.0
        )
        for sign in ("pos", "neg")
    }


def restored_advantages(
    level: str, scales: dict[str, float], token_advantages: torch.Tensor = ADVANTAGES
) -> torch.Tensor:
    """The advantages the loss uses with each sign scaled by hand: what a flag-off run needs to equal a flag-on run."""
    pos, neg = (torch.tensor(scales[sign], dtype=torch.float64).to(token_advantages.dtype) for sign in ("pos", "neg"))
    if level == "token":
        return token_advantages * torch.where(token_advantages >= 0, pos, neg)
    # Under sequence-level IS a row has one effective advantage, so the sign of the row mean picks the scale.
    return token_advantages * torch.where(token_advantages.mean(dim=1, keepdim=True) >= 0, pos, neg)


def counts_message(counts) -> str:
    return re.escape(f"{FLAG} requires exactly one synchronized model call per worker, got {tuple(counts)!r}")


class TestRatioRebalance(TestCasePlus):
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

    def run_rows(self, mode, level, layout, controls, dp_size=1, advantages=None, delta=None, extra_context=None):
        """The whole shared request as one model call (a worker's only call); returns (loss, grad, metrics)."""
        mode_config, mode_context, _ = MODES[mode]
        config = dict(
            loss_agg_mode=mode.split(" ")[0], importance_sampling_level=level, dp_size=dp_size, **mode_config
        )
        config.update(controls)
        context = {key: value[ALL_ROWS] if key in ROW_CONTEXT_KEYS else value for key, value in mode_context.items()}
        advantages = ADVANTAGES if advantages is None else advantages
        delta, mask, extra = (DELTA if delta is None else delta), LOSS_MASK, dict(extra_context or {})
        if layout == "packed":
            advantages, delta, mask = advantages.reshape(-1), delta.reshape(-1), mask.reshape(-1)
            extra = {key: value.reshape(-1) for key, value in extra.items()}
            context["cu_seqlens"] = torch.arange(0, (len(ALL_ROWS) + 1) * ROW_TOKENS, ROW_TOKENS, dtype=torch.int32)
        context.update(extra)
        return self.run_loss(advantages, delta, config, loss_mask=mask, **context)

    def test_flag_false_or_absent_is_statistics_only(self):
        for mode, level, layout, dp_size in itertools.product(MODES, LEVELS, LAYOUTS, (1, 4)):
            with self.subTest(mode=mode, level=level, layout=layout, dp_size=dp_size):
                controls = {**RATIO_GATES, "ratio_stats": True}
                absent = self.run_rows(mode, level, layout, controls, dp_size)
                false = self.run_rows(mode, level, layout, {**controls, FLAG: False}, dp_size)
                torch_assert_close(false[0], absent[0], rtol=0, atol=0)
                torch_assert_close(false[1], absent[1], rtol=0, atol=0)
                self.assertEqual(false[2], absent[2])
                # Only the echo of the flag is added to the statistics, and no restoration key appears.
                self.assertEqual(absent[2][FLAG], 0.0)
                self.assertFalse(set(REBALANCE_KEYS) & set(absent[2]))
        self.assertIsNone(RatioMasks.from_config({FLAG: False}))

    def test_nothing_dropped_is_bit_identical_to_flag_false(self):
        for mode, level, layout, dp_size in itertools.product(MODES, LEVELS, LAYOUTS, (1, 4)):
            for controls in ({"ratio_stats": True}, DROP_NOTHING):
                with self.subTest(mode=mode, level=level, layout=layout, dp_size=dp_size, controls=sorted(controls)):
                    off = self.run_rows(mode, level, layout, {**controls, FLAG: False}, dp_size)
                    on = self.run_rows(mode, level, layout, {**controls, FLAG: True}, dp_size)
                    self.assertEqual(on[2]["ratio_mask_dropped_token_count"], 0.0)
                    torch_assert_close(on[0], off[0], rtol=0, atol=0)
                    torch_assert_close(on[1], off[1], rtol=0, atol=0)
                    # Every metric but the echo and the restoration keys is the flag-false one, bit for bit.
                    shared = {key: value for key, value in on[2].items() if key not in (FLAG, *REBALANCE_KEYS)}
                    self.assertEqual(shared, {key: value for key, value in off[2].items() if key != FLAG})
                    for sign in ("pos", "neg"):
                        self.assertEqual(
                            on[2][f"ratio_rebalance_{sign}_post_mass_sum"], on[2][f"ratio_mask_{sign}_pre_mass_sum"]
                        )
                        self.assertEqual(on[2][f"ratio_rebalance_{sign}_unrestored_mass_sum"], 0.0)

    def test_nothing_dropped_without_gates_equals_no_ratio_controls_at_all(self):
        advantages = torch.tensor([[1.0, 3.0, -2.0, -2.0]])
        delta = torch.tensor([[0.1, -0.1, 2.0, 0.0]])
        base = self.run_loss(advantages, delta, {})
        loss, grad, _ = self.run_loss(advantages, delta, {FLAG: True})
        torch_assert_close(loss, base[0], rtol=0, atol=0)
        torch_assert_close(grad, base[1], rtol=0, atol=0)

    def check_restoration(
        self, mode, level, layout, controls, kept, dp_size=1, delta=None, loss_advantages=None, extra_context=None
    ):
        """Flag on against the exact references and against the same loss on advantages scaled by hand, flag off.

        ``loss_advantages`` are what the flag-on run is given; when ``controls`` add to them (the teacher term), the
        advantages the loss uses are those plus the term, and the by-hand run is given the scaled sum directly.
        """
        loss_advantages = ADVANTAGES if loss_advantages is None else loss_advantages
        teacher = "teacher_tau" in controls
        effective = ADVANTAGES + controls["teacher_tau"] * LOSS_MASK if teacher else loss_advantages
        expected = masses_for(mode, level, kept, effective)
        scales = sign_scales(mode, level, kept, effective)
        on = self.run_rows(
            mode, level, layout, {**controls, FLAG: True}, dp_size, loss_advantages, delta, extra_context
        )
        metrics = on[2]
        self.assertEqual({key: metrics[key] for key in MASS_KEYS}, expected)
        # The request's own masks drop exactly the tokens the test says (the references assume it).
        self.assertEqual(metrics["ratio_mask_dropped_token_count"], float((LOSS_MASK & ~kept).sum()))
        survivors = {sign for sign in ("pos", "neg") if expected[f"ratio_mask_{sign}_kept_mass_sum"] > 0}
        for sign in ("pos", "neg"):
            pre = expected[f"ratio_mask_{sign}_pre_mass_sum"]
            if sign in survivors:
                # Reported post mass is kept mass times the scale, in float64: it differs from pre by rounding only
                # (~1e-16 relative), while a wrong scale moves it by >= 0.1.
                torch_assert_close(metrics[f"ratio_rebalance_{sign}_post_mass_sum"], pre, rtol=0, atol=1e-12)
                self.assertEqual(metrics[f"ratio_rebalance_{sign}_unrestored_mass_sum"], 0.0)
            else:
                # Nothing of this sign survives: scale 1, nothing restored, and its whole pre mass is reported lost.
                self.assertEqual(scales[sign], 1.0)
                self.assertEqual(metrics[f"ratio_rebalance_{sign}_post_mass_sum"], 0.0)
                self.assertEqual(metrics[f"ratio_rebalance_{sign}_unrestored_mass_sum"], pre)
        self.assertTrue(torch.isfinite(on[0]) and torch.isfinite(on[1]).all())

        hand_controls = {key: value for key, value in controls.items() if not key.startswith("teacher_")}
        by_hand = self.run_rows(
            mode,
            level,
            layout,
            hand_controls,
            dp_size,
            restored_advantages(level, scales, effective),
            delta,
        )
        # The mass the restored advantages carry is the pre-mask mass, up to fp32 rounding of the scaled advantages
        # (6e-8 relative per term, sums of at most 20 terms of size <= 4): 1e-6 absolute.
        for sign in survivors:
            torch_assert_close(
                by_hand[2][f"ratio_mask_{sign}_kept_mass_sum"],
                expected[f"ratio_mask_{sign}_pre_mass_sum"],
                rtol=0,
                atol=1e-6,
            )
        if level == "token":
            # Same scalings, same fp32 operations: the objective is bit-equal.
            torch_assert_close(on[0], by_hand[0], rtol=0, atol=0)
            torch_assert_close(on[1], by_hand[1], rtol=0, atol=0)
        else:
            # A sequence's effective advantage is the mean of its scaled tokens by hand and the scaled mean in the
            # loss: they differ by one fp32 rounding (~1e-7 at |value| ~ 1).
            torch_assert_close(on[0], by_hand[0], rtol=0, atol=1e-6)
            torch_assert_close(on[1], by_hand[1], rtol=0, atol=1e-6)
        return survivors, scales

    def test_restoration_returns_each_signs_mass_to_its_premask_value(self):
        for mode, level, layout, dp_size in itertools.product(MODES, LEVELS, LAYOUTS, (1, 4)):
            with self.subTest(mode=mode, level=level, layout=layout, dp_size=dp_size):
                scales = sign_scales(mode, level, KEPT)
                self.assertTrue(all(scale > 1.0 for scale in scales.values()), "both signs must lose mass")
                self.check_restoration(mode, level, layout, RATIO_GATES, KEPT, dp_size)

    def test_a_sign_with_no_survivor_gets_scale_one_and_reports_its_lost_mass(self):
        # Token-level gates drop by the sign of the token's own advantage. Under token-level IS that is the sign the
        # mass is split by, so the sign loses everything; under sequence-level IS a row's effective sign can differ
        # from some of its tokens' signs, so a sign can keep mass from tokens the gate did not target.
        positive_side, negative_side = ADVANTAGES >= 0, ADVANTAGES < 0
        cases = (
            (NO_POSITIVE_SURVIVOR, LOSS_MASK & negative_side, {"pos"}),
            (NO_NEGATIVE_SURVIVOR, LOSS_MASK & positive_side, {"neg"}),
            ({**NO_POSITIVE_SURVIVOR, **NO_NEGATIVE_SURVIVOR}, LOSS_MASK & False, {"pos", "neg"}),
        )
        for (controls, kept, lost_at_token_level), mode, level, layout in itertools.product(
            cases, MODES, LEVELS, LAYOUTS
        ):
            with self.subTest(mode=mode, level=level, layout=layout, controls=sorted(controls)):
                survivors, _ = self.check_restoration(mode, level, layout, controls, kept)
                if level == "token" or not survivors:
                    self.assertEqual({"pos", "neg"} - survivors, lost_at_token_level)
                if not survivors:
                    # Both signs lost, scale 1 each: the objective is the flag-off one.
                    off = self.run_rows(mode, level, layout, {**controls, FLAG: False})
                    on = self.run_rows(mode, level, layout, {**controls, FLAG: True})
                    torch_assert_close(on[0], off[0], rtol=0, atol=0)
                    torch_assert_close(on[1], off[1], rtol=0, atol=0)

    def test_restoration_with_the_other_gates_and_the_teacher_term(self):
        teacher_context = {"teacher_log_probs_shifted": torch.full_like(ADVANTAGES, -1.0)}
        cases = {
            "probability gap": (PROB_GAP, KEPT_PROB_GAP, DELTA, None, None, LEVELS),
            "sequence gate on the mean log ratio": (SEQUENCE_GATES, KEPT_SEQUENCE, SEQUENCE_DELTA, None, None, LEVELS),
            "sequence gate on the mean k3": (SEQUENCE_K3_GATES, KEPT_SEQUENCE, SEQUENCE_DELTA, None, None, LEVELS),
            "ratio_m2_threshold": (M2_THRESHOLD, KEPT_M2, M2_DELTA, None, None, LEVELS),
            # The teacher term only exists at token level, where it makes the scale sign the one of A + term.
            "teacher term with the ratio band": (
                {**TEACHER, **RATIO_GATES},
                KEPT,
                DELTA,
                None,
                teacher_context,
                ("token",),
            ),
        }
        for (
            (name, (controls, kept, delta, advantages, context, levels)),
            mode,
            level,
            layout,
            dp_size,
        ) in itertools.product(cases.items(), MODES, LEVELS, LAYOUTS, (1, 4)):
            if level not in levels:
                continue
            with self.subTest(case=name, mode=mode, level=level, layout=layout, dp_size=dp_size):
                survivors, scales = self.check_restoration(
                    mode, level, layout, controls, kept, dp_size, delta, advantages, context
                )
                self.assertGreater(max(scales.values()), 1.0)  # the gates really cost a sign some mass
                self.assertIn("pos", survivors)  # and leave something of the positive side to scale
                if level == "token":
                    self.assertIn("neg", survivors)

    def test_mask_decisions_counts_and_k3_ignore_the_scaling_for_the_other_gates_too(self):
        teacher_context = {"teacher_log_probs_shifted": torch.full_like(ADVANTAGES, -1.0)}
        cases = (
            (PROB_GAP, DELTA, None),
            (SEQUENCE_GATES, SEQUENCE_DELTA, None),
            (M2_THRESHOLD, M2_DELTA, None),
            ({**TEACHER, **RATIO_GATES}, DELTA, teacher_context),
        )
        for (controls, delta, context), mode, layout in itertools.product(cases, MODES, LAYOUTS):
            with self.subTest(controls=sorted(controls), mode=mode, layout=layout):

                def run(flag):
                    controls_with_flag = {**controls, "ratio_stats": True, FLAG: flag}
                    return self.run_rows(mode, "token", layout, controls_with_flag, 1, None, delta, context)[2]

                off, on = run(False), run(True)
                self.assertGreater(off["ratio_mask_dropped_token_count"], 0.0)
                # Only the echo and the four restoration keys differ: drops, counts, histograms and the k3 sum do not.
                self.assertEqual(
                    {key for key in set(on) | set(off) if on.get(key) != off.get(key)}, {FLAG, *REBALANCE_KEYS}
                )

    def test_the_scale_is_not_capped(self):
        # Positive tokens only; an offset of +1 puts a token outside the [0.9, 1.1] ratio band, so it is dropped.
        survivor = float(torch.tensor(1e-3))  # the float32 value of one thousandth, which is what the loss sees
        one_in_100 = torch.cat([torch.ones(1, 99), torch.zeros(1, 1)], dim=1)
        # name: (advantages, offsets, tokens dropped, pre / kept)
        cases = {
            # One of 100 equal-magnitude tokens survives.
            "one survivor among 100 equal tokens": (torch.ones(1, 100), one_in_100, 99.0, 100.0),
            # The survivor's advantage is a thousandth of the dropped token's.
            "a survivor of a thousandth of the dropped advantage": (
                torch.tensor([[1e-3, 1.0]]),
                torch.tensor([[0.0, 1.0]]),
                1.0,
                (survivor + 1.0) / survivor,
            ),
        }
        for name, (advantages, delta, dropped, scale) in cases.items():
            with self.subTest(name):
                config = dict(batch_num_tokens=advantages.numel(), dp_size=1, ratio_mask_bounds_pos=[0.9, 1.1])
                loss, grad, metrics = self.run_loss(advantages, delta, {**config, FLAG: True})
                self.assertTrue(torch.isfinite(loss) and torch.isfinite(grad).all())
                self.assertEqual(metrics["ratio_mask_dropped_token_count"], dropped)
                pre, kept = metrics["ratio_mask_pos_pre_mass_sum"], metrics["ratio_mask_pos_kept_mass_sum"]
                # The scale is as large as the masses say, 100 and about 1001: nothing caps it. pre / kept is a float64
                # quotient, ~1e-16 relative from the analytic value.
                torch_assert_close(pre / kept, scale, rtol=1e-12, atol=0)
                # The restored mass is the pre-mask mass (float64 sums: ~1e-16 relative, so 1e-12 absolute at |mass| ~ 1).
                torch_assert_close(metrics["ratio_rebalance_pos_post_mass_sum"], pre, rtol=0, atol=1e-12)
                # The loss uses that scale: bit-equal to the flag-off loss on advantages scaled by hand.
                scaled = advantages * torch.tensor(scale, dtype=torch.float64).to(advantages.dtype)
                by_hand = self.run_loss(scaled, delta, config)
                torch_assert_close(loss, by_hand[0], rtol=0, atol=0)
                torch_assert_close(grad, by_hand[1], rtol=0, atol=0)

    def test_the_production_combiners_sum_the_restoration_metrics_and_keep_the_echo(self):
        runs = [
            self.run_rows("token-mean", "token", "padded", {**controls, FLAG: True})[2]
            for controls in (RATIO_GATES, NO_POSITIVE_SURVIVOR, NO_NEGATIVE_SURVIVOR)
        ]
        # Over these runs every restoration metric is non-zero somewhere (post of both signs, unrestored of both).
        for key in REBALANCE_KEYS:
            self.assertTrue(any(metrics[key] > 0 for metrics in runs), key)
        for metrics, combine in itertools.product(runs, (combine_metric_microbatches, combine_metric_shards)):
            combined = combine([metrics, metrics])
            for key in REBALANCE_KEYS:
                self.assertEqual(combined[key], 2 * metrics[key], msg=f"{combine.__name__} {key}")
            # The echo of the flag is not additive: two equal inputs keep it at 1.0.
            self.assertEqual(combined[FLAG], 1.0, msg=combine.__name__)

    def test_mask_decisions_and_the_k3_sum_ignore_the_scaling(self):
        for mode, level, layout in itertools.product(MODES, LEVELS, LAYOUTS):
            with self.subTest(mode=mode, level=level, layout=layout):
                off = self.run_rows(mode, level, layout, {**RATIO_GATES, FLAG: False})[2]
                on = self.run_rows(mode, level, layout, {**RATIO_GATES, FLAG: True})[2]
                self.assertGreater(off["ratio_mask_dropped_token_count"], 0.0)
                # Not one measurement of the drops, the populations, the histograms or the k3 sum moves.
                changed = {key for key in set(on) | set(off) if on.get(key) != off.get(key)}
                self.assertEqual(changed, {FLAG, *REBALANCE_KEYS})
                self.assertEqual(on[K3_KEY], off[K3_KEY])

    def test_an_explicit_false_alone_is_inert_for_both_flags_and_rejected_by_the_mixed_variants(self):
        # docs/rl.md: ratio_stats=false and ratio_mask_rebalance=false with no other ratio control are inert (no echo, no
        # metrics, no use_cispo_loss requirement), whereas the mixed variants reject any ratio-control key by presence.
        values = torch.full((1, 4), -2.0)
        context = dict(
            old_log_probs_shifted=values, advantages=torch.ones(1, 4), loss_mask=torch.ones(1, 4, dtype=torch.bool)
        )
        microbatch = {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "loss_mask": torch.ones((1, 4), dtype=torch.bool),
        }
        for config in ({FLAG: False}, {"ratio_stats": False}, {FLAG: False, "ratio_stats": False}):
            with self.subTest(config=config):
                self.assertIsNone(RatioMasks.from_config(config))
                metrics = grpo_loss({"logprobs": values}, context, {}, config, "cpu")[1]  # no use_cispo_loss
                self.assertFalse(any(key.startswith("ratio") for key in metrics))
                resolve_loss("ap_grpo").packed_reduction_callback([microbatch], config, "ap_grpo")
                for name in ("ap_grpo_mixed_v1", "grpo_mixed_v1"):
                    request = {
                        "input_ids": torch.ones((1, 4), dtype=torch.long),
                        "processing": {
                            "loss_fn": name,
                            "config": {"use_cispo_loss": True, "is_weight_clip_max": 5.0, **config},
                        },
                        "context": {
                            "loss_mask": torch.ones((1, 4), dtype=torch.bool),
                            "nll_mask": torch.zeros((1, 4), dtype=torch.bool),
                            "advantages": torch.ones((1, 4)),
                        },
                    }
                    with self.assertRaisesRegex(ValueError, f"{name} does not support ratio-mask keys"):
                        resolve_loss(name).batching_callback(request)
        # True, in contrast, activates the controls and needs CISPO.
        for config in ({FLAG: True}, {"ratio_stats": True}):
            with self.subTest(config=config):
                self.assertIsNotNone(RatioMasks.from_config(config))
                with self.assertRaisesRegex(ValueError, "need use_cispo_loss=True"):
                    grpo_loss({"logprobs": values}, context, {}, config, "cpu")

    def test_typed_flag_and_cispo_requirement(self):
        for bad in (None, 1, 0, "true"):
            with self.subTest(flag=bad), self.assertRaisesRegex(ValueError, f"{FLAG} must be a bool"):
                RatioMasks.from_config({FLAG: bad})
        request = {
            "input_ids": torch.ones((2, 4), dtype=torch.long),
            "processing": {"loss_fn": "ap_grpo", "config": {FLAG: True}},
            "context": {
                "loss_mask": torch.ones((2, 4), dtype=torch.bool),
                "advantages": torch.ones((2, 4)),
                "old_log_probs_shifted": torch.zeros((2, 4)),
            },
        }
        loss_object = resolve_loss("ap_grpo")
        loss_object.batching_callback(request)  # the key is a known ratio control, so the config passes batching
        # Resolving the packed loss reduction is where DSS and the native worker reject it before any forward.
        microbatch = {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "loss_mask": torch.ones((1, 4), dtype=torch.bool),
        }
        with self.assertRaisesRegex(ValueError, "need use_cispo_loss=True"):
            loss_object.packed_reduction_callback([microbatch], {FLAG: True}, "ap_grpo")
        loss_object.packed_reduction_callback([microbatch], {FLAG: True, "use_cispo_loss": True}, "ap_grpo")
        # The loss itself rejects it too.
        values = torch.full((1, 4), -2.0)
        context = dict(
            old_log_probs_shifted=values, advantages=torch.ones(1, 4), loss_mask=torch.ones(1, 4, dtype=torch.bool)
        )
        with self.assertRaisesRegex(ValueError, "need use_cispo_loss=True"):
            grpo_loss({"logprobs": values}, context, {}, {FLAG: True}, "cpu")

    def test_more_than_one_model_call_is_rejected_naming_the_flag_and_the_counts(self):
        config = {"use_cispo_loss": True, "is_weight_clip_max": 5.0, FLAG: True}
        for name in ("ap_grpo", "ap_grpo_echo_v1", "grpo", "grpo_echo_v1"):
            loss_object = resolve_loss(name)
            # What DSS hands the callback: one count per worker shard, the same request-wide number on every one.
            for counts in ([2] * WORLD_SIZE, [3] * WORLD_SIZE, [1, 2, 1, 1], [None] * WORLD_SIZE, []):
                with (
                    self.subTest(loss=name, counts=counts),
                    self.assertRaisesRegex(ValueError, counts_message(counts)),
                ):
                    loss_object.model_call_count_callback(counts, config)
            with self.subTest(loss=name, case="one call per worker"):
                loss_object.model_call_count_callback([1] * WORLD_SIZE, config)
            for off in ({**config, FLAG: False}, {key: value for key, value in config.items() if key != FLAG}):
                with self.subTest(loss=name, case="flag off or absent"):
                    loss_object.model_call_count_callback([2] * WORLD_SIZE, off)

    def test_the_guard_keeps_its_explicit_conditions(self):
        loss_object = resolve_loss("ap_grpo")
        # ratio_m2_threshold is guarded whenever it is not None, whatever its value: the guard itself also rejects 0, 0.0
        # and False, it does not rely on the later validation that rejects those values.
        for value in (0, 0.0, False, 0.5):
            with (
                self.subTest(ratio_m2_threshold=value),
                self.assertRaisesRegex(ValueError, counts_message([2, 2]).replace(FLAG, "ratio_m2_threshold")),
            ):
                loss_object.model_call_count_callback([2, 2], {"ratio_m2_threshold": value})
            loss_object.model_call_count_callback([1, 1], {"ratio_m2_threshold": value})
        # An explicit null does not trigger the guard.
        loss_object.model_call_count_callback([2, 2], {"ratio_m2_threshold": None})
        # The flag guards only when it is exactly True: other values are left to the typed validation, which rejects them
        # with its own message.
        for value in (False, 0, 1, "false", "true", None):
            with self.subTest(flag=value):
                loss_object.model_call_count_callback([2, 2], {FLAG: value})
                if value is not False and value is not None:
                    with self.assertRaisesRegex(ValueError, f"{FLAG} must be a bool"):
                        RatioMasks.from_config({FLAG: value})

    def test_every_single_call_control_that_is_set_is_named(self):
        loss_object = resolve_loss("ap_grpo")
        with self.assertRaisesRegex(ValueError, rf"^ratio_m2_threshold, {FLAG} requires exactly one"):
            loss_object.model_call_count_callback([2, 2], {"ratio_m2_threshold": 0.5, FLAG: True})
        with self.assertRaisesRegex(ValueError, "^ratio_m2_threshold requires exactly one"):
            loss_object.model_call_count_callback([2, 2], {"ratio_m2_threshold": 0.5, FLAG: False})

    def test_native_pipeline_rejects_a_split_request_before_any_forward(self):
        class Engine:
            global_rank = 0

            def __call__(self, *args, **kwargs):
                raise AssertionError("a split ratio_mask_rebalance request must fail before forward")

        batch = {
            "input_ids": torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
            "advantages": torch.ones(2, 4),
            "loss_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
            "old_log_probs_shifted": torch.zeros(2, 4),
        }
        processing = {
            "loss_fn": "ap_grpo",
            "post": [],
            "config": {"use_cispo_loss": True, "is_weight_clip_max": 5.0, FLAG: True, **RATIO_GATES},
        }
        # max_tokens_per_mb=3 splits the two rows into two microbatches, i.e. two model calls on this worker.
        with self.assertRaisesRegex(ValueError, counts_message([2])):
            run_pipeline(
                Engine(),
                (),
                batch,
                {"pad_token_id": 0},
                processing,
                "cpu",
                backward=True,
                pack=True,
                max_tokens_per_mb=3,
            )
