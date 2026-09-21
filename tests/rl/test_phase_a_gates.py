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

"""Phase A gates: registry hygiene, packed apply, metric pairing, zone names."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

import arctic_platform.rl.processors  # noqa: F401
import arctic_platform.sft.processor  # noqa: F401
from arctic_platform.common.registry import DECLARED_SUMMED_METRICS
from arctic_platform.common.registry import LOSS_FNS
from arctic_platform.common.registry import PACKED_LOSS_REDUCTION_ATTR
from arctic_platform.common.registry import POST_PROCESSORS
from arctic_platform.common.registry import PUBLIC_LOSS_FNS
from arctic_platform.common.registry import PUBLIC_POST_PROCESSORS
from arctic_platform.common.registry import SUMMED_METRICS_ATTR
from arctic_platform.common.registry import _is_public_registry_name
from arctic_platform.common.registry import is_declared_summed_metric
from arctic_platform.common.registry import register_loss_fn
from arctic_platform.common.registry import register_post_processor
from arctic_platform.common.registry import resolve_fn
from arctic_platform.common.utils.batch import combine_metric_microbatches
from arctic_platform.common.utils.batch import combine_metric_shards
from arctic_platform.rl.processors.causal_cross_entropy import causal_cross_entropy_loss
from arctic_platform.rl.processors.compute_logprobs import compute_logprobs_post
from arctic_platform.rl.processors.cortex_grpo import cortex_grpo_loss
from arctic_platform.rl.processors.grpo import _ECHO_CONFIG_DEFAULTS
from arctic_platform.rl.processors.grpo import _ECHO_CONFIG_KEYS
from arctic_platform.rl.processors.grpo import _ECHO_REQUIRED_CONFIG_KEYS
from arctic_platform.rl.processors.grpo import _GRPO_CONFIG_DEFAULTS
from arctic_platform.rl.processors.grpo import _GRPO_CONFIG_KEYS
from arctic_platform.rl.processors.grpo import ECHO_SUMMED_METRICS
from arctic_platform.rl.processors.grpo import _grpo_config_values
from arctic_platform.rl.processors.grpo import _internal_grpo_loss_fn
from arctic_platform.rl.processors.grpo import grpo_echo_v1_loss
from arctic_platform.rl.processors.grpo import grpo_loss
from arctic_platform.rl.processors.packed_reduction import apply_packed_loss_reduction
from arctic_platform.rl.processors.packed_reduction import combine_packed_losses
from arctic_platform.rl.processors.packed_reduction import combine_packed_metrics
from arctic_platform.rl.processors.packed_reduction import metric_is_summed
from arctic_platform.rl.processors.packed_reduction import resolve_packed_loss_reduction
from arctic_platform.rl.processors.pipeline import metric_is_summed as pipeline_metric_is_summed
from arctic_platform.rl.processors.pipeline import run_pipeline
from arctic_platform.testing_utils import TestCasePlus


class TestA1RegistryHygiene(TestCasePlus):
    def test_public_catalog_is_registered(self):
        public_losses = {name for name in LOSS_FNS if _is_public_registry_name(name) and "." not in name}
        public_posts = {name for name in POST_PROCESSORS if _is_public_registry_name(name) and "." not in name}
        self.assertEqual(public_losses, PUBLIC_LOSS_FNS)
        self.assertEqual(public_posts, PUBLIC_POST_PROCESSORS)

    def test_same_fn_reregister_is_idempotent(self):
        fn = LOSS_FNS["ap_grpo"]
        register_loss_fn("ap_grpo")(fn)
        self.assertIs(LOSS_FNS["ap_grpo"], fn)

    def test_same_fn_conflicting_packed_reduction_raises(self):
        fn = LOSS_FNS["ap_grpo"]

        def other_reduction(microbatches, config, loss_fn_name):
            raise AssertionError("must not replace the GRPO resolver")

        with self.assertRaises(ValueError):
            register_loss_fn("ap_grpo", packed_loss_reduction=other_reduction)(fn)
        self.assertIs(
            getattr(fn, PACKED_LOSS_REDUCTION_ATTR), getattr(LOSS_FNS["ap_grpo"], PACKED_LOSS_REDUCTION_ATTR)
        )

    def test_public_name_overwrite_raises(self):
        original = LOSS_FNS["ap_grpo"]

        def other(model_outputs, batch, meta, config, device):
            return model_outputs["logprobs"].sum(), {}

        with self.assertRaises(ValueError):
            register_loss_fn("ap_grpo")(other)
        self.assertIs(LOSS_FNS["ap_grpo"], original)

    def test_underscore_name_may_overwrite(self):
        @register_post_processor("_phase_a_tmp")
        def first(model_outputs, batch, meta, device):
            return {}

        @register_post_processor("_phase_a_tmp")
        def second(model_outputs, batch, meta, device):
            return {"x": 1}

        try:
            self.assertIs(POST_PROCESSORS["_phase_a_tmp"], second)
        finally:
            POST_PROCESSORS.pop("_phase_a_tmp", None)

    def test_removed_prefixed_cortex_names_suggest_zone_names(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_fn(LOSS_FNS, "cortex_grpo")
        self.assertIn("grpo", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            resolve_fn(POST_PROCESSORS, "cortex_compute_logprobs")
        self.assertIn("compute_logprobs", str(ctx.exception))


class TestA3PackedApply(TestCasePlus):
    def _mb(self, tokens: int, mask_ones: int) -> dict:
        return {
            "input_ids": torch.arange(tokens).view(1, tokens),
            "loss_mask": torch.tensor([[1] * mask_ones + [0] * (tokens - mask_ones)], dtype=torch.bool),
        }

    def test_token_mean_local_and_additive(self):
        mbs = [self._mb(4, 4), self._mb(4, 2)]
        local = resolve_packed_loss_reduction(
            {"loss_fn": "ap_grpo", "config": {"loss_agg_mode": "token-mean"}},
            mbs,
        )
        self.assertFalse(local.loss_is_additive)
        self.assertEqual(len(local.loss_scales), 2)
        additive = resolve_packed_loss_reduction(
            {"loss_fn": "ap_grpo", "config": {"loss_agg_mode": "token-mean", "batch_num_tokens": 6}},
            mbs,
        )
        self.assertTrue(additive.loss_is_additive)
        self.assertEqual(additive.loss_scales, (1.0, 1.0))

    def test_token_mean_reads_batch_num_tokens_from_microbatch(self):
        mb = self._mb(4, 4)
        mb["batch_num_tokens"] = 4
        reduction = resolve_packed_loss_reduction(
            {"loss_fn": "ap_grpo", "config": {"loss_agg_mode": "token-mean"}},
            [mb],
        )
        self.assertTrue(reduction.loss_is_additive)

    def test_sequence_mean_uses_active_counts(self):
        reduction = resolve_packed_loss_reduction(
            {"loss_fn": "ap_grpo", "config": {"loss_agg_mode": "seq-mean-token-mean"}},
            [
                {"input_ids": torch.ones(2, 2, dtype=torch.long), "loss_mask": torch.tensor([[1, 0], [1, 1]])},
                {"input_ids": torch.ones(2, 2, dtype=torch.long), "loss_mask": torch.tensor([[0, 0], [1, 0]])},
            ],
        )
        self.assertFalse(reduction.loss_is_additive)
        self.assertEqual(reduction.reporting_weights, (2.0, 1.0))

    def test_prompt_mean_without_weights_rejects_split(self):
        with self.assertRaises(ValueError):
            resolve_packed_loss_reduction(
                {"loss_fn": "ap_grpo", "config": {"loss_agg_mode": "prompt-mean"}},
                [self._mb(2, 2), self._mb(2, 2)],
            )

    def test_prompt_mean_with_sequence_weights_is_additive(self):
        mb0 = self._mb(2, 2)
        mb0["sequence_loss_weights"] = torch.tensor([0.5])
        mb1 = self._mb(2, 2)
        mb1["sequence_loss_weights"] = torch.tensor([1.5])
        reduction = resolve_packed_loss_reduction(
            {"loss_fn": "ap_grpo", "config": {"loss_agg_mode": "prompt-mean"}},
            [mb0, mb1],
        )
        self.assertTrue(reduction.loss_is_additive)
        self.assertEqual(reduction.reporting_weights, (0.5, 1.5))

    def test_trio_mismatch_across_microbatches_raises(self):
        mb0 = self._mb(2, 2)
        mb0["batch_num_tokens"] = 4.0
        mb1 = self._mb(2, 2)
        mb1["batch_num_tokens"] = 8.0
        with self.assertRaises(ValueError):
            resolve_packed_loss_reduction(
                {"loss_fn": "ap_grpo", "config": {"loss_agg_mode": "token-mean"}},
                [mb0, mb1],
            )

    def test_apply_scales_and_backward(self):
        seen = []

        class Engine:
            def backward(self, loss, scale_wrt_gas=True):
                seen.append((float(loss.detach()), scale_wrt_gas))

        loss = torch.tensor(4.0, requires_grad=True)
        out = apply_packed_loss_reduction(Engine(), loss, 0.25, backward=True)
        self.assertIsNone(out)
        self.assertEqual(seen, [(1.0, False)])

    def test_combine_additive_vs_mean(self):
        from arctic_platform.rl.processors.packed_reduction import PackedLossReduction

        additive = PackedLossReduction(loss_scales=(1.0, 1.0), reporting_weights=(3.0, 1.0), loss_is_additive=True)
        local = PackedLossReduction(loss_scales=(0.75, 0.25), reporting_weights=(3.0, 1.0), loss_is_additive=False)
        self.assertAlmostEqual(combine_packed_losses([0.2, 0.4], additive), 0.6)
        self.assertAlmostEqual(combine_packed_losses([0.2, 0.4], local), 0.25)

    def test_packing_preflight_before_forward(self):
        class Engine:
            def __call__(self, *args, **kwargs):
                raise AssertionError("invalid later microbatch must fail before forward")

        with self.assertRaises(ValueError):
            run_pipeline(
                Engine(),
                (),
                {
                    "input_ids": torch.tensor([[1, 2], [3, 4]]),
                    "attention_mask": torch.ones(2, 2, dtype=torch.long),
                },
                {"loss_mask": torch.tensor([[1.0, 0.0], [float("nan"), 0.0]])},
                {"loss_fn": "causal_cross_entropy", "post": [], "config": {}},
                "cpu",
                max_tokens_per_mb=2,
            )

    def test_missing_packed_metadata_rejects_multiple_microbatches(self):
        @register_loss_fn("_phase_a_no_packed")
        def _no_packed(model_outputs, batch, meta, config, device):
            return model_outputs["logprobs"].sum(), {}

        class Engine:
            def __call__(self, *args, **kwargs):
                raise AssertionError("missing reduction metadata must fail before forward")

        try:
            with self.assertRaises(ValueError):
                run_pipeline(
                    Engine(),
                    (),
                    {
                        "input_ids": torch.tensor([[1, 2], [3, 4]]),
                        "attention_mask": torch.ones(2, 2, dtype=torch.long),
                    },
                    {},
                    {"loss_fn": "_phase_a_no_packed", "config": {}},
                    "cpu",
                    max_tokens_per_mb=2,
                )
        finally:
            LOSS_FNS.pop("_phase_a_no_packed", None)


class TestA4Metrics(TestCasePlus):
    def test_metric_is_summed_pairing(self):
        self.assertTrue(metric_is_summed("loss_term_rl"))
        self.assertTrue(metric_is_summed("echo_environment_loss_sum"))
        self.assertTrue(metric_is_summed("echo_real_sequence_count"))
        self.assertFalse(metric_is_summed("loss"))
        self.assertFalse(metric_is_summed("entropy"))
        self.assertIs(pipeline_metric_is_summed, metric_is_summed)

    def test_echo_metrics_are_declared_at_registration(self):
        # The declaration is the contract; the naming convention must still
        # classify every declared name the same way, so removing a declaration
        # cannot silently turn a summed metric into a mean.
        self.assertEqual(
            getattr(LOSS_FNS["ap_grpo_echo_v1"], SUMMED_METRICS_ATTR),
            ECHO_SUMMED_METRICS,
        )
        self.assertEqual(
            getattr(LOSS_FNS["grpo_echo_v1"], SUMMED_METRICS_ATTR),
            ECHO_SUMMED_METRICS,
        )
        for key in ECHO_SUMMED_METRICS:
            self.assertTrue(is_declared_summed_metric(key), msg=key)
            self.assertTrue(metric_is_summed(key), msg=key)
            self.assertTrue(
                key.startswith("loss_term_") or key.endswith(("_sum", "_count")),
                msg=f"{key} is declared but no longer matches the fallback naming rule",
            )

    def test_declared_metric_is_summed_without_a_naming_hint(self):
        name = "_phase_a_declared_total"
        self.assertFalse(metric_is_summed(name))

        @register_loss_fn("_phase_a_declared", summed_metrics={name})
        def _declared_loss(model_outputs, batch, meta, config, device):
            return model_outputs["logprobs"].sum(), {name: 1.0}

        try:
            self.assertTrue(metric_is_summed(name))
            self.assertEqual(combine_packed_metrics([{name: 1.0}, {name: 3.0}], (1.0, 3.0))[name], 4.0)
            self.assertEqual(combine_metric_microbatches([{name: 1.0}, {name: 3.0}])[name], 4.0)
            self.assertEqual(combine_metric_shards([{name: 1.0}, {name: 3.0}])[name], 4.0)
        finally:
            DECLARED_SUMMED_METRICS.discard(name)
            LOSS_FNS.pop("_phase_a_declared", None)
        self.assertFalse(metric_is_summed(name))

    def test_naming_rule_still_applies_to_undeclared_keys(self):
        for key in ("loss_term_unregistered", "some_new_sum", "some_new_count"):
            self.assertFalse(is_declared_summed_metric(key), msg=key)
            self.assertTrue(metric_is_summed(key), msg=key)

    def test_combine_does_not_average_summed_keys(self):
        metrics = combine_packed_metrics(
            [
                {"loss_term_rl": 1.0, "entropy": 2.0},
                {"loss_term_rl": 3.0, "entropy": 4.0},
            ],
            (1.0, 3.0),
        )
        self.assertEqual(metrics["loss_term_rl"], 4.0)
        self.assertAlmostEqual(metrics["entropy"], (2.0 * 1.0 + 4.0 * 3.0) / 4.0)

    def test_combine_rejects_name_and_name_sum(self):
        with self.assertRaises(ValueError):
            combine_packed_metrics([{"loss": 1.0, "loss.sum": 2.0}], (1.0,))

    def test_gas_and_dp_sum_additive_metrics(self):
        gas = combine_metric_microbatches(
            [
                {"loss_term_rl": 1.0, "entropy": 2.0, "echo_real_sequence_count": 1.0},
                {"loss_term_rl": 3.0, "entropy": 4.0, "echo_real_sequence_count": 2.0},
            ]
        )
        self.assertEqual(gas["loss_term_rl"], 4.0)
        self.assertEqual(gas["echo_real_sequence_count"], 3.0)
        self.assertAlmostEqual(gas["entropy"], 3.0)

        dp = combine_metric_shards(
            [
                {"loss_term_rl": 1.0, "echo_environment_loss_sum": 0.5, "entropy": 1.0},
                {"loss_term_rl": 3.0, "echo_environment_loss_sum": 1.5, "entropy": 3.0},
            ]
        )
        self.assertEqual(dp["loss_term_rl"], 4.0)
        self.assertEqual(dp["echo_environment_loss_sum"], 2.0)
        self.assertAlmostEqual(dp["entropy"], 2.0)


class TestGrpoConfigContract(TestCasePlus):
    """The declared tables are the single source of the GRPO key schema and defaults."""

    # Arguments of _internal_grpo_loss_fn that come from tensors, not from config.
    NON_CONFIG_ARGS = frozenset(
        {
            "logprobs",
            "entropy",
            "input_data",
            "rollout_is_weights",
            "prompt_group_ids",
            "prompt_token_counts",
            "sequence_loss_weights",
        }
    )

    def test_tables_cover_every_config_argument(self):
        # A new knob on the loss must be declared in a table, or _grpo_loss
        # would never forward it and the *_echo_v1 schema would reject it.
        params = set(inspect.signature(_internal_grpo_loss_fn).parameters)
        declared = set(_GRPO_CONFIG_DEFAULTS) | set(_ECHO_CONFIG_DEFAULTS)
        self.assertEqual(params - declared, self.NON_CONFIG_ARGS)
        self.assertLessEqual(declared, params)

    def test_declared_defaults_match_the_loss_signature(self):
        params = inspect.signature(_internal_grpo_loss_fn).parameters
        declared = {**_GRPO_CONFIG_DEFAULTS, **_ECHO_CONFIG_DEFAULTS}
        for key, default in declared.items():
            signature_default = params[key].default
            # dp_size: None means "not supplied" and _resolve_dp_size maps it to
            # the signature default of 1. The eps/clip knobs have no signature
            # default at all, so the table is their only source.
            if key == "dp_size" or signature_default is inspect.Parameter.empty:
                continue
            self.assertEqual(default, signature_default, msg=key)

    def test_key_schema_is_derived_from_the_tables(self):
        self.assertEqual(_GRPO_CONFIG_KEYS, frozenset(_GRPO_CONFIG_DEFAULTS))
        self.assertEqual(_ECHO_CONFIG_KEYS, frozenset(_ECHO_CONFIG_DEFAULTS))
        self.assertLess(_ECHO_REQUIRED_CONFIG_KEYS, _ECHO_CONFIG_KEYS)
        self.assertFalse(_GRPO_CONFIG_KEYS & _ECHO_CONFIG_KEYS)

    def test_explicit_null_is_not_replaced_by_the_default(self):
        # Clients send explicit nulls over the wire; the math reads None as
        # "feature off", which is what config.get(key, default) did.
        self.assertEqual(_grpo_config_values({})["eps_clip"], 0.2)
        self.assertIsNone(_grpo_config_values({"eps_clip": None})["eps_clip"])
        self.assertNotIn("unknown", _grpo_config_values({"unknown": 1}))


class TestTrioPrecedenceIsPerLossName(TestCasePlus):
    """``grpo`` keeps the Cortex trio contract; ``ap_grpo`` keeps the AP one.

    The two disagree on purpose: Cortex lets the context win and raises on a
    conflict, AP fills missing keys from meta/batch and lets the config win.
    They are reachable only under different registered names, so a client that
    keeps sending ``grpo`` gets identical behavior on either backend. Pinning
    both here makes any future unification a deliberate, visible change.
    """

    def _call(self, loss_fn, config, meta):
        logprobs = torch.zeros(2, 3)
        batch = {
            "old_log_probs_shifted": torch.zeros(2, 3),
            "advantages": torch.ones(2, 3),
            "loss_mask": torch.ones(2, 3, dtype=torch.bool),
        }
        loss, _ = loss_fn({"logprobs": logprobs}, batch, dict(meta), dict(config), "cpu")
        return loss.item()

    def test_cortex_name_raises_on_a_context_config_conflict(self):
        with self.assertRaises(ValueError):
            self._call(
                cortex_grpo_loss,
                {"batch_num_tokens": 6, "dp_size": 2},
                {"batch_num_tokens": 12, "dp_size": 2},
            )

    def test_ap_name_lets_the_config_win(self):
        conflicting = self._call(
            grpo_loss,
            {"batch_num_tokens": 6, "dp_size": 2},
            {"batch_num_tokens": 12, "dp_size": 2},
        )
        config_only = self._call(grpo_loss, {"batch_num_tokens": 6, "dp_size": 2}, {})
        self.assertAlmostEqual(conflicting, config_only, places=6)

    def test_names_agree_when_only_one_side_supplies_the_trio(self):
        config = {"batch_num_tokens": 6, "dp_size": 2}
        self.assertAlmostEqual(
            self._call(grpo_loss, config, {}),
            self._call(cortex_grpo_loss, config, {}),
            places=6,
        )
        self.assertAlmostEqual(
            self._call(grpo_loss, {}, config),
            self._call(cortex_grpo_loss, {}, config),
            places=6,
        )


class TestA5Compat(TestCasePlus):
    def test_union_registry_prefixes_nonidentical_names(self):
        self.assertIn("grpo", LOSS_FNS)
        self.assertIn("compute_logprobs", POST_PROCESSORS)
        self.assertIn("ap_grpo", LOSS_FNS)
        self.assertIsNot(LOSS_FNS["ap_grpo"], LOSS_FNS["grpo"])
        self.assertIn("ap_compute_logprobs", POST_PROCESSORS)
        self.assertIs(POST_PROCESSORS["ap_compute_logprobs"], POST_PROCESSORS["compute_entropy_and_logprobs"])
        self.assertIsNot(POST_PROCESSORS["compute_logprobs"], POST_PROCESSORS["ap_compute_logprobs"])
        self.assertIs(LOSS_FNS["causal_cross_entropy"], causal_cross_entropy_loss)
        self.assertNotIn("cortex_grpo", LOSS_FNS)
        self.assertNotIn("cortex_compute_logprobs", POST_PROCESSORS)

    def test_stamp_only_dp_size_does_not_scale_token_mean(self):
        logprobs = torch.zeros(2, 3)
        batch = {
            "old_log_probs_shifted": torch.zeros(2, 3),
            "advantages": torch.ones(2, 3),
            "loss_mask": torch.ones(2, 3, dtype=torch.bool),
        }
        loss_local, _ = grpo_loss({"logprobs": logprobs}, batch, {}, {}, "cpu")
        loss_stamp, _ = grpo_loss({"logprobs": logprobs}, batch, {"dp_size": 8}, {}, "cpu")
        self.assertAlmostEqual(loss_local.item(), loss_stamp.item(), places=6)

    def test_stamped_dp_scales_sequence_mean_when_global_batch_size_present(self):
        logprobs = torch.zeros(2, 3)
        batch = {
            "old_log_probs_shifted": torch.zeros(2, 3),
            "advantages": torch.ones(2, 3),
            "loss_mask": torch.ones(2, 3, dtype=torch.bool),
        }
        config = {"loss_agg_mode": "seq-mean-token-mean"}
        loss_local, _ = grpo_loss({"logprobs": logprobs}, batch, {}, config, "cpu")
        loss_dp, _ = grpo_loss(
            {"logprobs": logprobs},
            batch,
            {"dp_size": 4, "global_batch_size": 2},
            config,
            "cpu",
        )
        self.assertAlmostEqual(loss_dp.item(), 4.0 * loss_local.item(), places=6)

    def test_cce_context_config_conflict_raises(self):
        with self.assertRaises(ValueError):
            causal_cross_entropy_loss(
                {"logprobs": torch.tensor([-1.0], requires_grad=True)},
                {"loss_mask": torch.ones(1)},
                {"batch_num_tokens": 3.0, "dp_size": 2},
                {"batch_num_tokens": 4.0, "dp_size": 2},
                "cpu",
            )

    def test_empty_policy_shard_still_runs_echo(self):
        batch = {
            "old_log_probs_shifted": torch.zeros(1, 3),
            "advantages": torch.ones(1, 3),
            "loss_mask": torch.zeros(1, 3, dtype=torch.bool),
            "sft_mask": torch.tensor([[0, 1, 0]], dtype=torch.bool),
            "echo_observation_mask": torch.tensor([[0, 1, 1]], dtype=torch.bool),
        }
        outputs = {"logprobs": torch.zeros(1, 3, requires_grad=True)}
        config = {"aux_ce_weight": 0.5, "echo_global_num_sequences": 1}
        loss, metrics = grpo_echo_v1_loss(outputs, batch, {}, config, "cpu")
        self.assertTrue(loss.requires_grad)
        self.assertIn("loss_term_aux", metrics)
        self.assertGreater(metrics["echo_environment_prediction_token_count"], 0.0)

    def test_cortex_compute_logprobs_prefers_labels_and_zeros_ignore_index(self):
        logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [-1.0, 0.0, 2.0]]])
        input_ids = torch.tensor([[0, 0, 0]])
        labels = torch.tensor([[1, 2, -100]])
        out = compute_logprobs_post(
            {"logits": logits},
            {"input_ids": input_ids, "labels": labels},
            {},
            "cpu",
        )
        expected = (
            torch.log_softmax(logits.float(), dim=-1)
            .gather(-1, labels.masked_fill(labels == -100, 0).unsqueeze(-1))
            .squeeze(-1)
        )
        expected[:, -1] = 0
        self.assertTrue(torch.allclose(out["logprobs"], expected))

    def test_cortex_compute_logprobs_passes_through_precomputed(self):
        precomputed = torch.randn(2, 4)
        out = compute_logprobs_post({"logprobs": precomputed}, {"input_ids": torch.arange(8).view(2, 4)}, {}, "cpu")
        self.assertEqual(out, {})

    def test_ap_config_wins_cortex_context_conflict_raises(self):
        logprobs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        batch = {
            "input_ids": torch.tensor([[1, 2]]),
            "old_log_probs_shifted": logprobs.detach(),
            "advantages": torch.ones(1, 2),
            "loss_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        meta = {"dp_size": 2, "batch_num_tokens": 4.0}
        config = {"dp_size": 1, "batch_num_tokens": 4.0}
        ap_loss, _ = grpo_loss({"logprobs": logprobs}, batch, meta, config, "cpu")
        self.assertTrue(torch.isfinite(ap_loss))
        with self.assertRaises(ValueError):
            cortex_grpo_loss({"logprobs": logprobs}, batch, meta, config, "cpu")

    def test_causal_cross_entropy_is_five_arg(self):
        import inspect

        self.assertEqual(len(inspect.signature(causal_cross_entropy_loss).parameters), 5)

    def test_weighted_cce_local_mean(self):
        logprobs = torch.tensor([-1.0, -2.0], requires_grad=True)
        weights = torch.tensor([0.5, 1.0])
        loss, metrics = causal_cross_entropy_loss(
            {"logprobs": logprobs},
            {"loss_mask": weights},
            {},
            {},
            "cpu",
        )
        self.assertEqual(metrics, {})
        self.assertAlmostEqual(loss.item(), (1.0 * 0.5 + 2.0 * 1.0) / 1.5)

    def test_cce_requires_dp_size_with_batch_num_tokens(self):
        with self.assertRaises(ValueError):
            causal_cross_entropy_loss(
                {"logprobs": torch.tensor([-1.0], requires_grad=True)},
                {"loss_mask": torch.ones(1)},
                {},
                {"batch_num_tokens": 3.0},
                "cpu",
            )

    def test_cce_ignores_global_batch_size(self):
        logprobs = torch.tensor([-1.0, -2.0], requires_grad=True)
        weights = torch.tensor([0.5, 1.0])
        loss, _ = causal_cross_entropy_loss(
            {"logprobs": logprobs},
            {"loss_mask": weights},
            {"batch_num_tokens": 3.0, "dp_size": 2, "global_batch_size": 4},
            {},
            "cpu",
        )
        self.assertAlmostEqual(loss.item(), 5.0 / 3.0)

    def test_cce_dp_from_meta(self):
        logprobs = torch.tensor([-1.0, -2.0], requires_grad=True)
        weights = torch.tensor([0.5, 1.0])
        loss, _ = causal_cross_entropy_loss(
            {"logprobs": logprobs},
            {"loss_mask": weights},
            {"batch_num_tokens": 3.0, "dp_size": 2},
            {},
            "cpu",
        )
        self.assertAlmostEqual(loss.item(), 5.0 / 3.0)

    def test_packing_honors_cce_and_filters_engine(self):
        class Engine:
            def __init__(self):
                self.last_kwargs = None
                self.backward_scale = []
                self.global_rank = 0

            def train(self):
                pass

            def eval(self):
                pass

            def __call__(self, *args, **kwargs):
                self.last_kwargs = dict(kwargs)
                ids = kwargs["input_ids"]
                if ids.ndim == 1:
                    ids = ids.unsqueeze(0)
                b, s = ids.shape[:2]
                table = torch.zeros(8)
                table[1] = -1.0
                table[2] = -2.0
                table[3] = -3.0
                logprobs = table[ids]
                return SimpleNamespace(logits=torch.zeros(b, s, 8), logprobs=logprobs)

            def backward(self, loss, scale_wrt_gas=True):
                self.backward_scale.append(scale_wrt_gas)

        engine = Engine()
        batch = {
            "input_ids": torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
            "loss_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]]),
        }
        out = run_pipeline(
            engine,
            (),
            batch,
            {"pad_token_id": 0},
            {"loss_fn": "causal_cross_entropy", "post": [], "config": {}},
            "cpu",
            backward=True,
            pack=True,
            max_tokens_per_mb=3,
        )
        self.assertIn("avg_loss", out)
        self.assertTrue(all(scale is False for scale in engine.backward_scale))
        self.assertNotIn("loss_mask", engine.last_kwargs)
        self.assertIn("input_ids", engine.last_kwargs)

    def test_cce_packed_resolve_accepts_global_batch_size(self):
        mb0 = {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "loss_mask": torch.ones(1, 2),
            "batch_num_tokens": 3.0,
            "dp_size": 2,
            "global_batch_size": 4,
        }
        mb1 = {**mb0, "loss_mask": torch.ones(1, 2) * 0.5}
        reduction = resolve_packed_loss_reduction(
            {"loss_fn": "causal_cross_entropy", "config": {}},
            [mb0, mb1],
        )
        self.assertTrue(reduction.loss_is_additive)

    def _cce_pack_engine(self):
        class Engine:
            def __init__(self):
                self.last_kwargs = None
                self.backward_losses = []
                self.global_rank = 0

            def train(self):
                pass

            def eval(self):
                pass

            def __call__(self, *args, **kwargs):
                self.last_kwargs = dict(kwargs)
                ids = kwargs["input_ids"]
                if ids.ndim == 1:
                    ids = ids.unsqueeze(0)
                table = torch.zeros(8)
                table[1] = -1.0
                table[2] = -2.0
                table[3] = -3.0
                table[4] = -4.0
                table[5] = -5.0
                return SimpleNamespace(logits=torch.zeros(*ids.shape, 8), logprobs=table[ids])

            def backward(self, loss, scale_wrt_gas=True):
                self.backward_losses.append((float(loss.detach()), scale_wrt_gas))

        return Engine()

    def test_cce_split_invariance_and_global_batch_size(self):
        batch = {
            "input_ids": torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
            "loss_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]]),
        }
        meta = {"pad_token_id": 0, "batch_num_tokens": 5.0, "dp_size": 2, "global_batch_size": 2}
        processing = {"loss_fn": "causal_cross_entropy", "post": [], "config": {}}
        one = run_pipeline(
            self._cce_pack_engine(), (), batch, meta, processing, "cpu", backward=True, pack=True, max_tokens_per_mb=16
        )
        split_engine = self._cce_pack_engine()
        two = run_pipeline(
            split_engine, (), batch, meta, processing, "cpu", backward=True, pack=True, max_tokens_per_mb=3
        )
        self.assertGreater(len(split_engine.backward_losses), 1)
        self.assertAlmostEqual(one["avg_loss"], two["avg_loss"], places=5)
        self.assertTrue(all(scale is False for _, scale in split_engine.backward_losses))


class TestA2FromPr111(TestCasePlus):
    def test_common_init_keeps_111_lazy_gate(self):
        """A2 landed in #111. This change set must not restore the import-time gate."""
        src = (Path(__file__).resolve().parents[2] / "arctic_platform" / "common" / "__init__.py").read_text()
        tree = ast.parse(src)
        module_calls = [
            node
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "require_any_dep_group"
        ]
        self.assertEqual(module_calls, [])
        self.assertIn("def _require_training_extras", src)
