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
"""CPU unit tests for SFT loss + pipeline (``processors/sft.py``).

No GPU / Ray / DeepSpeed — tiny deterministic tensors and a stub engine.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from arctic_platform.common.registry import LOSS_FNS
from arctic_platform.sft.processor import LOGIT_LOSS_FNS
from arctic_platform.sft.processor import SFT_GLOBAL_TOKEN_LOSS_FNS
from arctic_platform.sft.processor import SFT_LOSS_FNS
from arctic_platform.sft.processor import count_valid_target_tokens
from arctic_platform.sft.processor import run_sft_pipeline
from arctic_platform.sft.processor import sft_ce_loss
from arctic_platform.sft.processor import sft_loss
from arctic_platform.testing_utils import TestCasePlus


class TestSFTRegistry(TestCasePlus):
    def test_loss_fns_registered(self):
        self.assertIn("sft", LOSS_FNS)
        self.assertIn("sft_ce", LOSS_FNS)
        self.assertEqual(SFT_LOSS_FNS, {"sft", "sft_ce"})
        self.assertEqual(LOGIT_LOSS_FNS, {"sft_ce"})
        # Only sft_ce needs global-token injection; sft uses HF's per-shard mean.
        self.assertEqual(SFT_GLOBAL_TOKEN_LOSS_FNS, {"sft_ce"})


class TestSFTLoss(TestCasePlus):
    def test_reconstructs_sum_from_token_mean(self):
        # HF-style token-mean scalar of 2.0 over 3 valid targets → sum = 6.0
        labels = torch.tensor([[-100, 1, 2, 3, -100]])  # 3 valid after shift ([:,1:])
        loss, metrics = sft_loss(
            {"loss": torch.tensor(2.0)},
            {"labels": labels},
            {},
            {},
            "cpu",
        )
        self.assertAlmostEqual(loss.item(), 2.0, places=5)
        self.assertAlmostEqual(metrics["loss.sum"], 6.0, places=5)
        self.assertEqual(metrics["loss.tokens"], 3.0)

    def test_none_loss_raises(self):
        with self.assertRaises(ValueError):
            sft_loss(
                {"loss": None},
                {"labels": torch.ones(1, 4, dtype=torch.long)},
                {},
                {},
                "cpu",
            )


class TestSFTCELoss(TestCasePlus):
    def _perfect_logits(self, labels: torch.Tensor, vocab: int = 8) -> torch.Tensor:
        """Logits that put mass 1 on the shifted target token (ignore -100)."""
        b, s = labels.shape
        logits = torch.zeros(b, s, vocab)
        for i in range(b):
            for t in range(s - 1):
                tgt = int(labels[i, t + 1].item())
                if tgt != -100:
                    logits[i, t, tgt] = 20.0  # ~zero CE after softmax
        return logits

    def test_per_shard_token_mean(self):
        labels = torch.tensor([[-100, 1, 2, -100]])
        logits = self._perfect_logits(labels)
        loss, metrics = sft_ce_loss(
            {"logits": logits},
            {"labels": labels},
            {},
            {},
            "cpu",
        )
        # Two valid targets after shift; near-zero CE.
        self.assertEqual(metrics["loss.tokens"], 2.0)
        self.assertLess(loss.item(), 1e-4)
        self.assertAlmostEqual(metrics["loss.sum"], loss.item() * 2, places=4)

    def test_global_token_scaling(self):
        labels = torch.tensor([[-100, 1, 2, -100]])  # 2 local valid tokens
        logits = self._perfect_logits(labels)
        # Force a non-zero CE by flipping one target so scaling is observable.
        logits = torch.zeros_like(logits)
        logits[:, :, 0] = 10.0  # predict token 0 everywhere
        # targets are 1 and 2 → CE > 0
        loss, metrics = sft_ce_loss(
            {"logits": logits},
            {"labels": labels},
            {"global_num_tokens": 10, "dp_size": 2},
            {},
            "cpu",
        )
        # loss = ce_sum / 10 * 2
        expected = metrics["loss.sum"] / 10.0 * 2.0
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertEqual(metrics["loss.tokens"], 2.0)

    def test_missing_logits_raises(self):
        with self.assertRaises(ValueError):
            sft_ce_loss(
                {},
                {"labels": torch.ones(1, 3, dtype=torch.long)},
                {},
                {},
                "cpu",
            )

    def test_zero_global_tokens_falls_back_without_div_by_zero(self):
        # All targets masked ⇒ global_num_tokens == 0 across ranks. Scaling must
        # not divide by zero; it falls back to the per-shard token-mean (result
        # is a finite 0 here since ce_sum is 0).
        labels = torch.tensor([[-100, -100, -100, -100]])
        logits = torch.zeros(1, 4, 8)
        loss, metrics = sft_ce_loss(
            {"logits": logits},
            {"labels": labels},
            {"global_num_tokens": 0, "dp_size": 2},
            {},
            "cpu",
        )
        self.assertEqual(metrics["loss.tokens"], 0.0)
        self.assertTrue(torch.isfinite(loss))


class TestCountValidTargetTokens(TestCasePlus):
    """Valid-target accounting the worker all-reduces into ``global_num_tokens``.

    This is the ray-free core of ``_inject_sft_global_token_meta``: it must apply
    HF's ``labels[:, 1:]`` shift, sum across a list-of-microbatch (gas) shard, and
    signal "no labels" with ``None`` so the worker skips injection.
    """

    def test_dict_shard_counts_shifted_valid_targets(self):
        labels = torch.tensor([[-100, 1, 2, -100]])  # 2 valid after [:, 1:] shift
        self.assertEqual(count_valid_target_tokens({"labels": labels}), 2)

    def test_list_shard_sums_across_microbatches(self):
        labels = torch.tensor([[-100, 1, 2, -100]])  # 2 valid each
        self.assertEqual(count_valid_target_tokens([{"labels": labels}, {"labels": labels}]), 4)

    def test_missing_labels_returns_none(self):
        self.assertIsNone(count_valid_target_tokens({"input_ids": torch.zeros(1, 3, dtype=torch.long)}))

    def test_none_labels_value_returns_none(self):
        # Key present but value None must not crash the shift — treat as no labels.
        self.assertIsNone(count_valid_target_tokens({"labels": None}))

    def test_list_skips_none_labels_value(self):
        labels = torch.tensor([[-100, 1, 2, -100]])  # 2 valid
        self.assertEqual(count_valid_target_tokens([{"labels": None}, {"labels": labels}]), 2)

    def test_list_without_labels_returns_none(self):
        self.assertIsNone(count_valid_target_tokens([{"input_ids": torch.zeros(1, 3, dtype=torch.long)}]))

    def test_list_skips_microbatches_without_labels(self):
        labels = torch.tensor([[-100, 1, 2, -100]])  # 2 valid
        mixed = [{"input_ids": torch.zeros(1, 3, dtype=torch.long)}, {"labels": labels}]
        self.assertEqual(count_valid_target_tokens(mixed), 2)


class _StubEngine:
    """Minimal stand-in for DeepSpeedEngine used by ``run_sft_pipeline``."""

    def __init__(self, vocab: int = 8):
        self.vocab = vocab
        self.train_calls = 0
        self.eval_calls = 0
        self.backward_calls: list[torch.Tensor] = []
        self.last_kwargs: dict | None = None

    def train(self):
        self.train_calls += 1

    def eval(self):
        self.eval_calls += 1

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        ids = kwargs["input_ids"]
        b, s = ids.shape
        logits = torch.zeros(b, s, self.vocab)
        # Put mass on the next-token id so CE is near zero when labels are present.
        labels = kwargs.get("labels")
        loss = None
        if labels is not None:
            n_valid = (labels[:, 1:] != -100).sum().clamp(min=1)
            loss = torch.tensor(1.5, requires_grad=True)
            _ = n_valid  # keep referenced for shape/masking assertions if needed
        else:
            loss = None
        # Only surface hidden states when the pipeline asks (compute/memory CE).
        hidden_states = (torch.zeros(b, s, 4),) if kwargs.get("output_hidden_states") else None
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden_states)

    def backward(self, loss, scale_wrt_gas=True):
        self.backward_calls.append((loss.detach().clone(), scale_wrt_gas))


class TestRunSFTPipeline(TestCasePlus):
    def _batch(self):
        return {
            "input_ids": torch.arange(8).view(2, 4),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "labels": torch.tensor([[-100, 1, 2, 3], [-100, 4, 5, -100]]),
        }

    def test_sft_path_passes_labels_and_backward(self):
        engine = _StubEngine()
        out = run_sft_pipeline(
            engine,
            self._batch(),
            meta={},
            processing={"loss_fn": "sft"},
            device="cpu",
            backward=True,
        )
        self.assertEqual(engine.train_calls, 1)
        self.assertIn("labels", engine.last_kwargs)
        self.assertEqual(len(engine.backward_calls), 1)
        self.assertFalse(engine.backward_calls[0][1])  # scale_wrt_gas=False
        self.assertIn("loss.sum", out["metrics"])
        self.assertIn("loss.tokens", out["metrics"])
        self.assertAlmostEqual(out["avg_loss"], 1.5, places=5)

    def test_sft_ce_path_skips_labels_keeps_logits(self):
        engine = _StubEngine()
        out = run_sft_pipeline(
            engine,
            self._batch(),
            meta={},
            processing={"loss_fn": "sft_ce"},
            device="cpu",
            backward=False,
        )
        self.assertEqual(engine.eval_calls, 1)
        self.assertNotIn("labels", engine.last_kwargs)
        self.assertEqual(engine.backward_calls, [])
        self.assertIn("loss.tokens", out["metrics"])


class TestSftCeLogitsOptimizationRouting(TestCasePlus):
    """H8 routing: ``compute`` / ``memory`` take the hidden-state CE path and
    never keep the full ``[B, S, V]`` logits. Stubs the CE kernel so this stays
    a pure CPU routing test (client never runs those kernels).
    """

    def _batch(self):
        return {
            "input_ids": torch.arange(8).view(2, 4),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "labels": torch.tensor([[-100, 1, 2, 3], [-100, 4, 5, -100]]),
        }

    def _run_with_stub(self, mode):
        import arctic_platform.sft.processor as proc

        calls = {}

        def _stub(model, hidden, labels, *, mode, peak_mem_gib):
            calls["mode"] = mode
            calls["peak_mem_gib"] = peak_mem_gib
            calls["hidden_shape"] = tuple(hidden.shape)
            n_valid = int((labels[:, 1:] != -100).sum().item())
            return torch.tensor(6.0, requires_grad=True), n_valid

        orig = proc.sft_ce_sum_from_hidden
        proc.sft_ce_sum_from_hidden = _stub
        try:
            engine = _StubEngine()
            out = run_sft_pipeline(
                engine,
                self._batch(),
                meta={},
                processing={
                    "loss_fn": "sft_ce",
                    "config": {"logits_optimization": mode, "logits_optimization_peak_mem_size_in_gib": 3},
                },
                device="cpu",
                backward=True,
            )
            return engine, out, calls
        finally:
            proc.sft_ce_sum_from_hidden = orig

    def test_compute_routes_to_hidden_ce(self):
        engine, out, calls = self._run_with_stub("compute")
        self.assertEqual(calls["mode"], "compute")
        self.assertEqual(calls["peak_mem_gib"], 3)
        self.assertTrue(engine.last_kwargs.get("output_hidden_states"))
        self.assertEqual(engine.last_kwargs.get("logits_to_keep"), 1)
        self.assertNotIn("labels", engine.last_kwargs)  # HF CE skipped
        self.assertEqual(len(engine.backward_calls), 1)
        self.assertFalse(engine.backward_calls[0][1])  # scale_wrt_gas=False
        self.assertIn("loss.tokens", out["metrics"])

    def test_memory_routes_to_hidden_ce(self):
        engine, _out, calls = self._run_with_stub("memory")
        self.assertEqual(calls["mode"], "memory")
        self.assertTrue(engine.last_kwargs.get("output_hidden_states"))
        self.assertEqual(engine.last_kwargs.get("logits_to_keep"), 1)

    def test_none_keeps_full_logits_path(self):
        engine = _StubEngine()
        run_sft_pipeline(
            engine,
            self._batch(),
            meta={},
            processing={"loss_fn": "sft_ce", "config": {"logits_optimization": "none"}},
            device="cpu",
            backward=False,
        )
        # Classic path: no hidden-state request; full logits consumed by sft_ce_loss.
        self.assertNotIn("output_hidden_states", engine.last_kwargs)
        self.assertNotIn("logits_to_keep", engine.last_kwargs)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            run_sft_pipeline(
                _StubEngine(),
                self._batch(),
                meta={},
                processing={"loss_fn": "sft_ce", "config": {"logits_optimization": "bogus"}},
                device="cpu",
                backward=False,
            )


class TestSftCeSumFromHiddenErrors(TestCasePlus):
    def test_bad_mode_raises(self):
        from arctic_platform.sft.processor import sft_ce_sum_from_hidden

        model = SimpleNamespace(
            lm_head=torch.nn.Linear(4, 8, bias=False),
            config=SimpleNamespace(vocab_size=8),
        )
        with self.assertRaises(ValueError):
            sft_ce_sum_from_hidden(model, torch.randn(1, 3, 4), torch.zeros(1, 3, dtype=torch.long), mode="none")

