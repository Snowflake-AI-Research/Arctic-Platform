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
"""Adversarial / edge-case probes for the SFT path.

These tests intentionally try to break the implementation. Failures here are
documented bugs (or regressions), not flaky noise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from arctic_platform.sft.config import ArcticSFTClientConfig
from arctic_platform.sft.processor import SFT_LOSS_FNS
from arctic_platform.sft.processor import run_sft_pipeline
from arctic_platform.sft.processor import sft_ce_loss
from arctic_platform.sft.processor import sft_loss
from arctic_platform.testing_utils import TestCasePlus


class TestAllMaskedLabels(TestCasePlus):
    def test_sft_loss_all_neg100_zero_sum_and_tokens(self):
        """All-masked labels: good=0 → (sum=0, tokens=0) so the pair cancels cleanly."""
        labels = torch.full((1, 5), -100, dtype=torch.long)
        loss, metrics = sft_loss(
            {"loss": torch.tensor(3.0)},
            {"labels": labels},
            {},
            {},
            "cpu",
        )
        self.assertEqual(metrics["loss.tokens"], 0.0)
        # Fixed: no bogus loss*1 injected when there are no valid targets.
        self.assertEqual(metrics["loss.sum"], 0.0)

    def test_sft_ce_all_neg100_divides_by_one(self):
        labels = torch.full((1, 4), -100, dtype=torch.long)
        logits = torch.zeros(1, 4, 8)
        loss, metrics = sft_ce_loss(
            {"logits": logits},
            {"labels": labels},
            {},
            {},
            "cpu",
        )
        self.assertEqual(metrics["loss.tokens"], 0.0)
        # ce_sum is 0 over empty set; loss = 0/1 = 0 — OK numerically, but
        # callers must not treat this as a real training step.
        self.assertEqual(loss.item(), 0.0)


class TestMissingLabels(TestCasePlus):
    def test_sft_loss_missing_labels_raises_valueerror(self):
        # Fixed: a clear ValueError instead of a bare KeyError on batch['labels'].
        with self.assertRaises(ValueError):
            sft_loss({"loss": torch.tensor(1.0)}, {}, {}, {}, "cpu")

    def test_pipeline_missing_labels_raises_valueerror(self):
        class Eng:
            def train(self):
                pass

            def __call__(self, **kw):
                raise AssertionError("should fail before forward")

            def backward(self, *a, **k):
                pass

        batch = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            # labels intentionally absent
        }
        with self.assertRaises(ValueError):
            run_sft_pipeline(
                Eng(),
                batch,
                meta={},
                processing={"loss_fn": "sft"},
                device="cpu",
            )


class TestDispatchDrift(TestCasePlus):
    def test_worker_dispatches_via_sft_loss_fns(self):
        """deepspeed_worker resolves SFT vs GRPO against SFT_LOSS_FNS (no drift)."""
        worker_path = Path(__file__).resolve().parents[2] / "arctic_platform" / "common" / "deepspeed_worker.py"
        src = worker_path.read_text()
        # Fixed: dispatch resolves against the canonical registry set instead of
        # an inline ``loss_fn in ("sft", "sft_ce")`` tuple, so new SFT losses
        # route correctly without editing the worker.
        self.assertIn("use_sft_pipeline = loss_fn in SFT_LOSS_FNS", src)
        self.assertNotIn('loss_fn in ("sft", "sft_ce")', src)
        self.assertEqual(SFT_LOSS_FNS, {"sft", "sft_ce"})


class TestCheckpointPathRequired(TestCasePlus):
    def test_new_job_requires_checkpoint_path(self):
        """Fixed: client fails fast when starting a new job without a checkpoint_path."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            ArcticSFTClientConfig(model_name="m", training_gpus=1)

    def test_reconnect_does_not_require_checkpoint_path(self):
        # Reconnecting to an existing job inherits its path; none needed here.
        cfg = ArcticSFTClientConfig(model_name="m", training_gpus=1, training_job_id=3)
        self.assertIsNone(cfg.checkpoint_path)

    def test_to_rl_config_forwards_checkpoint(self):
        cfg = ArcticSFTClientConfig(model_name="m", training_gpus=1, checkpoint_path="/tmp/c")
        rl = cfg.to_rl_config()
        self.assertEqual(rl.checkpoint_path, "/tmp/c")


class TestInitFailureShutdown(TestCasePlus):
    def test_initialize_failure_calls_shutdown(self):
        from arctic_platform.client import JobHandles
        from arctic_platform.client import Request
        from arctic_platform.client import Transport
        from arctic_platform.sft.client import ArcticSFTClient

        class BoomTransport(Transport):
            def __init__(self, config):
                self.config = config
                self.shutdown_calls = 0
                self.jobs = JobHandles()

            def initialize(self):
                raise RuntimeError("init failed")

            def call(self, request: Request) -> dict:
                return {}

            def shutdown(self) -> None:
                self.shutdown_calls += 1

        # Patch via the module the client uses.
        import arctic_platform.sft.client as mod

        original = mod._make_transport
        boom = None

        def factory(cfg):
            nonlocal boom
            boom = BoomTransport(cfg)
            return boom

        mod._make_transport = factory
        try:
            with pytest.raises(RuntimeError, match="init failed"):
                ArcticSFTClient(ArcticSFTClientConfig(model_name="m", training_gpus=1, checkpoint_path="/tmp/c"))
            assert boom is not None
            assert boom.shutdown_calls == 1, f"expected shutdown on init failure, got {boom.shutdown_calls}"
        finally:
            mod._make_transport = original


class TestCommonPackageExports(TestCasePlus):
    def test_deepspeed_worker_lazily_exported_from_common_pkg(self):
        """Fixed: names in common.__all__ resolve via a lazy module __getattr__."""
        import arctic_platform.common as common

        self.assertIn("DeepSpeedWorker", common.__all__)
        # A bogus attribute still raises AttributeError (getattr didn't swallow it).
        with self.assertRaises(AttributeError):
            _ = common.does_not_exist
        try:
            worker = common.DeepSpeedWorker
        except ModuleNotFoundError:
            # ray/deepspeed not installed in this CPU env — lazy import is still wired.
            pytest.skip("ray/deepspeed not installed; lazy export path exercised")
        else:
            # It's a Ray @ray.remote ActorClass (no __name__), so assert identity
            # with the module attribute rather than a name string.
            import arctic_platform.common.deepspeed_worker as dw

            self.assertIs(worker, dw.DeepSpeedWorker)


class TestSftExampleIsCanonical(TestCasePlus):
    def test_example_uses_sft_client_not_rl(self):
        path = Path(__file__).resolve().parents[2] / "arctic_platform" / "sft" / "examples" / "sft_example.py"
        src = path.read_text()
        # Fixed: the example uses the SFT client + SFT wire format, no stale path.
        self.assertIn("ArcticSFTClient", src)
        self.assertNotIn("ArcticRLClient", src)
        self.assertNotIn("client/examples/sft_example.py", src)
        self.assertIn('"labels"', src)


class TestSamplePackingKwargs(TestCasePlus):
    def test_packed_batch_forwards_position_ids_omits_all_ones_mask(self):
        from arctic_platform.sft.processor import _batch_is_packed
        from arctic_platform.sft.processor import _build_sft_model_kwargs

        # Two packed docs: positions reset at the boundary (index 3).
        pos = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])
        batch = {
            "input_ids": torch.arange(7).unsqueeze(0),
            "labels": torch.full((1, 7), -100),
            "position_ids": pos,
            # Buggy client might still send all-ones — must be omitted.
            "attention_mask": torch.ones(1, 7, dtype=torch.long),
        }
        meta = {"sample_packing": True}
        self.assertTrue(_batch_is_packed(batch, meta))
        kw = _build_sft_model_kwargs(batch, meta, batch["labels"], need_logits=False)
        self.assertIn("position_ids", kw)
        self.assertNotIn("attention_mask", kw)
        self.assertIn("labels", kw)

    def test_packed_batch_keeps_nontrivial_mask(self):
        from arctic_platform.sft.processor import _build_sft_model_kwargs

        batch = {
            "input_ids": torch.arange(6).unsqueeze(0),
            "labels": torch.full((1, 6), -100),
            "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]),  # trailing pad
        }
        kw = _build_sft_model_kwargs(batch, {"sample_packing": True}, batch["labels"], need_logits=False)
        self.assertIn("position_ids", kw)
        self.assertIn("attention_mask", kw)

    def test_dense_batch_still_passes_attention_mask(self):
        from arctic_platform.sft.processor import _batch_is_packed
        from arctic_platform.sft.processor import _build_sft_model_kwargs

        batch = {
            "input_ids": torch.arange(4).unsqueeze(0),
            "labels": torch.full((1, 4), -100),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "position_ids": torch.tensor([[0, 1, 2, 3]]),  # no reset → dense
        }
        self.assertFalse(_batch_is_packed(batch, {}))
        kw = _build_sft_model_kwargs(batch, {}, batch["labels"], need_logits=False)
        self.assertIn("attention_mask", kw)
        self.assertIn("position_ids", kw)

    def test_pipeline_packed_without_mask_reaches_forward(self):
        seen = {}

        class Eng:
            def train(self):
                pass

            def __call__(self, **kw):
                seen.update(kw)

                class Out:
                    loss = torch.tensor(1.0)

                return Out()

            def backward(self, *a, **k):
                pass

        batch = {
            "input_ids": torch.arange(5).unsqueeze(0),
            "labels": torch.tensor([[-100, -100, 1, 2, 3]]),
            "position_ids": torch.tensor([[0, 1, 0, 1, 2]]),
        }
        run_sft_pipeline(
            Eng(),
            batch,
            meta={"sample_packing": True},
            processing={"loss_fn": "sft"},
            device="cpu",
        )
        self.assertIn("position_ids", seen)
        self.assertNotIn("attention_mask", seen)


class TestGasMicrobatchList(TestCasePlus):
    """H3: list-of-microbatches wire format (no concat → server re-split)."""

    def test_split_batch_list_shards_each_microbatch(self):
        from arctic_platform.common.utils.batch import _split_batch

        mb0 = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),  # 2 samples → 2 DP ranks
            "labels": torch.tensor([[-100, 2], [-100, 4]]),
            "attention_mask": torch.ones(2, 2, dtype=torch.long),
        }
        mb1 = {
            "input_ids": torch.tensor([[5], [6]]),  # different seq len — OK with list
            "labels": torch.tensor([[-100], [-100]]),
            "attention_mask": torch.ones(2, 1, dtype=torch.long),
        }
        envelope = {
            "batch": [mb0, mb1],
            "meta": {"pad_token_id": 0, "gas_microbatches": True},
            "processing": {"loss_fn": "sft"},
        }
        shards, reorder = _split_batch(envelope, num_workers=2)
        self.assertIsNone(reorder)
        self.assertEqual(len(shards), 2)
        # Each worker gets a list of 2 microbatches (GAS).
        self.assertIsInstance(shards[0]["batch"], list)
        self.assertEqual(len(shards[0]["batch"]), 2)
        self.assertEqual(shards[0]["batch"][0]["input_ids"].shape, (1, 2))
        self.assertEqual(shards[0]["batch"][1]["input_ids"].shape, (1, 1))
        self.assertEqual(shards[1]["batch"][0]["input_ids"].shape, (1, 2))
        self.assertTrue(shards[0]["meta"]["gas_microbatches"])

    def test_split_batch_legacy_dict_still_works(self):
        from arctic_platform.common.utils.batch import _split_batch

        envelope = {
            "batch": {
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "labels": torch.tensor([[-100, 2], [-100, 4]]),
                "attention_mask": torch.ones(2, 2, dtype=torch.long),
            },
            "meta": {"pad_token_id": 0},
            "processing": {"loss_fn": "sft"},
        }
        shards, _ = _split_batch(envelope, num_workers=2)
        self.assertIsInstance(shards[0]["batch"], dict)
        self.assertEqual(shards[0]["batch"]["input_ids"].shape, (1, 2))


class TestPerfFixesUnit(TestCasePlus):
    def test_h7_worker_sets_gas_accumulation_boundary(self):
        worker_path = Path(__file__).resolve().parents[2] / "arctic_platform" / "common" / "deepspeed_worker.py"
        src = worker_path.read_text()
        self.assertIn("set_gradient_accumulation_boundary", src)
        # Bound immediately before run_sft_pipeline in the SFT microbatch loop.
        idx_boundary = src.index("set_gradient_accumulation_boundary")
        idx_sft_run = src.index("run_sft_pipeline(", idx_boundary)
        self.assertLess(idx_sft_run - idx_boundary, 250)
        window = src[max(0, idx_boundary - 500) : idx_boundary]
        self.assertIn("if use_sft_pipeline:", window)

    def test_h4_move_batch_uses_non_blocking(self):
        worker_path = Path(__file__).resolve().parents[2] / "arctic_platform" / "common" / "deepspeed_worker.py"
        src = worker_path.read_text()
        self.assertIn("non_blocking=True", src)
        self.assertIn("pin_memory()", src)

    def test_h3_worker_accepts_microbatch_list_without_split_dict(self):
        worker_path = Path(__file__).resolve().parents[2] / "arctic_platform" / "common" / "deepspeed_worker.py"
        src = worker_path.read_text()
        self.assertIn("isinstance(batch_data, list)", src)
        self.assertIn("Received", src)  # length mismatch error

    def test_sft_profile_helpers(self):
        import os

        from arctic_platform.common.utils import sft_profile

        prev = os.environ.get("ARL_SFT_PROFILE")
        try:
            os.environ["ARL_SFT_PROFILE"] = "1"
            self.assertTrue(sft_profile.enabled())
            with sft_profile.timed("fwd"):
                pass
            metrics = sft_profile.merge_into_metrics({"loss.sum": 1.0})
            self.assertIn("_profile_ms", metrics)
            self.assertIn("fwd", metrics["_profile_ms"])
            self.assertEqual(metrics["loss.sum"], 1.0)
        finally:
            if prev is None:
                os.environ.pop("ARL_SFT_PROFILE", None)
            else:
                os.environ["ARL_SFT_PROFILE"] = prev
            sft_profile.take_last()  # clear any leftover


class TestShimsStillResolve(TestCasePlus):
    def test_old_client_shim(self):
        from arctic_platform.client.sft_client import ArcticSFTClient as C1
        from arctic_platform.sft import ArcticSFTClient as C2

        self.assertIs(C1, C2)

    def test_old_processor_shim(self):
        from arctic_platform.rl.processors.sft import run_sft_pipeline as R1
        from arctic_platform.sft import run_sft_pipeline as R2

        self.assertIs(R1, R2)
