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

"""Step-0 contracts: arity, isolation, packed inner call, decorator storage."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

import arctic_platform.rl.processors  # noqa: F401  # populate registries
import arctic_platform.sft.processor  # noqa: F401
from arctic_platform.common.registry import LOSS_FNS
from arctic_platform.common.registry import PACKED_LOSS_REDUCTION_ATTR
from arctic_platform.common.registry import POST_PROCESSORS
from arctic_platform.rl.processors.grpo import grpo_echo_v1_loss
from arctic_platform.rl.processors.grpo import grpo_loss
from arctic_platform.rl.processors.packed_reduction import PackedLossReduction
from arctic_platform.rl.processors.pipeline import _engine_forward_kwargs
from arctic_platform.rl.processors.pipeline import register_post_processor
from arctic_platform.rl.processors.pipeline import run_pipeline
from arctic_platform.testing_utils import TestCasePlus


def _positional_count(fn) -> int:
    return len(inspect.signature(fn).parameters)


class _StubEngine:
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
        return SimpleNamespace(logits=torch.zeros(b, s, 8), logprobs=torch.zeros(b, s))

    def backward(self, loss, scale_wrt_gas=True):
        self.backward_scale.append(scale_wrt_gas)


class TestArityGuard(TestCasePlus):
    def test_registered_arities(self):
        import arctic_platform.rl.processors  # noqa: F401
        import arctic_platform.sft.processor  # noqa: F401

        for name, fn in LOSS_FNS.items():
            params = inspect.signature(fn).parameters
            self.assertFalse(any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()), name)
            self.assertEqual(len(params), 5, name)
        for name, fn in POST_PROCESSORS.items():
            params = inspect.signature(fn).parameters
            self.assertFalse(any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()), name)
            self.assertEqual(len(params), 4, name)
        self.assertIn("ap_grpo", LOSS_FNS)
        self.assertIn("ap_grpo_echo_v1", LOSS_FNS)
        self.assertIn("grpo", LOSS_FNS)
        self.assertIn("grpo_echo_v1", LOSS_FNS)
        self.assertEqual(_positional_count(POST_PROCESSORS["identity"]), 4)


class TestPackedReductionResolverShape(TestCasePlus):
    def test_attribute_and_resolver_shape(self):
        fn = LOSS_FNS["ap_grpo"]
        resolver = getattr(fn, PACKED_LOSS_REDUCTION_ATTR)
        mb = {
            "input_ids": torch.arange(4).view(1, 4),
            "loss_mask": torch.ones(1, 4, dtype=torch.bool),
        }
        one = resolver([mb], {}, "ap_grpo")
        two = resolver([mb, mb], {}, "ap_grpo")
        self.assertIsInstance(one, PackedLossReduction)
        self.assertEqual(len(one.loss_scales), 1)
        self.assertEqual(len(two.loss_scales), 2)
        self.assertEqual(len(two.reporting_weights), 2)


class TestIsolation(TestCasePlus):
    def test_engine_kwargs_drop_loss_tensors(self):
        batch = {
            "input_ids": torch.arange(4).view(1, 4),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "advantages": torch.ones(1, 4),
            "old_log_probs": torch.zeros(1, 4),
        }
        meta = {
            "calculate_entropy": True,
            "loss_mask": torch.ones(1, 4, dtype=torch.bool),
            "old_log_probs_shifted": torch.zeros(1, 4),
            "actor_config": {"entropy_coeff": 0.0},
            "temperature": 1.0,
            "dp_size": 2,
        }
        kwargs = _engine_forward_kwargs(batch, meta)
        self.assertIn("input_ids", kwargs)
        self.assertIn("attention_mask", kwargs)
        self.assertIn("calculate_entropy", kwargs)
        self.assertNotIn("advantages", kwargs)
        self.assertNotIn("loss_mask", kwargs)
        self.assertNotIn("actor_config", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("dp_size", kwargs)

    def test_fwd_meta_keys_can_add_but_not_blocked_keys(self):
        blocked = ("advantages", "loss_mask", "actor_config", "temperature", "dp_size")
        batch = {
            "input_ids": torch.arange(4).view(1, 4),
            "advantages": torch.ones(1, 4),
            "loss_mask": torch.ones(1, 4),
        }
        meta = {
            "calculate_entropy": True,
            "use_cache": False,
            "actor_config": {"entropy_coeff": 0.1},
            "temperature": 1.0,
            "dp_size": 2,
            "fwd_meta_keys": ("use_cache",) + blocked,
        }
        kwargs = _engine_forward_kwargs(batch, meta)
        self.assertIn("input_ids", kwargs)
        self.assertIn("use_cache", kwargs)
        self.assertNotIn("fwd_meta_keys", kwargs)
        for key in blocked:
            self.assertNotIn(key, kwargs)

    def test_pack_false_forward_filters_batch_and_meta(self):
        engine = _StubEngine()
        batch = {
            "input_ids": torch.arange(8).view(2, 4),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "position_ids": torch.arange(4).repeat(2, 1),
            "advantages": torch.ones(2, 4),
            "old_log_probs": torch.zeros(2, 4),
        }
        meta = {
            "cu_seqlens": torch.tensor([0, 4, 8], dtype=torch.int32),
            "calculate_entropy": True,
            "loss_mask": torch.ones(2, 4, dtype=torch.bool),
            "old_log_probs_shifted": torch.zeros(2, 4),
            "pad_token_id": 0,
            "actor_config": {"entropy_coeff": 0.1},
        }
        run_pipeline(
            engine,
            (),
            batch,
            meta,
            {"loss_fn": None, "post": ["identity"], "config": {}},
            "cpu",
            backward=False,
            pack=False,
        )
        self.assertIsNotNone(engine.last_kwargs)
        self.assertNotIn("advantages", engine.last_kwargs)
        self.assertNotIn("loss_mask", engine.last_kwargs)
        self.assertNotIn("actor_config", engine.last_kwargs)
        self.assertIn("input_ids", engine.last_kwargs)
        self.assertIn("calculate_entropy", engine.last_kwargs)

    def test_scale_wrt_gas_still_false(self):
        engine = _StubEngine()
        batch = {
            "input_ids": torch.arange(4).view(1, 4),
            "attention_mask": torch.ones(1, 4),
            "position_ids": torch.arange(4).view(1, 4),
        }
        meta = {
            "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
            "loss_mask": torch.ones(1, 4, dtype=torch.bool),
            "advantages": torch.ones(1, 4),
            "old_log_probs_shifted": torch.zeros(1, 4),
        }
        run_pipeline(
            engine,
            (),
            batch,
            meta,
            {"loss_fn": "ap_grpo", "post": [], "config": {}},
            "cpu",
            backward=True,
            pack=False,
        )
        self.assertEqual(engine.backward_scale, [False])


class TestPackedInnerCall(TestCasePlus):
    def test_packing_path_no_typeerror_and_filters_forward(self):
        engine = _StubEngine()
        batch = {
            "input_ids": torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
            "advantages": torch.ones(2, 4),
            "loss_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
            "old_log_probs_shifted": torch.zeros(2, 4),
        }
        meta = {"pad_token_id": 0, "calculate_entropy": False}
        out = run_pipeline(
            engine,
            (),
            batch,
            meta,
            {"loss_fn": "ap_grpo", "post": [], "config": {}},
            "cpu",
            backward=True,
            pack=True,
            max_tokens_per_mb=3,
        )
        self.assertIn("avg_loss", out)
        self.assertNotIn("advantages", engine.last_kwargs)
        self.assertNotIn("loss_mask", engine.last_kwargs)
        self.assertIn("input_ids", engine.last_kwargs)
        self.assertIn("position_ids", engine.last_kwargs)
        self.assertIn("use_cache", engine.last_kwargs)


class TestGrpoContracts(TestCasePlus):
    def _ctx(self):
        return {
            "old_log_probs_shifted": torch.zeros(2, 3),
            "advantages": torch.ones(2, 3),
            "loss_mask": torch.ones(2, 3, dtype=torch.bool),
        }

    def _out(self):
        return {"logprobs": torch.zeros(2, 3)}

    def test_echo_keys_rejected_on_grpo(self):
        with self.assertRaises(ValueError):
            grpo_loss(self._out(), self._ctx(), {}, {"aux_ce_weight": 0.1}, "cpu")

    def test_echo_v1_requires_keys(self):
        with self.assertRaises(ValueError):
            grpo_echo_v1_loss(self._out(), self._ctx(), {}, {}, "cpu")

    def test_cispo_cpu(self):
        loss, metrics = grpo_loss(self._out(), self._ctx(), {}, {"use_cispo_loss": True}, "cpu")
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("clip_ratio", metrics)
        self.assertNotIn("loss", metrics)

    def test_dp_size_from_meta_not_config(self):
        outputs = {"logprobs": torch.zeros(2, 3)}
        tensors = self._ctx()
        loss_local, _ = grpo_loss(outputs, tensors, {}, {}, "cpu")
        loss_dp, _ = grpo_loss(outputs, tensors, {"dp_size": 4, "batch_num_tokens": 6}, {}, "cpu")
        # token-mean * dp_size / batch_num_tokens vs local count=6, dp=1
        self.assertNotAlmostEqual(loss_local.item(), loss_dp.item(), places=6)

    def test_nonfinite_logprobs_sanitized(self):
        outputs = {"logprobs": torch.tensor([[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]])}
        loss, metrics = grpo_loss(outputs, self._ctx(), {}, {}, "cpu")
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(metrics["entropy"] == metrics["entropy"])


class TestPostChainWithinCall(TestCasePlus):
    def test_later_post_sees_earlier_post_meta_without_mutating_caller(self):
        seen = {}

        @register_post_processor("_step0_write_flag")
        def _write_flag(model_outputs, batch, meta, device):
            return {"chain_flag": 7}

        @register_post_processor("_step0_read_flag")
        def _read_flag(model_outputs, batch, meta, device):
            seen["chain_flag"] = meta.get("chain_flag")
            seen["from_model"] = model_outputs.get("chain_flag")
            return {}

        engine = _StubEngine()
        caller_meta = {
            "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
            "pad_token_id": 0,
        }
        batch = {
            "input_ids": torch.arange(4).view(1, 4),
            "attention_mask": torch.ones(1, 4),
            "position_ids": torch.arange(4).view(1, 4),
        }
        try:
            run_pipeline(
                engine,
                (),
                batch,
                caller_meta,
                {"loss_fn": None, "post": ["_step0_write_flag", "_step0_read_flag"], "config": {}},
                "cpu",
                backward=False,
                pack=False,
            )
            self.assertEqual(seen["chain_flag"], 7)
            self.assertEqual(seen["from_model"], 7)
            self.assertNotIn("chain_flag", caller_meta)
        finally:
            POST_PROCESSORS.pop("_step0_write_flag", None)
            POST_PROCESSORS.pop("_step0_read_flag", None)
