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

"""CPU tests for no-ZoRRo ``logits_optimization=memory`` primitives."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from arctic_platform.common.utils.tiled_logits import fill_logits_opt_from_worker_config
from arctic_platform.common.utils.tiled_logits import memory_logprobs_entropy_from_hidden
from arctic_platform.common.utils.tiled_logits import sync_logits_num_shards
from arctic_platform.common.utils.tiled_logits import tiled_logprobs_entropy_from_hidden
from arctic_platform.rl.processors.pipeline import compute_entropy_and_logprobs_post
from arctic_platform.rl.processors.pipeline import run_pipeline
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import make_tied_lm_head_model
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close


class TestSyncLogitsNumShards(TestCasePlus):
    def test_noop_without_process_group(self):
        self.assertEqual(sync_logits_num_shards(7, torch.device("cpu")), 7)


class TestMemoryLogprobsFromHidden(TestCasePlus):
    def test_matches_untiled_when_one_shard(self):
        set_seed(0)
        hidden_size, vocab_size = 8, 16
        model = make_tied_lm_head_model(hidden_size, vocab_size, device="cpu")
        hidden = torch.randn(2, 4, hidden_size)
        labels = torch.randint(0, vocab_size, (2, 4))
        lp_none, ent_none = tiled_logprobs_entropy_from_hidden(
            model, hidden, labels, temperature=1.0, calculate_entropy=True
        )
        lp_mem, ent_mem = memory_logprobs_entropy_from_hidden(
            model,
            hidden,
            labels,
            temperature=1.0,
            calculate_entropy=True,
            peak_mem_gib=8.0,  # one shard
        )
        torch_assert_close(lp_mem, lp_none)
        torch_assert_close(ent_mem, ent_none)
        self.assertEqual(tuple(lp_mem.shape), tuple(labels.shape))

    def test_matches_untiled_with_many_shards(self):
        set_seed(0)
        hidden_size, vocab_size = 8, 16
        model = make_tied_lm_head_model(hidden_size, vocab_size, device="cpu")
        hidden = torch.randn(12, hidden_size)
        labels = torch.randint(0, vocab_size, (12,))
        lp_none, _ = tiled_logprobs_entropy_from_hidden(
            model, hidden, labels, temperature=1.0, calculate_entropy=False
        )
        lp_mem, _ = memory_logprobs_entropy_from_hidden(
            model,
            hidden,
            labels,
            temperature=1.0,
            calculate_entropy=False,
            peak_mem_gib=1e-9,  # force many shards (1 row per tile)
        )
        torch_assert_close(lp_mem, lp_none)

    def test_untied_embeddings_rejected(self):
        model = make_tied_lm_head_model(8, 16, device="cpu")
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        hidden = torch.randn(2, 8)
        labels = torch.randint(0, 16, (2,))
        with self.assertRaises(ValueError) as ctx:
            memory_logprobs_entropy_from_hidden(model, hidden, labels, calculate_entropy=False)
        self.assertIn("lm_head.weight", str(ctx.exception))

    def test_missing_input_embedding_rejected(self):
        model = make_tied_lm_head_model(8, 16, device="cpu")
        model.get_input_embeddings = None
        model.embed_tokens = None
        hidden = torch.randn(2, 8)
        labels = torch.randint(0, 16, (2,))
        with self.assertRaises(ValueError) as ctx:
            memory_logprobs_entropy_from_hidden(model, hidden, labels, calculate_entropy=False)
        self.assertIn("input embeddings", str(ctx.exception))

    def test_more_shards_than_rows_matches_untiled(self):
        set_seed(0)
        hidden_size, vocab_size = 8, 16
        model = make_tied_lm_head_model(hidden_size, vocab_size, device="cpu")
        labels = torch.randint(0, vocab_size, (2,))
        base = torch.randn(2, hidden_size)
        h_ref = base.detach().clone().requires_grad_(True)
        h_mem = base.detach().clone().requires_grad_(True)
        model.lm_head.weight.grad = None
        lp_ref, _ = tiled_logprobs_entropy_from_hidden(model, h_ref, labels, temperature=1.0, calculate_entropy=False)
        lp_ref.sum().backward()
        hidden_grad_ref = h_ref.grad.detach().clone()
        weight_grad_ref = model.lm_head.weight.grad.detach().clone()

        from arctic_platform.common.utils import tiled_logits as tiled_mod

        row_counts = []

        def _count(_module, inputs):
            row_counts.append(int(inputs[0].shape[0]))

        handle = model.lm_head.register_forward_pre_hook(_count)
        original_sync = tiled_mod.sync_logits_num_shards
        tiled_mod.sync_logits_num_shards = lambda num_shards, device: 5
        model.lm_head.weight.grad = None
        try:
            lp_mem, _ = memory_logprobs_entropy_from_hidden(
                model,
                h_mem,
                labels,
                temperature=1.0,
                calculate_entropy=False,
                peak_mem_gib=8.0,
            )
            lp_mem.sum().backward()
        finally:
            tiled_mod.sync_logits_num_shards = original_sync
            handle.remove()

        torch_assert_close(lp_mem, lp_ref.detach())
        torch_assert_close(h_mem.grad, hidden_grad_ref)
        torch_assert_close(model.lm_head.weight.grad, weight_grad_ref)
        self.assertIn(0, row_counts)
        self.assertGreaterEqual(row_counts.count(0), 3)


class TestComputeEntropyMemoryRaise(TestCasePlus):
    def test_memory_with_logits_still_raises(self):
        logits = torch.randn(1, 4, 8)
        batch = {"input_ids": torch.arange(4).view(1, 4)}
        meta = {"logits_optimization": "memory", "calculate_entropy": False}
        with self.assertRaises(ValueError) as ctx:
            compute_entropy_and_logprobs_post({"logits": logits}, batch, meta, "cpu")
        self.assertIn("must not receive full logits", str(ctx.exception))

    def test_memory_with_logprobs_passthrough(self):
        logprobs = torch.randn(1, 4)
        entropy = torch.randn(1, 4)
        out = compute_entropy_and_logprobs_post(
            {"logprobs": logprobs, "entropy": entropy},
            {"input_ids": torch.arange(4).view(1, 4)},
            {"logits_optimization": "memory", "calculate_entropy": True},
            "cpu",
        )
        self.assertIs(out["logprobs"], logprobs)
        self.assertIs(out["entropy"], entropy)

    def test_memory_with_neither_logits_nor_logprobs_raises(self):
        with self.assertRaises(ValueError) as ctx:
            compute_entropy_and_logprobs_post(
                {},
                {"input_ids": torch.arange(4).view(1, 4)},
                {"logits_optimization": "memory", "calculate_entropy": False},
                "cpu",
            )
        self.assertIn("neither logits nor logprobs", str(ctx.exception))


class _Backbone(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids=None, **kwargs):
        b, s = input_ids.shape
        hidden = torch.randn(b, s, self.hidden_size, dtype=torch.float32)
        hidden.requires_grad_(True)
        return SimpleNamespace(last_hidden_state=hidden)


class _CausalLM(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.model = _Backbone(hidden_size)
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.lm_head.weight.ds_grad_is_ready = True
        self.config = SimpleNamespace(vocab_size=vocab_size, tie_word_embeddings=True)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids=None, logits_to_keep=0, **kwargs):
        out = self.model(input_ids=input_ids, **kwargs)
        hidden = out.last_hidden_state
        keep = logits_to_keep if logits_to_keep else hidden.shape[1]
        logits = self.lm_head(hidden[:, -keep:, :])
        return SimpleNamespace(logits=logits)


class _Engine:
    global_rank = 0

    def __init__(self, module):
        self.module = module
        self.last_kwargs = None

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def __call__(self, *args, **kwargs):
        self.last_kwargs = kwargs
        return self.module(**kwargs)


class TestNoZorroPipelineMemoryRouting(TestCasePlus):
    def _batch(self):
        input_ids = torch.arange(8).view(2, 4)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "position_ids": torch.arange(4).unsqueeze(0).expand(2, -1),
            "prompts": input_ids[:, :0],
        }

    def _meta(self, opt):
        return {
            "zorro_train_enable": False,
            "pad_token_id": 0,
            "temperature": 1.0,
            "calculate_entropy": False,
            "logits_optimization": opt,
            "logits_optimization_peak_mem_size_in_gib": 8,
        }

    def test_memory_skips_full_logits_and_sets_logprobs(self):
        engine = _Engine(_CausalLM(8, 16))
        out = run_pipeline(
            engine,
            (),
            self._batch(),
            self._meta("memory"),
            {"post": ["apply_temperature", "compute_entropy_and_logprobs"], "loss_fn": None},
            "cpu",
            backward=False,
            pack=False,
            return_tensors=True,
        )
        self.assertEqual(engine.last_kwargs.get("logits_to_keep"), 1)
        self.assertIn("logprobs", out["batch"])
        self.assertNotIn("logits", out["batch"])
        self.assertEqual(tuple(out["batch"]["logprobs"].shape), (2, 4))
        self.assertEqual(len(engine.module.model._forward_hooks), 0)

    def test_ignored_logits_to_keep_raises(self):
        class _IgnoreKeep(_CausalLM):
            def forward(self, input_ids=None, logits_to_keep=0, **kwargs):
                out = self.model(input_ids=input_ids, **kwargs)
                logits = self.lm_head(out.last_hidden_state)
                return SimpleNamespace(logits=logits)

        engine = _Engine(_IgnoreKeep(8, 16))
        with self.assertRaises(RuntimeError) as ctx:
            run_pipeline(
                engine,
                (),
                self._batch(),
                self._meta("memory"),
                {"post": ["apply_temperature", "compute_entropy_and_logprobs"], "loss_fn": None},
                "cpu",
                backward=False,
                pack=False,
                return_tensors=True,
            )
        self.assertIn("logits_to_keep=1", str(ctx.exception))
        self.assertEqual(len(engine.module.model._forward_hooks), 0)

    def test_none_keeps_logits_path(self):
        engine = _Engine(_CausalLM(8, 16))
        out = run_pipeline(
            engine,
            (),
            self._batch(),
            self._meta("none"),
            {"post": ["apply_temperature", "compute_entropy_and_logprobs"], "loss_fn": None},
            "cpu",
            backward=False,
            pack=False,
            return_tensors=True,
        )
        self.assertNotIn("logits_to_keep", engine.last_kwargs or {})
        self.assertIn("logprobs", out["batch"])

    def test_worker_fill_inherits_memory_when_the_client_omits_it(self):
        engine = _Engine(_CausalLM(8, 16))
        meta = self._meta("none")
        del meta["logits_optimization"]
        meta = fill_logits_opt_from_worker_config(
            meta,
            {"logits_optimization": "memory", "logits_optimization_peak_mem_size_in_gib": 8},
        )
        out = run_pipeline(
            engine,
            (),
            self._batch(),
            meta,
            {"post": ["apply_temperature", "compute_entropy_and_logprobs"], "loss_fn": None},
            "cpu",
            backward=False,
            pack=False,
            return_tensors=True,
        )
        self.assertEqual(engine.last_kwargs.get("logits_to_keep"), 1)
        self.assertIn("logprobs", out["batch"])
        self.assertNotIn("logits", out["batch"])

    def test_explicit_none_keeps_full_logits_against_worker_memory(self):
        engine = _Engine(_CausalLM(8, 16))
        meta = fill_logits_opt_from_worker_config(
            self._meta("none"),
            {"logits_optimization": "memory", "logits_optimization_peak_mem_size_in_gib": 8},
        )
        out = run_pipeline(
            engine,
            (),
            self._batch(),
            meta,
            {"post": ["apply_temperature", "compute_entropy_and_logprobs"], "loss_fn": None},
            "cpu",
            backward=False,
            pack=False,
            return_tensors=True,
        )
        self.assertEqual(meta["logits_optimization"], "none")
        self.assertNotIn("logits_to_keep", engine.last_kwargs or {})
        self.assertIn("logprobs", out["batch"])

    def test_memory_defaults_missing_temperature(self):
        engine = _Engine(_CausalLM(8, 16))
        meta = self._meta("memory")
        del meta["temperature"]
        out = run_pipeline(
            engine,
            (),
            self._batch(),
            meta,
            {"post": ["apply_temperature", "compute_entropy_and_logprobs"], "loss_fn": None},
            "cpu",
            backward=False,
            pack=False,
            return_tensors=True,
        )
        self.assertIn("logprobs", out["batch"])
        self.assertEqual(tuple(out["batch"]["logprobs"].shape), (2, 4))

    def test_untied_rejected_before_forward(self):
        model = _CausalLM(8, 16)
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        engine = _Engine(model)
        engine.calls = 0
        original = engine.__call__

        def _counting(*args, **kwargs):
            engine.calls += 1
            return original(*args, **kwargs)

        engine.__call__ = _counting
        with self.assertRaises(ValueError) as ctx:
            run_pipeline(
                engine,
                (),
                self._batch(),
                self._meta("memory"),
                {"post": ["apply_temperature", "compute_entropy_and_logprobs"], "loss_fn": None},
                "cpu",
                backward=False,
                pack=False,
                return_tensors=True,
            )
        self.assertIn("lm_head.weight", str(ctx.exception))
        self.assertEqual(engine.calls, 0)

    def test_peft_wrapper_still_captures_hidden(self):
        base = _CausalLM(8, 16)

        class _PeftOuter(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.peft_config = {"default": object()}

            def get_base_model(self):
                return self.inner

            def forward(self, *args, **kwargs):
                # PEFT tuners call the inner module's forward directly, skipping its hooks.
                return self.inner.forward(*args, **kwargs)

        engine = _Engine(_PeftOuter(base))
        out = run_pipeline(
            engine,
            (),
            self._batch(),
            self._meta("memory"),
            {"post": ["apply_temperature", "compute_entropy_and_logprobs"], "loss_fn": None},
            "cpu",
            backward=False,
            pack=False,
            return_tensors=True,
        )
        self.assertEqual(engine.last_kwargs.get("logits_to_keep"), 1)
        self.assertIn("logprobs", out["batch"])
        self.assertEqual(tuple(out["batch"]["logprobs"].shape), (2, 4))
        self.assertEqual(len(base.model._forward_hooks), 0)


class TestFillLogitsOptFromWorkerConfig(TestCasePlus):
    def test_missing_or_none_inherits_worker_memory(self):
        worker = {
            "logits_optimization": "memory",
            "logits_optimization_peak_mem_size_in_gib": 8,
            "logits_compute_in_fp32": True,
        }
        for meta in ({"temperature": 1.0}, {"logits_optimization": None, "temperature": 1.0}):
            filled = fill_logits_opt_from_worker_config(meta, worker)
            self.assertEqual(filled["logits_optimization"], "memory")
            self.assertEqual(filled["logits_optimization_peak_mem_size_in_gib"], 8)
            self.assertTrue(filled["logits_compute_in_fp32"])
            self.assertEqual(filled["temperature"], 1.0)
            self.assertIsNot(filled, meta)

    def test_explicit_none_is_not_overridden(self):
        meta = {"logits_optimization": "none", "temperature": 1.0}
        filled = fill_logits_opt_from_worker_config(
            meta,
            {"logits_optimization": "memory", "logits_optimization_peak_mem_size_in_gib": 8},
        )
        self.assertIs(filled, meta)
        self.assertEqual(filled["logits_optimization"], "none")
        self.assertNotIn("logits_optimization_peak_mem_size_in_gib", filled)

    def test_client_memory_wins_over_worker_none(self):
        meta = {"logits_optimization": "memory"}
        filled = fill_logits_opt_from_worker_config(meta, {"logits_optimization": "none"})
        self.assertIs(filled, meta)
        self.assertEqual(filled["logits_optimization"], "memory")

    def test_noop_when_both_none(self):
        meta = {"logits_optimization": "none"}
        filled = fill_logits_opt_from_worker_config(meta, {"logits_optimization": "none"})
        self.assertIs(filled, meta)

    def test_missing_meta_key_inherits_worker(self):
        filled = fill_logits_opt_from_worker_config({}, {"logits_optimization": "compute"})
        self.assertEqual(filled["logits_optimization"], "compute")
