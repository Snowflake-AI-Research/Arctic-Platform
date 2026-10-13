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

# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU routing controls for models with configured chunked LM-head logprobs."""

from types import SimpleNamespace

import torch

from arctic_platform.model.implementations.gpu.lm_head import enable_chunked_lm_head_logprobs
from arctic_platform.sft.processor import run_sft_pipeline
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import torch_assert_equal


class _ChunkedEngine:
    def __init__(self, *, mapping=False, missing_logprobs=False):
        self.module = SimpleNamespace(_dss_chunked_lm_head_logprobs=True)
        self.mapping = mapping
        self.missing_logprobs = missing_logprobs
        self.parameter = torch.nn.Parameter(torch.tensor(1.25))
        self.last_kwargs = None
        self.training = None
        self.backward_scale = None

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        fields = {"loss": 9.0 * self.parameter}
        if kwargs.get("dss_compute_logprobs") and not self.missing_logprobs:
            targets = kwargs["labels"]
            fields["logprobs"] = -self.parameter.expand_as(targets) * (targets != -100)
        return fields if self.mapping else SimpleNamespace(**fields)

    def backward(self, loss, scale_wrt_gas=True):
        self.backward_scale = scale_wrt_gas
        loss.backward()


class _TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 4)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyDecoder()
        self.lm_head = torch.nn.Linear(4, 11)

    def forward(self, **kwargs):
        raise AssertionError("Configured chunked head must not invoke the dense forward")


class _RealChunkedEngine:
    def __init__(self, model):
        self.module = model

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def __call__(self, **kwargs):
        return self.module(**kwargs)

    def backward(self, loss, scale_wrt_gas=True):
        if scale_wrt_gas:
            raise AssertionError("The SFT loss is already globally normalized")
        loss.backward()


class TestSftChunkedRoute(TestCasePlus):
    def _batch(self):
        return {
            "input_ids": torch.arange(8).view(2, 4),
            "labels": torch.tensor([[-100, 1, 2, 3], [-100, 4, 5, -100]]),
            "position_ids": torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]]),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
        }

    def test_configured_logprobs_preserve_targets_normalization_and_backward(self):
        for shifted in (False, True):
            for backward in (False, True):
                for mapping in (False, True):
                    with self.subTest(shifted=shifted, backward=backward, mapping=mapping):
                        batch = self._batch()
                        original = batch["labels"].clone()
                        engine = _ChunkedEngine(mapping=mapping)
                        expected_targets = (
                            original if shifted else torch.tensor([[1, -100, 3, -100], [4, 5, -100, -100]])
                        )
                        valid_count = int((expected_targets != -100).sum())
                        result = run_sft_pipeline(
                            engine,
                            batch,
                            {"labels_are_shifted": shifted, "global_num_tokens": 10, "dp_size": 2},
                            {"loss_fn": "sft"},
                            "cpu",
                            backward=backward,
                        )
                        self.assertIs(engine.last_kwargs.get("dss_compute_logprobs"), True)
                        torch_assert_equal(engine.last_kwargs["labels"], expected_targets)
                        torch_assert_equal(batch["labels"], original)
                        self.assertEqual(result["metrics"]["loss.tokens"], valid_count)
                        self.assertEqual(result["metrics"]["loss.sum"], 1.25 * valid_count)
                        self.assertEqual(result["avg_loss"], 1.25 * valid_count / 10 * 2)
                        self.assertEqual(engine.training, backward)
                        if backward:
                            self.assertIs(engine.backward_scale, False)
                            torch_assert_equal(engine.parameter.grad, torch.tensor(valid_count / 10 * 2))
                        else:
                            self.assertIsNone(engine.parameter.grad)
                            self.assertIsNone(engine.backward_scale)

    def test_all_ignored_targets_keep_zero_model_gradient_connected(self):
        batch = self._batch()
        batch["labels"].fill_(-100)
        engine = _ChunkedEngine()
        result = run_sft_pipeline(engine, batch, {}, {"loss_fn": "sft"}, "cpu")
        self.assertIs(engine.last_kwargs.get("dss_compute_logprobs"), True)
        self.assertEqual(result["avg_loss"], 0.0)
        self.assertEqual(result["metrics"]["loss.sum"], 0.0)
        self.assertEqual(result["metrics"]["loss.tokens"], 0.0)
        torch_assert_equal(engine.parameter.grad, torch.tensor(0.0))

    def test_missing_configured_logprobs_rejected_without_dense_fallback(self):
        engine = _ChunkedEngine(missing_logprobs=True)
        with self.assertRaisesRegex(ValueError, "logprob"):
            run_sft_pipeline(engine, self._batch(), {}, {"loss_fn": "sft"}, "cpu")
        self.assertIsNone(engine.parameter.grad)

    def test_invalid_logprob_shape_rejected_before_backward(self):
        class InvalidShapeEngine(_ChunkedEngine):
            def __call__(self, **kwargs):
                outputs = super().__call__(**kwargs)
                if hasattr(outputs, "logprobs"):
                    outputs.logprobs = outputs.logprobs.reshape(-1)
                return outputs

        engine = InvalidShapeEngine()
        with self.assertRaisesRegex(ValueError, "logprob"):
            run_sft_pipeline(engine, self._batch(), {}, {"loss_fn": "sft"}, "cpu")
        self.assertIsNone(engine.parameter.grad)

    def test_real_chunked_head_matches_dense_ce_loss_and_all_gradients(self):
        for shifted in (False, True):
            for all_ignored in (False, True):
                with self.subTest(shifted=shifted, all_ignored=all_ignored), torch.random.fork_rng(devices=[]):
                    torch.manual_seed(19)
                    model, reference = _TinyModel(), _TinyModel()
                    reference.load_state_dict(model.state_dict())
                    enable_chunked_lm_head_logprobs(model, token_chunk_size=3, vocab_chunk_size=4, fp32_lm_head=True)
                    batch = self._batch()
                    if all_ignored:
                        batch["labels"].fill_(-100)
                    targets = batch["labels"].clone()
                    if not shifted:
                        targets[:, :-1] = batch["labels"][:, 1:]
                        targets[:, -1] = -100
                        targets[:, :-1].masked_fill_(batch["position_ids"][:, 1:] == 0, -100)
                    logits = reference.lm_head(reference.model(batch["input_ids"]).last_hidden_state)
                    expected_sum = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, 11), targets.reshape(-1), ignore_index=-100, reduction="sum"
                    )
                    expected_loss = expected_sum / 10 * 2
                    expected_loss.backward()
                    result = run_sft_pipeline(
                        _RealChunkedEngine(model),
                        batch,
                        {"labels_are_shifted": shifted, "global_num_tokens": 10, "dp_size": 2},
                        {"loss_fn": "sft"},
                        "cpu",
                    )
                    self.assertAlmostEqual(result["avg_loss"], expected_loss.item(), places=6)
                    self.assertAlmostEqual(result["metrics"]["loss.sum"], expected_sum.item(), places=5)
                    self.assertEqual(result["metrics"]["loss.tokens"], int((targets != -100).sum()))
                    for actual, expected in zip(model.parameters(), reference.parameters()):
                        self.assertIsNotNone(actual.grad)
                        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)

    def test_unconfigured_model_retains_model_loss_route(self):
        engine = _ChunkedEngine()
        engine.module._dss_chunked_lm_head_logprobs = False
        result = run_sft_pipeline(engine, self._batch(), {}, {"loss_fn": "sft"}, "cpu")
        self.assertNotIn("dss_compute_logprobs", engine.last_kwargs)
        self.assertEqual(result["avg_loss"], 11.25)
        torch_assert_equal(engine.parameter.grad, torch.tensor(9.0))
