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
"""Native PrimeRL chunked outputs must remain usable for SFT cross entropy."""

from types import SimpleNamespace

import pytest
import torch

from arctic_platform.model.implementations.moe.layers.lm_head import FusedCrossEntropyOutputLinear
from arctic_platform.model.implementations.moe.layers.lm_head import QuackFusedCrossEntropyOutputLinear
from arctic_platform.model.implementations.moe.layers.lm_head import inject_prime_lm_head
from arctic_platform.sft.processor import run_sft_pipeline


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 4)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(final_logit_softcapping=None)
        self.model = Decoder()
        self.lm_head = torch.nn.Linear(4, 11, bias=False)


class Engine:
    def __init__(self, model):
        self.module = model

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def __call__(self, **kwargs):
        return self.module(**kwargs)

    def backward(self, loss, scale_wrt_gas=True):
        assert scale_wrt_gas is False
        loss.backward()


@pytest.mark.parametrize("shifted", [False, True])
@pytest.mark.parametrize("ignored", [False, True])
@pytest.mark.parametrize("backward", [False, True])
def test_native_chunked_sft_matches_dense_loss_and_every_gradient(shifted, ignored, backward):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        model, reference = Model(), Model()
        reference.load_state_dict(model.state_dict())
        inject_prime_lm_head(model, chunk_size=3, fused_cross_entropy=False, fp32_lm_head=True)
        batch = {
            "input_ids": torch.arange(8).reshape(2, 4),
            "labels": torch.tensor([[-100, 1, 2, 3], [-100, 4, 5, -100]]),
            "position_ids": torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]]),
        }
        if ignored:
            batch["labels"].fill_(-100)
        original = batch["labels"].clone()
        targets = original.clone()
        if not shifted:
            targets[:, :-1] = original[:, 1:]
            targets[:, -1] = -100
            targets[:, :-1].masked_fill_(batch["position_ids"][:, 1:] == 0, -100)
        logits = reference.lm_head(reference.model(batch["input_ids"]).last_hidden_state)
        expected_sum = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 11), targets.reshape(-1), ignore_index=-100, reduction="sum"
        )
        expected_loss = expected_sum / 10 * 2
        if backward:
            expected_loss.backward()
        result = run_sft_pipeline(
            Engine(model),
            batch,
            {"labels_are_shifted": shifted, "global_num_tokens": 10, "dp_size": 2},
            {"loss_fn": "sft"},
            "cpu",
            backward=backward,
        )
        assert result["avg_loss"] == pytest.approx(expected_loss.item(), abs=1e-6)
        assert result["metrics"]["loss.sum"] == pytest.approx(expected_sum.item(), abs=1e-5)
        assert result["metrics"]["loss.tokens"] == int((targets != -100).sum())
        torch.testing.assert_close(batch["labels"], original, rtol=0, atol=0)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            if backward:
                assert actual.grad is not None
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
            else:
                assert actual.grad is None


@pytest.mark.parametrize("field", ["missing", "shape"])
def test_native_chunked_invalid_logprobs_rejected_before_backward(field):
    class InvalidModel(Model):
        def forward(self, **kwargs):
            if field == "missing":
                return {}
            return {"logprobs": torch.zeros(1, requires_grad=True)}

    model = InvalidModel()
    inject_prime_lm_head(model, chunk_size=3, fused_cross_entropy=False, fp32_lm_head=True)
    model.forward = InvalidModel.forward.__get__(model)
    batch = {"input_ids": torch.ones(1, 4, dtype=torch.long), "labels": torch.ones(1, 4, dtype=torch.long)}
    with pytest.raises(ValueError, match="logprob"):
        run_sft_pipeline(Engine(model), batch, {"labels_are_shifted": True}, {"loss_fn": "sft"}, "cpu")
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize(
    "head_class", [torch.nn.Linear, FusedCrossEntropyOutputLinear, QuackFusedCrossEntropyOutputLinear]
)
@pytest.mark.parametrize("native", [False, True])
def test_non_chunked_logprob_heads_retain_scalar_loss_route(head_class, native):
    class ScalarModel(Model):
        def forward(self, **kwargs):
            self.received = kwargs
            return {"loss": self.lm_head.weight.sum()}

    model = ScalarModel()
    # Scalar-route controls do not need the optional fused kernels to execute.
    head = head_class.__new__(head_class)
    torch.nn.Linear.__init__(head, 4, 11, bias=False)
    head.chunk_size = 3
    model.lm_head = head
    model._dss_native_lm_head_logprobs = native
    model._dss_chunked_lm_head_logprobs = True
    batch = {"input_ids": torch.ones(1, 4, dtype=torch.long), "labels": torch.tensor([[-100, 1, 2, 3]])}
    expected = model.lm_head.weight.detach().sum().item()
    result = run_sft_pipeline(Engine(model), batch, {}, {"loss_fn": "sft"}, "cpu")
    assert result["avg_loss"] == pytest.approx(expected)
    assert model.received["labels"] is batch["labels"]
    assert "dss_compute_logprobs" not in model.received
    torch.testing.assert_close(model.lm_head.weight.grad, torch.ones_like(model.lm_head.weight))
