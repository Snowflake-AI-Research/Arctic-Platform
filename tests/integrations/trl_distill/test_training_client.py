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

from __future__ import annotations

import torch

from arctic_platform.integrations.trl_distill import ArcticOPDOptimizer
from arctic_platform.integrations.trl_distill import ArcticOPDTrainingClient
from arctic_platform.integrations.trl_distill import RolloutSample
from arctic_platform.integrations.trl_distill.batch_layout import samples_to_train_batch
from arctic_platform.rl.processors.gather_logits import gather_logits_at_ids_post
from arctic_platform.rl.processors.gather_logits import weighted_gathered_logit_sum


def _sample() -> RolloutSample:
    return RolloutSample(
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        sampler_logprobs=[-0.4, -0.5],
        teacher_token_ids=[[3, 9], [4, 8]],
        teacher_logprobs=[[-0.2, -1.0], [-0.3, -2.0]],
    )


def test_samples_to_train_batch_shifts_support():
    batch = samples_to_train_batch([_sample()], pad_token_id=0)
    assert tuple(batch["input_ids"].shape) == (1, 4)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4]]
    # completion tokens 3,4 predicted at positions 1,2 (prompt_len-1)
    assert batch["loss_mask"].tolist() == [[False, True, True, False]]
    assert batch["gather_token_ids"][0, 1].tolist() == [3, 9]
    assert batch["gather_token_ids"][0, 2].tolist() == [4, 8]


def test_gather_and_surrogate_loss():
    logits = torch.zeros(1, 2, 5)
    logits[0, 0, 3] = 1.5
    logits[0, 0, 1] = 0.25
    ids = torch.tensor([[[3, 1], [0, 0]]])
    out = gather_logits_at_ids_post({"logits": logits}, {"gather_token_ids": ids}, {}, "cpu")
    assert out["gathered_logits"][0, 0].tolist() == [1.5, 0.25]
    weights = torch.zeros_like(out["gathered_logits"])
    weights[0, 0, 0] = 2.0
    loss, metrics = weighted_gathered_logit_sum(
        out, {"logit_weights": weights}, {}, {}, "cpu"
    )
    assert float(loss) == 3.0
    assert metrics["gathered_logit_sum"] == 3.0


class FakeTrainClient:
    def __init__(self, gathered: torch.Tensor):
        self.gathered = gathered
        self.fwd_bwd_batches: list[dict] = []
        self.steps = 0

    def fwd_no_grad(self, batch, processing=None, meta=None):
        assert processing["loss_fn"] is None
        assert "gather_logits_at_ids" in processing["post"]
        return {"batch": {"gathered_logits": self.gathered.clone()}}

    def fwd_bwd(self, batch, processing=None, meta=None):
        assert processing["loss_fn"] == "weighted_gathered_logit_sum"
        self.fwd_bwd_batches.append(batch)
        return {"metrics": {}}

    def step(self, learning_rate=None):
        self.steps += 1
        return {"metrics": {"learning_rate": learning_rate}}


def test_training_client_ships_logit_weights():
    gathered = torch.zeros(1, 4, 2)
    gathered[0, 1, 0] = 0.5
    backend = FakeTrainClient(gathered)
    client = ArcticOPDTrainingClient(backend)

    def loss_fn(logits, teacher, mask):
        return (logits[mask].sum()) * 2.0

    out = client.forward_samples([_sample()], loss_fn, pad_token_id=0)
    out.loss.backward()
    assert len(backend.fwd_bwd_batches) == 1
    weights = backend.fwd_bwd_batches[0]["logit_weights"]
    assert weights[0, 1, 0].item() == 2.0


def test_protocol_forward_backward_ignores_local_model():
    gathered = torch.zeros(1, 3, 2)
    gathered[0, 0, 0] = 1.0
    backend = FakeTrainClient(gathered)
    client = ArcticOPDTrainingClient(backend, pad_token_id=0)
    client.bind_teacher_support(torch.zeros(1, 3, 2, dtype=torch.long))
    input_ids = torch.tensor([[1, 2, 3]])
    position_ids = torch.tensor([[0, 1, 2]])
    completion_mask = torch.tensor([[0, 1, 1]])

    def loss_fn(log_probs):
        return log_probs.sum()

    out = client.forward_backward(
        object(), input_ids, position_ids, completion_mask, loss_fn
    )
    out.loss.backward()
    assert len(backend.fwd_bwd_batches) == 1
    client.clear_teacher_support()


def test_optimizer_steps_remote():
    backend = FakeTrainClient(torch.zeros(1, 1, 1))
    opt = ArcticOPDOptimizer(backend)
    opt.zero_grad()
    assert opt.step(1e-5)["metrics"]["learning_rate"] == 1e-5
    assert backend.steps == 1
