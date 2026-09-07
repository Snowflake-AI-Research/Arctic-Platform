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
from arctic_platform.integrations.trl_distill.jsd import generalized_jsd
from arctic_platform.rl.processors.gather_logits import gather_logits_at_ids_post
from arctic_platform.rl.processors.gather_logits import weighted_gathered_logit_sum
from arctic_platform.rl.processors.pipeline import padded_tensor_2d_dict_to_unpadded_tensor_1d_dict
from arctic_platform.rl.processors.pipeline import unpadded_tensor_1d_dict_to_padded_tensor_2d_dict


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
    assert batch["prompts"].tolist() == [[1, 2]]
    assert batch["loss_mask"].tolist() == [[False, True, True, False]]
    assert batch["gather_token_ids"][0, 1].tolist() == [3, 9]
    assert batch["gather_token_ids"][0, 2].tolist() == [4, 8]


def test_rollout_sample_exposes_trl_aliases():
    sample = _sample()
    assert sample.input_ids == [1, 2, 3, 4]
    assert sample.completion_mask == [0, 0, 1, 1]
    assert sample.teacher_topk_ids == [[], [], [3, 9], [4, 8]]
    assert sample.teacher_id == "default"


def test_gather_returns_logsumexp_and_surrogate_scales():
    logits = torch.zeros(1, 2, 5)
    logits[0, 0, 3] = 1.5
    logits[0, 0, 1] = 0.25
    ids = torch.tensor([[[3, 1], [0, 0]]])
    out = gather_logits_at_ids_post({"logits": logits}, {"gather_token_ids": ids}, {}, "cpu")
    assert out["gathered_logits"][0, 0].tolist() == [1.5, 0.25]
    assert out["logit_logsumexp"].shape == (1, 2)
    weights = torch.zeros_like(out["gathered_logits"])
    weights[0, 0, 0] = 2.0
    loss, metrics = weighted_gathered_logit_sum(
        out, {"logit_weights": weights}, {"dp_size": 2}, {}, "cpu"
    )
    assert float(loss) == 6.0
    assert metrics["gathered_logit_sum"] == 6.0


def test_pipeline_unpad_roundtrip_keeps_gather_support_dim():
    mask = torch.tensor([[True, True, False], [True, False, False]])
    gather = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    lse = torch.arange(6, dtype=torch.float).reshape(2, 3)
    prompts = torch.tensor([[9, 8], [7, 6]])
    packed = padded_tensor_2d_dict_to_unpadded_tensor_1d_dict(
        {
            "gather_token_ids": gather.clone(),
            "logit_logsumexp": lse.clone(),
            "prompts": prompts.clone(),
        },
        mask,
    )
    assert tuple(packed["gather_token_ids"].shape) == (1, 3, 4)
    assert tuple(packed["logit_logsumexp"].shape) == (1, 3)
    assert packed["prompts"].tolist() == prompts.tolist()
    restored = unpadded_tensor_1d_dict_to_padded_tensor_2d_dict(
        {
            "gathered_logits": packed["gather_token_ids"].float(),
            "logit_logsumexp": packed["logit_logsumexp"],
        },
        mask,
        pad_value=0,
    )
    assert restored["gathered_logits"][mask].tolist() == gather[mask].float().tolist()
    assert restored["logit_logsumexp"][mask].tolist() == lse[mask].tolist()


def test_jsd_is_zero_when_student_matches_teacher():
    logits = torch.tensor([[[1.0, 0.0], [0.5, -0.5]]])
    teacher = torch.log_softmax(logits, dim=-1)
    lse = torch.logsumexp(logits.float(), dim=-1)
    mask = torch.tensor([[True, True]])
    for add_tail in (True, False):
        loss = generalized_jsd(
            logits, teacher, mask, beta=0.0, add_tail_bucket=add_tail, student_logit_logsumexp=lse
        )
        assert torch.isfinite(loss)
        assert float(loss.abs()) < 1e-5


def test_jsd_requires_logsumexp():
    logits = torch.zeros(1, 1, 2)
    teacher = torch.tensor([[[-0.2, -1.0]]])
    mask = torch.tensor([[True]])
    try:
        generalized_jsd(logits, teacher, mask)
    except ValueError as exc:
        assert "student_logit_logsumexp" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_surrogate_matches_autograd_through_full_vocab():
    torch.manual_seed(0)
    hidden = torch.randn(1, 3, 4)
    weight = torch.randn(4, 8, requires_grad=True)
    ids = torch.tensor([[[0, 1, 2], [1, 2, 3], [0, 2, 4]]])
    teacher = torch.log_softmax(torch.randn(1, 3, 3), dim=-1)
    mask = torch.tensor([[True, True, False]])

    logits = hidden @ weight
    gathered = torch.gather(logits, dim=-1, index=ids)
    lse = torch.logsumexp(logits.float(), dim=-1)
    loss = generalized_jsd(gathered, teacher, mask, student_logit_logsumexp=lse)
    loss.backward()
    true_grad = weight.grad.detach().clone()

    weight_2 = weight.detach().clone().requires_grad_(True)
    logits_2 = hidden @ weight_2
    gathered_2 = torch.gather(logits_2, dim=-1, index=ids)
    lse_2 = torch.logsumexp(logits_2.float(), dim=-1)
    leaf_g = gathered_2.detach().requires_grad_(True)
    leaf_lse = lse_2.detach().requires_grad_(True)
    loss_2 = generalized_jsd(leaf_g, teacher, mask, student_logit_logsumexp=leaf_lse)
    grad_g, grad_lse = torch.autograd.grad(loss_2, (leaf_g, leaf_lse))
    surrogate = (gathered_2 * grad_g.detach()).sum() + (lse_2 * grad_lse.detach()).sum()
    surrogate.backward()
    torch.testing.assert_close(weight_2.grad, true_grad, rtol=1e-5, atol=1e-5)


class FakeTrainClient:
    def __init__(self, gathered: torch.Tensor, logsumexp: torch.Tensor | None = None):
        self.gathered = gathered
        self.logsumexp = logsumexp if logsumexp is not None else torch.zeros(gathered.shape[:2])
        self.fwd_bwd_batches: list[dict] = []
        self.steps = 0

    def fwd_no_grad(self, batch, processing=None, meta=None):
        assert processing["loss_fn"] is None
        assert "gather_logits_at_ids" in processing["post"]
        assert "prompts" in batch
        assert meta["pad_token_id"] == 0
        return {"batch": {"gathered_logits": self.gathered.clone(), "logit_logsumexp": self.logsumexp.clone()}}

    def fwd_bwd(self, batch, processing=None, meta=None):
        assert processing["loss_fn"] == "weighted_gathered_logit_sum"
        self.fwd_bwd_batches.append(batch)
        return {"metrics": {}}

    def step(self, learning_rate=None):
        self.steps += 1
        return {"metrics": {"learning_rate": learning_rate}}


def test_training_client_ships_logit_and_lse_weights():
    gathered = torch.zeros(1, 4, 2)
    gathered[0, 1, 0] = 0.5
    lse = torch.zeros(1, 4)
    backend = FakeTrainClient(gathered, lse)
    client = ArcticOPDTrainingClient(backend)

    def loss_fn(logits, teacher, mask, *, student_logit_logsumexp):
        return (logits[mask].sum()) * 2.0 + 0.0 * student_logit_logsumexp.sum()

    out = client.forward_samples([_sample()], loss_fn, pad_token_id=0)
    out.loss.backward()
    assert len(backend.fwd_bwd_batches) == 1
    weights = backend.fwd_bwd_batches[0]["logit_weights"]
    assert weights[0, 1, 0].item() == 2.0
    assert "logsumexp_weights" in backend.fwd_bwd_batches[0]


def test_protocol_forward_backward_uses_teacher_support():
    gathered = torch.zeros(1, 3, 2)
    gathered[0, 1, 0] = 1.0
    gathered[0, 1, 1] = 0.0
    lse = torch.logsumexp(torch.stack([gathered, torch.zeros_like(gathered)], dim=-1), dim=-1)
    # Use a consistent lse for [B,S]
    lse = torch.zeros(1, 3)
    backend = FakeTrainClient(gathered, lse)
    client = ArcticOPDTrainingClient(backend, pad_token_id=0)
    input_ids = torch.tensor([[1, 2, 3]])
    position_ids = torch.tensor([[0, 1, 2]])
    token_mask = torch.tensor([False, True])
    target_ids = torch.tensor([[2, 7]])
    teacher = torch.log_softmax(torch.tensor([[1.0, 0.0]]), dim=-1)
    candidate_mask = torch.tensor([[True, True]])
    out = client.forward_backward(
        object(),
        input_ids,
        position_ids,
        token_mask,
        target_ids,
        teacher,
        candidate_mask,
        None,
        beta=0.0,
        teacher_temperature=1.0,
        add_tail_bucket=True,
        num_teachers=1,
    )
    out.loss.backward()
    assert len(backend.fwd_bwd_batches) == 1
    shipped = backend.fwd_bwd_batches[0]
    assert shipped["gather_token_ids"][0, 1].tolist() == [2, 7]
    assert "logsumexp_weights" in shipped
    assert "logit_weights" in shipped


def test_optimizer_steps_remote():
    backend = FakeTrainClient(torch.zeros(1, 1, 1))
    opt = ArcticOPDOptimizer(backend)
    opt.zero_grad()
    assert opt.step(1e-5)["metrics"]["learning_rate"] == 1e-5
    assert backend.steps == 1
