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

import pytest
import torch

from arctic_platform.common.utils.batch import finalize_fwd_bwd_metrics
from arctic_platform.common.utils.batch import promote_batch_dim_to_batch
from arctic_platform.rl.processors import REQUIRES_TOKEN_LOGPROBS
from arctic_platform.rl.processors import resolve_loss
from arctic_platform.rl.processors.packed_reduction import combine_packed_metrics
from arctic_platform.rl.processors.packed_reduction import resolve_packed_loss_reduction
from arctic_platform.testing_utils import torch_assert_close


def _context() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "loss_mask": torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.bool),
        "logprob_weights": torch.tensor([[0.0, -0.25, 0.5], [0.0, 1.25, 0.0]], dtype=torch.float32),
    }


def test_weighted_logprob_sum_exact_value_gradient_and_dp_scale() -> None:
    objective = resolve_loss("weighted_logprob_sum")
    assert objective.has_capability(REQUIRES_TOKEN_LOGPROBS)

    logprobs = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], requires_grad=True)
    context = _context()
    loss, metrics = objective.loss(
        {"logprobs": logprobs},
        context,
        {},
        {"batch_num_tokens": 3.0, "dp_size": 2, "global_batch_size": 2},
        "cpu",
    )
    expected = (logprobs.detach()[context["loss_mask"]] * context["logprob_weights"][context["loss_mask"]]).sum() * 2
    assert loss.item() == pytest.approx(expected.item())
    assert metrics == {"loss.sum": expected.item() / 2, "loss.tokens": 3.0}
    loss.backward()
    assert torch.equal(
        logprobs.grad,
        context["logprob_weights"] * context["loss_mask"] * 2,
    )
    with pytest.raises(ValueError, match="requires an explicit dp_size"):
        objective.loss(
            {"logprobs": logprobs},
            context,
            {},
            {"batch_num_tokens": 3.0},
            "cpu",
        )


def test_weighted_logprob_sum_rejects_nonfinite_logprobs_outside_mask() -> None:
    logprobs = torch.zeros(2, 3)
    logprobs[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite logprob"):
        resolve_loss("weighted_logprob_sum").loss(
            model_outputs={"logprobs": logprobs.requires_grad_(True)},
            batch=_context(),
            meta={},
            config={},
            device="cpu",
        )


def test_weighted_logprob_sum_metrics_remove_dp_training_scale() -> None:
    objective = resolve_loss("weighted_logprob_sum")
    rank_results = []
    for logprob, weight in ((-2.0, 0.25), (-3.0, -0.5)):
        loss, metrics = objective.loss(
            model_outputs={"logprobs": torch.tensor([logprob], requires_grad=True)},
            batch={
                "loss_mask": torch.tensor([True]),
                "logprob_weights": torch.tensor([weight]),
            },
            meta={},
            config={"dp_size": 2},
            device="cpu",
        )
        rank_results.append({"avg_loss": loss.item(), "metrics": metrics})

    metrics, avg_loss = finalize_fwd_bwd_metrics(rank_results)
    assert metrics["loss"] == pytest.approx(0.5)
    assert avg_loss == pytest.approx(0.5)


def test_weighted_logprob_sum_packed_reduction_is_additive() -> None:
    first = _context()
    second = {
        "input_ids": torch.tensor([[7, 8]]),
        "loss_mask": torch.tensor([[1, 0]], dtype=torch.bool),
        "logprob_weights": torch.tensor([[0.75, 0.0]], dtype=torch.float32),
    }
    reduction = resolve_packed_loss_reduction(
        {
            "loss_fn": "weighted_logprob_sum",
            "config": {"batch_num_tokens": 4.0, "dp_size": 1},
        },
        [first, second],
    )
    assert reduction.loss_scales == (1.0, 1.0)
    assert reduction.reporting_weights == (3.0, 1.0)
    assert reduction.loss_is_additive is True


def test_weighted_logprob_sum_accepts_packed_singleton_batch_dimension() -> None:
    logprobs = torch.tensor([[-2.0, -3.0]], requires_grad=True)
    weights = torch.tensor([[0.25, 0.0]], dtype=torch.float32)
    loss, _ = resolve_loss("weighted_logprob_sum").loss(
        {"logprobs": logprobs},
        {
            "loss_mask": torch.tensor([[True, False]]),
            "logprob_weights": weights,
            "cu_seqlens": torch.tensor([0, 2]),
        },
        {},
        {},
        "cpu",
    )

    loss.backward()
    assert loss.item() == pytest.approx(-0.5)
    assert torch.equal(logprobs.grad, weights)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda frame: frame.update(logprob_weights=frame["logprob_weights"].half()),
            "float32",
        ),
        (
            lambda frame: frame["logprob_weights"].__setitem__((0, 0), 1.0),
            "zero outside loss_mask",
        ),
        (
            lambda frame: frame["logprob_weights"].__setitem__((0, 1), float("nan")),
            "must be finite",
        ),
        (
            lambda frame: frame.update(loss_mask=frame["loss_mask"].float() * 0.5),
            "must be binary",
        ),
    ],
)
def test_weighted_logprob_sum_rejects_invalid_frames(mutation, match: str) -> None:
    context = _context()
    mutation(context)
    with pytest.raises(ValueError, match=match):
        resolve_loss("weighted_logprob_sum").loss(
            {"logprobs": torch.zeros(2, 3, requires_grad=True)},
            context,
            {},
            {},
            "cpu",
        )


def test_logprob_weights_are_dp_sharded_with_other_batch_tensors() -> None:
    frame = _context()
    batch, meta = promote_batch_dim_to_batch(
        {"input_ids": frame["input_ids"]},
        {
            "loss_mask": frame["loss_mask"],
            "logprob_weights": frame["logprob_weights"],
            "pad_token_id": 0,
        },
    )
    assert "logprob_weights" in batch
    assert "logprob_weights" not in meta
    assert meta == {"pad_token_id": 0}


def test_dpo_gradient_matches_when_pair_crosses_dp_shards() -> None:
    """Distributed weighted backward must equal full-pair DPO backward."""
    chosen, rejected = 0, 1
    beta = 0.1
    dp_size = 2

    def token_logprobs(theta: torch.Tensor) -> torch.Tensor:
        return torch.stack([-1.0 + 2 * theta, -1.0 - theta])

    reference_logprobs = token_logprobs(torch.tensor(0.0))

    def dpo_loss(logprobs: torch.Tensor) -> torch.Tensor:
        margin = beta * (
            (logprobs[chosen] - reference_logprobs[chosen]) - (logprobs[rejected] - reference_logprobs[rejected])
        )
        return -torch.nn.functional.logsigmoid(margin)

    # 1. Baseline: backpropagate DPO with the complete pair.
    theta = torch.tensor(0.2, requires_grad=True)
    policy_logprobs = token_logprobs(theta)
    full_pair_dpo_gradient = torch.autograd.grad(dpo_loss(policy_logprobs), theta)[0]

    # 2. Client: convert DPO into fixed dL/dlogprob weights before sharding.
    client_logprobs = policy_logprobs.detach().requires_grad_(True)
    client_logprob_weights = torch.autograd.grad(dpo_loss(client_logprobs), client_logprobs)[0]

    # 3. Server: In cases where chosen and rejected land on separate model replicas.
    objective = resolve_loss("weighted_logprob_sum")

    def replica_gradient(response: int) -> torch.Tensor:
        replica_theta = theta.detach().clone().requires_grad_(True)
        replica_logprob = token_logprobs(replica_theta)[response : response + 1]
        replica_loss, _ = objective.loss(
            model_outputs={"logprobs": replica_logprob},
            batch={
                "loss_mask": torch.ones_like(replica_logprob, dtype=torch.bool),
                "logprob_weights": client_logprob_weights[response : response + 1],
            },
            meta={},
            config={"dp_size": dp_size},
            device="cpu",
        )
        return torch.autograd.grad(replica_loss, replica_theta)[0]

    chosen_replica_gradient = replica_gradient(chosen)
    rejected_replica_gradient = replica_gradient(rejected)

    # 4. Simulate DeepSpeed's cross-replica gradient average; no engine is started.
    distributed_gradient = (chosen_replica_gradient + rejected_replica_gradient) / dp_size
    torch_assert_close(
        distributed_gradient,
        full_pair_dpo_gradient,
        rtol=0,
        atol=1e-7,
    )


def test_weighted_loss_metrics_add_across_unequal_packed_microbatches() -> None:
    resolve_loss("weighted_logprob_sum")
    combined = combine_packed_metrics(
        [{"loss.sum": 1.0, "loss.tokens": 1.0}, {"loss.sum": 9.0, "loss.tokens": 3.0}],
        (1.0, 3.0),
    )
    assert combined == {"loss.sum": 10.0, "loss.tokens": 4.0}
    assert combined["loss.sum"] / combined["loss.tokens"] == 2.5


@pytest.mark.parametrize("active_counts", [(1, 1, 3, 0), (0, 0, 0, 0)])
def test_cortex_response_pools_weighted_metrics_across_packed_workers(active_counts) -> None:
    objective = resolve_loss("weighted_logprob_sum")
    worker_results = []
    for worker in range(2):
        packed_metrics = []
        reporting_weights = []
        worker_loss = 0.0
        for index in range(worker * 2, worker * 2 + 2):
            count = active_counts[index]
            score, weight = ((-2.0, 0.5), (-3.0, -0.5), (-4.0, 0.25), (-5.0, 0.75))[index]
            mask = torch.arange(3) < count
            loss, metrics = objective.loss(
                {"logprobs": torch.full((3,), score, requires_grad=True)},
                {"loss_mask": mask, "logprob_weights": mask.float() * weight},
                {},
                {"dp_size": 2},
                "cpu",
            )
            worker_loss += loss.item()
            packed_metrics.append(metrics)
            reporting_weights.append(float(count))
        worker_results.append(
            {"loss": worker_loss, "metrics": combine_packed_metrics(packed_metrics, reporting_weights)}
        )

    # Cortex initializes metrics from worker zero, then invokes the loss callback.
    response_metrics = dict(worker_results[0]["metrics"], unrelated_metric=7.0)
    objective.metrics_callback([result["metrics"] for result in worker_results], response_metrics)
    expected_sum = -2.5 if any(active_counts) else 0.0
    expected_tokens = float(sum(active_counts))
    assert response_metrics == {
        "loss.sum": expected_sum,
        "loss.tokens": expected_tokens,
        "loss": expected_sum / expected_tokens if expected_tokens else 0.0,
        "unrelated_metric": 7.0,
    }
    assert sum(result["loss"] for result in worker_results) / 2 == expected_sum
