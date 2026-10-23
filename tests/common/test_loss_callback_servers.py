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

"""Loss callbacks at the real Ray forward/backward boundaries."""

from __future__ import annotations

import asyncio
import pickle

import pytest
import torch

from arctic_platform.rl.processors import BaseLoss


def _request() -> dict:
    return {
        "kwargs": {
            "input_ids": torch.arange(8).reshape(4, 2),
            "attention_mask": torch.ones(4, 2, dtype=torch.long),
        },
        "context": {
            "kd_mask": torch.tensor([[1.0, 0.0]]).expand(4, -1).clone(),
            "loss_mask": torch.tensor([[True, False]]).expand(4, -1).clone(),
        },
        "processing": {
            "loss_fn": "grpo",
            "config": {"kd_coef": 0.5},
        },
    }


class _RecordingLoss:
    def __init__(self, loss, events):
        self.loss = loss
        self.events = events

    def batching_callback(self, request):
        self.events.append("batching")
        self.loss.batching_callback(request)

    def metrics_callback(self, worker_metrics, metrics):
        assert self.events[0] == "batching"
        assert sum(event.startswith("worker-") for event in self.events) == 2
        self.events.append("metrics")
        self.loss.metrics_callback(worker_metrics, metrics)
        metrics["callback_probe_sum"] = sum(worker["probe"] for worker in worker_metrics)

    def reporting_callback(self, worker_metrics, metrics, avg_loss):
        return self.loss.reporting_callback(worker_metrics, metrics, avg_loss)


class _SPLogicalLoss(_RecordingLoss):
    def metrics_callback(self, worker_metrics, metrics):
        self.loss.metrics_callback(worker_metrics, metrics)
        self.events.append(("metrics", len(worker_metrics)))

    def reporting_callback(self, worker_metrics, metrics, avg_loss):
        self.events.append(("reporting", len(worker_metrics)))
        return self.loss.reporting_callback(worker_metrics, metrics, avg_loss)


def _sp_execution_server(monkeypatch):
    import arctic_platform.common.ray_server as ray_server
    import arctic_platform.rl.processors.base_loss as base_loss_module

    events = []
    loss = _SPLogicalLoss(base_loss_module.resolve_loss("grpo"), events)
    monkeypatch.setattr(base_loss_module, "resolve_loss", lambda name: loss)
    groups = ((2, 0), (3, 1))
    records = {
        rank: {"rank": rank, "world_size": 4, "sp_size": 2, "sp_rank": local, "sp_ranks": list(group)}
        for group in groups
        for local, rank in enumerate(group)
    }
    server = object.__new__(ray_server.ArcticRLRayServer)
    server.jobs = {1: {"job_type": "training", "sp_size": 2}}

    class Membership:
        def __init__(self, rank):
            self.rank = rank

        def remote(self):
            events.append(("membership", self.rank))
            return dict(records[self.rank])

    class Model:
        def __init__(self, rank):
            self.rank = rank

        def remote(self, shard):
            events.append(("worker", self.rank))
            assert sum(isinstance(event, tuple) and event[0] == "membership" for event in events) == 4
            assert shard["meta"]["dp_size"] == 2
            return {
                "_sequence_parallel_info": dict(records[self.rank]),
                "avg_loss": 1.0 if self.rank in (3, 1) else 2.0,
                "metrics": {"token_count": float(shard["batch"]["input_ids"].shape[0])},
                "batch": {"output": shard["batch"]["input_ids"]},
            }

    server.training_workers = [
        type(
            "Worker",
            (),
            {
                "get_sequence_parallel_info": Membership(rank),
                "forward_backward": Model(rank),
                "forward_no_grad": Model(rank),
            },
        )()
        for rank in (1, 2, 3, 0)
    ]
    monkeypatch.setattr(ray_server.ray, "get", lambda refs: refs)
    request = {
        "batch": {
            "input_ids": torch.arange(20).reshape(5, 4),
            "attention_mask": torch.ones(5, 4, dtype=torch.long),
            "loss_mask": torch.tensor([[1, 1, 1, 0]]).expand(5, -1).clone(),
            "advantages": torch.ones(5, 4),
        },
        "meta": {"return_fwd_batch": True},
        "processing": {"loss_fn": "grpo", "post": ["compute_logprobs"], "config": {"loss_agg_mode": "token-mean"}},
    }
    return server, records, request, events


@pytest.mark.parametrize("backward", [True, False])
def test_sp_execution_ray_collects_logical_callbacks_and_rows_once(monkeypatch, backward):
    server, _, request, events = _sp_execution_server(monkeypatch)
    method = server.forward_backward if backward else server.forward
    result = asyncio.run(method(1, request))
    assert result["avg_loss"] == 3.0
    assert result["metrics"]["token_count"] == 5.0
    assert torch.equal(result["batch"]["output"], request["batch"]["input_ids"])
    assert events[-2:] == [("reporting", 2), ("metrics", 2)]
    assert len([event for event in events if isinstance(event, tuple) and event[0] == "worker"]) == 4


def test_sp_execution_ray_rejects_returned_membership_drift(monkeypatch):
    server, _, request, events = _sp_execution_server(monkeypatch)
    model = server.training_workers[0].forward_backward
    original = model.remote

    def changed(shard):
        result = original(shard)
        result["_sequence_parallel_info"]["rank"] = 3
        return result

    model.remote = changed
    with pytest.raises(ValueError, match="dispatched worker"):
        asyncio.run(server.forward_backward(1, request))
    assert not any(isinstance(event, tuple) and event[0] == "metrics" for event in events)


def test_sp_execution_ray_rejects_unsupported_route_before_membership_rpc(monkeypatch):
    server, _, request, events = _sp_execution_server(monkeypatch)
    request["processing"]["post"] = ["compute_entropy_and_logprobs"]
    with pytest.raises(ValueError, match="selected compute_logprobs"):
        asyncio.run(server.forward_backward(1, request))
    assert not any(isinstance(event, tuple) for event in events)


class _BatchingStateLoss(BaseLoss):
    name = "_w10_batching_state"
    events: list[str] = []

    def __init__(self):
        self.ready = False

    def batching_callback(self, request):
        self.ready = True
        self.events.append("batching")

    def validation_callback(self, context, config):
        assert self.ready
        self.events.append("validation")

    def model_forward_callback(self, model_kwargs, context, config, output_keys):
        assert self.ready
        self.events.append("model_forward")
        output_keys.append("objective_score")

    def loss(self, model_outputs, batch, meta, config, device):
        assert self.ready
        self.events.append("loss")
        return -model_outputs["objective_score"].mean(), {"state_ready": 1.0}

    def output_callback(self, model_outputs):
        assert self.ready
        self.events.append("output")
        model_outputs.pop("objective_score")

    def metrics_callback(self, worker_metrics, metrics):
        assert self.ready
        self.events.append("metrics")
        metrics["state_ready"] = sum(result["state_ready"] for result in worker_metrics)


def _patch_loss(monkeypatch, events):
    import arctic_platform.rl.processors.base_loss as base_loss_module

    loss = _RecordingLoss(base_loss_module.resolve_loss("grpo"), events)
    monkeypatch.setattr(base_loss_module, "resolve_loss", lambda name: loss)


def _worker_result(index, shard):
    return {
        "avg_loss": float(index + 1),
        "batch": {"output": shard["batch"]["input_ids"]},
        "metrics": {
            "probe": float(index + 1),
            "kd_sum": float(index + 1),
        },
    }


def _assert_shard(shard, index, events):
    events.append(f"worker-{index}")
    assert shard["processing"]["config"]["kd_batch_num_tokens"] == 4.0
    assert shard["batch"]["kd_mask"].shape == (2, 2)
    return _worker_result(index, shard)


def _assert_response(response, events):
    assert events[0] == "batching"
    assert set(events[1:-1]) == {"worker-0", "worker-1"}
    assert events[-1] == "metrics"
    assert response["avg_loss"] == 3.0
    assert response["metrics"]["kd_sum"] == 3.0
    assert response["metrics"]["callback_probe_sum"] == 3.0


def test_ray_forward_backward_runs_loss_callbacks_around_real_split(monkeypatch):
    import arctic_platform.common.ray_server as ray_server

    events = []
    _patch_loss(monkeypatch, events)

    class Remote:
        def __init__(self, index):
            self.index = index

        def remote(self, shard):
            return _assert_shard(shard, self.index, events)

    server = object.__new__(ray_server.ArcticRLRayServer)
    server.jobs = {1: {"job_type": "training", "sp_size": 1}}
    server.training_workers = [type("Worker", (), {"forward_backward": Remote(index)})() for index in range(2)]
    monkeypatch.setattr(ray_server.ray, "get", lambda refs: refs)

    response = asyncio.run(server.forward_backward(1, _request()))

    _assert_response(response, events)


def test_ray_forward_runs_loss_callbacks_around_real_split(monkeypatch):
    import arctic_platform.common.ray_server as ray_server

    events = []
    _patch_loss(monkeypatch, events)

    class Remote:
        def __init__(self, index):
            self.index = index

        def remote(self, shard):
            return _assert_shard(shard, self.index, events)

    server = object.__new__(ray_server.ArcticRLRayServer)
    server.jobs = {1: {"job_type": "training", "sp_size": 1}}
    server.training_workers = [type("Worker", (), {"forward_no_grad": Remote(index)})() for index in range(2)]
    monkeypatch.setattr(ray_server.ray, "get", lambda refs: refs)

    response = asyncio.run(server.forward(1, _request()))

    _assert_response(response, events)
    assert response["batch"]["output"].shape == (4, 2)


class _GroupedPolicyEngine:
    global_rank = 0

    def __init__(self, value=0.0):
        self.parameter = torch.tensor(float(value), requires_grad=True)
        self.group_shapes = []

    def gradient_accumulation_steps(self):
        return 1

    def train(self):
        pass

    def eval(self):
        pass

    def __call__(self, input_ids, group_token_ids=None, **_kwargs):
        logprobs = self.parameter.expand_as(input_ids)
        if group_token_ids is None:
            return {"logprobs": logprobs}
        self.group_shapes.append(tuple(group_token_ids.shape))
        width = group_token_ids.shape[-1]
        group_logits = torch.cat(
            (
                self.parameter.expand(*input_ids.shape, width),
                self.parameter.new_zeros(*input_ids.shape, 1),
            ),
            dim=-1,
        )
        return {
            "logprobs": logprobs,
            "group_log_probs": group_logits.log_softmax(-1),
        }

    def backward(self, loss, scale_wrt_gas=False):
        assert scale_wrt_gas is False
        loss.backward()


class _StatefulLossEngine(_GroupedPolicyEngine):
    def __call__(self, input_ids, **_kwargs):
        return {"objective_score": self.parameter.expand_as(input_ids)}


def _cpu_worker(engine):
    from arctic_platform.common.deepspeed_worker import DeepSpeedWorker

    worker_class = DeepSpeedWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_class)
    worker.rank = 0
    worker.world_size = 1
    worker.sp_size = 1
    worker.engine = engine
    worker._device = torch.device("cpu")
    worker.cpu_device = torch.device("cpu")
    return worker


def _native_grpo_kd_request():
    rows, sequence_length, width = 4, 3, 2
    attention_mask = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=torch.long,
    )
    target_mask = attention_mask.bool() & torch.roll(attention_mask.bool(), shifts=-1, dims=-1)
    target_mask[..., -1] = False
    return {
        "kwargs": {
            "input_ids": torch.ones(rows, sequence_length, dtype=torch.long),
            "attention_mask": attention_mask,
            "labels": torch.where(
                target_mask,
                torch.ones_like(attention_mask),
                torch.full_like(attention_mask, -100),
            ),
            "dss_compute_logprobs": True,
        },
        "context": {
            "pad_token_id": 0,
            "old_log_probs_shifted": torch.zeros(rows, sequence_length),
            "advantages": torch.ones(rows, sequence_length),
            "loss_mask": target_mask,
            "kd_mask": target_mask.float(),
            "teacher_token_ids": torch.arange(width).expand(rows, sequence_length, width),
            "teacher_log_probs": torch.full(
                (rows, sequence_length, width),
                torch.log(torch.tensor(0.2)),
            ),
            "teacher_tail_log_prob": torch.full((rows, sequence_length), torch.log(torch.tensor(0.6))),
        },
        "processing": {
            "loss_fn": "grpo",
            "config": {
                "use_cispo_loss": True,
                "is_weight_clip_max": 5.0,
                "kd_coef": 0.5,
            },
        },
    }


def _native_grouped_distillation_request():
    request = _native_grpo_kd_request()
    request["processing"] = {
        "loss_fn": "grouped_distillation",
        "config": {"kd_coef": 0.25},
    }
    return request


def test_ray_preserves_batching_state_through_serialized_worker_execution(monkeypatch):
    import arctic_platform.common.ray_server as ray_server

    _BatchingStateLoss.events = []
    worker = _cpu_worker(_StatefulLossEngine())

    class Remote:
        def remote(self, shard):
            worker_shard = pickle.loads(pickle.dumps(shard))
            return worker.forward_backward(worker_shard)

    server = object.__new__(ray_server.ArcticRLRayServer)
    server.jobs = {1: {"job_type": "training", "sp_size": 1}}
    server.training_workers = [type("Worker", (), {"forward_backward": Remote()})()]
    monkeypatch.setattr(ray_server.ray, "get", lambda refs: refs)
    request = {
        "batch": {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        },
        "meta": {"pad_token_id": 0},
        "processing": {"loss_fn": _BatchingStateLoss.name, "config": {}},
    }

    response = asyncio.run(server.forward_backward(1, request))

    assert response["avg_loss"] == 0.0
    assert response["metrics"]["state_ready"] == 1.0
    assert _BatchingStateLoss.events == [
        "batching",
        "validation",
        "model_forward",
        "loss",
        "output",
        "metrics",
    ]


def test_ray_grpo_kd_unpads_teacher_groups_for_multiple_rows_per_worker(monkeypatch):
    import arctic_platform.common.ray_server as ray_server

    request = _native_grpo_kd_request()
    workers = [_cpu_worker(_GroupedPolicyEngine()) for _ in range(2)]
    received = []

    class Remote:
        def __init__(self, worker):
            self.worker = worker

        def remote(self, shard):
            received.append(shard)
            return self.worker.forward_backward(shard)

    server = object.__new__(ray_server.ArcticRLRayServer)
    server.jobs = {1: {"job_type": "training", "sp_size": 1}}
    server.training_workers = [type("Worker", (), {"forward_backward": Remote(worker)})() for worker in workers]
    monkeypatch.setattr(ray_server.ray, "get", lambda refs: refs)

    response = asyncio.run(server.forward_backward(1, request))

    base_request = _native_grpo_kd_request()
    base_request["processing"]["config"].pop("kd_coef")
    base_workers = [_cpu_worker(_GroupedPolicyEngine()) for _ in range(2)]
    server.training_workers = [type("Worker", (), {"forward_backward": Remote(worker)})() for worker in base_workers]
    base_response = asyncio.run(server.forward_backward(1, base_request))

    assert received[0]["processing"]["config"]["dp_size"] is None
    assert received[0]["processing"]["config"]["kd_batch_num_tokens"] == 4.0
    assert [shard["meta"]["dp_size"] for shard in received[:2]] == [2, 2]
    assert [worker.engine.group_shapes for worker in workers] == [[(1, 5, 2)], [(1, 3, 2)]]
    assert response["metrics"]["kd_weight_sum"] == 4.0
    assert response["avg_loss"] - base_response["avg_loss"] == pytest.approx(response["metrics"]["loss_term_kd"])
    assert "_avg_loss_correction_sum" not in response["metrics"]
    assert all(torch.isfinite(worker.engine.parameter.grad) for worker in workers)


def test_ray_grouped_distillation_reports_global_objective_without_dp_gradient_scale(monkeypatch):
    import arctic_platform.common.ray_server as ray_server

    request = _native_grouped_distillation_request()
    workers = [_cpu_worker(_GroupedPolicyEngine(-2.0)) for _ in range(2)]

    class Remote:
        def __init__(self, worker):
            self.worker = worker

        def remote(self, shard):
            return self.worker.forward_backward(shard)

    server = object.__new__(ray_server.ArcticRLRayServer)
    server.jobs = {1: {"job_type": "training", "sp_size": 1}}
    server.training_workers = [type("Worker", (), {"forward_backward": Remote(worker)})() for worker in workers]
    monkeypatch.setattr(ray_server.ray, "get", lambda refs: refs)

    response = asyncio.run(server.forward_backward(1, request))

    parameter = torch.tensor(-2.0, requires_grad=True)
    teacher = torch.tensor([0.2, 0.2, 0.6])
    student_log_probs = torch.stack((parameter, parameter, parameter.new_zeros(()))).log_softmax(0)
    kd = (teacher * (teacher.log() - student_log_probs)).sum()
    objective = 0.75 * -parameter + 0.25 * kd
    (objective_grad,) = torch.autograd.grad(objective, parameter)

    assert response["avg_loss"] == pytest.approx(objective.item())
    assert "_avg_loss_correction_sum" not in response["metrics"]
    assert [worker.engine.parameter.grad.item() for worker in workers] == pytest.approx(
        [2 * 3 / 4 * objective_grad.item(), 2 * 1 / 4 * objective_grad.item()]
    )
