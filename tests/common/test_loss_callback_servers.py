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

"""Loss callbacks at the real Ray and HTTP forward/backward boundaries."""

from __future__ import annotations

import asyncio

import torch

from arctic_platform import wire


def _request() -> dict:
    return {
        "kwargs": {
            "input_ids": torch.arange(8).reshape(4, 2),
            "attention_mask": torch.ones(4, 2, dtype=torch.long),
        },
        "context": {"kd_mask": torch.ones(4, 2)},
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
    assert shard["processing"]["config"]["kd_batch_num_tokens"] == 8.0
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


def test_http_forward_backward_runs_loss_callbacks_around_real_split(monkeypatch):
    import arctic_platform.common.http_server as http_server

    events = []
    _patch_loss(monkeypatch, events)

    class Remote:
        def __init__(self, index):
            self.index = index

        async def _call(self, shard):
            return _assert_shard(shard, self.index, events)

        def remote(self, shard):
            return self._call(shard)

    http_server.app.state.jobs = {1: {"job_type": "training", "sp_size": 1}}
    http_server.app.state.training_workers = [
        type("Worker", (), {"forward_backward": Remote(index)})() for index in range(2)
    ]

    response = asyncio.run(
        http_server.forward_backward(
            job_id=1,
            body=wire.dumps(_request()),
        )
    )

    _assert_response(wire.loads(response.body), events)


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


def test_http_forward_runs_loss_callbacks_around_real_split(monkeypatch):
    import arctic_platform.common.http_server as http_server

    events = []
    _patch_loss(monkeypatch, events)

    class Remote:
        def __init__(self, index):
            self.index = index

        async def _call(self, shard):
            return _assert_shard(shard, self.index, events)

        def remote(self, shard):
            return self._call(shard)

    http_server.app.state.jobs = {1: {"job_type": "training", "sp_size": 1}}
    http_server.app.state.training_workers = [
        type("Worker", (), {"forward_no_grad": Remote(index)})() for index in range(2)
    ]

    response = asyncio.run(
        http_server.forward(
            job_id=1,
            body=wire.dumps(_request()),
        )
    )
    decoded = wire.loads(response.body)

    _assert_response(decoded, events)
    assert decoded["batch"]["output"].shape == (4, 2)


class _GroupedPolicyEngine:
    global_rank = 0

    def __init__(self):
        self.parameter = torch.tensor(0.0, requires_grad=True)

    def gradient_accumulation_steps(self):
        return 1

    def train(self):
        pass

    def eval(self):
        pass

    def __call__(self, input_ids, group_token_ids=None, **_kwargs):
        assert group_token_ids is not None
        logprobs = self.parameter.expand_as(input_ids)
        candidate = torch.nn.functional.logsigmoid(self.parameter)
        tail = torch.nn.functional.logsigmoid(-self.parameter)
        group_log_probs = torch.stack((candidate, tail)).expand(*input_ids.shape, 2)
        return {
            "logprobs": logprobs,
            "group_log_probs": group_log_probs,
        }

    def backward(self, loss, scale_wrt_gas=False):
        assert scale_wrt_gas is False
        loss.backward()


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
    rows, sequence_length = 2, 2
    return {
        "kwargs": {
            "input_ids": torch.ones(rows, sequence_length, dtype=torch.long),
            "attention_mask": torch.ones(rows, sequence_length, dtype=torch.long),
            "labels": torch.ones(rows, sequence_length, dtype=torch.long),
            "dss_compute_logprobs": True,
        },
        "context": {
            "pad_token_id": 0,
            "old_log_probs_shifted": torch.zeros(rows, sequence_length),
            "advantages": torch.ones(rows, sequence_length),
            "loss_mask": torch.ones(rows, sequence_length, dtype=torch.bool),
            "kd_mask": torch.ones(rows, sequence_length),
            "teacher_token_ids": torch.zeros(rows, sequence_length, 1, dtype=torch.long),
            "teacher_log_probs": torch.full((rows, sequence_length, 1), torch.log(torch.tensor(0.4))),
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


def test_ray_grpo_kd_uses_native_split_dp_size_in_real_worker(monkeypatch):
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

    assert request["processing"]["config"]["dp_size"] is None
    assert request["processing"]["config"]["kd_batch_num_tokens"] == 4.0
    assert [shard["meta"]["dp_size"] for shard in received] == [2, 2]
    assert response["metrics"]["kd_weight_sum"] == 4.0
    assert torch.isfinite(torch.tensor(response["avg_loss"]))
    assert all(torch.isfinite(worker.engine.parameter.grad) for worker in workers)
