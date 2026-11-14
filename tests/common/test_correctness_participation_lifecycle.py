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

"""Participation capture on the correctness gateway covers step 1 and leaves later optimizer steps unobserved.

The correctness gateway (``python -m arctic_platform.debug.training``) arms the capture for every training job it
creates, including multi-step checks such as ``e2e-sft-train-validate``. CPU only: a ``torch.nn.Linear`` stands in
for the model, and plain classes stand in for the DSS worker and coordinator.
"""

import asyncio
import sys
import types

import pytest
import torch

from arctic_platform.debug import training as debug_training


class FakeEngine:
    def __init__(self):
        self.module = torch.nn.Linear(4, 3)
        self.global_steps = 0


class FakeDeepSpeedWorker:
    def __init__(self):
        self._engine = FakeEngine()
        self.rank = 0
        self.job_id = "job-0"
        self.steps = 0

    def fwd_bwd(self, batch):
        loss = self._engine.module(batch).square().mean()
        loss.backward()
        return {"avg_loss": float(loss)}

    def step(self, learning_rate=None):
        with torch.no_grad():
            for parameter in self._engine.module.parameters():
                parameter -= 0.1 * parameter.grad
                parameter.grad = None
        self._engine.global_steps += 1
        self.steps += 1
        return {"global_steps": self._engine.global_steps}

    def destroy(self, remove_checkpoint_scratch=False):
        return None


def test_worker_keeps_the_first_step_receipt_across_later_steps():
    worker = debug_training.participation_worker_class(FakeDeepSpeedWorker)()
    batch = torch.ones(2, 4)

    worker.fwd_bwd(batch)
    assert worker.step(1e-3) == {"global_steps": 1}
    receipt = worker.correctness_participation_receipt()
    assert receipt["rank"] == 0 and receipt["job_id"] == "job-0"
    states = {entry["state"] for entry in receipt["after"]["participation"]["parameters"].values()}
    assert states == {"active"}
    assert worker._correctness_participation.capture.handles == []

    for expected in (2, 3):
        worker.fwd_bwd(batch)
        assert worker.step(1e-3) == {"global_steps": expected}
    assert worker.correctness_participation_receipt() == receipt
    assert worker.steps == 3


def test_worker_refuses_step_before_backward():
    worker = debug_training.participation_worker_class(FakeDeepSpeedWorker)()
    with pytest.raises(ValueError, match="requires backward before step"):
        worker.step(1e-3)


class FakeReceiptHandle:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = 0

    async def remote(self):
        self.calls += 1
        return self.receipt


class FakeWorkerActor:
    def __init__(self, receipt):
        self.correctness_participation_receipt = FakeReceiptHandle(receipt)


class FakeCoordinatorBase:
    def __init__(self, workers, steps):
        self._workers = workers
        self._info = types.SimpleNamespace(job_id="job-0")
        self._steps = iter(steps)

    async def step(self, learning_rate=None):
        return {"global_steps": next(self._steps)}


def coordinator_class(monkeypatch):
    fake_ray = types.ModuleType("ray")
    fake_ray.remote = lambda **_options: lambda cls: cls
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    fake_training = types.SimpleNamespace(
        _DeepSpeedWorker=FakeDeepSpeedWorker,
        GPUTrainingJob=types.SimpleNamespace(
            __ray_metadata__=types.SimpleNamespace(modified_class=FakeCoordinatorBase)
        ),
    )
    merged = []

    def merge(receipts, *, job_id, world_size):
        merged.append((receipts, job_id, world_size))
        return {"merged": len(receipts)}

    monkeypatch.setattr(debug_training, "merge_worker_receipts", merge)
    return debug_training.participation_coordinator_class(fake_training), merged


def test_coordinator_attaches_participation_to_step_one_only(monkeypatch):
    coordinator_cls, merged = coordinator_class(monkeypatch)
    workers = [FakeWorkerActor({"rank": rank}) for rank in range(2)]
    coordinator = coordinator_cls(workers, steps=[1, 2, 3])

    first = asyncio.run(coordinator.step(1e-3))
    assert first == {"global_steps": 1, "participation": {"merged": 2}}
    assert merged == [([{"rank": 0}, {"rank": 1}], "job-0", 2)]

    for expected in (2, 3):
        assert asyncio.run(coordinator.step(1e-3)) == {"global_steps": expected}
    assert len(merged) == 1
    assert [worker.correctness_participation_receipt.calls for worker in workers] == [1, 1]


def test_coordinator_refuses_a_step_that_did_not_complete(monkeypatch):
    coordinator_cls, _ = coordinator_class(monkeypatch)
    coordinator = coordinator_cls([FakeWorkerActor({"rank": 0})], steps=[0])
    with pytest.raises(ValueError, match="requires a completed optimizer step"):
        asyncio.run(coordinator.step(1e-3))
