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

"""A LoRA training job on the correctness gateway reports the same step-1 participation inventory as a full one.

Every LoRA payload carries an adapter initialization request, so the gateway creates its coordinator through the
original or shared adapter coordinator rather than the plain participation coordinator. The gradient and optimizer
checks then require the inventory from that coordinator's step response.

CPU only: this module stands in for ``dss.ray_dss.jobs.gpu.training``; its coordinator base constructs the worker class
the adapter coordinator binds the way DSS creates each worker actor, with ``cfg`` alone and the rank in ``RANK``, and a
``torch.nn.Linear`` stands in for the model.
"""

import asyncio
import contextlib
import os
import sys
import types
from copy import deepcopy
from unittest import mock

import pytest
import torch

from arctic_platform.correctness.harness import lora
from arctic_platform.correctness.harness.optimizer import PARTICIPATION_SCHEMA
from arctic_platform.correctness.harness.optimizer import validate_participation
from arctic_platform.debug import training as debug_training

JOB_ID = "job-0"
WORLD_SIZE = 2


class FakeEngine:
    def __init__(self):
        self.module = torch.nn.Linear(4, 3)
        self.global_steps = 0


class FakeDeepSpeedWorker:
    def __init__(self, cfg):
        self._engine = FakeEngine()
        self.rank = int(os.environ["RANK"])
        self.job_id = cfg["job_id"]

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
        return {"global_steps": self._engine.global_steps}


def initialize_training_engine(model, rank):
    return model


class _Remote:
    def __init__(self, call):
        self.call = call

    async def remote(self, *args):
        return self.call(*args)


class FakeWorkerActor:
    def __init__(self, worker, adapter_receipt):
        self.worker = worker
        self.original_adapter_receipt = _Remote(lambda: deepcopy(adapter_receipt))
        self.shared_adapter_receipt = _Remote(lambda: deepcopy(adapter_receipt))

    def __getattr__(self, name):
        return _Remote(getattr(self.worker, name))


class FakeCoordinatorBase:
    adapter_receipts = None

    def __init__(self):
        self._workers = []
        self._info = types.SimpleNamespace(job_id=JOB_ID, config=None)

    async def _spawn_workers(self, cfg, ranks, master_addr, master_port, source_stage_info):
        # The adapter coordinator rebinds this module global for the duration of the native spawn.
        worker_class = _DeepSpeedWorker
        for rank, _ in enumerate(ranks):
            # A worker actor imports ``training`` in its own process, where ``_DeepSpeedWorker`` is the native class.
            native = mock.patch.object(sys.modules[__name__], "_DeepSpeedWorker", FakeDeepSpeedWorker)
            with mock.patch.dict(os.environ, {"RANK": str(rank)}), native:
                worker = worker_class(cfg)
            self._workers.append(FakeWorkerActor(worker, self.adapter_receipts[rank]))

    async def _kill_workers(self, hosts):
        self._workers = []

    async def step(self, learning_rate=None):
        return {"global_steps": 1}


_DeepSpeedWorker = FakeDeepSpeedWorker
GPUTrainingJob = types.SimpleNamespace(__ray_metadata__=types.SimpleNamespace(modified_class=FakeCoordinatorBase))
IDENTITY = {"schema": "shared-peft-adapter-v1"}
ORIGINAL_METADATA = {"identity_sha256": "a" * 64, "initialization": {"identity_sha256": "b" * 64, "tensors": []}}


def original_receipt(rank):
    return {
        "schema": "original-peft-worker-initialization-v1",
        "job_id": JOB_ID,
        "rank": rank,
        "engine_initializations": 1,
        "original_identity_sha256": ORIGINAL_METADATA["identity_sha256"],
        "shared_identity": IDENTITY,
        "initialization": {
            "identity_sha256": ORIGINAL_METADATA["initialization"]["identity_sha256"],
            "verified_tensors": [],
        },
    }


def shared_receipt(rank):
    return {
        "schema": "shared-peft-worker-initialization-v1",
        "job_id": JOB_ID,
        "rank": rank,
        "engine_initializations": 1,
        "shared_identity": IDENTITY,
    }


def adapter_scope(receipt):
    @contextlib.contextmanager
    def scope(training, constructor, files, identity, *metadata, job_id, rank):
        assert constructor is FakeDeepSpeedWorker.__init__ and job_id == JOB_ID
        yield receipt(rank)

    return scope


def adapter_coordinator(monkeypatch, kind):
    training = sys.modules[__name__]
    fake_ray = types.ModuleType("ray")
    fake_ray.remote = lambda **_options: lambda cls: cls
    fake_ray.get = lambda reference: reference
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "dss.ray_dss.jobs.gpu", types.SimpleNamespace(training=training))
    monkeypatch.setenv(debug_training.CAPTURE_ENV, "1")
    request = {"training_gpus": WORLD_SIZE, "identity": IDENTITY}
    if kind == "original":
        monkeypatch.setattr(lora, "native_original_worker_adapter_scope", adapter_scope(original_receipt))
        request.update(sources=lora._runtime_sources(training), original_metadata=ORIGINAL_METADATA)
        monkeypatch.setattr(lora, "_request", lambda config: deepcopy(request))
        coordinator_class = lora._coordinator_class(training, request, files_ref=None)
        receipts = [original_receipt(rank) for rank in range(WORLD_SIZE)]
    else:
        request.update(sources=lora._shared_runtime_sources(training))
        monkeypatch.setattr(lora, "_shared_request", lambda config: deepcopy(request))
        monkeypatch.setattr(lora, "native_shared_worker_adapter_scope", adapter_scope(shared_receipt))
        coordinator_class = lora._shared_coordinator_class(training, request, files_ref=None)
        receipts = [shared_receipt(rank) for rank in range(WORLD_SIZE)]
    monkeypatch.setattr(FakeCoordinatorBase, "adapter_receipts", receipts)
    return coordinator_class()


@pytest.mark.parametrize("kind", ["original", "shared"])
def test_adapter_job_step_one_carries_participation_inventory(monkeypatch, kind):
    coordinator = adapter_coordinator(monkeypatch, kind)
    ranks = [("127.0.0.1", rank) for rank in range(WORLD_SIZE)]
    asyncio.run(coordinator._spawn_workers({"job_id": JOB_ID}, ranks, "127.0.0.1", 0, None))
    batch = torch.ones(2, 4)
    for actor in coordinator._workers:
        actor.worker.fwd_bwd(batch)
        actor.worker.step(1e-3)

    response = asyncio.run(coordinator.step(1e-3))

    assert response["global_steps"] == 1
    participation = response.get("participation")
    assert participation is not None, "LoRA adapter job step response carries no participation inventory"
    assert participation["schema"] == PARTICIPATION_SCHEMA
    assert participation["training_child_id"] == JOB_ID
    assert [receipt["rank"] for receipt in participation["rank_receipts"]] == list(range(WORLD_SIZE))
    states = {entry["state"] for entry in validate_participation(participation).values()}
    assert states == {"active"}
