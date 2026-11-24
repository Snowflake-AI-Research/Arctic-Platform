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

"""CPU-only checks of the native AP worker's retained capacity guard."""

from types import SimpleNamespace

import pytest
import torch


def worker_module():
    pytest.importorskip("ray")
    pytest.importorskip("deepspeed")
    from arctic_platform.common import deepspeed_worker

    return deepspeed_worker


@pytest.mark.parametrize(
    "training",
    [
        {"debug": {"gpu_mem_preallocate_gib": 20, "gradient_norms_per_param": True}},
        {"prime_rl": {"debug": {"gpu_mem_preallocate_gib": 20}}},
    ],
)
def test_training_margin_reaches_model_and_survives_engine_initialization(monkeypatch, training):
    mod = worker_module()
    cls = mod.DeepSpeedWorker.__ray_metadata__.modified_class
    worker = object.__new__(cls)
    worker.rank, worker.world_size, worker.master_port = 0, 1, 12345
    reserve = torch.empty(1)
    model = torch.nn.Identity()
    model._arctic_gpu_mem_preallocation = reserve
    observed = []
    monkeypatch.setattr(mod.deepspeed, "init_distributed", lambda: None)
    monkeypatch.setattr(mod, "pr0", lambda *args: None)
    monkeypatch.setattr(mod, "sp_size_from_job_config", lambda config: 1)
    monkeypatch.setattr(worker, "ds_training_config", lambda *args: {})

    def build(spec):
        observed.append(spec.debug)
        return SimpleNamespace(model=model)

    def initialize(**kwargs):
        assert worker._gpu_mem_preallocation is reserve
        assert kwargs["model"] is model
        return SimpleNamespace(local_rank=0, global_steps=0), None, None, None

    monkeypatch.setattr(mod, "build_model", build)
    monkeypatch.setattr(mod.deepspeed, "initialize", initialize)
    monkeypatch.setattr(mod, "get_accelerator", lambda: SimpleNamespace(device_name=lambda rank: "cuda:0"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda gpu: SimpleNamespace(uuid="test"))
    assert worker.initialize(
        "localhost",
        {
            "model_name": "x",
            "job_type": "train",
            "training_config": training,
            "ds_worker_config": {"attn_implementation": "sdpa"},
        },
    )
    assert observed == [{"gpu_mem_preallocate_gib": 20}]
    assert worker._gpu_mem_preallocation is reserve


@pytest.mark.parametrize("sender_fails", [False, True])
@pytest.mark.parametrize("communicator_fails", [False, True])
def test_destroy_retains_buffer_through_sender_and_communicator_teardown(
    monkeypatch, sender_fails, communicator_fails
):
    mod = worker_module()
    cls = mod.DeepSpeedWorker.__ray_metadata__.modified_class
    worker = object.__new__(cls)
    reserve = object()
    worker._gpu_mem_preallocation = reserve
    worker.engine = object()
    events = []

    def teardown(name):
        assert worker._gpu_mem_preallocation is reserve
        events.append(name)
        if (name == "sender" and sender_fails) or (name == "communicator" and communicator_fails):
            raise RuntimeError(name)

    worker._weight_sender = SimpleNamespace(destroy=lambda: teardown("sender"))
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mod.dist, "destroy_process_group", lambda: teardown("communicator"))
    if sender_fails or communicator_fails:
        with pytest.raises(RuntimeError, match="communicator" if communicator_fails else "sender"):
            worker.destroy()
    else:
        assert worker.destroy()
    assert events == ["sender", "communicator"]
    assert worker._weight_sender is None
    assert worker.engine is None
    assert worker._gpu_mem_preallocation is None
