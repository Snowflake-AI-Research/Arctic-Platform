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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""New CPU-only worker construction controls, with no native ranks or model loads."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def runtime(monkeypatch):
    pytest.importorskip("ray")
    pytest.importorskip("deepspeed")
    from arctic_platform.common import deepspeed_worker as mod
    from arctic_platform.model.parallel import groups

    worker = object.__new__(mod.DeepSpeedWorker.__ray_metadata__.modified_class)
    worker.rank, worker.world_size, worker.master_port = 0, 16, 12345
    ep, sp, mpu = object(), object(), object()
    calls = []
    monkeypatch.setattr(mod.socket, "gethostname", lambda: "host-a")

    def gather(output, host):
        assert host == "host-a"
        output[:] = ["host-a"] * 8 + ["host-b"] * 8

    def setup(sp_size, ep_size, **kwargs):
        calls.append((sp_size, ep_size, kwargs))
        return groups.SPGroups(mpu if sp_size > 1 else None, sp if sp_size > 1 else None, sp_size, 0)

    monkeypatch.setattr(mod.dist, "all_gather_object", gather)
    monkeypatch.setattr(groups, "setup_parallel_groups", setup)
    monkeypatch.setattr(groups, "expert_parallel_group", lambda size: ep)
    return SimpleNamespace(mod=mod, worker=worker, ep=ep, sp=sp, mpu=mpu, calls=calls)


def _spec(ep=8, sp=8):
    return SimpleNamespace(parallelism=SimpleNamespace(expert_parallel=ep, sequence_parallel=sp))


@pytest.mark.parametrize("job_config", [{}, {"training_config": {}}])
def test_worker_training_config_does_not_insert_accumulation_precision(runtime, job_config):
    config = {}

    result = runtime.worker.ds_training_config(job_config, config, {})

    assert result is config
    assert "data_types" not in result


@pytest.mark.parametrize("data_types", [{}, {"grad_accum_dtype": "bf16"}, {"grad_accum_dtype": "fp32"}])
def test_worker_training_config_preserves_explicit_accumulation_precision(runtime, data_types):
    config = {"data_types": deepcopy(data_types)}

    result = runtime.worker.ds_training_config({}, config, {})

    assert result["data_types"] == data_types


def test_worker_groups_derive_actual_host_blocks_and_forward_layout(runtime):
    config = {"zero_optimization": {"stage": 2}}
    result = runtime.worker._setup_model_parallel_groups(_spec(), config, {"ep_group_layout": "packed"})
    assert runtime.calls == [(8, 8, {"world_size": 16, "gpus_per_node": 8, "ep_group_layout": "packed"})]
    assert result == {"ep_group": runtime.ep, "sp_group": runtime.sp}
    assert runtime.worker.mpu is runtime.mpu
    assert config["sequence_parallel_size"] == 8


@pytest.mark.parametrize("worker_config", [{}, {"ep_group_layout": "auto"}, {"ep_group_layout": "packed"}])
def test_worker_groups_ep_only_uses_replicated_setup(runtime, worker_config):
    original = deepcopy(worker_config)
    result = runtime.worker._setup_model_parallel_groups(_spec(sp=1), {}, worker_config)
    assert runtime.calls == [(1, 8, {"world_size": 16, "gpus_per_node": None, **worker_config})]
    assert worker_config == original
    assert result == {"ep_group": runtime.ep, "sp_group": None}
    assert runtime.worker.mpu is None


def test_worker_groups_dense_route_does_not_create_or_gather(runtime, monkeypatch):
    monkeypatch.setattr(runtime.mod.dist, "all_gather_object", lambda *args: pytest.fail("unexpected collective"))
    assert runtime.worker._setup_model_parallel_groups(_spec(ep=1, sp=1), {}, {}) is None
    assert runtime.calls == []


@pytest.mark.parametrize(
    "hosts",
    [
        ["host-a"] * 7 + ["host-b"] * 9,
        ["host-a", "host-b"] * 8,
        ["host-a"] * 8 + [None] * 8,
    ],
)
def test_worker_groups_unsupported_placement_fails_before_creation(runtime, monkeypatch, hosts):
    monkeypatch.setattr(
        runtime.mod.dist, "all_gather_object", lambda output, host: output.__setitem__(slice(None), hosts)
    )
    with pytest.raises(ValueError):
        runtime.worker._setup_model_parallel_groups(_spec(), {}, {})
    assert runtime.calls == []


@pytest.mark.parametrize(
    "spec,config",
    [
        (_spec(ep=6), {}),
        (_spec(sp=3), {}),
        (_spec(), {"zero_optimization": {"stage": 3}}),
        (_spec(), {"sequence_parallel_size": 4}),
    ],
)
def test_worker_groups_invalid_degrees_or_engine_selector_fail(runtime, spec, config):
    with pytest.raises(ValueError):
        runtime.worker._setup_model_parallel_groups(spec, config, {})
    assert runtime.calls == []


def test_worker_initialize_passes_native_groups_and_mpu_without_changing_request(runtime, monkeypatch):
    mod, worker = runtime.mod, runtime.worker
    model = torch.nn.Linear(1, 1)
    events = []
    real_setup = worker._setup_model_parallel_groups
    spec = _spec()
    monkeypatch.setattr(mod.deepspeed, "init_distributed", lambda: events.append("distributed"))
    monkeypatch.setattr(mod, "pr0", lambda *args: None)
    monkeypatch.setattr(mod.ModelSpec, "from_ds_worker_config", lambda *args: spec)
    monkeypatch.setattr(worker, "ds_training_config", lambda job, config, worker_config: config)

    def setup(*args):
        events.append("groups")
        return real_setup(*args)

    def build(selected, *, parallel_groups):
        events.append("model")
        assert selected is spec
        assert parallel_groups == {"ep_group": runtime.ep, "sp_group": runtime.sp}
        return SimpleNamespace(model=model)

    def initialize(**kwargs):
        events.append("engine")
        assert kwargs["model"] is model
        assert list(kwargs["model_parameters"]) == list(model.parameters())
        assert kwargs["mpu"] is runtime.mpu
        assert kwargs["config"]["sequence_parallel_size"] == 8
        assert kwargs["config"]["bf16"] == {"enabled": True, "bf16_optimizer_states": True}
        return SimpleNamespace(local_rank=0, global_steps=0), None, None, None

    monkeypatch.setattr(worker, "_setup_model_parallel_groups", setup)
    monkeypatch.setattr(mod, "build_model", build)
    monkeypatch.setattr(mod, "preserve_fp8_params_through_ds_cast", lambda selected: events.append("preserve"))
    monkeypatch.setattr(mod.deepspeed, "initialize", initialize)
    monkeypatch.setattr(mod, "get_accelerator", lambda: SimpleNamespace(device_name=lambda rank: "cpu"))
    monkeypatch.setattr(mod.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(mod.torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="synthetic"))
    request = {
        "model_name": "synthetic",
        "job_type": "training",
        "ds_worker_config": {"attn_implementation": "sparse_mla", "ep_size": 8, "sp_size": 8},
        "ds_config": {"zero_optimization": {"stage": 2}, "bf16": {"enabled": True, "bf16_optimizer_states": True}},
    }
    original = deepcopy(request)
    assert worker.initialize("localhost", request)
    assert request == original
    assert events == ["distributed", "groups", "model", "preserve", "engine"]


@pytest.mark.parametrize(
    "worker_config,ds_config",
    [
        ({"sp_size": 8, "sequence_parallel": 4}, {}),
        ({"sp_size": 8}, {"sequence_parallel_size": 4}),
        ({"sp_size": True}, {}),
    ],
)
def test_worker_sp_alias_conflicts_fail_before_model(runtime, monkeypatch, worker_config, ds_config):
    monkeypatch.setattr(runtime.mod.deepspeed, "init_distributed", lambda: None)
    monkeypatch.setattr(runtime.mod, "pr0", lambda *args: None)
    monkeypatch.setattr(runtime.mod, "build_model", lambda *args, **kwargs: pytest.fail("unexpected model"))
    request = {
        "model_name": "synthetic",
        "ds_worker_config": worker_config,
        "ds_config": ds_config,
    }
    with pytest.raises(ValueError):
        runtime.worker.initialize("localhost", request)
    assert runtime.calls == []


@pytest.fixture
def routing_worker(runtime, monkeypatch):
    worker = runtime.worker
    worker.engine = object()
    worker.sp_group, worker.sp_size, worker.sp_rank = runtime.sp, 8, 0
    monkeypatch.setattr(runtime.mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime.mod.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(runtime.mod.dist, "get_world_size", lambda group=None: 16 if group is None else 8)
    monkeypatch.setattr(runtime.mod.dist, "get_process_group_ranks", lambda group: list(range(0, 16, 2)))
    return runtime


def test_worker_sp_routing_retains_actual_noncontiguous_order(routing_worker):
    assert routing_worker.worker.get_sequence_parallel_info() == {
        "rank": 0,
        "world_size": 16,
        "sp_size": 8,
        "sp_rank": 0,
        "sp_ranks": list(range(0, 16, 2)),
    }


def test_worker_sp_routing_dense_domain_is_single_rank(routing_worker):
    worker = routing_worker.worker
    worker.sp_size, worker.sp_group = 1, None
    assert worker.get_sequence_parallel_info() == {
        "rank": 0,
        "world_size": 16,
        "sp_size": 1,
        "sp_rank": 0,
        "sp_ranks": [0],
    }


@pytest.mark.parametrize("fault", ["engine", "group", "rank", "size", "initialized", "duplicates"])
def test_worker_sp_routing_rejects_uninitialized_or_inconsistent_membership(routing_worker, monkeypatch, fault):
    worker, dist = routing_worker.worker, routing_worker.mod.dist
    if fault == "engine":
        worker.engine = None
    elif fault == "group":
        worker.sp_group = None
    elif fault == "rank":
        worker.sp_rank = 1
    elif fault == "size":
        worker.sp_size = 4
    elif fault == "initialized":
        monkeypatch.setattr(dist, "is_initialized", lambda: False)
    else:
        monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [0] * 8)
    with pytest.raises(RuntimeError):
        worker.get_sequence_parallel_info()
