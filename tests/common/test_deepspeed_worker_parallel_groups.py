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
"""Worker group wiring, including tiny native CPU/Gloo engine controls.

The native controls exercise worker group setup and the maintained engine adapter,
not Ray GPU-worker startup, expert dispatch kernels, or full-model numerics.
"""

import os
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import execute_subprocess_async
from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import require_deepspeed
from arctic_platform.testing_utils import reserve_free_port
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close


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


@pytest.mark.parametrize(
    "aliases",
    [
        {"sp_size": 8},
        {"sp_size": None, "sequence_parallel": 8},
        {"sp_size": 8, "sequence_parallel": None},
    ],
)
def test_worker_initialize_passes_native_groups_and_mpu_without_changing_request(runtime, monkeypatch, aliases):
    mod, worker = runtime.mod, runtime.worker
    model = torch.nn.Linear(1, 1)
    events = []
    real_setup = worker._setup_model_parallel_groups
    spec = _spec()
    monkeypatch.setattr(mod.deepspeed, "init_distributed", lambda: events.append("distributed"))
    monkeypatch.setattr(mod, "pr0", lambda *args: None)
    monkeypatch.setattr(mod, "ds_worker_moe_model_spec", lambda *args: spec)
    monkeypatch.setattr(worker, "ds_training_config", lambda job, config, worker_config: config)

    def setup(*args):
        events.append("groups")
        return real_setup(*args)

    def build(selected, *, parallel_groups):
        events.append("model")
        assert selected is spec
        assert parallel_groups == {"ep_group": runtime.ep, "sp_group": runtime.sp}
        return SimpleNamespace(model=model)

    def initialize(selected, config, **kwargs):
        events.append("engine")
        assert selected is model
        assert kwargs["mpu"] is runtime.mpu
        assert kwargs["ep_size"] == 8
        assert config["sequence_parallel_size"] == 8
        assert config["train_batch_size"] == 2
        assert config["bf16"] == {"enabled": True, "bf16_optimizer_states": True}
        return SimpleNamespace(local_rank=0, global_steps=0), None, None

    monkeypatch.setattr(worker, "_setup_model_parallel_groups", setup)
    monkeypatch.setattr(mod, "build_model", build)
    monkeypatch.setattr(mod, "initialize_training_engine", initialize)
    monkeypatch.setattr(mod, "get_accelerator", lambda: SimpleNamespace(device_name=lambda rank: "cpu"))
    monkeypatch.setattr(mod.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(mod.torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="synthetic"))
    request = {
        "model_name": "synthetic",
        "job_type": "training",
        "ds_worker_config": {"attn_implementation": "sparse_mla", "ep_size": 8, **aliases},
        "ds_config": {"zero_optimization": {"stage": 2}, "bf16": {"enabled": True, "bf16_optimizer_states": True}},
    }
    original = deepcopy(request)
    assert worker.initialize("localhost", request)
    assert request == original
    assert events == ["distributed", "groups", "model", "engine"]


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


def test_forward_only_moe_keeps_dtype_guard_and_groups_without_optimizer(runtime, monkeypatch):
    mod, worker = runtime.mod, runtime.worker
    model = torch.nn.Linear(1, 1)
    events = []
    monkeypatch.setattr(mod.deepspeed, "init_distributed", lambda: None)
    monkeypatch.setattr(mod, "pr0", lambda *args: None)
    monkeypatch.setattr(worker, "ds_inference_config", lambda *args: {})
    monkeypatch.setattr(mod, "ds_worker_moe_model_spec", lambda *args: _spec(ep=8, sp=1))

    def build(spec, *, parallel_groups):
        assert parallel_groups == {"ep_group": runtime.ep, "sp_group": None}
        return SimpleNamespace(model=model)

    def preserve(selected):
        assert selected is model
        events.append("preserve")

    def initialize(**kwargs):
        assert kwargs["model"] is model
        assert "optimizer" not in kwargs and "model_parameters" not in kwargs
        assert "mpu" not in kwargs
        events.append("engine")
        return SimpleNamespace(local_rank=0, global_steps=0), None, None, None

    monkeypatch.setattr(mod, "build_model", build)
    monkeypatch.setattr(mod, "preserve_fp8_params_through_ds_cast", preserve)
    monkeypatch.setattr(mod.deepspeed, "initialize", initialize)
    monkeypatch.setattr(mod, "initialize_training_engine", lambda *args, **kwargs: pytest.fail("training engine"))
    monkeypatch.setattr(mod, "get_accelerator", lambda: SimpleNamespace(device_name=lambda rank: "cpu"))
    monkeypatch.setattr(mod.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(mod.torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="synthetic"))
    request = {
        "model_name": "synthetic",
        "job_type": "log_prob",
        "ds_worker_config": {"attn_implementation": "sparse_mla", "ep_size": 8, "sp_size": 1},
    }
    original = deepcopy(request)
    assert worker.initialize("localhost", request)
    assert request == original
    assert events == ["preserve", "engine"]


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


class TestNativeWorkerParallelGroups(TestCasePlus):
    @require_deepspeed
    def test_cpu_dense_engine_batch_gradients_and_norm(self):
        self._check_topology(1, 1)

    @require_deepspeed
    def test_cpu_ep_only_engine_batch_gradients_and_norm(self):
        self._check_topology(2, 1)

    @require_deepspeed
    def test_cpu_sp_only_engine_batch_gradients_and_norm(self):
        self._check_topology(1, 2)

    def _check_topology(self, ep_size, sp_size):
        pytest.importorskip("ray")
        env = self.get_env()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["DS_ACCELERATOR"] = "cpu"
        env["OMP_NUM_THREADS"] = "1"
        env["NATIVE_WORKER_EP"] = str(ep_size)
        env["NATIVE_WORKER_SP"] = str(sp_size)
        port = reserve_free_port(get_unique_port_number() + 1, span=7)
        execute_subprocess_async(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--nproc-per-node=4",
                "--master-addr=127.0.0.1",
                f"--master-port={port}",
                __file__,
                "--native-worker-groups",
            ],
            env=env,
            timeout=180,
        )


def _native_worker_parallel_groups():
    import deepspeed
    import torch.distributed as dist
    from deepspeed.runtime.config import DeepSpeedConfig
    from deepspeed.utils import safe_get_full_grad

    from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
    from arctic_platform.common.deepspeed_engine import build_deepspeed_config
    from arctic_platform.common.deepspeed_engine import initialize_training_engine
    from arctic_platform.common.deepspeed_worker import DeepSpeedWorker
    from arctic_platform.common.optimizer import OptimizerSettings

    deepspeed.init_distributed("gloo", auto_mpi_discovery=False)
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        ep_size, sp_size = int(os.environ["NATIVE_WORKER_EP"]), int(os.environ["NATIVE_WORKER_SP"])
        assert world == 4 and not torch.cuda.is_available()
        dp_size = world // sp_size
        worker = object.__new__(DeepSpeedWorker.__ray_metadata__.modified_class)
        worker.rank, worker.world_size = rank, world
        setup_config = {"zero_optimization": {"stage": 2}}
        parallel_groups = worker._setup_model_parallel_groups(_spec(ep=ep_size, sp=sp_size), setup_config, {})
        assert (worker.mpu is None) == (sp_size == 1)
        if ep_size > 1:
            assert dist.get_world_size(worker.ep_group) == ep_size
            assert parallel_groups["ep_group"] is worker.ep_group
        if sp_size > 1:
            assert dist.get_world_size(worker.sp_group) == sp_size
            assert worker.mpu.get_sequence_data_parallel_world_size() == world
        for gas in (1, 3):
            set_seed(7)
            model = torch.nn.Linear(1, 1)
            reference = deepcopy(model)
            spec = TrainingEngineSpec(
                dtype="float32",
                world_size=world,
                sequence_parallel=sp_size,
                train_micro_batch_size_per_gpu=2,
                gradient_accumulation_steps=gas,
                gradient_clipping=0,
                optimizer=OptimizerSettings(name="torch_adamw", lr=0.001, fused=False),
                ds_overrides={
                    "zero_allow_untested_optimizer": True,
                    "zero_optimization": {"stage": 2, "reduce_scatter": True},
                },
            )
            config = build_deepspeed_config(spec)
            expected_batch = dp_size * 2 * gas
            assert config["train_batch_size"] == expected_batch
            assert DeepSpeedConfig(config, mpu=worker.mpu).world_size == dp_size
            engine, _, _ = initialize_training_engine(
                model,
                config,
                optimizer=spec.optimizer,
                ep_size=ep_size,
                mpu=worker.mpu,
                model_path="tiny-cpu-policy",
                rank=rank,
            )
            assert engine.seq_dp_world_size // engine.sequence_parallel_size == dp_size
            if sp_size == 1:
                assert engine.dp_world_size == world
            assert engine.train_batch_size() == expected_batch
            assert engine.gradient_accumulation_steps() == gas
            assert engine.train_micro_batch_size_per_gpu() == 2
            # Distinct logical-DP rows; SP ranks hold disjoint tokens of the same rows.
            inputs = torch.arange(expected_batch * 4, dtype=torch.float32).reshape(dp_size, gas, 2, 4, 1) / 32
            expected_loss = reference(inputs).square().mean()
            expected_loss.backward()
            replica = rank // sp_size
            local_rank = worker.sp_rank
            observed_loss = torch.zeros(())
            for micro in range(gas):
                tokens = inputs[replica, micro, :, local_rank * (4 // sp_size) : (local_rank + 1) * (4 // sp_size)]
                loss = engine(tokens).square().sum() / (2 * 4)
                observed_loss += loss.detach() / (gas * dp_size)
                engine.backward(loss)
                if micro + 1 < gas:
                    engine.step()
            dist.all_reduce(observed_loss)
            # FP32 collective ordering can differ from the full-batch reference.
            torch_assert_close(observed_loss, expected_loss.detach(), rtol=0, atol=1e-6)
            for parameter, expected in zip(model.parameters(), reference.parameters()):
                torch_assert_close(safe_get_full_grad(parameter), expected.grad, rtol=0, atol=1e-6)
            expected_norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in reference.parameters()]))
            engine.step()
            torch_assert_close(torch.as_tensor(engine.get_global_grad_norm()), expected_norm, rtol=0, atol=1e-6)
            if rank == 0:
                print(
                    f"NATIVE_WORKER_GROUPS_PASS ep={ep_size} sp={sp_size} dp={dp_size} "
                    f"gas={gas} batch={expected_batch} loss={observed_loss.item()} norm={expected_norm.item()}",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__" and "--native-worker-groups" in sys.argv:
    _native_worker_parallel_groups()
