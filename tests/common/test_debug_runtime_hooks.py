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

"""CPU routing controls for opt-in runtime hooks, not GPU or collective parity."""

import asyncio
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.common import deepspeed_engine as engine_module
from arctic_platform.common import deepspeed_worker as worker_module
from arctic_platform.common import ray_server as server_module
from arctic_platform.common.optimizer import OptimizerSettings
from arctic_platform.debug import lora
from arctic_platform.debug import weight_sync

Worker = worker_module.DeepSpeedWorker.__ray_metadata__.modified_class


@pytest.fixture(autouse=True)
def debug_disabled(monkeypatch):
    for name in (
        lora.REQUEST_ENV,
        lora.DIGEST_ENV,
        lora.PATH_ENV,
        lora.SHA_ENV,
        lora.INITIALIZATION_ENV,
        lora.INITIALIZATION_SHA_ENV,
        lora.MODEL_CONSTRUCTION_ENV,
        lora.MODEL_CONSTRUCTION_SHA_ENV,
        weight_sync.VERIFY_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def worker():
    selected = object.__new__(Worker)
    selected.rank, selected.world_size, selected.master_port = 0, 1, 12345
    selected.engine = SimpleNamespace(module=torch.nn.Linear(2, 1, bias=False))
    selected._weight_sender = None
    selected._device = "cpu"
    selected._gpu_mem_preallocation = None
    return selected


def server():
    selected = object.__new__(server_module.ArcticRLRayServer)
    selected.training_workers = []
    selected._training_op_lock = asyncio.Lock()
    selected._training_failed = False
    selected._training_failure = None
    selected._shutdown_complete = False
    selected.jobs = {}
    selected.weight_sync_ready = False
    return selected


class Remote:
    def __init__(self, value):
        self.value = value

    async def remote(self, *args, **kwargs):
        return self.value


@pytest.mark.parametrize("prebuilt", [False, True])
def test_disabled_engine_hooks_preserve_native_arguments_and_returns(monkeypatch, prebuilt):
    model = torch.nn.Linear(2, 1)
    config = {"train_batch_size": 1, "optimizer": {"type": "AdamW", "params": {"lr": 0.01}}}
    built, native_engine, mpu = object(), object(), object()
    calls = []
    monkeypatch.setattr(engine_module, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))
    monkeypatch.setattr(engine_module, "build_training_optimizer", lambda *args, **kwargs: built)
    monkeypatch.setattr(engine_module, "optimizer_choice", lambda *args, **kwargs: {"implementation": "test"})

    def initialize(**kwargs):
        calls.append(kwargs)
        return native_engine, built, None, None

    monkeypatch.setattr(worker_module.deepspeed, "initialize", initialize)
    result = engine_module.initialize_training_engine(
        model,
        config,
        optimizer=OptimizerSettings(lr=0.01) if prebuilt else None,
        ep_size=1,
        mpu=mpu,
        model_path="dense",
        rank=0,
    )
    assert result[0] is native_engine and result[1] is built
    assert result[2] == ({"implementation": "test"} if prebuilt else None)
    assert len(calls) == 1 and calls[0]["config"] is config
    assert calls[0]["model"] is model and calls[0]["mpu"] is mpu
    if prebuilt:
        assert calls[0]["optimizer"] is built and "model_parameters" not in calls[0]
    else:
        assert list(calls[0]["model_parameters"]) == list(model.parameters())
        assert "optimizer" not in calls[0]


def test_engine_hook_order_and_trace_config_reach_native_initialize(monkeypatch):
    events = []
    model, config = torch.nn.Linear(2, 1), {"train_batch_size": 1}
    broadcast, traced, native = dict(config), dict(config), dict(config)
    monkeypatch.setattr(engine_module, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))

    def configure(value):
        assert value is config
        events.append("broadcast")
        return broadcast

    def router(value, selected_model, rank):
        assert value is broadcast and selected_model is model and rank == 3
        events.append("router")
        return traced

    @contextmanager
    def trace(value, rank):
        assert value is traced and rank == 3
        events.append("trace-enter")
        yield native
        events.append("trace-exit")

    def initialize(**kwargs):
        assert kwargs["config"] is native
        events.append("initialize")
        return object(), None, None, None

    monkeypatch.setattr(engine_module, "maybe_configure_broadcast_training", configure)
    monkeypatch.setattr(engine_module, "maybe_configure_router_replay_trace", router)
    monkeypatch.setattr(engine_module, "maybe_trace_zero_initialization", trace)
    monkeypatch.setattr(worker_module.deepspeed, "initialize", initialize)
    engine_module.initialize_training_engine(
        model, config, optimizer=None, ep_size=1, mpu=None, model_path="dense", rank=3
    )
    assert events == ["broadcast", "router", "trace-enter", "initialize", "trace-exit"]


def test_engine_consumes_disabled_broadcast_option_without_mutating_recipe(monkeypatch):
    config = {"train_batch_size": 1, weight_sync.TRAINING_OPTION: False}
    seen = []
    monkeypatch.setattr(engine_module, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))

    def initialize(**kwargs):
        seen.append(kwargs["config"])
        return object(), None, None, None

    monkeypatch.setattr(worker_module.deepspeed, "initialize", initialize)
    engine_module.initialize_training_engine(
        torch.nn.Linear(2, 1), config, optimizer=None, ep_size=1, mpu=None, model_path="dense", rank=0
    )
    assert seen == [{"train_batch_size": 1}]
    assert config == {"train_batch_size": 1, weight_sync.TRAINING_OPTION: False}
    assert not weight_sync.enabled()


def test_disabled_lora_decorator_keeps_original_worker_class():
    assert lora.maybe_lora_training_worker(Worker) is Worker
    assert not hasattr(Worker, "prepare_lora_weights")


def test_uninitialized_sender_still_gets_weights(monkeypatch):
    selected = worker()
    calls = []
    monkeypatch.setattr(selected, "get_weights", lambda: calls.append("gather") or [])
    assert selected.send_weights() == {"status": "not_initialized"}
    assert calls == ["gather"]


class Sender:
    def __init__(self):
        self.calls = []
        self.result = {"status": "ok"}

    def send(self, weights):
        self.calls.append(weights)
        return self.result


def test_disabled_send_keeps_transport_result_and_exact_tensor_list(monkeypatch):
    selected, sender = worker(), Sender()
    selected._weight_sender = sender
    weights = selected.get_weights()
    monkeypatch.setattr(selected, "get_weights", lambda: weights)
    result = selected.send_weights()
    assert result is sender.result
    assert sender.calls == [weights]
    assert result["sent_l2_sq"] == selected._weights_l2_sq(weights)
    assert set(result) == {"status", "sent_l2_sq"}


def test_enabled_send_observation_hashes_actual_transmitted_tensors(monkeypatch):
    selected, sender = worker(), Sender()
    selected._weight_sender = sender
    monkeypatch.setenv(weight_sync.VERIFY_ENV, "1")
    result = selected.send_weights()
    assert len(sender.calls) == 1
    assert result["sent_manifest"] == {name: weight_sync.tensor_manifest(value) for name, value in sender.calls[0]}
    assert result["trainer_parameters"] == {"numel": 2, "count": 1}
    assert result["evidence_sources"]
    assert result is not sender.result


def test_enabled_send_detects_changed_tensors_without_retry(monkeypatch):
    selected = worker()
    calls = []

    class MutatingSender:
        def send(self, weights):
            calls.append(weights)
            weights[0][1].add_(1)
            return {"status": "ok"}

    selected._weight_sender = MutatingSender()
    monkeypatch.setenv(weight_sync.VERIFY_ENV, "1")
    with pytest.raises(RuntimeError, match="actually sent tensors changed"):
        selected.send_weights()
    assert len(calls) == 1


@pytest.mark.parametrize("job_type", ["train", "log_prob"])
@pytest.mark.parametrize("ep_size", [1, 8])
def test_worker_admission_precedes_model_and_verification_follows_engine(monkeypatch, job_type, ep_size):
    selected, events = worker(), []
    model = selected.engine.module
    spec = SimpleNamespace(parallelism=SimpleNamespace(expert_parallel=ep_size, sequence_parallel=1))
    groups = None
    if hasattr(selected, "_setup_model_parallel_groups"):
        groups = {"ep_group": object(), "sp_group": None}
        monkeypatch.setattr(selected, "_setup_model_parallel_groups", lambda *args: groups)
    monkeypatch.setattr(worker_module.deepspeed, "init_distributed", lambda: None)
    monkeypatch.setattr(worker_module, "pr0", lambda *args: None)
    monkeypatch.setattr(worker_module, "ds_worker_dense_model_spec", lambda *args, **kwargs: spec)
    monkeypatch.setattr(worker_module, "ds_worker_moe_model_spec", lambda *args, **kwargs: spec)
    monkeypatch.setattr(worker_module, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0), raising=False)
    selected.mpu = None

    def admit(owner, selected_spec, config, selected_groups, build, initialize):
        assert owner is selected and selected_spec is spec and selected_groups is groups
        assert build is worker_module.build_model and initialize is worker_module.deepspeed.initialize
        events.append("admit")

    def build(selected_spec, *, parallel_groups=None):
        assert selected_spec is spec and parallel_groups is groups
        events.append("model")
        return SimpleNamespace(model=model)

    def initialize(*args, **kwargs):
        events.append("engine")
        native = SimpleNamespace(local_rank=0, global_steps=0)
        return (native, None, None) if args else (native, None, None, None)

    def verify(owner):
        assert owner is selected and owner.engine.global_steps == 0
        events.append("verify")

    monkeypatch.setattr(worker_module, "maybe_admit_worker_construction", admit)
    monkeypatch.setattr(worker_module, "build_model", build)
    monkeypatch.setattr(worker_module, "initialize_training_engine", initialize)
    monkeypatch.setattr(worker_module.deepspeed, "initialize", initialize)
    monkeypatch.setattr(worker_module, "maybe_verify_worker_engine", verify)
    monkeypatch.setattr(worker_module, "get_accelerator", lambda: SimpleNamespace(device_name=lambda rank: "cpu"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda rank: SimpleNamespace(uuid="test"))
    assert selected.initialize(
        "localhost",
        {
            "model_name": "dense",
            "job_type": job_type,
            "ds_worker_config": {"attn_implementation": "sdpa", "ep_size": ep_size},
        },
    )
    assert events == ["admit", "model", "engine", "verify"]


def test_sender_hook_observes_initialized_native_sender(monkeypatch):
    selected, events = worker(), []
    group, schedule = object(), object()
    sender_module = SimpleNamespace(WeightSender=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setitem(sys.modules, "arctic_platform.inference.server.weight_sync.sender", sender_module)

    def record(owner, selected_group, selected_schedule, base_port):
        assert owner is selected and owner._weight_sender.group is group
        assert selected_group is group and selected_schedule is schedule and base_port == 23456
        events.append("record")

    monkeypatch.setattr(worker_module, "maybe_record_lora_sender", record)
    assert selected.init_weight_sender(group, schedule, "localhost", 23456, 4096)
    assert events == ["record"]


def test_worker_release_precedes_native_cleanup(monkeypatch):
    selected, events = worker(), []
    selected._weight_sender = SimpleNamespace(destroy=lambda: events.append("sender"))
    monkeypatch.setattr(worker_module, "maybe_release_lora_worker", lambda owner: events.append("release"))
    monkeypatch.setattr(worker_module.dist, "is_initialized", lambda: False)
    assert selected.destroy()
    assert events == ["release", "sender"]
    assert selected.engine is selected._weight_sender is selected._gpu_mem_preallocation is None


def test_enabled_initialization_requires_native_topology_and_keeps_owned_cleanup(monkeypatch):
    selected, events = worker(), []
    selected.sp_size, selected.sp_rank, selected.sp_group = 1, 0, None
    recipe = {"train_batch_size": 1}

    def verify_engine(engine, *, sp_info, sp_group):
        assert engine is selected.engine and sp_group is None
        assert sp_info == {"rank": 0, "world_size": 1, "sp_size": 1, "sp_rank": 0, "sp_ranks": [0]}
        events.append("runtime")
        return {"verified": True}

    slot = {
        "subject": {"worker": selected, "ds_config": recipe},
        "status": "armed",
        "worker_initialize": selected.initialize.__func__,
        "ds_config": dict(recipe),
        "runtime": SimpleNamespace(verify_engine=verify_engine),
        "destroy": lambda: events.append("owned-cleanup"),
    }
    selected._ap_initialization_admission = slot
    monkeypatch.setattr(lora, "_guard_initialization", lambda slot: None)
    monkeypatch.setattr(lora, "_worker_cleanup_matches", lambda slot, owner: owner is selected)
    monkeypatch.setattr(lora, "_verify_initialization_capacity", lambda *args: {"capacity": True})
    monkeypatch.setattr(worker_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(worker_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(worker_module.dist, "get_world_size", lambda: 1)

    if hasattr(selected, "get_sequence_parallel_info"):
        assert lora.maybe_verify_worker_engine(selected) == {"capacity": True}
        assert slot["status"] == "verified" and events == ["runtime"]
    else:
        # Main has no native topology RPC; debug admission must not fabricate one.
        with pytest.raises(AttributeError, match="get_sequence_parallel_info"):
            lora.maybe_verify_worker_engine(selected)
        assert slot["status"] == "failed" and events == ["owned-cleanup"]


def test_replica_pool_hook_wraps_the_actual_lazy_import(monkeypatch):
    original, decorated = type("Pool", (), {}), type("DecoratedPool", (), {})
    monkeypatch.setattr(server_module, "require_any_dep_group", lambda *args: None)
    monkeypatch.setitem(
        sys.modules, "arctic_platform.inference.server.replica_pool", SimpleNamespace(ReplicaPool=original)
    )
    monkeypatch.setattr(server_module, "maybe_lora_replica_pool", lambda cls: decorated if cls is original else None)
    assert server_module._replica_pool_cls() is decorated


def sync_runtime(monkeypatch):
    selected, events = server(), []
    group = SimpleNamespace(group_id=0, sender_train_rank=0, world_size=2, replica_ids=[0])
    schedule = SimpleNamespace(groups=[group])
    monkeypatch.setattr(
        server_module, "_transfer_schedule_cls", lambda: SimpleNamespace(build=lambda **kwargs: schedule)
    )
    sender = SimpleNamespace(
        get_ip=Remote("localhost"),
        max_param_bytes=Remote(4096),
        init_weight_sender=Remote(True),
        send_weights=Remote({"sent_l2_sq": 4.0}),
    )
    selected.arctic_rl_ray_server_state = SimpleNamespace(wake_inference=Remote(None))

    async def sync_weights(**kwargs):
        events.append(kwargs)
        return {"workers": [{"recv_l2_sq": 4.0}]}

    pool = SimpleNamespace(num_replicas=1, tp_size=1, sync_weights=sync_weights)
    return selected, sender, pool, events


def test_disabled_nccl_hooks_preserve_original_receipt_and_routing(monkeypatch):
    selected, sender, pool, events = sync_runtime(monkeypatch)
    result = asyncio.run(selected._sync_weights_nccl([sender], pool))
    assert result["status"] == "ok" and result["checksum_match"] is True
    assert "targets" not in result and "weight_format" not in result
    assert len(events) == 1
    assert events[0]["groups"][0]["master_addr"] == "localhost"
    assert selected.weight_sync_ready is True and selected._training_failed is False


@pytest.mark.parametrize("failure", ["send", "receive", "cancel"])
def test_inline_awaiter_preserves_peer_cancellation_and_latch(monkeypatch, failure):
    selected, sender, pool, _ = sync_runtime(monkeypatch)
    started, cancelled = set(), set()

    async def transfer(side):
        started.add(side)
        while len(started) != 2:
            await asyncio.sleep(0)
        try:
            if side == failure:
                raise RuntimeError(f"{side} failed")
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add(side)
            raise

    async def run():
        sender.send_weights = SimpleNamespace(remote=lambda: transfer("send"))
        pool.sync_weights = lambda **kwargs: transfer("receive")
        operation = asyncio.create_task(selected._sync_weights_nccl([sender], pool))
        while len(started) != 2:
            await asyncio.sleep(0)
        if failure == "cancel":
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
        else:
            with pytest.raises(RuntimeError, match=f"{failure} failed"):
                await operation
        for _ in range(4):
            await asyncio.sleep(0)
        # A sole completed sender failure has no unfinished training peer to cancel. Keep that native policy.
        assert selected._training_failed is (failure != "send")
        assert {"send", "receive"} - {failure} <= cancelled

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_server_hooks_receive_original_response_and_ordered_sender_objects(monkeypatch):
    selected, sender, pool, _ = sync_runtime(monkeypatch)
    received, sent = {"workers": [{"recv_l2_sq": 4.0}]}, {"sent_l2_sq": 4.0}
    sender.send_weights = Remote(sent)

    async def receive(**kwargs):
        return received

    def annotate(owner, result, response, send_results):
        assert owner is selected and response is received
        assert len(send_results) == 1 and send_results[0] is sent
        return result

    pool.sync_weights = receive
    monkeypatch.setattr(server_module, "maybe_lora_sync_result", annotate)
    assert asyncio.run(selected._sync_weights_nccl([sender], pool))["checksum_match"]
    assert selected._training_failed is False
