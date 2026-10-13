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

"""Weight transfer must materialize complete tensors on every training rank."""

import ast
import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def production_method(filename, class_name, name, **namespace):
    # Isolate the actual method from GPU-only module initialization.
    path = ROOT / "arctic_platform/common" / filename
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in cls.body if getattr(node, "name", None) == name)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class Broadcast:
    def __init__(self, rank, expected):
        self.rank = rank
        self.expected = expected
        self.events = []
        self.tensors = iter(tensor for _, tensor in expected)

    def get_rank(self):
        return self.rank

    def broadcast_object_list(self, payload, src):
        assert src == 0
        metadata = [(name, tuple(tensor.shape), tensor.dtype) for name, tensor in self.expected]
        if self.rank == 0:
            assert payload == [metadata]
        else:
            assert payload == [None]
            payload[0] = metadata
        self.events.append("metadata")

    def broadcast(self, tensor, src):
        assert src == 0
        expected = next(self.tensors)
        assert tensor.shape == expected.shape and tensor.dtype == expected.dtype
        if self.rank == 0:
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
        else:
            tensor.copy_(expected)
        self.events.append("tensor")


@pytest.mark.parametrize("rank", [0, 1, 7, 15])
@pytest.mark.parametrize("empty", [False, True])
def test_hf_converter_runs_collectively_and_replicates_complete_weights(rank, empty):
    expected = (
        []
        if empty
        else [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.arange(6.0).reshape(2, 3)),
            ("model.layers.0.linear_attn.in_proj_qkvz.weight", torch.arange(12.0).reshape(4, 3).bfloat16()),
        ]
    )
    events = []

    def converted():
        events.append("expert_gather")
        if rank == 0:
            yield from expected

    raw = torch.nn.Parameter(torch.zeros(1))
    model = SimpleNamespace(
        _iter_full_hf_weights=converted,
        named_parameters=lambda: [("model.layers.0._checkpoint_wrapped_module.mlp.experts.w1", raw)],
    )
    transport = Broadcast(rank, expected)
    get_weights = production_method(
        "deepspeed_worker.py", "DeepSpeedWorker", "get_weights", torch=torch, dist=transport
    )
    actual = get_weights(SimpleNamespace(engine=SimpleNamespace(module=model), _device="cpu"))
    assert [name for name, _ in actual] == [name for name, _ in expected]
    for (_, left), (_, right) in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert events == ["expert_gather"]
    assert transport.events == ["metadata"] + ["tensor"] * len(expected)


def test_unsharded_dense_weights_preserve_names_and_storage():
    param = torch.nn.Parameter(torch.arange(5.0))
    worker = SimpleNamespace(
        engine=SimpleNamespace(module=SimpleNamespace(named_parameters=lambda: [("linear.weight", param)]))
    )
    get_weights = production_method("deepspeed_worker.py", "DeepSpeedWorker", "get_weights", torch=torch)
    weights = get_weights(worker)
    assert weights[0][0] == "linear.weight"
    assert weights[0][1].data_ptr() == param.data_ptr()


def test_zero3_transfer_owns_storage_after_gather_release():
    param = torch.nn.Parameter(torch.empty(0))
    param.ds_id = 0
    full = torch.arange(6.0)

    @contextmanager
    def gathered(params, enabled):
        assert params == [param] and enabled
        param.data = full.clone()
        try:
            yield
        finally:
            param.data.fill_(-1)
            param.data = torch.empty(0)

    get_weights = production_method(
        "deepspeed_worker.py",
        "DeepSpeedWorker",
        "get_weights",
        torch=torch,
        deepspeed=SimpleNamespace(zero=SimpleNamespace(GatheredParameters=gathered)),
    )
    worker = SimpleNamespace(
        engine=SimpleNamespace(module=SimpleNamespace(named_parameters=lambda: [("linear.weight", param)]))
    )
    weights = get_weights(worker)
    torch.testing.assert_close(weights[0][1], full, rtol=0, atol=0)
    assert weights[0][1].data_ptr() != full.data_ptr()


@pytest.mark.parametrize("empty", [False, True])
def test_bucket_fits_converted_fused_experts(empty):
    raw = torch.zeros(2, 3)
    fused = torch.zeros(8, 6)
    events = []
    worker = SimpleNamespace(
        engine=SimpleNamespace(module=SimpleNamespace(_iter_full_hf_weights=object(), parameters=lambda: [raw])),
        get_weights=lambda: events.append("converted") or ([] if empty else [("gate_up_proj", fused)]),
    )
    maximum = production_method("deepspeed_worker.py", "DeepSpeedWorker", "max_param_bytes")
    assert maximum(worker) == (0 if empty else fused.numel() * fused.element_size())
    assert events == ["converted"]


@pytest.mark.parametrize("zero3", [False, True])
def test_dense_bucket_preserves_full_zero3_parameter_size(zero3):
    param = torch.nn.Parameter(torch.zeros(2))
    if zero3:
        param.ds_id = 3
        param.ds_numel = 14
    worker = SimpleNamespace(engine=SimpleNamespace(module=SimpleNamespace(parameters=lambda: [param])))
    maximum = production_method("deepspeed_worker.py", "DeepSpeedWorker", "max_param_bytes")
    assert maximum(worker) == (56 if zero3 else 8)


@pytest.mark.parametrize("has_sender", [False, True])
def test_non_senders_materialize_without_transmitting(has_sender):
    events = []
    weights = [("weight", torch.ones(1))]
    sender = SimpleNamespace(send=lambda actual: events.append(("send", actual)) or {"status": "done"})
    worker = SimpleNamespace(
        get_weights=lambda: events.append("prepare") or weights, _weight_sender=sender if has_sender else None
    )
    send = production_method("deepspeed_worker.py", "DeepSpeedWorker", "send_weights")
    assert send(worker)["status"] == ("done" if has_sender else "not_initialized")
    assert events == (["prepare", ("send", weights)] if has_sender else ["prepare"])


class Remote:
    def __init__(self, value, events, tag):
        self.value, self.events, self.tag = value, events, tag

    def remote(self, *args, **kwargs):
        self.events.append((self.tag, args))

        async def result():
            return self.value

        return result()


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("width", [4, 8, 16])
def test_server_materializes_on_all_ranks_but_initializes_only_scheduled_senders(ready, width):
    events = []
    groups = [
        SimpleNamespace(sender_train_rank=rank, group_id=rank, world_size=2, replica_ids=[rank]) for rank in range(2)
    ]
    schedule = SimpleNamespace(groups=groups)
    workers = [
        SimpleNamespace(
            get_ip=Remote("10.0.0.1", events, ("ip", rank)),
            max_param_bytes=Remote(4096 if rank == width - 1 else 1024, events, ("max", rank)),
            init_weight_sender=Remote(True, events, ("init", rank)),
            send_weights=Remote({"status": "done"}, events, ("prepare", rank)),
        )
        for rank in range(width)
    ]
    server = SimpleNamespace(
        arctic_rl_ray_server_state=SimpleNamespace(wake_inference=Remote(None, events, "wake")),
        weight_sync_ready=ready,
        weight_sync_bucket_size=4096,
    )

    async def receive(**kwargs):
        assert len(kwargs["groups"]) == 2
        assert kwargs["bucket_size"] == 4096
        assert all(group["master_port"] == 29500 for group in kwargs["groups"])
        events.append(("receive", ()))
        return {}

    sync = production_method(
        "ray_server.py",
        "ArcticRLRayServer",
        "_sync_weights_nccl",
        asyncio=asyncio,
        ReplicaPool=object,
        _transfer_schedule_cls=lambda: SimpleNamespace(build=lambda **kwargs: schedule),
        _WEIGHT_SYNC_BASE_PORT=29500,
        _WEIGHT_SYNC_BUCKET_SIZE=2048,
        time=SimpleNamespace(monotonic=lambda: 0),
        logger=SimpleNamespace(info=lambda *args: None),
    )
    pool = SimpleNamespace(num_replicas=2, tp_size=1, sync_weights=receive)
    assert asyncio.run(sync(server, workers, pool)) == {"status": "ok"}
    tags = [tag for tag, _ in events]
    assert [tag for tag in tags if isinstance(tag, tuple) and tag[0] == "prepare"] == [
        ("prepare", rank) for rank in range(width)
    ]
    assert [tag for tag in tags if isinstance(tag, tuple) and tag[0] == "max"] == (
        [] if ready else [("max", rank) for rank in range(width)]
    )
    init = [(tag, args) for tag, args in events if isinstance(tag, tuple) and tag[0] == "init"]
    assert [tag for tag, _ in init] == ([] if ready else [("init", 0), ("init", 1)])
    assert all(args[-1] == 4096 for _, args in init)
