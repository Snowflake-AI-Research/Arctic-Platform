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

"""Replay captures must reach the exact training rows without copying the pool."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]


def production_definition(filename, name, class_name=None, **namespace):
    # Isolate real definitions from Ray/CUDA initialization on CPU test hosts.
    path = ROOT / "arctic_platform/common" / filename
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name)
    definition = next(node for node in nodes if getattr(node, "name", None) == name)
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def state_method(name):
    return production_definition(
        "ray_server.py",
        name,
        "ArcticRLRayServerState",
        asyncio=asyncio,
        Any=Any,
        ReplicaPool=object,
        _ROUTER_REPLAY_PORT=29610,
        unpack_batch=lambda shard: ({}, shard["batch"], {}, {}),
    )


def worker_method(name):
    return production_definition("deepspeed_worker.py", name, "DeepSpeedWorker", torch=torch)


class Remote:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


@pytest.mark.parametrize("native", [False, True])
def test_native_pool_stays_in_owning_state_actor(native):
    pool = object()
    state = SimpleNamespace(_router_replay_native=native, sampling_pool=pool)
    assert asyncio.run(state_method("get_sampling_pool")(state)) is (None if native else pool)


@pytest.mark.parametrize(
    "prompts,parameters",
    [
        ([[1, 2], [3]], [{"max_tokens": 1}, {"max_tokens": 2}]),
        (["text", [4, 5]], [None, {"temperature": 0.0}]),
        (["text"], {"max_tokens": 1}),
    ],
)
def test_generation_schema_preserves_token_prompts_and_parameter_rows(prompts, parameters):
    request_type = production_definition("utils/server_models.py", "GenerateRequest", BaseModel=BaseModel, Any=Any)
    request = request_type(prompts=prompts, sampling_params=parameters)
    assert request.prompts == prompts
    assert request.sampling_params == parameters


@pytest.mark.parametrize("rows,chunks", [(7, 3), (11, 4), (10, 4), (6, 3), (4, 4)])
def test_replay_ids_and_tensor_rows_share_uneven_shard_boundaries(rows, chunks):
    split = production_definition("utils/batch.py", "_split_value", torch=torch, Any=Any)
    tensors = split(torch.arange(rows), chunks)
    ids = split([f"rr1:{row}" for row in range(rows)], chunks)
    assert ids == [[f"rr1:{row}" for row in tensor.tolist()] for tensor in tensors]


def test_receiver_assembly_uses_original_noncontiguous_sequence_positions():
    captures = [
        torch.arange(20).reshape(5, 2, 2).to(torch.uint8),
        torch.arange(40, 56).reshape(4, 2, 2).to(torch.uint8),
    ]
    mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
    positions = torch.tensor([[0, 3, 1], [2, 4, -1]])
    assembled = worker_method("_assemble_routed_experts")(captures, mask, "cpu", position_ids=positions)
    assert assembled.shape == (2, 3, 2, 2)
    assert assembled.dtype == torch.int64
    torch.testing.assert_close(assembled[0, 1:], captures[0][[3, 1]].long())
    torch.testing.assert_close(assembled[1, 0], captures[1][2].long())
    assert assembled[0, 0].count_nonzero() == 0
    assert assembled[1, 1:].count_nonzero() == 0


def test_receiver_consumes_exact_cache_entries_before_packing():
    capture = torch.arange(12).reshape(3, 2, 2).to(torch.uint8)
    cache = {"rr1:a": capture, "rr1:b": capture + 40}
    worker = SimpleNamespace(
        _device="cpu",
        _router_replay_cache_rx=SimpleNamespace(pop=cache.pop),
        _assemble_routed_experts=worker_method("_assemble_routed_experts"),
    )
    batch = {
        "attention_mask": torch.ones(2, 4),
        "rr_sample_ids": ["rr1:b", "rr1:a"],
        "rr_replay_meta": [{"capture_len": 3}] * 2,
        "rr_discard": True,
    }
    worker_method("_consume_router_replay")(worker, batch)
    assert cache == {}
    assert not any(key.startswith("rr_") for key in batch)
    torch.testing.assert_close(batch["routed_experts"][0, :3], (capture + 40).long())
    torch.testing.assert_close(batch["routed_experts"][1, :3], capture.long())
    assert batch["routed_experts"][:, 3].count_nonzero() == 0


def test_ordinary_batch_does_not_need_replay_cache():
    batch = {"input_ids": torch.ones(1, 2, dtype=torch.long)}
    worker_method("_consume_router_replay")(SimpleNamespace(_router_replay_cache_rx=None), batch)
    assert list(batch) == ["input_ids"]


@pytest.mark.parametrize("failed_validation", [False, True])
def test_transport_owns_one_store_and_validates_captures_before_discard(failed_validation):
    events = []

    async def address():
        return "10.0.0.1"

    async def initialize(*args, **kwargs):
        events.append(("initialize", args, kwargs))
        return {"status": "ok"}

    async def receive(ids, **kwargs):
        events.append(("receive", ids, kwargs))
        return {"status": "ok", "validations": [{"valid": not failed_validation} for _ in ids]}

    async def close():
        events.append("close")
        return {"status": "ok"}

    async def pool_initialize(**kwargs):
        events.append(("pool-initialize", kwargs))
        return {"workers": [{"status": "ok"}]}

    async def send():
        events.append("send")
        return {"workers": [{"status": "ok"}]}

    async def discard(ids):
        events.append(("discard", ids))
        return {"status": "ok"}

    workers = [
        SimpleNamespace(
            get_ip=Remote(address),
            init_router_replay=Remote(initialize),
            recv_router_replay=Remote(receive),
            close_router_replay=Remote(close),
        )
        for _ in range(2)
    ]
    state = SimpleNamespace(
        jobs={1: {"job_type": "training"}, 2: {"job_type": "sampling"}},
        training_workers=workers,
        sampling_pool=SimpleNamespace(
            _workers=[object()],
            init_router_replay=pool_initialize,
            send_router_replay=send,
            discard_router_replay=discard,
            close_router_replay=close,
        ),
        _router_replay_jobs=None,
        _router_replay_lock=asyncio.Lock(),
    )
    for name in (
        "_complete_router_replay",
        "_init_router_replay",
        "_receive_router_replay",
        "_require_router_replay_results",
        "close_router_replay",
    ):
        function = state_method(name)
        setattr(state, name, lambda *args, function=function: function(state, *args))
    shards = [
        {"batch": {"rr_sample_ids": [f"rr1:{rank}"], "rr_replay_meta": [{"capture_len": 3}], "rr_discard": True}}
        for rank in range(2)
    ]
    if failed_validation:
        with pytest.raises(RuntimeError, match="invalid capture"):
            asyncio.run(state_method("receive_router_replay")(state, 1, 2, shards))
        assert state._router_replay_jobs is None
        assert events.count("close") == 3
        assert not any(isinstance(event, tuple) and event[0] == "discard" for event in events)
    else:
        result = asyncio.run(state_method("receive_router_replay")(state, 1, 2, shards))
        assert result["status"] == "ok"
        assert state._router_replay_jobs == (1, 2)
        assert ("discard", ["rr1:0", "rr1:1"]) in events
    initializes = [event for event in events if isinstance(event, tuple) and event[0] == "initialize"]
    assert initializes == [("initialize", ("10.0.0.1", 29610, rank, 3), {"is_server": rank == 0}) for rank in range(2)]
