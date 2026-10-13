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

"""Ensure the DeepEP lifecycle observer never changes the observed calls."""

import asyncio
import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.diagnostics.deepep_trace import buffer_call_trace
from arctic_platform.correctness.diagnostics.deepep_trace import tensor_metadata
from arctic_platform.correctness.diagnostics.deepep_trace import trace_exceptions
from arctic_platform.correctness.diagnostics.deepep_trace import traced_workers


def test_tensor_metadata_does_not_read_or_synchronize_values(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("trace must not read tensor values")

    for name in ("cpu", "item", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    tensor = torch.empty(4, 3, dtype=torch.bfloat16).t()
    metadata = tensor_metadata(tensor)
    assert metadata["shape"] == [3, 4]
    assert metadata["dtype"] == "torch.bfloat16"
    assert metadata["stride"] == [1, 3]
    assert metadata["contiguous"] is False
    assert tensor_metadata(None) is None


@pytest.mark.parametrize("operation", ["dispatch", "combine"])
def test_trace_preserves_arguments_result_and_method_bindings(operation):
    records, calls = [], []
    result = (torch.empty(2, 3), None, object())

    class Buffer:
        num_nvl_bytes = 256
        num_rdma_bytes = 0

        def dispatch(self, *args, **kwargs):
            calls.append((args, kwargs))
            return result

        combine = dispatch

    original_dispatch, original_combine = Buffer.dispatch, Buffer.combine
    tensor, handle = torch.empty(4, 3), object()
    with buffer_call_trace(Buffer, records.append):
        actual = getattr(Buffer(), operation)(tensor, handle=handle, async_finish=True)
    assert actual is result
    assert calls == [((tensor,), {"handle": handle, "async_finish": True})]
    assert [entry["phase"] for entry in records] == ["enter", "return"]
    assert all(entry["operation"] == operation and entry["call"] == 1 for entry in records)
    assert records[0]["cached"] is True
    assert records[0]["x"]["shape"] == [4, 3]
    assert records[1]["x"]["shape"] == [2, 3]
    assert Buffer.dispatch is original_dispatch
    assert Buffer.combine is original_combine


def test_trace_reports_original_exception_and_restores_binding():
    records = []
    error = RuntimeError("kernel failed")

    class Buffer:
        def dispatch(self, *args, **kwargs):
            raise error

        combine = dispatch

    original = Buffer.dispatch
    with pytest.raises(RuntimeError) as caught:
        with buffer_call_trace(Buffer, records.append):
            Buffer().dispatch(x=torch.empty(1, 3))
    assert caught.value is error
    assert [entry["phase"] for entry in records] == ["enter", "error"]
    assert records[-1]["error"] == "RuntimeError: kernel failed"
    assert "raise error" in records[-1]["traceback"]
    assert Buffer.dispatch is original


def test_trace_call_numbers_span_both_operations():
    records = []

    class Buffer:
        def dispatch(self, **kwargs):
            return (kwargs["x"],)

        combine = dispatch

    with buffer_call_trace(Buffer, records.append):
        buffer = Buffer()
        buffer.dispatch(x=torch.empty(1, 2))
        buffer.combine(x=torch.empty(1, 2), handle=SimpleNamespace())
    assert [entry["call"] for entry in records] == [1, 1, 2, 2]


def test_layout_trace_preserves_return_and_records_original_exception():
    records = []
    error = RuntimeError("layout kernel failed")

    class Buffer:
        def get_dispatch_layout(self, topk_idx, num_experts):
            if num_experts == 0:
                raise error
            return topk_idx, None, None, None, object()

    original = Buffer.get_dispatch_layout
    tensor = torch.zeros(2, 4, dtype=torch.int64)
    with buffer_call_trace(Buffer, records.append, operations=("get_dispatch_layout",)):
        result = Buffer().get_dispatch_layout(topk_idx=tensor, num_experts=4)
        assert result[0] is tensor
        with pytest.raises(RuntimeError) as caught:
            Buffer().get_dispatch_layout(topk_idx=tensor, num_experts=0)
    assert caught.value is error
    assert Buffer.get_dispatch_layout is original
    assert [entry["phase"] for entry in records] == ["enter", "return", "enter", "error"]
    assert "layout kernel failed" in records[-1]["traceback"]


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_worker_exception_trace_keeps_original_exception(error_type):
    records = []
    error = error_type("original worker error")
    with pytest.raises(error_type) as caught:
        with trace_exceptions(records.append, "forward_backward"):
            raise error
    assert caught.value is error
    assert len(records) == 1
    assert records[0]["operation"] == "forward_backward"
    assert records[0]["phase"] == "error"
    assert records[0]["error"] == f"{error_type.__name__}: original worker error"
    assert "raise error" in records[0]["traceback"]


def test_worker_exception_trace_does_not_emit_on_success():
    records = []
    with trace_exceptions(records.append, "initialize"):
        pass
    assert records == []


@pytest.mark.parametrize(
    "failure,permutation_errors,serialized",
    [
        (failure, enabled, False)
        for failure in (None, "initialize", "forward_backward", "server")
        for enabled in (False, True)
    ]
    + [(None, True, True)],
)
def test_trace_worker_and_delayed_server_preserve_bindings_and_exceptions(
    monkeypatch, tmp_path, failure, permutation_errors, serialized
):
    import arctic_platform.common as common

    error = RuntimeError("original worker failure")
    permutation = types.ModuleType("arctic_platform.model.implementations.moe.distributed.deepep")

    def permute(*args):
        if failure == "forward_backward":
            raise error
        return args[0]

    permutation.permute_tokens = permute
    distributed = types.ModuleType("arctic_platform.model.implementations.moe.distributed")
    distributed.deepep = permutation
    monkeypatch.setitem(sys.modules, permutation.__name__, permutation)
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)

    class Buffer:
        def dispatch(self, *args, **kwargs):
            return (torch.empty(1, 2),)

        combine = dispatch
        get_dispatch_layout = dispatch

    class Worker:
        rank = 3

        def initialize(self, master_addr, job_config):
            if failure == "initialize":
                raise error
            return job_config

        def forward_backward(self, batch):
            permutation.permute_tokens(torch.ones(1, 2), torch.tensor([[-2, -1]]), torch.ones(1, 2), 1)
            return batch

    original_worker = SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=Worker))
    workers = types.ModuleType("arctic_platform.common.deepspeed_worker")
    workers.DeepSpeedWorker = original_worker
    server = types.ModuleType("arctic_platform.common.ray_server")
    server.DeepSpeedWorker = original_worker

    class Server:
        async def initialize(self, payload):
            assert workers.DeepSpeedWorker is server.DeepSpeedWorker
            assert workers.DeepSpeedWorker is not original_worker
            if failure == "server":
                raise error
            return server.DeepSpeedWorker

    server.ArcticRLRayServerState = Server

    def remote(cls):
        if serialized:
            import cloudpickle

            return cloudpickle.loads(cloudpickle.dumps(cls))
        return cls

    for name, module in (
        ("ray", SimpleNamespace(remote=remote)),
        ("deep_ep", SimpleNamespace(Buffer=Buffer)),
        (workers.__name__, workers),
        (server.__name__, server),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(common, "deepspeed_worker", workers, raising=False)
    monkeypatch.setattr(common, "ray_server", server, raising=False)
    worker = None
    try:
        with traced_workers(tmp_path, permutation_errors=permutation_errors):
            traced_worker = workers.DeepSpeedWorker
            delayed_server = server.ArcticRLRayServerState
            # A freshly imported actor sees original worker bindings.
            workers.DeepSpeedWorker = server.DeepSpeedWorker = original_worker
            if failure == "server":
                with pytest.raises(RuntimeError) as caught:
                    asyncio.run(delayed_server().initialize({}))
                assert caught.value is error
            else:
                assert asyncio.run(delayed_server().initialize({})) is traced_worker
                worker = traced_worker()
                if failure == "initialize":
                    with pytest.raises(RuntimeError) as caught:
                        worker.initialize("master", {})
                    assert permutation.permute_tokens is permute
                else:
                    config, batch = {}, {}
                    assert worker.initialize("master", config) is config
                    assert (permutation.permute_tokens is not permute) is permutation_errors
                    if failure == "forward_backward":
                        with pytest.raises(RuntimeError) as caught:
                            worker.forward_backward(batch)
                    else:
                        assert worker.forward_backward(batch) is batch
                if failure is not None:
                    assert caught.value is error
                    records = [json.loads(line) for line in worker._trace_path.read_text().splitlines()]
                    assert records[-1]["operation"] == failure
                    assert records[-1]["rank"] == 3
                    assert "raise error" in records[-1]["traceback"]
                    observed = [record for record in records if record["operation"] == "permute_tokens"]
                    assert bool(observed) is (permutation_errors and failure == "forward_backward")
                    if observed:
                        assert observed[0]["index_values"]["other_negative_sample"] == [-2]
            assert workers.DeepSpeedWorker is server.DeepSpeedWorker is original_worker
    finally:
        if worker is not None:
            worker._trace_context.__exit__(None, None, None)
    assert workers.DeepSpeedWorker is server.DeepSpeedWorker is original_worker
    assert server.ArcticRLRayServerState is Server
    assert permutation.permute_tokens is permute
