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

import gc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.testing_utils import require_torch_gpu


@pytest.fixture
def deepep(monkeypatch):
    return _load_deepep(monkeypatch)


def _load_deepep(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "arctic_platform/model/implementations/moe"
    package_name = "_deepep_lifetime_tests"
    for name in (package_name, f"{package_name}.distributed"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    # Load the real tensor helpers without importing the optional distributed package.
    modules = {}
    for relative in (
        "token_combine",
        "routing_weights",
        "distributed.token_permute",
        "distributed.combine_replay",
        "distributed.deepep",
    ):
        name = f"{package_name}.{relative}"
        spec = importlib.util.spec_from_file_location(name, root / f"{relative.replace('.', '/')}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        modules[relative] = (spec, module)

    for relative, (spec, module) in modules.items():
        if relative != "distributed.deepep":
            spec.loader.exec_module(module)

    extension = ModuleType("deep_ep")
    extension.Buffer = object
    extension_utils = ModuleType("deep_ep.utils")
    extension_utils.EventHandle = object
    extension_utils.EventOverlap = object

    class CudaLibrary:
        def __init__(self, *args):
            pass

        def define(self, schema):
            pass

    # Only extension imports and CUDA registration are stubbed; lifetime logic is unchanged.
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "deep_ep", extension)
        imports.setitem(sys.modules, "deep_ep.utils", extension_utils)
        imports.setattr(torch.library, "Library", CudaLibrary)
        imports.setattr(torch.library, "impl", lambda *args: lambda function: function)
        imports.setattr(torch.library, "register_autograd", lambda *args, **kwargs: None)
        spec, module = modules["distributed.deepep"]
        spec.loader.exec_module(module)
    return modules["distributed.deepep"][1]


@pytest.fixture
def dispatched(monkeypatch, deepep):
    event_calls = []
    event = SimpleNamespace(current_stream_wait=lambda: event_calls.append("wait"))
    handle = object()
    buffer = SimpleNamespace(
        dispatch=lambda **kwargs: (kwargs["x"], kwargs["topk_idx"], kwargs["topk_weights"], [1], handle, event)
    )
    monkeypatch.setattr(deepep, "_buffer", buffer)
    monkeypatch.setattr(deepep, "_handle_cache", {})
    monkeypatch.setattr(deepep, "_pending_dispatch_events", {})
    monkeypatch.setattr(deepep, "_new_event_overlap", lambda: None)
    finalizers = []
    finalize = deepep.weakref.finalize

    def record_finalizer(*args, **kwargs):
        result = finalize(*args, **kwargs)
        finalizers.append(result)
        return result

    monkeypatch.setattr(deepep.weakref, "finalize", record_finalizer)
    args = [torch.ones(1, 1)] * 7
    output = deepep._dispatch_op_impl(*args)
    return SimpleNamespace(output=output, event_calls=event_calls, handle=handle, args=args, finalizers=finalizers)


def test_live_dispatch_does_not_finalize_during_interpreter_shutdown(dispatched):
    assert len(dispatched.finalizers) == 1
    assert dispatched.finalizers[0].alive
    assert dispatched.finalizers[0].atexit is False


def test_abandoned_dispatch_releases_handle_and_waits_for_event(dispatched, deepep):
    key = dispatched.output[-1].item()
    assert deepep._handle_cache[key] is dispatched.handle
    assert key in deepep._pending_dispatch_events
    dispatched.output = None
    gc.collect()
    assert key not in deepep._handle_cache
    assert key not in deepep._pending_dispatch_events
    assert dispatched.event_calls == ["wait"]


def test_live_dispatch_remains_available_and_autograd_owns_its_handle(dispatched, deepep):
    key = dispatched.output[-1].item()
    ctx = SimpleNamespace()
    deepep._dispatch_setup_context(ctx, dispatched.args, dispatched.output)
    gc.collect()
    assert deepep._handle_cache[key] is ctx.saved_handle
    assert dispatched.event_calls == []
    dispatched.output = None
    gc.collect()
    assert key not in deepep._handle_cache
    assert ctx.saved_handle is dispatched.handle


def test_consumed_dispatch_is_not_waited_or_released_twice(dispatched, deepep):
    key = dispatched.output[-1].item()
    deepep._sync_dispatch(key)
    saved = deepep._handle_cache.pop(key)
    dispatched.output = None
    gc.collect()
    assert saved is dispatched.handle
    assert dispatched.event_calls == ["wait"]
    assert deepep._handle_cache == {}
    assert deepep._pending_dispatch_events == {}


@require_torch_gpu
def test_abandoned_dispatch_through_registered_cuda_op(monkeypatch):
    deepep = _load_deepep(monkeypatch)
    events = []
    handle = object()

    def dispatch(**kwargs):
        return kwargs["x"].clone(), kwargs["topk_idx"].clone(), kwargs["topk_weights"].clone(), [1], handle, event

    event = SimpleNamespace(current_stream_wait=lambda: events.append("wait"))
    monkeypatch.setattr(deepep, "_buffer", SimpleNamespace(dispatch=dispatch))
    monkeypatch.setattr(deepep, "_new_event_overlap", lambda: None)
    deepep._deepep_cuda_ops_registered = False
    deepep.register_deepep_cuda_ops()
    for requires_grad in (False, True):
        x = torch.ones(1, 8, device="cuda", requires_grad=requires_grad)
        indices = torch.zeros(1, 1, dtype=torch.int64, device="cuda")
        weights = torch.ones(1, 1, device="cuda")
        counts = torch.ones(1, dtype=torch.int32, device="cuda")
        output = torch.ops.ap_deepep.dispatch(x, indices, weights, counts, counts, indices.bool(), counts)
        key = output[-1].item()
        assert deepep._handle_cache[key] is handle
        assert key in deepep._pending_dispatch_events
        del output
        gc.collect()
        torch.cuda.synchronize()
        assert key not in deepep._handle_cache
        assert key not in deepep._pending_dispatch_events
    assert events == ["wait", "wait"]
