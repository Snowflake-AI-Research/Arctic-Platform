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

"""CPU integration contracts for private debug hooks, not vLLM deployment or GPU kernels."""

import importlib
import os
import subprocess
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch
from test_weight_sync_direct_writer import _FakeEngine
from torch import nn

from arctic_platform.debug import inference
from arctic_platform.debug import weight_sync
from arctic_platform.inference.server.weight_sync.receiver import WeightSyncExtension


class _PlacesOnlyWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def load_weights(self, weights):
        name, tensor = next(iter(weights))
        if name == "weight":
            self.weight.data.copy_(tensor)
            return {"weight"}
        return set()


def _install_linear_stubs(monkeypatch):
    linear = ModuleType("vllm.model_executor.layers.linear")
    linear.MergedColumnParallelLinear = type("MergedColumnParallelLinear", (nn.Module,), {})
    linear.QKVParallelLinear = type("QKVParallelLinear", (nn.Module,), {})
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", linear)


@pytest.fixture(autouse=True)
def disabled_by_default(monkeypatch):
    monkeypatch.delenv(weight_sync.VERIFY_ENV, raising=False)
    monkeypatch.delenv(inference.STARTUP_REQUEST_ENV, raising=False)
    monkeypatch.delenv(inference.ATTEMPT_ENV, raising=False)
    assert weight_sync._LORA_BUILD_EVIDENCE.get() is None
    yield
    assert weight_sync._LORA_BUILD_EVIDENCE.get() is None


def test_debug_gateway_imports_do_not_require_correctness():
    code = """
import importlib.abc
import sys

class NoCorrectness(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("arctic_platform.correctness"):
            raise AssertionError("ordinary mode imported correctness")

sys.meta_path.insert(0, NoCorrectness())
import arctic_platform.debug.weight_sync
import arctic_platform.debug.inference
import arctic_platform.debug.training
"""
    result = subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_disabled_lora_builder_preserves_identity_and_does_not_hash(monkeypatch):
    tensors = {"raw": torch.ones(1)}
    sentinel = object()
    calls = []
    monkeypatch.setattr(weight_sync, "tensor_manifest", lambda value: pytest.fail("disabled hashing"))

    def builder(**kwargs):
        calls.append(kwargs)
        weight_sync.maybe_record_lora_remapping(tensors, "prefix", lambda *_: pytest.fail("disabled remap"))
        return sentinel

    model, evidence = weight_sync.maybe_build_lora_model(builder, tensors=tensors, other=sentinel)

    assert model is sentinel
    assert evidence is None
    assert calls == [{"tensors": tensors, "other": sentinel}]


def test_disabled_native_result_and_exception_objects_are_unchanged():
    result = {"native": object()}
    assert weight_sync.maybe_broadcast_receiver_result(result, None, rank=3) is result
    assert weight_sync.maybe_lora_built_receiver_result(result, None) is result
    error = RuntimeError("native loading failed")
    with pytest.raises(RuntimeError) as raised:
        weight_sync.maybe_retain_weight_sync_failure(None, error)
    assert raised.value is error


def test_disabled_broadcast_calls_native_send_once_with_original_stream():
    weights = iter([("weight", torch.ones(1))])
    result = {"native": object()}
    calls = []

    def send(stream):
        calls.append(stream)
        return result

    assert weight_sync.maybe_send_broadcast_weights(object(), weights, send) is result
    assert calls == [weights]


def test_broadcast_wrapper_annotates_only_after_native_send(monkeypatch):
    from contextlib import contextmanager

    events = []
    stream = object()
    result = {"native": 4}

    @contextmanager
    def observe(sender, weights, send):
        evidence = {}
        events.append("enter")
        yield stream, evidence
        events.append("exit")
        evidence["observed"] = True

    def send(value):
        assert value is stream
        events.append("send")
        return result

    monkeypatch.setattr(weight_sync, "maybe_observe_broadcast_send", observe)
    actual = weight_sync.maybe_send_broadcast_weights(object(), object(), send)
    assert events == ["enter", "send", "exit"]
    assert actual == {"native": 4, "observed": True}
    assert result == {"native": 4}


@pytest.mark.parametrize("batched", [False, True])
def test_receiver_enabled_shadow_has_exact_destination_evidence(monkeypatch, batched):
    monkeypatch.setenv(weight_sync.VERIFY_ENV, "1")
    _install_linear_stubs(monkeypatch)
    model = _PlacesOnlyWeight()
    receiver = WeightSyncExtension()
    receiver.device = torch.device("cpu")
    receiver.parallel_config = SimpleNamespace(enable_expert_parallel=False, pipeline_parallel_size=1)
    engine = _FakeEngine([("weight", torch.tensor([7.0]))])
    verifier = weight_sync.maybe_begin_broadcast_load(receiver, model, engine)

    loader = receiver._load_batched if batched else receiver._load_direct
    loaded = weight_sync.maybe_run_broadcast_load(loader, model, engine, verifier)
    actual = weight_sync.maybe_broadcast_receiver_result({"loaded": loaded[0]}, verifier, rank=2)
    evidence = actual["weight_sync_verification"]

    assert actual["loaded"] == 1
    assert evidence["rank"] == 2
    assert evidence["received"]["weight"]["destinations"] == ["weight"]
    assert evidence["destinations"]["weight"] == {"uncovered": 0, "mismatched": 0}
    assert evidence["changed_after_verification"] == []
    torch.testing.assert_close(model.weight, torch.tensor([7.0]), rtol=0, atol=0)


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_broadcast_dispatch_preserves_native_arguments_results_and_errors(observed, raises):
    model, engine, verifier, result = object(), object(), object(), object()
    error = ValueError("native load error")
    calls = []

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        if raises:
            raise error
        return result

    def run():
        return weight_sync.maybe_run_broadcast_load(loader, model, engine, verifier if observed else None)

    if raises:
        with pytest.raises(ValueError) as raised:
            run()
        assert raised.value is error
    else:
        assert run() is result
    assert calls == [((model, engine), {"verifier": verifier} if observed else {})]


def test_disabled_fp8_receiver_preserves_main_updater_arguments_and_pending_warning(monkeypatch, caplog):
    from arctic_platform.inference.server.weight_sync import utils

    model = object()
    tensor = torch.ones(1)
    calls = []

    class Updater:
        pending = 1

        def __init__(self, selected_model, dtype, device):
            calls.append((selected_model, dtype, device))

        def feed(self, name, value):
            calls.append((name, value))

    monkeypatch.setattr(utils, "_FP8InplaceUpdater", Updater)
    receiver = SimpleNamespace(
        device=torch.device("cpu"),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    assert WeightSyncExtension._load_fp8(receiver, model, _FakeEngine([("weight", tensor)])) == (1, 1.0)
    assert calls == [(model, torch.bfloat16, receiver.device), ("weight", tensor)]
    assert "FP8 updater has 1 incomplete modules" in caplog.text


@pytest.mark.parametrize("dequantization", [False, True])
def test_fp8_staging_adapter_tracks_native_buffers_without_public_api(dequantization):
    buffer = torch.zeros(2, 3)
    half = torch.zeros(3, 5)
    updater = SimpleNamespace(_bufs={"proj": buffer})
    if dequantization:
        updater._dequant = {"wire": {"weight": half, "destination": "target.weight"}}
    verifier = SimpleNamespace(parameters={"proj.weight": None, "target.weight": None})
    staged = weight_sync._fp8_staging(verifier, updater)
    expected = {weight_sync._storage(buffer): ["proj.weight"]}
    if dequantization:
        expected[weight_sync._storage(half)] = ["target.weight"]
    assert staged() == expected
    updater._bufs.clear()
    if dequantization:
        updater._dequant.clear()
    assert staged() == {}
    assert not hasattr(updater, "staged")


def test_enabled_lora_remap_overwrite_and_packed_slots(monkeypatch):
    monkeypatch.setenv(weight_sync.VERIFY_ENV, "1")
    lora_model_module = ModuleType("vllm.lora.lora_model")
    lora_utils_module = ModuleType("vllm.lora.utils")
    lora_model_module.LoRAModel = SimpleNamespace(_should_skip_module=lambda name, prefixes: name.startswith(prefixes))
    lora_utils_module.is_base_embedding_weights = lambda name: name.startswith("embedding")
    lora_utils_module.parse_fine_tuned_lora_name = lambda name, mapper: (name, True)
    monkeypatch.setitem(sys.modules, "vllm.lora", ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.lora_model", lora_model_module)
    monkeypatch.setitem(sys.modules, "vllm.lora.utils", lora_utils_module)
    tensors = {name: torch.tensor([float(index)]) for index, name in enumerate(["first", "last", "skip", "embed"])}
    remapped = {"first": "layer.experts.w1", "last": "layer.experts.w1", "skip": "skip.layer", "embed": "embedding"}
    factor = torch.ones(1)

    def builder(**kwargs):
        assert kwargs["tensors"] is tensors
        weight_sync.maybe_record_lora_remapping(tensors, "", lambda name, prefix: remapped[name])
        model = SimpleNamespace(loras={"layer.experts.w1": SimpleNamespace(lora_a=factor)})
        weight_sync.maybe_record_built_lora_factors(model, None, ("skip",))
        assert weight_sync.maybe_pack_fused_expert_loras(lambda _: 1, model) == 1
        return model

    model, evidence = weight_sync.maybe_build_lora_model(builder, tensors=tensors)
    assert evidence.factors == {"first": None, "last": ("layer.experts", "lora_a", 0), "skip": None, "embed": None}
    assert evidence.received["last"] == weight_sync.tensor_manifest(tensors["last"])
    manager = SimpleNamespace(
        modules={"layer.experts"},
        get_adapter=lambda identifier: model,
        _get_lora_layer_weights=lambda adapter, name: SimpleNamespace(lora_a=[factor, None, None]),
    )
    result = weight_sync.maybe_lora_built_receiver_result(
        {"native": True}, evidence, manager=manager, lora_int_id=1, rank=0, world_size=1
    )
    entries = result["weight_sync_verification"]["received"]
    assert entries["first"]["destinations"] == []
    assert entries["last"]["destinations"] == ["layer.experts.lora_a[0]"]
    assert entries["skip"]["destinations"] == []


def test_enabled_lora_builder_failure_resets_context(monkeypatch):
    monkeypatch.setenv(weight_sync.VERIFY_ENV, "1")
    error = ValueError("native builder error")

    def builder(**kwargs):
        assert weight_sync._LORA_BUILD_EVIDENCE.get() is not None
        raise error

    with pytest.raises(ValueError) as raised:
        weight_sync.maybe_build_lora_model(builder, tensors={"one": torch.ones(1)})
    assert raised.value is error
    assert weight_sync._LORA_BUILD_EVIDENCE.get() is None


@pytest.mark.parametrize("outcome", ["ready", "retry", "error"])
def test_startup_state_helpers_preserve_attempt_outcome_and_error(monkeypatch, outcome):
    attempt = object()
    error = ValueError("startup")
    calls = []
    monkeypatch.setattr(inference, "maybe_begin_inference_startup_attempt", lambda ordinal: attempt)
    monkeypatch.setattr(inference, "maybe_finish_inference_startup_attempt", lambda *args: calls.append(args))
    state = inference.maybe_begin_inference_startup_attempt_state(3)
    if outcome == "retry":
        inference.maybe_record_inference_startup_retry(state, error)
    elif outcome == "error":
        inference.maybe_record_inference_startup_failure(state, error)
    inference.maybe_complete_inference_startup_attempt(state)
    assert calls == [(attempt, outcome, None if outcome == "ready" else error)]


def test_disabled_startup_state_helpers_are_noops(monkeypatch):
    monkeypatch.setattr(inference, "maybe_finish_inference_startup_attempt", lambda *_: pytest.fail("disabled finish"))
    state = inference.maybe_begin_inference_startup_attempt_state(0)
    assert state is None
    inference.maybe_record_inference_startup_failure(state, ValueError("unused"))
    inference.maybe_record_inference_startup_retry(state, ValueError("unused"))
    inference.maybe_complete_inference_startup_attempt(state)


def test_cache_hook_observes_exact_consumed_tensor(monkeypatch):
    cache_module = importlib.import_module("arctic_platform.inference.server.router_replay.cache")
    observed = []
    monkeypatch.setattr(cache_module, "maybe_record_router_replay_received", lambda *args: observed.append(args))
    cache = cache_module.RouterReplayCacheRX("cpu", 1024)
    cache.put("sample", torch.ones(2, 1, 1, dtype=torch.uint8))
    held = cache.get("sample")
    actual = cache.pop("sample")
    assert actual is held
    assert observed == [("sample", held)]
    assert len(cache) == 0


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_single_call_load_helpers_preserve_native_result_exception_and_observation(fp8, observed, raises):
    from contextlib import contextmanager

    events = []
    tensor = torch.ones(1)
    result = object()
    error = RuntimeError("native caller failure")

    def load(name, value):
        events.append("load")
        assert name == "weight"
        assert value is tensor
        if raises:
            raise error
        return result

    @contextmanager
    def observe(name, value, *, loader, **kwargs):
        assert name == "weight"
        assert value is tensor
        assert loader is load
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    verifier = SimpleNamespace(observe_load=observe, parameters={}) if observed else None
    updater = SimpleNamespace(feed=load, _bufs={}, _dequant={})
    replay = SimpleNamespace(feed=load, _bufs={}, _dequant={})

    def run():
        if fp8:
            return weight_sync.maybe_feed_fp8_tensor(verifier, updater, replay, "weight", tensor)
        return weight_sync.maybe_load_broadcast_tensor(verifier, load, "weight", tensor)

    if raises:
        with pytest.raises(RuntimeError) as raised:
            run()
        assert raised.value is error
    else:
        assert run() is result
    assert events == (["enter", "load", "exit"] if observed else ["load"])
