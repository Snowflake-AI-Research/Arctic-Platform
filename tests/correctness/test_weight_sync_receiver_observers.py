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

"""Native receiver calls must work with their optional observations removed."""

import ast
import logging
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.common.peft import normalize_lora_peft_config
from arctic_platform.common.weight_checksum import tensor_l2_sq
from arctic_platform.debug import weight_sync as evidence
from arctic_platform.testing_utils import torch_assert_equal

_SOURCE = Path(__file__).resolve().parents[2] / "arctic_platform/inference/server/weight_sync/receiver.py"


def _optional_call(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith("maybe_")


class _RemoveObservations(ast.NodeTransformer):
    def visit_Expr(self, node):
        return None if _optional_call(node.value) else self.generic_visit(node)

    def visit_Assign(self, node):
        return None if _optional_call(node.value) else self.generic_visit(node)

    def visit_With(self, node):
        node = self.generic_visit(node)
        if all(_optional_call(item.context_expr) for item in node.items):
            return node.body
        return node


@pytest.fixture(params=[False, True], ids=["debug-disabled", "debug-removed"])
def receiver(request, monkeypatch):
    monkeypatch.delenv(evidence.VERIFY_ENV, raising=False)
    source = ast.parse(_SOURCE.read_text())
    # Compile the real receiver definitions without importing the optional vLLM/NCCL package.
    source.body = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    source.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    if request.param:
        source = _RemoveObservations().visit(source)
        assert not any(isinstance(node, ast.Name) and node.id.startswith("maybe_") for node in ast.walk(source))
    namespace = {
        **(
            {name: value for name, value in vars(evidence).items() if name.startswith("maybe_")}
            if not request.param
            else {}
        ),
        "__name__": __name__,
        "torch": torch,
        "logger": logging.getLogger(__name__),
        "os": os,
        "time": time,
        "_tensor_l2_sq": tensor_l2_sq,
        "normalize_lora_peft_config": normalize_lora_peft_config,
    }
    ast.fix_missing_locations(source)
    exec(compile(source, str(_SOURCE), "exec"), namespace)
    namespace["_FUSED_EXPERT_PROJS"] = ("w1", "w2", "w3")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return SimpleNamespace(**namespace)


def _module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    vars(module).update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        self.calls = []

    def load_weights(self, weights):
        self.calls.append(weights)
        self.weight.copy_(weights[0][1])
        return {"weight"}


class _Engine:
    def __init__(self, value):
        self.value = value

    def receive_weights(self):
        yield "weight", self.value


def _worker(receiver, model):
    worker = receiver.WeightSyncExtension()
    worker.device = torch.device("cpu")
    worker.model_config = SimpleNamespace(dtype=torch.float32, quantization=None)
    worker.model_runner = SimpleNamespace(model=model)
    return worker


@pytest.mark.parametrize("path", ["direct", "fused", "model"])
def test_native_tensor_writes_survive_removing_observations(receiver, monkeypatch, path):
    model = _Model()
    value = torch.tensor([3.0, 4.0])
    writes = []

    class Writer:
        def __init__(self, model, device):
            self.model = model

        def get_view(self, name):
            return self.model.weight.data

        def get_destination(self, name):
            return "weight"

        def destination_name(self, name):
            return "weight" if path == "fused" else None

        def feed(self, name, tensor):
            writes.append((name, tensor))
            self.model.weight.copy_(tensor)

    _module(
        monkeypatch,
        "arctic_platform.inference.server.weight_sync.utils",
        _DirectParamWriter=Writer,
        _ShardAwareFusedWriter=Writer,
        _is_lora_adapter_name=lambda name: False,
    )
    worker = _worker(receiver, model)
    result = (worker._load_direct if path == "direct" else worker._load_batched)(model, _Engine(value))
    assert result == (1, 25.0, 25.0, {"weight": ["weight"]}, [])
    torch_assert_equal(model.weight, value)
    assert len(writes) == (1 if path == "fused" else 0)
    assert len(model.calls) == (1 if path == "model" else 0)
    if writes:
        assert writes[0][0] == "weight" and writes[0][1] is value
    if model.calls:
        assert model.calls[0][0][0] == "weight" and model.calls[0][0][1] is value


@pytest.mark.parametrize("failure", [False, True])
def test_native_fp8_feed_survives_removing_observations(receiver, monkeypatch, failure):
    model = _Model()
    value = torch.tensor([3.0, 4.0])
    feeds = []

    class Updater:
        pending = 0
        unplaced = ()

        def __init__(self, model, dtype, device):
            pass

        def feed(self, name, tensor):
            feeds.append((name, tensor))
            if failure:
                raise ValueError("native quantizer failed")
            model.weight.copy_(tensor)

    _module(
        monkeypatch,
        "arctic_platform.inference.server.weight_sync.utils",
        _FP8InplaceUpdater=Updater,
        _is_lora_adapter_name=lambda name: False,
    )
    worker = _worker(receiver, model)
    if failure:
        with pytest.raises(RuntimeError, match="FP8 weight sync failed") as caught:
            worker._load_fp8(model, _Engine(value))
        assert isinstance(caught.value.__cause__, ValueError)
    else:
        assert worker._load_fp8(model, _Engine(value)) == (1, 25.0)
        torch_assert_equal(model.weight, value)
    assert len(feeds) == 1 and feeds[0][0] == "weight" and feeds[0][1] is value


@pytest.mark.parametrize(
    "weight_format,quantization,tp,path",
    [
        ("hf", None, 1, "batched"),
        ("vllm", "fp8", 1, "fp8"),
        ("vllm", None, 1, "direct"),
        ("vllm", None, 2, "batched"),
    ],
)
def test_native_loader_dispatch_survives_removing_observations(
    receiver, monkeypatch, weight_format, quantization, tp, path
):
    model = _Model()
    worker = _worker(receiver, model)
    worker.model_config.quantization = quantization
    engine = object()
    calls = []
    worker._get_or_create_broadcast_engine = lambda *args: engine
    worker._build_result = lambda *args: {"status": "done", "path": args[2], "loaded": args[3]}

    def load(actual_model, actual_engine):
        assert actual_model is model and actual_engine is engine
        calls.append(path)
        return (1, 25.0) if path == "fp8" else (1, 25.0, 25.0, {"weight": ["weight"]}, [])

    setattr(worker, f"_load_{path}", load)
    _module(
        monkeypatch,
        "vllm.distributed.parallel_state",
        get_tensor_model_parallel_world_size=lambda: tp,
        get_world_group=lambda: SimpleNamespace(rank=0),
    )
    _module(monkeypatch, "arctic_platform.inference.envs", ARCTIC_INFERENCE_DUMP_PARAM_L2=False)
    result = worker.sync_weights_broadcast("localhost", 12345, 1, 2, weight_format=weight_format)
    assert calls == [path]
    assert result["status"] == "done" and result["path"] == path and result["loaded"] == 1
    assert "weight_sync_verification" not in result


def test_native_lora_builder_and_pack_survive_removing_observations(receiver, monkeypatch):
    packed = object()
    calls = []
    loras = {f"experts.{projection}": object() for projection in ("w1", "w2", "w3")}
    original_parts = list(loras.values())
    adapter = SimpleNamespace(loras=loras)

    class LoRAModel:
        @staticmethod
        def from_lora_tensors(**kwargs):
            calls.append(kwargs)
            return adapter

    class PackedWeights:
        @staticmethod
        def pack_moe_stacked(parts, module):
            assert parts == original_parts and module == "experts"
            calls.append("pack")
            return packed

    _module(monkeypatch, "vllm.lora.lora_model", LoRAModel=LoRAModel)
    _module(monkeypatch, "vllm.lora.lora_weights", PackedLoRALayerWeights=PackedWeights)
    _module(monkeypatch, "vllm.lora.peft_helper", PEFTHelper=SimpleNamespace(from_dict=lambda value: value))
    result = receiver._build_lora_model(
        7,
        {"weight": torch.ones(2)},
        {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 2, "lora_alpha": 2, "bias": "none", "target_modules": []},
        torch.device("cpu"),
        torch.float32,
    )
    assert result is adapter and adapter.loras == {"experts": packed}
    assert len(calls) == 2 and calls[0]["lora_model_id"] == 7 and calls[1] == "pack"


@pytest.mark.parametrize("evict_first", [False, True])
@pytest.mark.parametrize("failure", [None, "build", "activate"])
def test_native_lora_sync_survives_removing_observations(receiver, monkeypatch, evict_first, failure):
    model = _Model()
    worker = _worker(receiver, model)
    value = torch.tensor([3.0, 4.0])
    engine = _Engine(value)
    adapter = SimpleNamespace(loras={})
    events = []

    def build(**kwargs):
        events.append("build")
        torch_assert_equal(kwargs["tensors"]["weight"], value)
        assert kwargs["tensors"]["weight"] is not value
        assert kwargs["lora_int_id"] == 7 and kwargs["model"] is model
        if failure == "build":
            raise ValueError("native adapter build failed")
        return adapter

    def remove(adapter_id):
        assert adapter_id == 7
        events.append("remove")
        return True

    def add(actual):
        assert actual is adapter
        events.append("add")
        return True

    def activate(adapter_id):
        assert adapter_id == 7
        events.append("activate")
        return failure != "activate"

    engine.release_staging_buffers = lambda: events.append("release")
    worker._get_or_create_broadcast_engine = lambda *args: engine
    worker._lora_adapter_manager = lambda: SimpleNamespace(
        remove_adapter=remove, add_adapter=add, activate_adapter=activate
    )
    worker._build_result = lambda *args: {"status": "done", "path": args[2], "loaded": args[3]}
    monkeypatch.setitem(worker.sync_lora_weights_broadcast.__func__.__globals__, "_build_lora_model", build)
    _module(monkeypatch, "vllm.distributed.parallel_state", get_world_group=lambda: SimpleNamespace(rank=0))
    _module(monkeypatch, "vllm.utils.platform_utils", is_pin_memory_available=lambda: False)
    config = {
        "peft_type": "Lora",
        "task_type": "CAUSAL_LM",
        "r": 2,
        "lora_alpha": 2,
        "bias": "none",
        "target_modules": ["q_proj"],
    }
    if failure:
        error = ValueError if failure == "build" else RuntimeError
        with pytest.raises(error, match="native adapter build failed|failed to activate synced LoRA"):
            worker.sync_lora_weights_broadcast("localhost", 12345, 1, 2, 7, "adapter", config, evict_first=evict_first)
        assert "release" not in events
    else:
        result = worker.sync_lora_weights_broadcast(
            "localhost", 12345, 1, 2, 7, "adapter", config, evict_first=evict_first
        )
        assert result["status"] == "done" and result["loaded"] == 1 and result["recv_l2_sq"] == 25.0
        assert result["lora_added"] and result["lora_activated"] and result["lora_replaced"]
        assert "weight_sync_verification" not in result
        assert events == (["remove", "build"] if evict_first else ["build", "remove"]) + ["add", "activate", "release"]


@pytest.mark.parametrize("failure", [False, True])
def test_broadcast_observer_resets_context_after_native_failure(monkeypatch, failure):
    sentinel = object()
    monkeypatch.setattr(evidence, "maybe_begin_broadcast_load", lambda *args: sentinel)
    assert evidence.maybe_current_broadcast_verifier() is None

    def load():
        assert evidence.maybe_current_broadcast_verifier() is sentinel
        if failure:
            raise ValueError("native loader failed")

    if failure:
        with pytest.raises(ValueError, match="native loader failed"):
            with evidence.maybe_observe_broadcast_load(None, None, None):
                load()
    else:
        with evidence.maybe_observe_broadcast_load(None, None, None):
            load()
    assert evidence.maybe_current_broadcast_verifier() is None


@pytest.mark.parametrize("failure", [False, True])
def test_lora_pack_observer_records_only_successful_native_packing(monkeypatch, failure):
    monkeypatch.setenv(evidence.VERIFY_ENV, "1")
    model = SimpleNamespace(loras={"experts.w1": object(), "experts.w2": object(), "experts.w3": object()})
    assert evidence._LORA_BUILD_EVIDENCE.get() is None
    with evidence.maybe_observe_lora_build({}) as record:
        record.factors["source"] = ("experts.w2", "lora_a", None)
        if failure:
            with pytest.raises(ValueError, match="native pack failed"):
                with evidence.maybe_observe_fused_expert_loras(model):
                    raise ValueError("native pack failed")
            assert record.factors["source"] == ("experts.w2", "lora_a", None)
        else:
            with evidence.maybe_observe_fused_expert_loras(model):
                model.loras = {"experts": object()}
            assert record.factors["source"] == ("experts", "lora_a", 1)
    assert evidence._LORA_BUILD_EVIDENCE.get() is None


def test_receiver_debug_helpers_do_not_replace_native_return_values():
    source = ast.parse(_SOURCE.read_text())
    assert not any(isinstance(node, ast.Return) and _optional_call(node.value) for node in ast.walk(source))
    for name in (
        "maybe_pack_fused_expert_loras",
        "maybe_feed_fp8_tensor",
        "maybe_load_broadcast_tensor",
        "maybe_run_broadcast_load",
        "maybe_build_lora_model",
    ):
        assert name not in _SOURCE.read_text()


def test_broadcast_observer_restores_nested_context(monkeypatch):
    monkeypatch.setattr(evidence, "maybe_begin_broadcast_load", lambda receiver, *args: receiver)
    outer, inner = object(), object()
    with evidence.maybe_observe_broadcast_load(outer, None, None):
        with pytest.raises(ValueError, match="inner load failed"):
            with evidence.maybe_observe_broadcast_load(inner, None, None):
                assert evidence.maybe_current_broadcast_verifier() is inner
                raise ValueError("inner load failed")
        assert evidence.maybe_current_broadcast_verifier() is outer
    assert evidence.maybe_current_broadcast_verifier() is None


def test_receiver_result_observers_preserve_native_dict_identity(monkeypatch):
    result = {"status": "done"}
    assert evidence.maybe_broadcast_receiver_result(result, None, rank=2) is result
    assert evidence.maybe_lora_built_receiver_result(result, None) is result
    verifier = SimpleNamespace(finish=lambda: {"status": "ok"})
    assert evidence.maybe_broadcast_receiver_result(result, verifier, rank=2) is result
    assert result == {"status": "done", "rank": 2, "weight_sync_verification": {"status": "ok", "rank": 2}}
    monkeypatch.setattr(evidence, "maybe_lora_receiver_result", lambda *args, **kwargs: {"adapter_evidence": True})
    assert evidence.maybe_lora_built_receiver_result(result, SimpleNamespace(received={}, factors={})) is result
    assert result["status"] == "done" and result["adapter_evidence"] is True
