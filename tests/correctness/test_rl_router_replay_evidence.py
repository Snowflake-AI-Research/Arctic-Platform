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

"""Controls for independently configured router inventories and request lifecycle."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.debug import training as evidence


class TokenChoiceTopKRouter(torch.nn.Module):
    def forward(self, hidden, *, routed_experts=None):
        return hidden, routed_experts, torch.ones(1)


def _model(*layers, num_hidden_layers=6, dense_layers=(0, 1, 3, 4)):
    model = torch.nn.Module()
    model.config = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": num_hidden_layers,
        "num_experts": 6,
        "num_experts_per_tok": 1,
        "mlp_only_layers": list(dense_layers),
        "decoder_sparse_step": 1,
    }
    model.layers = torch.nn.ModuleDict()
    for layer in layers:
        block = torch.nn.Module()
        block.router = TokenChoiceTopKRouter()
        model.layers[str(layer)] = block
    return model


@pytest.fixture(autouse=True)
def _trace_cleanup():
    evidence.maybe_finish_router_replay_trace(0)
    try:
        yield
    finally:
        evidence.maybe_finish_router_replay_trace(0)


@pytest.fixture
def _configured(tmp_path):
    model = _model(2, 5)
    option = {"request_id": "inventory-control", "output_dir": "/data-fast/tmp/router-replay-selftest"}
    result = evidence.maybe_configure_router_replay_trace(
        {evidence.TRAINING_OPTION: option, "bf16": {"enabled": True}}, model, 0
    )
    assert result == {"bf16": {"enabled": True}}
    evidence._publication["output_dir"] = tmp_path
    engine = SimpleNamespace(global_rank=0)
    kwargs = {
        "input_ids": torch.tensor([[1, 2]]),
        "position_ids": torch.tensor([[0, 1]]),
        "routed_experts": torch.tensor([[[[2]] * 6, [[3]] * 6]]),
    }
    return model, engine, kwargs, tmp_path


def test_router_inventory_is_independent_of_observations_and_allows_layer_gaps():
    model = _model(5, 2)
    model.layers["2"].router._dss_router_replay_layer = 999
    trace = evidence.start(model)
    exported = trace.export(0)
    assert exported["expected_layers"] == [2, 5]
    assert exported["model_inventory"] == evidence.router_replay_model_inventory(model.config)
    assert exported["model_inventory"]["expected_layers"] == [2, 5]
    assert model.config["mlp_only_layers"] == [0, 1, 3, 4]
    assert exported["calls"] == [] and exported["unattributed_router_calls"] == []
    model.layers["2"].router(torch.ones(1), routed_experts=torch.tensor([[3]]))
    assert trace.unattributed[0]["layer"] == 2
    assert trace.export(0)["expected_layers"] == [2, 5]


@pytest.mark.parametrize("fault", ["empty", "unknown", "duplicate"])
def test_router_inventory_rejects_invalid_layers_before_installing_any_hooks(fault):
    model = _model() if fault == "empty" else _model(2, dense_layers=(0, 1, 3, 4, 5))
    if fault == "unknown":
        model.router = TokenChoiceTopKRouter()
    elif fault == "duplicate":
        model.layers["2"].second_router = TokenChoiceTopKRouter()
    with pytest.raises(ValueError, match="inventory"):
        evidence.start(model)
    assert evidence.active() is None
    assert all(len(module._forward_hooks) == 0 for module in model.modules())


@pytest.mark.parametrize("missing_layer", [2, 5])
def test_router_inventory_rejects_a_missing_architectural_layer_before_installing_any_hooks(missing_layer):
    model = _model(2, 5)
    inventory = evidence.router_replay_model_inventory(model.config)
    assert inventory["expected_layers"] == [2, 5]
    del model.layers[str(missing_layer)].router
    assert evidence.router_replay_model_inventory(model.config) == inventory
    with pytest.raises(ValueError, match="architectural roster"):
        evidence.start(model)
    assert evidence.active() is None and evidence._publication is None
    assert all(len(module._forward_hooks) == 0 for module in model.modules())


@pytest.mark.parametrize("model_type", [None, "unverified_moe"])
def test_router_inventory_rejects_missing_or_unsupported_config_before_installing_any_hooks(model_type):
    model = _model(2, 5)
    if model_type is None:
        del model.config
        message = "independently bound config"
    else:
        model.config["model_type"] = model_type
        message = "unverified router-replay layer predicate"
    with pytest.raises(NotImplementedError, match=message):
        evidence.start(model)
    assert evidence.active() is None and evidence._publication is None
    assert all(len(module._forward_hooks) == 0 for module in model.modules())


def test_router_inventory_rejects_an_unregistered_router_even_with_a_known_attribute():
    trace = evidence.start(_model(2, dense_layers=(0, 1, 3, 4, 5)))
    unknown = TokenChoiceTopKRouter()
    unknown._dss_router_replay_layer = 2
    with pytest.raises(ValueError, match="unknown inventory layer"):
        trace.record_router(unknown, torch.tensor([[1]]), replayed=True)
    assert trace.unattributed == []


def test_router_inventory_partial_hook_installation_is_cleaned_up(monkeypatch):
    model = _model(2, 5)

    def failed_registration(*args, **kwargs):
        raise RuntimeError("hook installation failed")

    monkeypatch.setattr(model.layers["5"].router, "register_forward_hook", failed_registration)
    with pytest.raises(RuntimeError, match="hook installation failed"):
        evidence.start(model)
    assert evidence.active() is None
    assert all(len(module._forward_hooks) == 0 for module in model.modules())


def test_router_inventory_is_retained_in_every_immutable_completed_call(_configured):
    model, engine, kwargs, output = _configured
    inventory = evidence.active().export(0)["model_inventory"]
    assert inventory == evidence.router_replay_model_inventory(model.config)
    evidence.maybe_record_router_replay_received("rr1:sample", torch.tensor([[[2]] * 6]))
    for index in range(2):
        evidence.maybe_begin_router_replay_call(engine, kwargs, backward="loss_only")
        model.layers["2"].router(torch.ones(2), routed_experts=torch.tensor([[2], [3]]))
        assert not (output / f"rank-0-call-{index}.json").exists()
        evidence.maybe_complete_router_replay_call(engine)
        record = json.loads((output / f"rank-0-call-{index}.json").read_text())
        assert record["schema"] == "arctic-platform-router-replay-call-v2"
        assert record["expected_layers"] == [2, 5]
        assert record["model_inventory"] == inventory
        assert record["call_index"] == index
        assert record["received"] == {"rr1:sample": [[[2]] * 6]}
        assert record["calls"][0]["input_ids"] == [1, 2]
        assert record["calls"][0]["routed_experts"] == [[[2]] * 6, [[3]] * 6]
        assert record["calls"][0]["router"][0]["layer"] == 2


def test_router_cleanup_discards_incomplete_call_and_allows_a_new_job(_configured):
    model, engine, kwargs, output = _configured
    trace = evidence.active()
    evidence.maybe_begin_router_replay_call(engine, kwargs, backward=True)
    model.layers["2"].router(torch.ones(2), routed_experts=torch.tensor([[2], [3]]))
    evidence.maybe_finish_router_replay_trace(0)
    evidence.maybe_finish_router_replay_trace(0)
    assert evidence.active() is None and evidence._publication is None
    assert list(output.iterdir()) == []
    assert all(len(module._forward_hooks) == 0 for module in model.modules())
    count = len(trace.calls[0]["router"])
    model.layers["2"].router(torch.ones(2), routed_experts=torch.tensor([[4], [5]]))
    assert len(trace.calls[0]["router"]) == count
    fresh = evidence.start(_model(9, num_hidden_layers=10, dense_layers=tuple(range(9))))
    assert fresh.export(0)["expected_layers"] == [9]
    assert fresh.received == {} and fresh.calls == []


def test_router_cleanup_in_an_error_finally_preserves_the_actual_error(_configured):
    model, engine, kwargs, output = _configured
    with pytest.raises(RuntimeError, match="actual backward failed"):
        try:
            evidence.maybe_begin_router_replay_call(engine, kwargs, backward=True)
            raise RuntimeError("actual backward failed")
        finally:
            evidence.maybe_finish_router_replay_trace(0)
    assert evidence.active() is None and evidence._publication is None
    assert all(len(module._forward_hooks) == 0 for module in model.modules())
    assert list(output.iterdir()) == []


def test_router_finish_cleans_hooks_even_when_export_fails(_configured):
    model, _, _, _ = _configured
    with pytest.raises(TypeError):
        evidence.finish(object())
    assert evidence.active() is None and evidence._publication is None
    assert all(len(module._forward_hooks) == 0 for module in model.modules())


def test_router_cleanup_wrong_rank_cannot_clear_another_request(_configured):
    model, _, _, _ = _configured
    trace = evidence.active()
    with pytest.raises(RuntimeError, match="cleanup rank differs"):
        evidence.maybe_finish_router_replay_trace(1)
    assert evidence.active() is trace and evidence._publication is not None
    assert len(model.layers["2"].router._forward_hooks) == 1
    evidence.maybe_finish_router_replay_trace(0)
    assert evidence.active() is None and evidence._publication is None


def test_router_disabled_helpers_do_not_inspect_models_or_inputs():
    config = {"bf16": {"enabled": True}}
    assert evidence.maybe_configure_router_replay_trace(config, None, 0) is config
    evidence.maybe_record_router_replay_received("sample", None)
    evidence.maybe_begin_router_replay_call(None, None, backward=True)
    evidence.maybe_complete_router_replay_call(None)
    evidence.maybe_finish_router_replay_trace(0)
    assert evidence.active() is None and evidence._publication is None


@pytest.mark.parametrize("completed", [False, True])
def test_router_invocation_outside_an_open_call_cannot_escape_published_evidence(_configured, completed):
    model, engine, kwargs, output = _configured
    if completed:
        evidence.maybe_begin_router_replay_call(engine, kwargs, backward=True)
        model.layers["2"].router(torch.ones(2), routed_experts=torch.tensor([[2], [3]]))
        evidence.maybe_complete_router_replay_call(engine)
        path = output / "rank-0-call-0.json"
        original = path.read_bytes()
    trace = evidence.active()
    with pytest.raises(RuntimeError, match="outside an open model call"):
        model.layers["2"].router(torch.ones(1), routed_experts=torch.tensor([[4]]))
    assert trace.unattributed[-1]["indices"] == [[4]]
    if completed:
        assert path.read_bytes() == original
        assert len(trace.calls[0]["router"]) == 1
    else:
        assert list(output.iterdir()) == []


@pytest.mark.parametrize("native_failure", [False, True])
def test_router_observer_close_failure_cannot_prevent_native_worker_destroy(_configured, monkeypatch, native_failure):
    events = []
    trace = evidence.active()
    close = trace.close

    def failed_close():
        events.append("close")
        close()
        raise RuntimeError("observer close failed")

    def send_destroy():
        events.append("sender")
        if native_failure:
            raise ValueError("native sender destroy failed")

    monkeypatch.setattr(trace, "close", failed_close)
    # Exercise the installed teardown method without importing the optional Ray training stack.
    source = Path(__file__).resolve().parents[2] / "arctic_platform/common/deepspeed_worker.py"
    worker_class = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in worker_class.body if isinstance(node, ast.FunctionDef) and node.name == "destroy")
    namespace = {
        "maybe_finish_router_replay_after_destroy": evidence.maybe_finish_router_replay_after_destroy,
        "dist": SimpleNamespace(is_initialized=lambda: True, destroy_process_group=lambda: events.append("group")),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    worker = SimpleNamespace(
        _weight_sender=SimpleNamespace(destroy=send_destroy),
        engine=object(),
        _gpu_mem_preallocation=object(),
    )
    if native_failure:
        with pytest.raises(ValueError, match="native sender destroy failed"):
            namespace["destroy"](worker)
        assert events == ["sender", "close"]
    else:
        with pytest.raises(RuntimeError, match="observer close failed"):
            namespace["destroy"](worker)
        assert events == ["sender", "group", "close"]
        assert worker._weight_sender is None
        assert worker.engine is None and worker._gpu_mem_preallocation is None
    assert evidence.active() is None and evidence._publication is None
