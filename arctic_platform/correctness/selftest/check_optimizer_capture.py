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

"""Optimizer captures must be readable outside rank zero's node-local filesystem."""

import ast
import asyncio
import contextlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.harness import optimizer_capture


@pytest.fixture
def capture_worker(monkeypatch, tmp_path):
    import arctic_platform.common as common_package

    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model.weight.grad = torch.full_like(model.weight, 2.0)
    accumulated = {id(model.weight): torch.full(model.weight.shape, 6.25, dtype=torch.float32)}
    events = []

    class Worker:
        def _bf16_accumulated_gradients(self):
            events.append("accumulated")
            return accumulated

    worker_module = types.ModuleType("arctic_platform.common.deepspeed_worker")
    worker_module.DeepSpeedWorker = SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=Worker))
    worker_module._worker_debug_config = lambda config: config.get("debug", {})
    ray_module = types.ModuleType("ray")
    ray_module.remote = lambda worker: worker
    deepspeed_utils = types.ModuleType("deepspeed.utils")
    deepspeed_utils.safe_get_full_fp32_param = lambda parameter: parameter.detach().float()

    def full_gradient(parameter):
        events.append("fallback")
        return parameter.grad

    deepspeed_utils.safe_get_full_grad = full_gradient
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    monkeypatch.setitem(sys.modules, ray_module.__name__, ray_module)
    monkeypatch.setitem(sys.modules, deepspeed_utils.__name__, deepspeed_utils)
    monkeypatch.setattr(common_package, "deepspeed_worker", worker_module, raising=False)
    monkeypatch.setattr(optimizer_capture, "_OPTIMIZER_CAPTURE_WORKER", None)
    monkeypatch.setattr(optimizer_capture.dist, "barrier", lambda: None)
    actor = optimizer_capture._optimizer_capture_worker_class()()
    actor._optimizer_state_output_dir = str(tmp_path / "capture")
    actor.rank = 0
    actor.engine = SimpleNamespace(
        module=model, optimizer=SimpleNamespace(state={}), global_steps=0, get_lr=lambda: [0.0001]
    )
    actor._full_adam_moments = lambda parameter: (None, None)
    return SimpleNamespace(
        actor=actor,
        model=model,
        accumulated=accumulated,
        events=events,
        artifact=tmp_path / "capture/before/000000.pt",
    )


def test_bf16_capture_uses_reduced_accumulation_not_rank_local_parameter_grad(capture_worker):
    state = capture_worker
    state.actor._snapshot_optimizer_state_parameters()
    saved = torch.load(state.artifact, weights_only=True)
    assert saved["gradient"].dtype == torch.float32
    assert torch.equal(saved["gradient"], state.accumulated[id(state.model.weight)])
    assert state.events == ["accumulated"]
    assert torch.equal(state.model.weight.grad, torch.full_like(state.model.weight, 2.0))


def test_missing_bf16_accumulation_does_not_fall_back_to_local_gradient(capture_worker):
    state = capture_worker
    state.accumulated.clear()
    with pytest.raises(RuntimeError, match="no reduced gradient"):
        state.actor._snapshot_optimizer_state_parameters()
    assert state.events == ["accumulated"]
    assert not state.artifact.exists()


def test_non_bf16_capture_keeps_deepspeed_full_gradient_fallback(capture_worker):
    state = capture_worker
    state.actor._bf16_accumulated_gradients = lambda: None
    state.actor._snapshot_optimizer_state_parameters()
    saved = torch.load(state.artifact, weights_only=True)
    assert torch.equal(saved["gradient"], state.model.weight.grad)
    assert state.events == ["fallback"]


def test_disabled_optimizer_capture_reads_no_gradients(capture_worker):
    state = capture_worker
    state.actor._optimizer_state_output_dir = None
    state.actor._snapshot_optimizer_state_parameters()
    assert state.events == []
    assert not state.artifact.exists()


def test_inactive_bf16_parameter_records_absent_gradient_without_local_fallback(capture_worker):
    state = capture_worker
    parameter = state.model.weight
    parameter.grad = None
    state.accumulated.clear()
    state.actor.engine.optimizer.bf16_groups = [[parameter]]
    state.actor.engine.optimizer.fp32_groups_has_gradients = [[False]]
    state.actor._snapshot_optimizer_state_parameters()
    saved = torch.load(state.artifact, weights_only=True)
    assert saved["gradient"] is None
    assert torch.equal(saved["fp32_master"], parameter.detach().float())
    assert state.events == ["accumulated"]


def test_inactive_gradient_does_not_hide_post_step_optimizer_state(capture_worker):
    state = capture_worker
    parameter = state.model.weight
    parameter.grad = None
    state.accumulated.clear()
    state.actor.engine.optimizer.bf16_groups = [[parameter]]
    state.actor.engine.optimizer.fp32_groups_has_gradients = [[False]]
    state.actor._snapshot_optimizer_state_parameters()
    moments = (torch.full_like(parameter, 0.5, dtype=torch.float32), torch.ones_like(parameter, dtype=torch.float32))
    state.actor._full_adam_moments = lambda parameter: moments
    state.actor.engine.global_steps = 1
    metrics = state.actor._optimizer_state_artifact_metrics()
    manifest = Path(metrics["optimizer_state_manifest"])
    saved = torch.load(manifest.parent / "000000.pt", weights_only=True)
    assert saved["gradient"] is None
    assert torch.equal(saved["exp_avg"], moments[0])
    assert torch.equal(saved["exp_avg_sq"], moments[1])


def test_missing_participating_bf16_buffer_with_no_local_grad_still_errors(capture_worker):
    state = capture_worker
    parameter = state.model.weight
    parameter.grad = None
    state.accumulated.clear()
    state.actor.engine.optimizer.bf16_groups = [[parameter]]
    state.actor.engine.optimizer.fp32_groups_has_gradients = [[True]]
    with pytest.raises(RuntimeError, match="no reduced gradient"):
        state.actor._snapshot_optimizer_state_parameters()
    assert state.events == ["accumulated"]
    assert not state.artifact.exists()


@pytest.fixture
def capture_server(monkeypatch):
    import arctic_platform.common as common_package

    files = {"head": {}, "peer": {}}
    events = []
    manifest = "/data-fast/tmp/capture/manifest.json"
    files["peer"] = {manifest: b'{"parameters": []}', "/data-fast/tmp/capture/000000.pt": b"tensor"}
    response = {"job_id": 7, "metrics": {"optimizer_state_manifest": [manifest]}, "batch": {}}
    failures = {}
    nodes = [
        {"Alive": True, "NodeManagerAddress": "head"},
        {"Alive": True, "NodeManagerAddress": "peer"},
        {"Alive": False, "NodeManagerAddress": "old"},
    ]

    class Worker:
        def __init__(self, host):
            self.host = host
            self.routable_ip = SimpleNamespace(remote=lambda: self.host)
            self.export_node_checkpoint = SimpleNamespace(remote=self.export)
            self.require_checkpoint_files = SimpleNamespace(remote=self.require)

        def export(self, root, peers):
            events.append(("export", self.host, root, peers))
            if failures.get("export"):
                raise RuntimeError("artifact transfer failed")
            selected = {path: content for path, content in files[self.host].items() if Path(path).is_relative_to(root)}
            for peer in peers:
                files[peer].update(selected)
            return [str(Path(path).relative_to(root)) for path in selected]

        def require(self, root, relative_paths):
            events.append(("require", self.host))
            if failures.get("require"):
                raise FileNotFoundError("artifact unreadable after publication")
            for relative_path in relative_paths:
                if str(Path(root) / relative_path) not in files[self.host]:
                    raise FileNotFoundError(relative_path)

    class ServerState:
        async def initialize(self, payload):
            return {"job_id": 7}

    class Server:
        async def step(self, job_id, body=None):
            events.append(("step", job_id, body))
            if failures.get("step"):
                raise RuntimeError("optimizer step failed")
            return response

    worker_module = types.ModuleType("arctic_platform.common.deepspeed_worker")
    worker_module.DeepSpeedWorker = object()
    server_module = types.ModuleType("arctic_platform.common.ray_server")
    server_module.DeepSpeedWorker = worker_module.DeepSpeedWorker
    server_module.ArcticRLRayServerState = ServerState
    server_module.ArcticRLRayServer = Server
    compatibility_module = types.ModuleType("arctic_platform.rl.ray_server")
    compatibility_module.DeepSpeedWorker = worker_module.DeepSpeedWorker
    compatibility_module.ArcticRLRayServerState = ServerState
    compatibility_module.ArcticRLRayServer = Server
    ray_module = types.ModuleType("ray")
    ray_module.get = lambda value: value
    ray_module.nodes = lambda: nodes
    for module in (worker_module, server_module, compatibility_module, ray_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(common_package, "deepspeed_worker", worker_module, raising=False)
    monkeypatch.setattr(common_package, "ray_server", server_module, raising=False)
    production_worker = worker_module.DeepSpeedWorker
    capture_worker = object()
    monkeypatch.setattr(optimizer_capture, "_optimizer_capture_worker_class", lambda: capture_worker)
    with optimizer_capture.optimizer_capture_worker():
        actor = server_module.ArcticRLRayServer()
    actor.training_workers = [Worker("peer"), Worker("peer"), Worker("head"), Worker("head")]
    return SimpleNamespace(
        actor=actor,
        files=files,
        events=events,
        manifest=manifest,
        response=response,
        failures=failures,
        nodes=nodes,
        server_module=server_module,
        compatibility_module=compatibility_module,
        worker_module=worker_module,
        state_class=ServerState,
        server_class=Server,
        production_worker=production_worker,
        capture_worker=capture_worker,
    )


def test_publication_fixture_matches_product_request_server():
    source = Path(__file__).resolve().parents[2] / "common/ray_server.py"
    owners = [
        node.name
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef)
        and any(isinstance(member, ast.AsyncFunctionDef) and member.name == "step" for member in node.body)
    ]
    assert owners == ["ArcticRLRayServer"]


def test_capture_context_restores_both_server_classes_after_exception(capture_server):
    state = capture_server
    with pytest.raises(RuntimeError, match="context failed"):
        with optimizer_capture.optimizer_capture_worker():
            assert state.server_module.ArcticRLRayServer is not state.server_class
            assert not hasattr(state.server_module.ArcticRLRayServerState, "step")
            assert state.worker_module.DeepSpeedWorker is state.capture_worker
            raise RuntimeError("context failed")
    assert state.server_module.ArcticRLRayServer is state.server_class
    assert state.server_module.ArcticRLRayServerState is state.state_class
    assert state.worker_module.DeepSpeedWorker is state.production_worker
    assert state.server_module.DeepSpeedWorker is state.production_worker


def test_nested_capture_context_restores_outer_bindings(capture_server):
    state = capture_server
    with optimizer_capture.optimizer_capture_worker():
        outer_server = state.server_module.ArcticRLRayServer
        outer_state = state.server_module.ArcticRLRayServerState
        with optimizer_capture.optimizer_capture_worker():
            assert state.server_module.ArcticRLRayServer is not outer_server
            assert state.server_module.ArcticRLRayServerState is not outer_state
        assert state.server_module.ArcticRLRayServer is outer_server
        assert state.server_module.ArcticRLRayServerState is outer_state
        assert state.worker_module.DeepSpeedWorker is state.capture_worker
    assert state.server_module.ArcticRLRayServer is state.server_class
    assert state.server_module.ArcticRLRayServerState is state.state_class
    assert state.worker_module.DeepSpeedWorker is state.production_worker


def test_peer_rank_zero_capture_is_published_before_step_response(capture_server):
    state = capture_server
    assert state.manifest not in state.files["head"]
    result = asyncio.run(state.actor.step(7, {"learning_rate": 0.001}))
    assert result is state.response
    assert state.files["head"] == state.files["peer"]
    assert state.events[0] == ("step", 7, {"learning_rate": 0.001})
    assert state.events[1] == ("export", "peer", "/data-fast/tmp/capture", ["head"])
    assert sorted(event[1] for event in state.events if event[0] == "require") == ["head", "peer"]
    assert sum(event[0] == "step" for event in state.events) == 1


def test_local_capture_uses_no_peer_transfer(capture_server):
    state = capture_server
    state.nodes[:] = [state.nodes[1]]
    state.actor.training_workers = state.actor.training_workers[:2]
    result = asyncio.run(state.actor.step(7))
    assert result is state.response
    assert ("export", "peer", "/data-fast/tmp/capture", []) in state.events


def test_step_without_optimizer_capture_leaves_product_response_unchanged(capture_server):
    state = capture_server
    state.response["metrics"] = {"grad_norm": 1.0}
    assert asyncio.run(state.actor.step(7)) is state.response
    assert state.events == [("step", 7, None)]


@pytest.mark.parametrize("failure", ["step", "export", "require"])
def test_capture_failures_are_not_successful_step_responses(capture_server, failure):
    state = capture_server
    state.failures[failure] = True
    with pytest.raises((RuntimeError, FileNotFoundError)):
        asyncio.run(state.actor.step(7))
    assert sum(event[0] == "step" for event in state.events) == 1


def test_workers_outside_live_allocation_are_rejected_before_export(capture_server):
    state = capture_server
    state.nodes.pop(1)
    with pytest.raises(RuntimeError, match="outside"):
        asyncio.run(state.actor.step(7))
    assert not any(event[0] == "export" for event in state.events)


def test_manifest_is_required_in_exported_files(capture_server):
    state = capture_server
    state.files["peer"].pop(state.manifest)
    with pytest.raises(FileNotFoundError, match="manifest"):
        asyncio.run(state.actor.step(7))


@pytest.mark.parametrize("as_string", [False, True])
def test_manifest_scalar_and_duplicate_metrics_publish_once(capture_server, as_string):
    state = capture_server
    state.response["metrics"]["optimizer_state_manifest"] = (
        state.manifest if as_string else [state.manifest, state.manifest]
    )
    assert asyncio.run(state.actor.step(7)) is state.response
    assert sum(event[0] == "export" for event in state.events) == 1


def test_empty_capture_worker_list_is_rejected(capture_server):
    state = capture_server
    state.actor.training_workers = []
    with pytest.raises(RuntimeError, match="no training workers"):
        asyncio.run(state.actor.step(7))
    assert not any(event[0] == "export" for event in state.events)


def test_relative_manifest_is_rejected_before_publication(capture_server):
    state = capture_server
    state.response["metrics"]["optimizer_state_manifest"] = ["manifest.json"]
    with pytest.raises(ValueError, match="absolute"):
        asyncio.run(state.actor.step(7))
    assert not any(event[0] == "export" for event in state.events)


def test_head_without_training_workers_still_receives_capture(capture_server):
    state = capture_server
    state.actor.training_workers = state.actor.training_workers[:2]
    asyncio.run(state.actor.step(7))
    assert state.files["head"] == state.files["peer"]
    assert ("export", "peer", "/data-fast/tmp/capture", ["head"]) in state.events


@pytest.mark.parametrize("fail_context", [False, True])
def test_preloaded_compatibility_server_publishes_and_restores(capture_server, monkeypatch, fail_context):
    state = capture_server
    compatibility = types.ModuleType("arctic_platform.rl.ray_server")
    compatibility.DeepSpeedWorker = state.production_worker
    compatibility.ArcticRLRayServer = state.server_class
    compatibility.ArcticRLRayServerState = state.state_class
    monkeypatch.setitem(sys.modules, compatibility.__name__, compatibility)
    expected = pytest.raises(RuntimeError, match="context failed") if fail_context else contextlib.nullcontext()
    with expected:
        with optimizer_capture.optimizer_capture_worker():
            assert compatibility.ArcticRLRayServer is state.server_module.ArcticRLRayServer
            assert compatibility.ArcticRLRayServerState is state.server_module.ArcticRLRayServerState
            server = compatibility.ArcticRLRayServer()
            server.training_workers = state.actor.training_workers
            assert asyncio.run(server.step(7)) is state.response
            assert state.files["head"] == state.files["peer"]
            if fail_context:
                raise RuntimeError("context failed")
    assert compatibility.ArcticRLRayServer is state.server_class
    assert compatibility.ArcticRLRayServerState is state.state_class
    assert compatibility.DeepSpeedWorker is state.production_worker


def test_lazy_compatibility_import_restores_production_classes(capture_server, monkeypatch):
    state = capture_server
    monkeypatch.delitem(sys.modules, "arctic_platform.rl.ray_server")
    with optimizer_capture.optimizer_capture_worker():
        compatibility = sys.modules["arctic_platform.rl.ray_server"]
        assert compatibility.ArcticRLRayServer is state.server_module.ArcticRLRayServer
        assert compatibility.ArcticRLRayServerState is state.server_module.ArcticRLRayServerState
    assert compatibility.ArcticRLRayServer is state.server_class
    assert compatibility.ArcticRLRayServerState is state.state_class
    assert compatibility.DeepSpeedWorker is state.production_worker
    monkeypatch.setitem(sys.modules, compatibility.__name__, state.compatibility_module)


def test_nested_context_restores_compatibility_outer_bindings(capture_server):
    state = capture_server
    compatibility = state.compatibility_module
    with optimizer_capture.optimizer_capture_worker():
        outer_server = compatibility.ArcticRLRayServer
        outer_state = compatibility.ArcticRLRayServerState
        with optimizer_capture.optimizer_capture_worker():
            assert compatibility.ArcticRLRayServer is state.server_module.ArcticRLRayServer
            assert compatibility.ArcticRLRayServerState is state.server_module.ArcticRLRayServerState
            assert compatibility.ArcticRLRayServer is not outer_server
        assert compatibility.ArcticRLRayServer is outer_server
        assert compatibility.ArcticRLRayServerState is outer_state
    assert compatibility.ArcticRLRayServer is state.server_class
    assert compatibility.ArcticRLRayServerState is state.state_class
