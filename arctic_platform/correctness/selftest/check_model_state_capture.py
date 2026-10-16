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

"""Fingerprint every rank's model without writing full checkpoint tensors."""

from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def capture_module():
    return importlib.import_module("arctic_platform.correctness.harness.model_state_capture")


def _engine():
    model = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    model.weight.requires_grad_(False)
    model.weight.allreduce = False
    model.weight.group_name = "ep_size_8"
    model.register_buffer("persistent", torch.arange(4))
    model.register_buffer("runtime", torch.ones(2), persistent=False)
    return SimpleNamespace(module=model, global_steps=10, has_moe_layers=False, num_experts=[])


def test_manifest_includes_frozen_experts_and_all_buffers(capture_module):
    manifest = capture_module.model_state_manifest(_engine(), rank=7)

    assert manifest["rank"] == 7
    assert manifest["engine"]["global_steps"] == 10
    assert manifest["engine"]["has_moe_layers"] is False
    entries = {entry["name"]: entry for entry in manifest["tensors"]}
    assert set(entries) == {"weight", "persistent", "runtime"}
    assert entries["weight"]["requires_grad"] is False
    assert entries["weight"]["group_name"] == "ep_size_8"
    assert entries["weight"]["allreduce"] is False
    assert entries["weight"]["shape"] == [3, 4]
    assert entries["weight"]["dtype"] == "torch.bfloat16"
    assert entries["weight"]["kind"] == "parameter"
    assert entries["persistent"]["persistent"] is True
    assert entries["runtime"]["persistent"] is False


def test_hash_is_exact_layout_independent_and_handles_scalars(capture_module):
    tensor = torch.arange(80, dtype=torch.bfloat16).reshape(8, 10).t()
    digest = capture_module.tensor_digest(tensor, chunk_bytes=8)
    assert digest == capture_module.tensor_digest(tensor.contiguous(), chunk_bytes=8)
    changed = tensor.clone()
    changed[0, 0] += 1
    assert digest != capture_module.tensor_digest(changed, chunk_bytes=8)
    assert capture_module.tensor_digest(torch.tensor(2.0))
    assert capture_module.tensor_digest(torch.empty(0))


def test_hash_cpu_transfers_are_bounded(capture_module, monkeypatch):
    original_cpu = torch.Tensor.cpu
    transfers = []

    def cpu(tensor, *args, **kwargs):
        transfers.append(tensor.numel() * tensor.element_size())
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    capture_module.tensor_digest(torch.arange(128).reshape(8, 16).t(), chunk_bytes=32)
    assert transfers and max(transfers) <= 32


def test_save_load_hooks_capture_at_boundary_without_changing_results(capture_module, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "deepspeed", SimpleNamespace(__version__="test-installed"))
    events = []

    class Base:
        def save_checkpoint(self, path, export_hf=False):
            events.append(("save", path, export_hf))
            self.engine.module.weight.data.add_(1)
            return {"path": path}

        def load_checkpoint(self, path):
            events.append(("load", path))
            self.engine.module.weight.data.add_(2)
            return 10

    class Worker(capture_module.ModelStateCaptureMixin, Base):
        pass

    worker = Worker()
    worker.engine, worker.rank = _engine(), 3
    worker._model_state_output_dir = str(tmp_path)
    before = capture_module.tensor_digest(worker.engine.module.weight)

    assert worker.save_checkpoint("/checkpoint", export_hf=True) == {"path": "/checkpoint"}
    assert worker.load_checkpoint("/checkpoint") == 10
    assert events == [("save", "/checkpoint", True), ("load", "/checkpoint")]
    saved = json.loads((tmp_path / "before-save" / "rank-00003.json").read_text())
    loaded = json.loads((tmp_path / "after-load" / "rank-00003.json").read_text())
    assert saved["tensors"][0]["sha256"] == before
    assert saved["deepspeed_version"] == "test-installed"
    assert loaded["tensors"][0]["sha256"] == capture_module.tensor_digest(worker.engine.module.weight)


def test_hooks_are_inert_when_not_requested(capture_module, tmp_path):
    class Base:
        def save_checkpoint(self, path, export_hf=False):
            return {"path": path}

        def load_checkpoint(self, path):
            return 0

    class Worker(capture_module.ModelStateCaptureMixin, Base):
        _model_state_output_dir = None

    worker = Worker()
    assert worker.save_checkpoint("/checkpoint") == {"path": "/checkpoint"}
    assert worker.load_checkpoint("/checkpoint") == 0
    assert list(tmp_path.iterdir()) == []


def _write_pair(capture_module, tmp_path, changed_rank=None):
    for rank in range(2):
        engine = _engine()
        left = capture_module.model_state_manifest(engine, rank)
        if rank == changed_rank:
            engine.module.weight.data[0, 0] += 1
        right = capture_module.model_state_manifest(engine, rank)
        for phase, manifest in (("before-save", left), ("after-load", right)):
            directory = tmp_path / phase
            directory.mkdir(exist_ok=True)
            (directory / f"rank-{rank:05d}.json").write_text(json.dumps(manifest))


def test_comparison_detects_frozen_rank_local_corruption(capture_module, tmp_path):
    _write_pair(capture_module, tmp_path, changed_rank=1)
    result = capture_module.compare_model_state(tmp_path / "before-save", tmp_path / "after-load", ranks=2)
    assert result["exact"] is False
    assert result["compared_tensors"] == 6
    assert result["differences"] == [{"rank": 1, "kind": "parameter", "name": "weight", "fields": ["sha256"]}]


def test_comparison_rejects_incomplete_rank_coverage(capture_module, tmp_path):
    _write_pair(capture_module, tmp_path)
    (tmp_path / "after-load" / "rank-00001.json").unlink()
    with pytest.raises(RuntimeError, match="rank coverage"):
        capture_module.compare_model_state(tmp_path / "before-save", tmp_path / "after-load", ranks=2)


def test_comparison_reports_exact_model_state(capture_module, tmp_path):
    _write_pair(capture_module, tmp_path)
    result = capture_module.compare_model_state(tmp_path / "before-save", tmp_path / "after-load", ranks=2)
    assert result["exact"] is True
    assert result["differences"] == []
    assert result["compared_tensors"] == 6
