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

"""Correctness-only rank-local model fingerprints across checkpoint boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def tensor_digest(tensor: torch.Tensor, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    if chunk_bytes < tensor.element_size():
        raise ValueError("chunk_bytes must hold at least one tensor element")
    digest = hashlib.sha256()

    def update(value):
        if value.numel() * value.element_size() <= chunk_bytes:
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw))
            return
        # Split by logical dimensions to bound copies even for noncontiguous tensors.
        dimension = next(index for index, size in enumerate(value.shape) if size > 1)
        midpoint = value.shape[dimension] // 2
        update(value.narrow(dimension, 0, midpoint))
        update(value.narrow(dimension, midpoint, value.shape[dimension] - midpoint))

    update(tensor)
    return digest.hexdigest()


def model_state_manifest(engine, rank: int) -> dict:
    model = engine.module
    persistent_buffers = {
        f"{prefix}.{name}" if prefix else name
        for prefix, module in model.named_modules()
        for name in module._buffers
        if name not in module._non_persistent_buffers_set
    }
    entries = []
    for kind, tensors in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in sorted(tensors):
            entry = {
                "name": name,
                "kind": kind,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "requires_grad": bool(tensor.requires_grad),
                "group_name": getattr(tensor, "group_name", None),
                "allreduce": getattr(tensor, "allreduce", True),
                "sha256": tensor_digest(tensor),
            }
            if kind == "buffer":
                entry["persistent"] = name in persistent_buffers
            entries.append(entry)
    return {
        "rank": int(rank),
        "engine": {
            "global_steps": int(engine.global_steps),
            "has_moe_layers": bool(engine.has_moe_layers),
            "num_experts": list(engine.num_experts),
        },
        "tensors": entries,
    }


class ModelStateCaptureMixin:
    """Record local tensors immediately before save and after load when opted in."""

    _model_state_output_dir = None

    def _capture_model_state(self, phase: str) -> None:
        if self._model_state_output_dir is None:
            return
        import deepspeed

        manifest = model_state_manifest(self.engine, self.rank)
        manifest["deepspeed_version"] = deepspeed.__version__
        directory = Path(self._model_state_output_dir) / phase
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"rank-{self.rank:05d}.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"MODEL_STATE phase={phase} rank={self.rank} tensors={len(manifest['tensors'])} "
            f"has_moe_layers={manifest['engine']['has_moe_layers']} "
            f"step={manifest['engine']['global_steps']} deepspeed={manifest['deepspeed_version']}",
            flush=True,
        )

    def save_checkpoint(self, path: str, export_hf: bool = False) -> dict:
        self._capture_model_state("before-save")
        return super().save_checkpoint(path, export_hf=export_hf)

    def load_checkpoint(self, path: str) -> int:
        restored_step = super().load_checkpoint(path)
        self._capture_model_state("after-load")
        return restored_step


def compare_model_state(left: Path, right: Path, *, ranks: int) -> dict:
    expected = {f"rank-{rank:05d}.json" for rank in range(ranks)}
    for directory in (left, right):
        if {path.name for path in directory.glob("rank-*.json")} != expected:
            raise RuntimeError(f"incomplete model-state rank coverage in {directory}")
    differences = []
    compared_tensors = 0
    engines = []
    for rank in range(ranks):
        filename = f"rank-{rank:05d}.json"
        before, after = (json.loads((directory / filename).read_text()) for directory in (left, right))
        if before["rank"] != rank or after["rank"] != rank:
            raise RuntimeError(f"model-state rank mismatch in {filename}")
        engines.append({"rank": rank, "before": before["engine"], "after": after["engine"]})
        for field in ("engine", "deepspeed_version"):
            if before.get(field) != after.get(field):
                differences.append({"rank": rank, "kind": "metadata", "name": field, "fields": [field]})
        before_entries, after_entries = (
            {(entry["kind"], entry["name"]): entry for entry in manifest["tensors"]} for manifest in (before, after)
        )
        for kind, name in sorted(before_entries.keys() | after_entries.keys()):
            first, second = before_entries.get((kind, name)), after_entries.get((kind, name))
            compared_tensors += 1
            if first is None or second is None:
                fields = ["missing"]
            else:
                fields = sorted(key for key in first.keys() | second.keys() if first.get(key) != second.get(key))
            if fields:
                differences.append({"rank": rank, "kind": kind, "name": name, "fields": fields})
    return {
        "exact": not differences,
        "ranks": ranks,
        "compared_tensors": compared_tensors,
        "engines": engines,
        "differences": differences,
    }
