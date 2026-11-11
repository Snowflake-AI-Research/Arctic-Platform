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

"""Official Inkling2 checkpoint loading; MXFP8 routed experts are decoded to the training dtype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Literal

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch.distributed.tensor import DTensor

from arctic_platform.model.loader import canonical_parameter_name

MXFP8_BLOCK = 32


def weight_targets(name: str) -> list[tuple[str, Literal["gate", "up"] | None]]:
    """Model parameters a checkpoint tensor fills, with the gate or up rows of an interleaved ``w13_weight``.

    Block scales are read with their weights, and the vision tower is not trained.
    """
    if not name.startswith("model.llm.") or name.endswith("_weight_scale"):
        return []
    if name == "model.llm.unembed.weight":
        return [("lm_head.weight", None)]
    name = name.replace("model.llm.", "model.", 1).replace("model.embed.", "model.embed_tokens.", 1)
    if name.endswith(".mlp.experts.w13_weight"):
        prefix = name.removesuffix("w13_weight")
        return [(prefix + "w1", "gate"), (prefix + "w3", "up")]
    if name.endswith(".mlp.experts.w2_weight"):
        return [(name.removesuffix("w2_weight") + "w2", None)]
    return [(name, None)]


def decode_mxfp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """FP32 weights of E4M3 ``values`` with one UE8M0 ``scales`` exponent per 32 consecutive inputs."""
    return values.float() * torch.exp2(scales.float() - 127).repeat_interleave(MXFP8_BLOCK, dim=-1)


class CheckpointReader:
    """Reads (row slices of) official checkpoint tensors, decoding MXFP8 expert weights."""

    def __init__(self, name_or_path: str):
        path = Path(name_or_path)
        if not path.is_dir():
            path = Path(snapshot_download(name_or_path, allow_patterns=["*.json", "model*.safetensors"]))
        index = path / "model.safetensors.index.json"
        if index.is_file():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(path / "model.safetensors", framework="pt", device="cpu") as source:
                self.weight_map = dict.fromkeys(source.keys(), "model.safetensors")
        self.path = path
        self._files: dict[str, Any] = {}

    def names(self):
        return sorted(self.weight_map)

    def _slice(self, name):
        filename = self.weight_map[name]
        if filename not in self._files:
            self._files[filename] = safe_open(self.path / filename, framework="pt", device="cpu")
        return self._files[filename].get_slice(name)

    def read(self, name: str, rows: slice | None = None, device=None) -> torch.Tensor:
        source = self._slice(name)
        value = source[rows] if rows is not None else source[:]
        value = value.to(device) if device is not None else value
        scale_name = name + "_scale"
        if value.dtype == torch.float8_e4m3fn:
            scale = self._slice(scale_name)
            scale = (scale[rows] if rows is not None else scale[:]).to(value.device)
            if scale.dtype != torch.uint8:
                raise ValueError(f"{name}: only MXFP8 experts (UE8M0 block scales) are supported")
            return decode_mxfp8(value, scale)
        if value.dtype == torch.uint8:
            raise ValueError(f"{name}: NVFP4 experts are not supported for training; use the MXFP8 checkpoint")
        return value

    def close(self):
        self._files.clear()


@torch.no_grad()
def load_weights(model, config, parallel_dims):
    reader = CheckpointReader(config.name)
    device = torch.cuda.current_device()
    model.to_empty(device=device)
    targets = {canonical_parameter_name(name): tensor for name, tensor in model.state_dict().items()}
    loaded = set()
    for name in reader.names():
        for target_name, part in weight_targets(name):
            if target_name not in targets:
                raise ValueError(f"Unsupported Inkling2 checkpoint weight: {name} -> {target_name}")
            if target_name in loaded:
                raise ValueError(f"Duplicate Inkling2 checkpoint weight: {target_name}")
            target = targets[target_name]
            rows = None
            local = target
            if isinstance(target, DTensor):
                rank = target.device_mesh.get_local_rank()
                local = target.to_local()
                rows = slice(rank * local.shape[0], (rank + 1) * local.shape[0])
            value = reader.read(name, rows, device=device)
            if part is not None:
                value = value[:, 0::2] if part == "gate" else value[:, 1::2]
            if value.shape != local.shape:
                raise ValueError(f"Inkling2 weight {name}: checkpoint shape {value.shape}, expected {local.shape}")
            local.copy_(value)
            loaded.add(target_name)
            del value
    reader.close()
    missing = targets.keys() - loaded
    if missing:
        raise ValueError(f"Missing Inkling2 checkpoint weights: {sorted(missing)}")
    model.init_buffers_post_meta()
