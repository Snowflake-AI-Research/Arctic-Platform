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
"""Load HF safetensors into the existing, optionally EP-sharded model storage."""

import json
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor
from torch.distributed.checkpoint.utils import _DistWrapper
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

# Families with layer-local converters; packed DeepSeek checkpoints retain their cache path.
_DIRECT_HF_MODEL_TYPES = {
    "qwen3_moe",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "glm4_moe",
    "glm_moe_dsa",
    "minimax",
    "minimax_m2",
    "afmoe",
    "nemotron_h",
    "qwen4_exp",
    "glm5_next",
}


def supports_direct_hf_loading(model: torch.nn.Module) -> bool:
    return getattr(model.config, "model_type", None) in _DIRECT_HF_MODEL_TYPES


_LAYER_RE = re.compile(r"^(?:language_model\.)?(?:model(?:\.language_model)?|backbone)\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"^(.*\.experts\.)(\d+)(\..*)$")
_EXPERT_WEIGHT_RE = re.compile(r"\.experts\.w[123]$")
_PLE_SHARD_RE = re.compile(r"^(.*\.ple\.ple_embedding\.ngram_embedding)\.shard_(\d+)\.weight$")


def _layer_index(name: str) -> int:
    match = _LAYER_RE.match(name)
    return int(match[1]) if match is not None else -1


def _local_region(tensor: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    if isinstance(tensor, DTensor):
        _, offsets = compute_local_shape_and_global_offset(tensor.shape, tensor.device_mesh, tensor.placements)
        return tensor.to_local(), offsets
    return tensor, (0,) * tensor.ndim


class _SafetensorsReader:
    def __init__(self, path: Path):
        self.files: dict[str, Path] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self._shard: safe_open | None = None
        self._shard_path: Path | None = None
        index_path = path / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            weight_map = index["weight_map"]
            paths = sorted({path / name for name in weight_map.values()})
        else:
            weight_map = None
            paths = sorted(path.glob("*.safetensors"))
        for shard_path in paths:
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                names = shard.keys()
                for name in names:
                    if weight_map is not None and weight_map.get(name) != str(shard_path.relative_to(path)):
                        continue
                    if name in self.files:
                        raise ValueError(f"Duplicate checkpoint tensor {name!r}")
                    self.files[name] = shard_path
                    self.shapes[name] = tuple(shard.get_slice(name).get_shape())
        if weight_map is not None and (missing := weight_map.keys() - self.files.keys()):
            raise KeyError(f"Checkpoint index references missing tensors: {sorted(missing)[:8]}")
        if not self.files:
            raise FileNotFoundError(f"No safetensors weights in {path}")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self) -> None:
        if self._shard is not None:
            self._shard.__exit__(None, None, None)
            self._shard = None
            self._shard_path = None

    def read(self, name: str, selection: tuple[slice, ...] | None = None) -> Tensor:
        path = self.files[name]
        if self._shard_path != path:
            self.close()
            self._shard = safe_open(path, framework="pt", device="cpu")
            self._shard.__enter__()
            self._shard_path = path
        shard = self._shard
        assert shard is not None
        if selection is None:
            return shard.get_tensor(name)
        return shard.get_slice(name)[selection]


def _expert_range(targets: dict[str, Tensor]) -> tuple[int, int, int] | None:
    regions = set()
    for name, target in targets.items():
        if _EXPERT_WEIGHT_RE.search(name) is None or target.ndim != 3:
            continue
        local, offsets = _local_region(target)
        if offsets[1:] != (0, 0) or local.shape[1:] != target.shape[1:]:
            raise ValueError(f"Direct HF expert loading requires sharding only the expert axis: {name}")
        regions.add((offsets[0], offsets[0] + local.shape[0], target.shape[0]))
    if len(regions) > 1:
        raise ValueError(f"Inconsistent expert shards in layer: {sorted(regions)}")
    return next(iter(regions), None)


def _copy_tensor(name: str, source: Tensor, target: Tensor) -> None:
    local, offsets = _local_region(target)
    if source.shape == target.shape and source.shape != local.shape:
        source = source[tuple(slice(offset, offset + size) for offset, size in zip(offsets, local.shape))]
    if source.shape != local.shape:
        raise ValueError(
            f"Checkpoint shape mismatch for {name}: source={tuple(source.shape)}, target={tuple(local.shape)}"
        )
    local.copy_(source)


def _load_ple_shards(
    reader: _SafetensorsReader,
    names: list[str],
    targets: dict[str, Tensor],
) -> set[str]:
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for name in names:
        match = _PLE_SHARD_RE.match(name)
        if match is not None:
            groups[f"{match[1]}.weight"].append((int(match[2]), name))
    loaded = set()
    for name, shards in groups.items():
        if name not in targets:
            continue
        shards.sort()
        if [index for index, _ in shards] != list(range(len(shards))):
            raise KeyError(f"Non-contiguous Qwen3.8 PLE shards for {name}")
        target = targets[name]
        shapes = [reader.shapes[key] for _, key in shards]
        if (
            any(shape[1:] != tuple(target.shape[1:]) for shape in shapes)
            or sum(shape[0] for shape in shapes) != target.shape[0]
        ):
            raise ValueError(
                f"Checkpoint PLE shape mismatch for {name}: shards={shapes}, target={tuple(target.shape)}"
            )
        local, offsets = _local_region(target)
        row_start, row_stop = offsets[0], offsets[0] + local.shape[0]
        shard_start = 0
        for _, key in shards:
            shard_stop = shard_start + reader.shapes[key][0]
            start, stop = max(row_start, shard_start), min(row_stop, shard_stop)
            if start < stop:
                selection = (slice(start - shard_start, stop - shard_start),) + tuple(
                    slice(offset, offset + size) for offset, size in zip(offsets[1:], local.shape[1:])
                )
                local[start - row_start : stop - row_start].copy_(reader.read(key, selection))
            shard_start = shard_stop
        loaded.add(name)
    return loaded


def _load_hf_weights(
    state_dict: dict[str, Tensor],
    reader: _SafetensorsReader,
    convert_layer_to_prime: Callable[[dict[str, Tensor], int], dict[str, Tensor]],
) -> None:
    source_layers: dict[int, list[str]] = defaultdict(list)
    target_layers: dict[int, dict[str, Tensor]] = defaultdict(dict)
    for name in reader.files:
        source_layers[_layer_index(name)].append(name)
    for name, tensor in state_dict.items():
        target_layers[_layer_index(name)][name] = tensor
    loaded: set[str] = set()
    for layer_idx, names in sorted(source_layers.items()):
        targets = target_layers.get(layer_idx, {})
        if not targets:
            continue
        loaded.update(_load_ple_shards(reader, names, targets))
        expert_range = _expert_range(targets)
        if expert_range is not None:
            start, stop, count = expert_range
            ids = {int(match[2]) for name in names if (match := _EXPERT_RE.match(name)) is not None}
            if ids and ids != set(range(count)):
                raise KeyError(f"Checkpoint expert ids for layer {layer_idx} do not match {count} experts")
            if start == stop:
                raise ValueError(f"Empty expert shard in layer {layer_idx}")

        # Global weights (including the vision encoder) need only namespace renames.
        batches = [[name] for name in names] if layer_idx == -1 else [names]
        for batch in batches:
            layer_state: dict[str, Tensor] = {}
            for name in batch:
                if _PLE_SHARD_RE.match(name) is not None:
                    continue
                shape = reader.shapes[name]
                selection: tuple[slice, ...] | None = None
                destination_name = name
                match = _EXPERT_RE.match(name)
                if expert_range is not None and match is not None:
                    expert_id = int(match[2])
                    if not start <= expert_id < stop:
                        continue
                    destination_name = f"{match[1]}{expert_id - start}{match[3]}"
                elif expert_range is not None and ".experts." in name and len(shape) == 3:
                    if shape[0] != count:
                        raise ValueError(f"Checkpoint expert count mismatch for {name}: {shape[0]} != {count}")
                    selection = (slice(start, stop), slice(None), slice(None))
                elif name in targets:
                    local, offsets = _local_region(targets[name])
                    if shape != tuple(targets[name].shape):
                        raise ValueError(
                            f"Checkpoint shape mismatch for {name}: {shape} != {tuple(targets[name].shape)}"
                        )
                    if shape != tuple(local.shape):
                        selection = tuple(slice(offset, offset + size) for offset, size in zip(offsets, local.shape))
                if layer_idx == -1:
                    metadata = {name: torch.empty(shape, device="meta")}
                    convert_layer_to_prime(metadata, -1)
                    if not metadata.keys() & targets.keys():
                        continue
                layer_state[destination_name] = reader.read(name, selection)
            if not layer_state:
                continue
            convert_layer_to_prime(layer_state, layer_idx)
            while layer_state:
                name, tensor = layer_state.popitem()
                if name in targets:
                    _copy_tensor(name, tensor, targets[name])
                    loaded.add(name)
                del tensor
            del layer_state
    if missing := state_dict.keys() - loaded:
        raise KeyError(f"HF checkpoint is missing training tensors: {sorted(missing)[:8]}")


@torch.no_grad()
def load_hf_weights(
    state_dict: dict[str, Tensor],
    path: Path,
    convert_layer_to_prime: Callable[[dict[str, Tensor], int], dict[str, Tensor]],
) -> None:
    """Convert one local layer at a time; never write an intermediate checkpoint."""

    def load():
        with _SafetensorsReader(path) as reader:
            _load_hf_weights(state_dict, reader, convert_layer_to_prime)

    if torch.distributed.is_initialized():
        _DistWrapper(group=None, use_dist=True, coordinator_rank=0).all_gather("load_hf_weights", load)
    else:
        load()
