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

"""Hugging Face weight streams: checkpoints, and the trainer-to-sampler weight sync of an RL step."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from arctic_platform.model_plugin.build import build
from arctic_platform.model_plugin.build import materialize
from arctic_platform.model_plugin.experts import EXPERT_PARALLEL_SIZE_ATTR
from arctic_platform.model_plugin.marks import weight_sync_skipped
from arctic_platform.model_plugin.plugin import ModelPlugin

# On a loaded model: the checkpoint file holding each pass-through tensor, to copy into every export.
PASSTHROUGH_ATTR = "_ap_passthrough"
CHECKPOINT_FILE = "model.safetensors"
INDEX_FILE = "model.safetensors.index.json"
STAMP_PLUGIN = "arctic_plugin"
STAMP_VERSION = "arctic_plugin_version"
STAMP_CONTRACT = "arctic_plugin_contract"

logger = logging.getLogger(__name__)

_LAYER = re.compile(r"^model\.layers\.(\d+)\.")


def layer_index(name: str) -> int:
    """The decoder layer a trainer parameter belongs to, or -1 for embeddings, final norm and head."""
    match = _LAYER.match(name)
    return int(match.group(1)) if match else -1


def _full_tensor(tensor: torch.Tensor, ep_group) -> torch.Tensor:
    ep_size = getattr(tensor, EXPERT_PARALLEL_SIZE_ATTR, 1)
    tensor = tensor.detach()
    if ep_size == 1:
        return tensor
    if ep_group is None or dist.get_world_size(ep_group) != ep_size:
        raise ValueError(f"an expert tensor sharded {ep_size} ways needs its expert-parallel group to export")
    parts = [torch.empty_like(tensor) for _ in range(ep_size)]
    dist.all_gather(parts, tensor.contiguous(), group=ep_group)
    return torch.cat(parts, dim=0)


def iter_hf_weights(
    model: nn.Module, plugin: ModelPlugin, *, for_sampler: bool, ep_group=None
) -> Iterator[tuple[str, torch.Tensor]]:
    """Full tensors under Hugging Face names, one decoder layer at a time.

    Expert-parallel shards are gathered first, so every rank of ``ep_group`` must iterate. ``for_sampler`` drops
    tensors marked ``weight_sync=False``. Every tensor keeps its own dtype, so an fp32-marked tensor of a bf16 model
    streams, and is saved, as fp32.
    """
    skipped = weight_sync_skipped(model) if for_sampler else set()
    groups: dict[int, dict[str, torch.Tensor]] = {}
    for name, tensor in model.state_dict(keep_vars=True).items():
        if name in skipped:
            continue
        groups.setdefault(layer_index(name), {})[name] = _full_tensor(tensor, ep_group)
    for index in sorted(groups, key=lambda i: (i < 0, i)):
        layer = groups[index]
        if index >= 0 and plugin.convert_layer_to_hf is not None:
            layer = plugin.convert_layer_to_hf(layer, index, model.config)
        yield from layer.items()


def _load_layers(model: nn.Module, plugin: ModelPlugin, layers: Iterable[tuple[int, dict[str, torch.Tensor]]]) -> None:
    """Load one decoder layer at a time, so peak extra memory is one layer. Every trainer tensor must be set."""
    expected = set(model.state_dict())
    loaded: set[str] = set()
    for index, layer in layers:
        if index >= 0 and plugin.convert_layer_to_prime is not None:
            layer = plugin.convert_layer_to_prime(layer, index, model.config)
        unexpected = set(layer) - expected
        if len(unexpected) > 0:
            raise ValueError(f"checkpoint tensors match no trainer tensor: {sorted(unexpected)}")
        model.load_state_dict(layer, strict=False)
        loaded.update(layer)
    missing = expected - loaded
    if len(missing) > 0:
        raise ValueError(f"checkpoint does not set trainer tensors: {sorted(missing)}")


def load_hf_weights(model: nn.Module, plugin: ModelPlugin, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Load Hugging Face-named tensors into a freshly built, unsharded trainer model. Every tensor must be set."""
    groups: dict[int, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        groups.setdefault(layer_index(name), {})[name] = tensor
    _load_layers(model, plugin, groups.items())


def checkpoint_stamp(plugin: ModelPlugin) -> dict[str, str]:
    """Safetensors metadata recording which plugin wrote a checkpoint."""
    manifest = plugin.manifest
    return {
        "format": "pt",
        STAMP_PLUGIN: manifest.name,
        STAMP_VERSION: manifest.version,
        STAMP_CONTRACT: manifest.contract,
    }


def _shards(weights: dict[str, torch.Tensor], max_shard_bytes: int | None) -> list[dict[str, torch.Tensor]]:
    shards: list[dict[str, torch.Tensor]] = [{}]
    size = 0
    for name, tensor in weights.items():
        nbytes = tensor.numel() * tensor.element_size()
        if max_shard_bytes is not None and len(shards[-1]) > 0 and size + nbytes > max_shard_bytes:
            shards.append({})
            size = 0
        shards[-1][name] = tensor
        size += nbytes
    return shards


def save_hf_checkpoint(
    model: nn.Module,
    plugin: ModelPlugin,
    config: Any,
    path: str | Path,
    *,
    ep_group=None,
    max_shard_bytes: int | None = None,
) -> None:
    """Write ``config.json`` and safetensors under Hugging Face names, split into shards of at most
    ``max_shard_bytes`` with a ``model.safetensors.index.json``, each stamped with the plugin that wrote it."""
    from safetensors.torch import save_file

    weights = {
        name: tensor.contiguous()
        for name, tensor in iter_hf_weights(model, plugin, for_sampler=False, ep_group=ep_group)
    }
    if ep_group is not None and dist.get_rank(ep_group) != 0:
        return
    # Read before writing anything, so exporting over the source directory is safe.
    weights.update(_read_tensors(getattr(model, PASSTHROUGH_ATTR, {})))
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(path)
    stamp = checkpoint_stamp(plugin)
    shards = _shards(weights, max_shard_bytes)
    weight_map = {}
    if len(shards) == 1:
        save_file(shards[0], str(path / CHECKPOINT_FILE), metadata=stamp)
        weight_map = dict.fromkeys(shards[0], CHECKPOINT_FILE)
    else:
        for number, shard in enumerate(shards, start=1):
            file_name = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
            save_file(shard, str(path / file_name), metadata=stamp)
            weight_map.update(dict.fromkeys(shard, file_name))
        total = sum(tensor.numel() * tensor.element_size() for tensor in weights.values())
        index = {"metadata": {"total_size": total, **stamp}, "weight_map": weight_map}
        (path / INDEX_FILE).write_text(json.dumps(index, indent=2) + "\n")
    # Later exports copy pass-through tensors from this one, so the source checkpoint need not outlive the run.
    passthrough = getattr(model, PASSTHROUGH_ATTR, {})
    setattr(model, PASSTHROUGH_ATTR, {name: str(path / weight_map[name]) for name in passthrough})


def _read_tensors(files: dict[str, str]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    tensors = {}
    for file in sorted(set(files.values())):
        with safe_open(file, framework="pt") as f:
            tensors.update({name: f.get_tensor(name) for name, source in files.items() if source == file})
    return tensors


def checkpoint_files(path: str | Path) -> dict[str, Path]:
    """The file holding each tensor of a single-file or sharded Hugging Face checkpoint."""
    from safetensors import safe_open

    path = Path(path)
    if (path / INDEX_FILE).is_file():
        weight_map = json.loads((path / INDEX_FILE).read_text())["weight_map"]
        return {name: path / file_name for name, file_name in weight_map.items()}
    single = path / CHECKPOINT_FILE
    if not single.is_file():
        raise FileNotFoundError(f"{path} has neither {CHECKPOINT_FILE} nor {INDEX_FILE}")
    with safe_open(str(single), framework="pt") as f:
        return dict.fromkeys(f.keys(), single)


def read_stamp(path: str | Path) -> dict[str, str]:
    """The plugin stamp of a checkpoint, or ``{}`` if it was not written by a plugin."""
    from safetensors import safe_open

    first = next(iter(sorted(set(checkpoint_files(path).values()))))
    with safe_open(str(first), framework="pt") as f:
        metadata = f.metadata() or {}
    return {key: value for key, value in metadata.items() if key.startswith("arctic_plugin")}


def iter_checkpoint_layers(
    path: str | Path, skip: set[str] = frozenset()
) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
    """A checkpoint's tensors one decoder layer at a time, reading only that layer from disk, except ``skip``."""
    from safetensors import safe_open

    by_layer: dict[int, dict[str, Path]] = {}
    for name, file in checkpoint_files(path).items():
        if name not in skip:
            by_layer.setdefault(layer_index(name), {})[name] = file
    with ExitStack() as stack:
        handles: dict[Path, Any] = {}
        for index in sorted(by_layer, key=lambda i: (i < 0, i)):
            layer = {}
            for name, file in by_layer[index].items():
                if file not in handles:
                    handles[file] = stack.enter_context(safe_open(str(file), framework="pt"))
                layer[name] = handles[file].get_tensor(name)
            yield index, layer


_SAFETENSORS_FLOAT_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}


def checkpoint_dtype(path: str | Path, skip: set[str] = frozenset()) -> torch.dtype | None:
    """The floating dtype holding most of a checkpoint's elements, read from safetensors headers; ``None`` if the
    checkpoint has no floating tensors."""
    from safetensors import safe_open

    elements: dict[torch.dtype, int] = {}
    by_file: dict[Path, list[str]] = {}
    for name, file in checkpoint_files(path).items():
        if name not in skip:
            by_file.setdefault(file, []).append(name)
    for file, names in by_file.items():
        with safe_open(str(file), framework="pt") as f:
            for name in names:
                tensor = f.get_slice(name)
                dtype = _SAFETENSORS_FLOAT_DTYPES.get(tensor.get_dtype())
                if dtype is not None:
                    shape = tensor.get_shape()
                    elements[dtype] = elements.get(dtype, 0) + (torch.Size(shape).numel())
    return max(elements, key=elements.__getitem__) if len(elements) > 0 else None


def load_hf_checkpoint(
    plugin: ModelPlugin, path: str | Path, *, options: Any = None, device: str | torch.device = "cpu"
) -> tuple[Any, nn.Module]:
    """Build the trainer model from ``config.json`` on the meta device, allocate it on ``device`` in the dtype the
        checkpoint stores, and load the checkpoint into it layer by layer. Building on the meta device is what the
    platform does at scale; this helper then allocates the whole model on ``device``.

        Building in the stored dtype makes a save and reload exact: loading bf16 tensors into an fp32 model would
        compute in fp32 and change the logits. Tensors marked ``dtype="fp32"`` stay fp32, as they were saved.
    """
    stamp = read_stamp(path)
    if STAMP_PLUGIN in stamp and stamp[STAMP_PLUGIN] != plugin.manifest.name:
        raise ValueError(f"{path} was written by plugin {stamp[STAMP_PLUGIN]}, not {plugin.manifest.name}")
    if stamp.get(STAMP_VERSION, plugin.manifest.version) != plugin.manifest.version:
        logger.info(
            "loading %s, written by %s %s, with version %s",
            path,
            stamp[STAMP_PLUGIN],
            stamp[STAMP_VERSION],
            plugin.manifest.version,
        )
    config = plugin.config_class.from_pretrained(path)
    passthrough = {
        name: str(file)
        for name, file in checkpoint_files(path).items()
        if plugin.manifest.checkpoint.is_passthrough(name)
    }
    dtype = checkpoint_dtype(path, skip=set(passthrough))  # a large pass-through tower must not set the model's dtype
    model = materialize(build(plugin, config, options, meta=True), device, dtype)
    _load_layers(model, plugin, iter_checkpoint_layers(path, skip=set(passthrough)))
    setattr(model, PASSTHROUGH_ATTR, passthrough)
    return config, model
