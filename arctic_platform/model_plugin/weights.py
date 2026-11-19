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
from typing import TypeVar

import torch
import torch.distributed as dist
from torch import nn

from arctic_platform.model_plugin.experts import EXPERT_PARALLEL_SIZE_ATTR
from arctic_platform.model_plugin.plugin import ModelPlugin

# The platform's existing marker; the weight sync skips any tensor that carries it.
SKIP_WEIGHT_SYNC_ATTR = "_dss_skip_weight_sync"
CHECKPOINT_FILE = "model.safetensors"
INDEX_FILE = "model.safetensors.index.json"
STAMP_PLUGIN = "arctic_plugin"
STAMP_VERSION = "arctic_plugin_version"
STAMP_CONTRACT = "arctic_plugin_contract"

logger = logging.getLogger(__name__)

_LAYER = re.compile(r"^model\.layers\.(\d+)\.")

T = TypeVar("T", torch.Tensor, nn.Module)


def skip_weight_sync(obj: T) -> T:
    """Keep a parameter, or every tensor of a module, in checkpoints but out of the weight sync.

    Use it for trainer-only state the sampler never reads. Mark a module rather than a buffer: moving a model
    between devices or dtypes replaces buffer objects, and a marker set on the old object would be lost.
    """
    setattr(obj, SKIP_WEIGHT_SYNC_ATTR, True)
    return obj


def weight_sync_skipped(model: nn.Module) -> set[str]:
    """Names of the trainer state tensors the weight sync leaves out."""
    skipped = {
        name
        for name, tensor in model.state_dict(keep_vars=True).items()
        if getattr(tensor, SKIP_WEIGHT_SYNC_ATTR, False)
    }
    for path, module in model.named_modules():
        if getattr(module, SKIP_WEIGHT_SYNC_ATTR, False):
            skipped.update(f"{path}.{name}" for name in module.state_dict())
    return skipped


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
    tensors marked with :func:`skip_weight_sync`.
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
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(path)
    stamp = checkpoint_stamp(plugin)
    shards = _shards(weights, max_shard_bytes)
    if len(shards) == 1:
        save_file(shards[0], str(path / CHECKPOINT_FILE), metadata=stamp)
        return
    weight_map = {}
    for number, shard in enumerate(shards, start=1):
        file_name = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(path / file_name), metadata=stamp)
        weight_map.update(dict.fromkeys(shard, file_name))
    total = sum(tensor.numel() * tensor.element_size() for tensor in weights.values())
    index = {"metadata": {"total_size": total, **stamp}, "weight_map": weight_map}
    (path / INDEX_FILE).write_text(json.dumps(index, indent=2) + "\n")


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


def iter_checkpoint_layers(path: str | Path) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
    """A checkpoint's tensors one decoder layer at a time, reading only that layer from disk."""
    from safetensors import safe_open

    by_layer: dict[int, dict[str, Path]] = {}
    for name, file in checkpoint_files(path).items():
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


def load_hf_checkpoint(plugin: ModelPlugin, path: str | Path) -> tuple[Any, nn.Module]:
    """Build the trainer model from ``config.json`` and load the checkpoint into it layer by layer."""
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
    model = plugin.build_model(config)
    _load_layers(model, plugin, iter_checkpoint_layers(path))
    return config, model
