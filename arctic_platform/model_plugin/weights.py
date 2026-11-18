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

import re
from collections.abc import Iterable
from collections.abc import Iterator
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


def load_hf_weights(model: nn.Module, plugin: ModelPlugin, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Load Hugging Face-named tensors into a freshly built, unsharded trainer model. Every parameter must be set."""
    groups: dict[int, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        groups.setdefault(layer_index(name), {})[name] = tensor
    state: dict[str, torch.Tensor] = {}
    for index, layer in groups.items():
        if index >= 0 and plugin.convert_layer_to_prime is not None:
            layer = plugin.convert_layer_to_prime(layer, index, model.config)
        state.update(layer)
    model.load_state_dict(state, strict=True)


def save_hf_checkpoint(model: nn.Module, plugin: ModelPlugin, config: Any, path: str | Path, *, ep_group=None) -> None:
    from safetensors.torch import save_file

    weights = {
        name: tensor.contiguous()
        for name, tensor in iter_hf_weights(model, plugin, for_sampler=False, ep_group=ep_group)
    }
    if ep_group is None or dist.get_rank(ep_group) == 0:
        Path(path).mkdir(parents=True, exist_ok=True)
        config.save_pretrained(path)
        save_file(weights, str(Path(path) / CHECKPOINT_FILE))


def load_hf_checkpoint(plugin: ModelPlugin, path: str | Path) -> tuple[Any, nn.Module]:
    from safetensors.torch import load_file

    config = plugin.config_class.from_pretrained(path)
    model = plugin.build_model(config)
    load_hf_weights(model, plugin, load_file(str(Path(path) / CHECKPOINT_FILE)).items())
    return config, model
