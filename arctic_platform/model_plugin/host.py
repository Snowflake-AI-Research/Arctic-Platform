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

"""The platform side of the contract: binding a plugin model's layers and running it packed and in parallel.

This is the reference host conformance runs against. The training service drives the same hooks in the same
order inside its DeepSpeed workers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from arctic_platform.common.packing import IGNORE_INDEX
from arctic_platform.common.packing import cu_seqlens_from_position_ids
from arctic_platform.model_plugin.experts import EXPERT_PARALLEL_SIZE_ATTR
from arctic_platform.model_plugin.marks import apply_marks
from arctic_platform.model_plugin.mixers import Mixer
from arctic_platform.model_plugin.parallel import group_size
from arctic_platform.model_plugin.parallel import shard_sequence
from arctic_platform.model_plugin.plugin import ModelPlugin
from arctic_platform.model_plugin.plugin import ParallelismInfo


@dataclass
class LayerBinding:
    """Which mixer runs each sequence-mixing module, and which modules are expert layers."""

    info: ParallelismInfo
    mixers: dict[str, tuple[str, Mixer]] = field(default_factory=dict)
    experts: list[str] = field(default_factory=list)


def bind_layers(model: nn.Module, plugin: ModelPlugin, info: ParallelismInfo) -> LayerBinding:
    """Bind every module that declares ``ap_layer_type`` to its layer type's mixer, and find the expert layers.

    The bound modules must match the config's ``layer_types`` and ``sublayers`` exactly, and the expert layers its
    ``moe_layers``: a layer the platform cannot account for fails here, before a job starts.
    """
    mixers = plugin.layer_mixers()
    sublayer_types = set(plugin.manifest.sublayer_types)
    unknown = (set(info.layer_types) - set(plugin.manifest.layer_types)) | (set(info.sublayers) - sublayer_types)
    if len(unknown) > 0:
        raise ValueError(f"config uses layer types the manifest does not declare: {sorted(unknown)}")
    binding = LayerBinding(info=info)
    for path, module in model.named_modules():
        layer_type = getattr(module, "ap_layer_type", None)
        if layer_type is None:
            continue
        mixer = mixers.get(layer_type)
        if mixer is None:
            raise ValueError(f"{path} declares layer type {layer_type!r}, which the manifest does not declare")
        if not mixer.is_instance(module):
            raise ValueError(f"{path} declares layer type {layer_type!r} but its mixer does not accept it")
        binding.mixers[path] = (layer_type, mixer)
    bound = Counter(layer_type for layer_type, _ in binding.mixers.values())
    expected = Counter(info.layer_types) + Counter(info.sublayers)
    if bound != expected:
        raise ValueError(f"model has sequence-mixing modules {dict(bound)} but its config declares {dict(expected)}")
    if plugin.expert_layer is not None:
        binding.experts = [path for path, module in model.named_modules() if plugin.expert_layer.is_instance(module)]
    if len(binding.experts) != len(info.moe_layers):
        raise ValueError(
            f"model has {len(binding.experts)} expert layers but its config declares {len(info.moe_layers)}"
        )
    return binding


def prepare_model(
    model: nn.Module,
    plugin: ModelPlugin,
    config: Any,
    *,
    sp_group=None,
    ep_group=None,
    ep_group_name: str | None = None,
) -> LayerBinding:
    """Bind the model's layers and apply the requested sequence and expert parallelism.

    Expert parameters are tagged for DeepSpeed's expert-data-parallel gradient reduction under ``ep_group_name``,
    by default ``ep_size_<ep>``, the name DeepSpeed registers an expert-parallel degree under. Run the model under
    DeepSpeed with :func:`~arctic_platform.model_plugin.deepspeed_engine.initialize_deepspeed`.
    """
    info = plugin.parallelism(config)
    binding = bind_layers(model, plugin, info)
    modules = dict(model.named_modules())

    sp_size = group_size(sp_group)
    if sp_size > 1:
        if not plugin.manifest.capabilities.sequence_parallel:
            raise ValueError(f"{plugin.manifest.name} does not declare sequence_parallel")
        for mixer in {id(mixer): mixer for _, mixer in binding.mixers.values()}.values():
            mixer.validate_sp(info, sp_size)
        for path, (_, mixer) in binding.mixers.items():
            mixer.apply_sequence_parallel(modules[path], sp_group)

    ep_size = group_size(ep_group)
    if ep_size > 1:
        expert_layer = plugin.expert_layer
        if expert_layer is None or not plugin.manifest.capabilities.expert_parallel:
            raise ValueError(f"{plugin.manifest.name} does not declare expert_parallel")
        # DeepSpeed's registry key for an expert-parallel degree (arctic_platform.model.parallel.groups.ep_group_name,
        # not imported so this module stays free of DeepSpeed).
        ep_group_name = ep_group_name or f"ep_size_{ep_size}"
        for path in binding.experts:
            module = modules[path]
            num_experts = expert_layer.num_experts(module)
            names = list(expert_layer.expert_parameters(module))
            expert_layer.apply_expert_parallel(module, ep_group)
            for name, param in expert_layer.expert_parameters(module).items():
                if name not in names or param.shape[0] * ep_size != num_experts:
                    raise ValueError(
                        f"{path}.{name}: expected {num_experts // ep_size} local experts after sharding, "
                        f"got shape {tuple(param.shape)}"
                    )
                # DeepSpeed reduces these over the expert-data-parallel group instead of the whole world.
                setattr(param, "allreduce", False)
                setattr(param, "group_name", ep_group_name)
                setattr(param, EXPERT_PARALLEL_SIZE_ATTR, ep_size)
    apply_marks(model)  # after expert sharding, which replaces the expert parameters
    return binding


def pack_sequences(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """``input_ids`` and ``position_ids``, each ``[1, total]`` on the sequences' device, with positions restarting at
    0 for every row."""
    input_ids = torch.cat(sequences).unsqueeze(0)
    position_ids = torch.cat([torch.arange(len(sequence), device=sequence.device) for sequence in sequences])
    position_ids = position_ids.unsqueeze(0)
    return input_ids, position_ids


def forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    sp_group=None,
    routed_experts: dict[int, torch.Tensor] | None = None,
    return_routing: bool = False,
):
    """Run a packed ``[1, T]`` batch, passing each rank its sequence shard and the full sequence's boundaries.

    Returns the model's final hidden states ``[1, T_local, D]``, with the routing when ``return_routing``; turn them
    into log-probabilities with :func:`token_logprobs`, or into logits with :func:`compute_logits`.
    """
    cu_seqlens = cu_seqlens_from_position_ids(position_ids)
    local_routes = None
    if routed_experts is not None:
        local_routes = {layer: shard_sequence(route, sp_group, dim=0) for layer, route in routed_experts.items()}
    return model(
        shard_sequence(input_ids, sp_group),
        shard_sequence(position_ids, sp_group),
        cu_seqlens=cu_seqlens,
        routed_experts=local_routes,
        return_routing=return_routing,
    )


def next_token_labels(input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """The token each position predicts, with ``IGNORE_INDEX`` at the last token of every packed row."""
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[:, :-1] = input_ids[:, 1:]
    row_end = torch.zeros_like(input_ids, dtype=torch.bool)
    row_end[:, :-1] = position_ids[:, 1:] == 0
    row_end[:, -1] = True
    return labels.masked_fill(row_end, IGNORE_INDEX)


# Tokens per chunk when turning hidden states into log-probabilities: the full logits of a 256K-token row over a
# 152K vocabulary would take ~159 GB in fp32.
DEFAULT_LOGPROB_CHUNK = 4096


def compute_logits(model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Logits of ``hidden`` ``[..., D]``: the model's ``compute_logits`` if it defines one (output scaling, logit
    softcapping), else ``lm_head(hidden)``."""
    override = getattr(model, "compute_logits", None)
    return override(hidden) if override is not None else model.lm_head(hidden)


def _chunk_logprobs(model: nn.Module, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels != IGNORE_INDEX
    logits = compute_logits(model, hidden).float()
    picked = torch.log_softmax(logits, dim=-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return picked * valid


def token_logprobs(
    model: nn.Module, hidden: torch.Tensor, labels: torch.Tensor, *, chunk_size: int = DEFAULT_LOGPROB_CHUNK
) -> torch.Tensor:
    """Per-token log-probabilities ``[1, T]`` of ``labels`` from hidden states ``[1, T, D]``; zero where a label is
    ``IGNORE_INDEX``.

    Logits are computed ``chunk_size`` tokens at a time and recomputed in backward, so at most one chunk's logits
    exist at once, in forward or backward.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if hidden.shape[1] <= chunk_size:
        return _chunk_logprobs(model, hidden, labels)
    # One split, so backward assembles the hidden-state gradient once instead of zero-filling it per chunk.
    chunks = zip(hidden.split(chunk_size, dim=1), labels.split(chunk_size, dim=1))
    return torch.cat([checkpoint(_chunk_logprobs, model, h, y, use_reentrant=False) for h, y in chunks], dim=1)
