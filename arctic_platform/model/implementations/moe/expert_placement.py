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
"""Static expert placement for expert parallelism (EPLB-style, after prime-rl's ``expert_placement.py``).

EP shards each MoE layer's experts contiguously: EP rank ``r`` owns the physical slots ``[r * L, (r + 1) * L)``
(``L = E / ep``), and the dispatch backends and both replica planners (``expert_replicas.py``, ``ultraep_replicas.py``)
take that owner layout as given. A placement relabels which logical expert each slot holds, ``perm[slot] = logical
expert``, so that every node and rank receives about the same routed load. Only the dispatch target changes:

- the router (scores, top-k, routing weights, ``expert_bias``) and the ``tokens_per_expert`` buffer (router metrics,
  load-balancing bias) stay in logical ids;
- ``MoE`` maps the router's choices and per-expert counts to slots (``MoE._to_expert_slots``) before any replica
  planner and the dispatch, so those see contiguous owners in slot space, and combine returns rows to their tokens
  as before;
- the loader moves the expert rows into slot order once, right after the checkpoint load and before DeepSpeed builds
  its ZeRO partitions (one all-to-all over the EP group per tensor), so gradients and optimizer state are in slot
  order too;
- every export of EP-sharded expert tensors gathers them back into logical order (``cat_expert_shards``,
  ``expert_export_rows``). Resumable checkpoints keep slot order and resume only under the same placement.

The placement is the job's ``loader_options.expert_placement`` or, when that is unset, ``ARL_EXPERT_PLACEMENT``: a
JSON file path, the JSON itself, or (loader options only) the parsed object: ``{"ep_size": ep, "gpus_per_node": g
(optional), "layers": {"<decoder layer index>": perm, ...}}``, fitted offline from router histograms (DSS
``debug.router_tokens_per_expert``). Layers the file omits keep the contiguous placement; a listed layer that is not a
``MoE`` layer of the model is an error. With ``gpus_per_node`` the loader checks that EP rank ``r`` sits on node
``r // gpus_per_node``, the layout a node-balanced placement was fitted for.
"""

from __future__ import annotations

import json
import os
import re
import socket
import zlib
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

PLACEMENT_ENV = "ARL_EXPERT_PLACEMENT"
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# Set on every placed expert tensor (and expert LoRA factor): logical expert id -> slot, on the CPU, for the exports.
_SLOT_ATTR = "_arl_expert_slot"


# The job's ``loader_options.expert_placement`` (``runtime_options.apply_moe_runtime_options``); None leaves
# ``ARL_EXPERT_PLACEMENT`` in charge.
_configured: str | dict | None = None


def configure(value: str | dict | None) -> None:
    """Set this process's placement from the job's loader options; None keeps ``ARL_EXPERT_PLACEMENT``."""
    global _configured
    _configured = value


def placement_value() -> str | dict | None:
    """The placement this process applies: the configured one, else ``ARL_EXPERT_PLACEMENT``, else None."""
    if _configured is not None:
        return _configured
    return os.environ.get(PLACEMENT_ENV) or None


def read_placement(value: str | dict) -> dict:
    """The placement object of ``value``: a JSON file path, the JSON text, or the object itself."""
    if isinstance(value, dict):
        spec = value
    else:
        text = value if value.lstrip().startswith("{") else Path(value).read_text()
        spec = json.loads(text)
    if not isinstance(spec, dict) or not isinstance(spec.get("layers"), dict):
        raise ValueError(f"an expert placement must hold {{'layers': {{layer: perm}}}}, got {type(spec).__name__}")
    return spec


def inverse(perm: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(perm)
    out[perm] = torch.arange(perm.numel(), dtype=perm.dtype, device=perm.device)
    return out


def _moe_layers(model: nn.Module) -> list[tuple[int, nn.Module]]:
    from .layers.moe import MoE

    layers = []
    for name, module in model.named_modules():
        if isinstance(module, MoE):
            match = _LAYER_RE.search(name + ".")
            if match is not None:
                layers.append((int(match.group(1)), module))
    return layers


def _expert_tensors(experts: nn.Module, local_experts: int) -> list[torch.Tensor]:
    """Every parameter and buffer of ``experts`` stored one row per local expert."""
    tensors = [t for _, t in experts.named_parameters(recurse=True)] + [t for _, t in experts.named_buffers()]
    return [t for t in tensors if t is not None and t.ndim >= 1 and t.shape[0] == local_experts]


def _permute_rows(local: torch.Tensor, slot_expert: torch.Tensor, group) -> None:
    """Move logical experts so that slot ``s`` of the EP-sharded ``local`` holds logical expert ``slot_expert[s]``.
    The rows start in contiguous logical order (slot == logical id)."""
    ep, rank = dist.get_world_size(group), dist.get_rank(group)
    per = local.shape[0]
    owner = slot_expert // per  # the rank holding each slot's expert now
    send_rows, send_sizes, recv_slots, recv_sizes = [], [], [], []
    for dst in range(ep):
        wanted = slot_expert[dst * per:(dst + 1) * per]
        mine = owner[dst * per:(dst + 1) * per] == rank
        send_rows.append(wanted[mine] % per)
        send_sizes.append(int(mine.sum()))
    my_owner = owner[rank * per:(rank + 1) * per]
    for src in range(ep):
        from_src = my_owner == src
        recv_slots.append(torch.nonzero(from_src).flatten())
        recv_sizes.append(int(from_src.sum()))
    data = local.data
    wire = data.view(torch.uint8) if data.element_size() == 1 else data  # NCCL has no float8 all-to-all
    send = wire[torch.cat(send_rows).to(wire.device)].contiguous()
    recv = torch.empty_like(wire)
    dist.all_to_all_single(recv, send, recv_sizes, send_sizes, group=group)
    wire[torch.cat(recv_slots).to(wire.device)] = recv


def _check_nodes(spec: dict, group, device: torch.device, logger) -> None:
    gpus = spec.get("gpus_per_node")
    if not gpus:
        return
    ep = dist.get_world_size(group)
    host = torch.tensor([zlib.crc32(socket.gethostname().encode())], dtype=torch.int64, device=device)
    hosts = [torch.empty_like(host) for _ in range(ep)]
    dist.all_gather(hosts, host, group=group)
    ids: dict[int, int] = {}
    node_of = [ids.setdefault(int(h), len(ids)) for h in hosts]
    if node_of != [r // int(gpus) for r in range(ep)] and logger is not None:
        logger.warning(
            f"the expert placement balances nodes of {gpus} consecutive EP ranks, but the EP ranks' nodes are {node_of}: "
            "the placement is still exact, but its node balance does not hold"
        )


@torch.no_grad()
def apply_expert_placement(model: nn.Module, logger=None) -> int:
    """Install the job's placement (``placement_value``) on the loaded model; returns the number of MoE layers placed.

    Runs after the expert weights are loaded (contiguous, local tensors) and before PEFT adapters or the optimizer
    exist, on every rank in lockstep (the weight moves are EP-group collectives)."""
    value = placement_value()
    if not value:
        return 0
    from .distributed.expert_parallel import get_ep_group
    from .layers.moe import _unwrap_experts

    spec = read_placement(value)
    rows = {int(layer): row for layer, row in spec["layers"].items()}
    layers = _moe_layers(model)
    unknown = sorted(set(rows) - {layer for layer, _ in layers})
    if unknown:
        raise ValueError(f"the expert placement places layers {unknown}, which are not MoE layers of this model")
    placed, group, device = 0, None, None
    for layer, block in layers:
        row = rows.get(layer)
        if row is None:
            continue
        if block.expert_slot is not None:
            raise RuntimeError(f"layer {layer} is already placed; its rows are no longer in logical order")
        experts = _unwrap_experts(block.experts)
        num_experts = experts.num_experts
        if sorted(row) != list(range(num_experts)):
            raise ValueError(f"expert placement row of layer {layer} is not a permutation of {num_experts} experts")
        group = get_ep_group(experts)
        if group is None:
            raise ValueError(f"an expert placement needs expert parallelism; layer {layer} has no EP group")
        ep = dist.get_world_size(group)
        if int(spec.get("ep_size", ep)) != ep:
            raise ValueError(f"the expert placement was fitted for EP {spec['ep_size']}, the job runs EP {ep}")
        slot_expert = torch.tensor(row, dtype=torch.long)
        expert_slot = inverse(slot_expert)
        tensors = _expert_tensors(experts, num_experts // ep)
        if not tensors:
            raise RuntimeError(f"layer {layer}: no expert tensors hold {num_experts // ep} local experts")
        for tensor in tensors:
            _permute_rows(tensor, slot_expert, group)
            setattr(tensor, _SLOT_ATTR, expert_slot)
        device = tensors[0].device
        block.expert_slot = expert_slot.to(device)
        block.slot_expert = slot_expert.to(device)
        placed += 1
    if group is not None:
        _check_nodes(spec, group, device, logger)
    if logger is not None:
        logger.info(f"Expert placement: {placed} MoE layers placed")
    return placed


def tag_expert_adapter(parameter: torch.Tensor, block: nn.Module) -> None:
    """Expert LoRA factors created after placement are stored in slot order too; tag them for the export."""
    expert_slot = getattr(block, "expert_slot", None)
    if expert_slot is not None:
        setattr(parameter, _SLOT_ATTR, expert_slot.cpu())


def cat_expert_shards(tensor: torch.Tensor, shards: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """``torch.cat(shards, dim)`` of the EP group's shards (in EP rank order) of ``tensor``, with the experts in
    logical order when ``tensor`` carries a placement. Along ``dim`` each shard holds its local experts, expert-major,
    with the same number of entries each."""
    expert_slot = getattr(tensor, _SLOT_ATTR, None)
    if expert_slot is None:
        return torch.cat(shards, dim=dim)
    per = expert_slot.numel() // len(shards)
    k = shards[0].shape[dim] // per
    return torch.cat([shards[s // per].narrow(dim, (s % per) * k, k) for s in expert_slot.tolist()], dim=dim)


def expert_export_rows(tensor: torch.Tensor, num_rows: int) -> torch.Tensor | None:
    """For each of the ``num_rows`` rows of ``tensor`` gathered over its EP group in rank (slot) order, its row in
    logical expert order; None without a placement."""
    expert_slot = getattr(tensor, _SLOT_ATTR, None)
    if expert_slot is None:
        return None
    k = num_rows // expert_slot.numel()
    return (inverse(expert_slot).unsqueeze(1) * k + torch.arange(k)).flatten()
