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

"""Preserve EP-tagged weights outside DeepSpeed's replicated model-state file."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _expert_items_and_ranks(model):
    items = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if getattr(parameter, "allreduce", True) is False and getattr(parameter, "group_name", None)
    ]
    if not items:
        return items, 0, 0
    group_names = {parameter.group_name for _, parameter in items}
    if len(group_names) != 1:
        raise ValueError(f"expert checkpoint requires one EP group, found {sorted(group_names)}")
    import deepspeed.utils.groups as groups

    group_name = next(iter(group_names))
    ep_rank = dist.get_rank(group=groups._get_expert_parallel_group(group_name))
    edp_rank = dist.get_rank(group=groups._get_expert_data_parallel_group(group_name))
    return items, ep_rank, edp_rank


def save_expert_parallel_weights(model, checkpoint_dir: str | Path) -> None:
    """Save one weight file per expert rank, including frozen parameters."""
    items, ep_rank, edp_rank = _expert_items_and_ranks(model)
    if not items:
        return
    if edp_rank == 0:
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / f"experts_ep_rank_{ep_rank:03d}.pt"
        torch.save({name: parameter.detach().cpu() for name, parameter in items}, filename)
        logger.info("Saved %d expert parameters for EP rank %d to %s", len(items), ep_rank, filename)
    dist.barrier()


def load_expert_parallel_weights(model, checkpoint_dir: str | Path) -> None:
    """Restore local expert weights after DeepSpeed loads model and optimizer state."""
    items, ep_rank, _ = _expert_items_and_ranks(model)
    if not items:
        return
    filename = Path(checkpoint_dir) / f"experts_ep_rank_{ep_rank:03d}.pt"
    shard = torch.load(filename, map_location="cpu", weights_only=True)
    expected = {name for name, _ in items}
    if set(shard) != expected:
        raise ValueError(
            f"expert parameter names differ in {filename}: "
            f"missing={sorted(expected - set(shard))}, unexpected={sorted(set(shard) - expected)}"
        )
    # Validate the whole shard before changing any live parameter.
    for name, parameter in items:
        tensor = shard[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
            raise ValueError(f"expert parameter shape or dtype mismatch for {name!r} in {filename}")
    with torch.no_grad():
        for name, parameter in items:
            parameter.copy_(shard[name])
    logger.info("Loaded %d expert parameters for EP rank %d from %s", len(items), ep_rank, filename)
    dist.barrier()
