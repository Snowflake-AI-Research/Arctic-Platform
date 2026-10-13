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
"""Rank-layout math for the combined sequence-parallel, expert-parallel, and data-parallel mesh.

Torch-free, and the shared source of truth for process-group creation and sequence-parallel batch
sharding. Callers that shard tokens must use these same rank lists. Two topologies live here.

Node-local-EP (2D-SP): DeepEP's all-to-all needs every EP group to be an intra-node NVLink block, but the
default orthogonal mesh strides EP by SP. Whenever that stride would put EP peers on different nodes
(typical once ``sp_size`` fills a node, e.g. ``sp=8, ep=2`` on 32 GPUs or ``sp=8, ep=8`` on 128 GPUs),
node-local-EP keeps EP inside a node and lets SP span nodes:

    local_rank     = rank % gpus_per_node
    ep_position    = local_rank %  ep                    # expert shard (0..ep-1)
    intra_sp_index = local_rank // ep
    slot           = node * sp_per_node + intra_sp_index # unique among same-expert ranks
    sp_rank        = slot %  sp_size
    replica        = slot // sp_size                     # extra DP along same-expert ranks
    dp_index       = replica * ep + ep_position

    * EP groups = intra-node ep-blocks (e.g. ep=2 -> {0,1},{2,3},...).
    * SP groups = ranks sharing an ep_position, ordered by slot, then chunked into ``sp_size``.

When ``world == sp*ep`` there is one SP group per expert shard, so expert-DP == SP group. With extra
DP replicas, expert-DP is every rank with that ``ep_position`` (``intra_node_ep_replica_groups``).

Host-complete EP: ``ep`` is an integer number of full nodes (``ep % gpus_per_node == 0`` and ``ep`` does not
fit in one node). Consecutive EP blocks fill every host they span (``ep=16`` on 8-GPU nodes is two nodes of
8). The orthogonal mesh can still *place* some of those shapes by striding EP by SP, but that packing leaves
too few ranks per host for UCCL's NVLink-domain buffer sizing. This layout keeps contiguous SP and consecutive
EP blocks; expert-DP is again ``intra_node_ep_replica_groups``.

Decoupled intra-node-EP (bottom section): the EP axis as a function of ``ep`` only, independent of SP.
"""

from __future__ import annotations

from typing import List, Tuple

EP_GROUP_LAYOUTS = frozenset({"auto", "packed", "mesh"})


def normalize_ep_group_layout(value) -> str:
    layout = "auto" if value is None else value
    if layout not in EP_GROUP_LAYOUTS:
        raise ValueError(
            f"ep_group_layout must be one of {sorted(EP_GROUP_LAYOUTS)}; got {value!r}"
        )
    return layout


def use_node_local_ep(world_size: int, sp_size: int, ep_size: int, gpus_per_node: int) -> bool:
    """True iff node-local-EP applies: multi-node, EP fits in a node, orthogonal EP would leave the node.

    Covers the original ``world == sp*ep`` (DP=1) shapes and the same placement with extra DP, such as
    ``sp=8, ep=2`` on 32 GPUs and ``sp=8, ep=8`` on 128 GPUs. Leaves shapes the orthogonal mesh cannot
    place to the decoupled path, and leaves intra-node orthogonal EP (e.g. ``sp=4, ep=2`` on 8-GPU nodes)
    on the default mesh. Full-world SP falls out of the mesh-placement check rather than needing its own:
    it makes ``world // sp`` equal 1, which is never divisible by an ``ep`` of 2 or more.

    The last check is where the mesh would put EP: its groups are aligned spans of ``sp * ep`` ranks tiling
    from rank 0, so all of them stay on-node exactly when that product divides ``gpus_per_node``. Rank 0's
    group alone would not answer -- at ``sp=6, ep=2`` on 8-GPU nodes ``{0,6}`` is on node 0, ``{2,8}`` is not.
    """
    if ep_size == 1 or sp_size == 1:
        return False
    if gpus_per_node <= 0 or world_size <= gpus_per_node:
        return False
    if world_size % gpus_per_node != 0 or gpus_per_node % ep_size != 0:
        return False
    if world_size % sp_size != 0:
        return False
    if (world_size // sp_size) % ep_size != 0:
        return False
    return gpus_per_node % (sp_size * ep_size) != 0


def use_intra_node_ep_decoupled(
    world_size: int, sp_size: int, ep_size: int, gpus_per_node: int
) -> bool:
    if sp_size == 1 or ep_size == 1:
        return False
    if gpus_per_node <= 0 or gpus_per_node % ep_size != 0:
        return False
    if world_size % gpus_per_node != 0:
        return False
    if world_size % sp_size != 0 or world_size % ep_size != 0:
        return False
    return sp_size == world_size or (world_size // sp_size) % ep_size != 0


def use_host_complete_ep(world_size: int, sp_size: int, ep_size: int, gpus_per_node: int) -> bool:
    """True iff EP is packed as consecutive whole-node blocks that fill every host they span.

    Node-local EP cannot apply: ``ep`` does not divide ``gpus_per_node``. The orthogonal mesh would stride
    EP by SP and leave a partial NVLink domain on each host (``sp=2, ep=16`` on 8-GPU nodes is 4 ranks per
    host across 4 hosts). Consecutive blocks of ``ep`` ranks restore a full ``gpus_per_node`` of EP peers on
    every host in the group, which is what UCCL sizes internode buffers for. ``ep`` must be a power of two so
    the UCCL size-hint tables accept the group.

    ``sp_size == 1`` already uses consecutive EP via DeepSpeed's expert-plus-data-parallel groups, so this
    path is only the SP+EP overlay.
    """
    if ep_size == 1 or sp_size == 1:
        return False
    if gpus_per_node <= 0 or world_size <= gpus_per_node:
        return False
    if world_size % gpus_per_node != 0:
        return False
    if world_size % sp_size != 0 or world_size % ep_size != 0:
        return False
    if ep_size % gpus_per_node != 0:
        return False
    if gpus_per_node % ep_size == 0:
        return False
    if ep_size & (ep_size - 1):
        return False
    return True


def resolve_ep_group_topology(
    world_size: int,
    sp_size: int,
    ep_size: int,
    gpus_per_node: int,
    ep_group_layout: str = "auto",
) -> str:
    layout = normalize_ep_group_layout(ep_group_layout)
    if sp_size <= 1 or ep_size <= 1:
        return "standard"

    mesh_valid = (
        world_size % sp_size == 0
        and sp_size != world_size
        and (world_size // sp_size) % ep_size == 0
    )
    if layout == "mesh":
        if mesh_valid:
            return "orthogonal"
        raise ValueError(
            f"ep_group_layout='mesh' requires (world_size // sp_size) % ep_size == 0 "
            f"and sp_size != world_size; got world={world_size}, sp={sp_size}, ep={ep_size}"
        )

    if layout == "packed":
        if gpus_per_node <= 0:
            raise ValueError("ep_group_layout='packed' requires gpus_per_node > 0")
        if world_size % gpus_per_node != 0:
            raise ValueError("ep_group_layout='packed' requires a whole number of GPU nodes")
        if gpus_per_node % ep_size == 0:
            return "node-local" if mesh_valid else "decoupled"
        if use_host_complete_ep(world_size, sp_size, ep_size, gpus_per_node):
            return "host-complete"
        raise ValueError(
            f"ep_group_layout='packed' requires ep_size to divide gpus_per_node or be a "
            f"power-of-two multiple of it; got ep={ep_size}, gpus_per_node={gpus_per_node}"
        )

    if use_node_local_ep(world_size, sp_size, ep_size, gpus_per_node):
        return "node-local"
    if use_intra_node_ep_decoupled(world_size, sp_size, ep_size, gpus_per_node):
        return "decoupled"
    if use_host_complete_ep(world_size, sp_size, ep_size, gpus_per_node):
        return "host-complete"
    if mesh_valid:
        return "orthogonal"
    raise ValueError(
        f"unsupported SP/EP topology (sp={sp_size}, ep={ep_size}, "
        f"world={world_size}, gpus_per_node={gpus_per_node}, ep_group_layout={layout!r})"
    )


def host_complete_ep_groups(world_size: int, ep_size: int) -> List[List[int]]:
    """Consecutive EP blocks of size ``ep`` (one per replica), each an integer number of full nodes."""
    return [list(range(start, start + ep_size)) for start in range(0, world_size, ep_size)]


def rank_to_node_sp_ep(rank: int, ep_size: int, gpus_per_node: int) -> Tuple[int, int, int]:
    """Decompose a global ``rank`` into ``(node, intra_sp_index, ep_position)`` (see module docstring)."""
    node = rank // gpus_per_node
    local_rank = rank % gpus_per_node
    intra_sp_index = local_rank // ep_size
    ep_position = local_rank % ep_size
    return node, intra_sp_index, ep_position


def same_expert_slot(rank: int, ep_size: int, gpus_per_node: int) -> int:
    """Index of ``rank`` among ranks that share its expert shard, ordered by ``(node, intra_sp)``."""
    node, intra_sp_index, _ep_position = rank_to_node_sp_ep(rank, ep_size, gpus_per_node)
    sp_per_node = gpus_per_node // ep_size
    return node * sp_per_node + intra_sp_index


def sp_groups(world_size: int, ep_size: int, gpus_per_node: int, sp_size: int) -> List[List[int]]:
    """SP rank groups: same-expert ranks ordered by slot, then chunked into ``sp_size``.

    ``use_node_local_ep`` makes ``world`` a multiple of ``sp * ep``, so the chunks always come out even.
    """
    sp_per_node = gpus_per_node // ep_size
    num_nodes = world_size // gpus_per_node
    by_expert = [
        [
            node * gpus_per_node + intra_sp * ep_size + ep_position
            for node in range(num_nodes)
            for intra_sp in range(sp_per_node)
        ]
        for ep_position in range(ep_size)
    ]
    return [
        ranks[start : start + sp_size]
        for ranks in by_expert
        for start in range(0, len(ranks), sp_size)
    ]


def ep_groups(world_size: int, ep_size: int, gpus_per_node: int) -> List[List[int]]:
    """EP rank groups (one per (node, intra-SP index)), fully intra-node."""
    sp_per_node = gpus_per_node // ep_size
    num_nodes = world_size // gpus_per_node
    return [
        [node * gpus_per_node + intra_sp * ep_size + ep_position for ep_position in range(ep_size)]
        for node in range(num_nodes)
        for intra_sp in range(sp_per_node)
    ]


def rank_to_dp_sp(
    world_size: int, ep_size: int, gpus_per_node: int, sp_size: int
) -> List[Tuple[int, int]]:
    """Per-rank ``(dp_index, sp_rank)`` for the data plane.

    ``dp_index = replica * ep + ep_position`` selects the DP sample; ``sp_rank`` is the token window.
    """

    def coords(rank: int) -> Tuple[int, int]:
        _node, _intra_sp_index, ep_position = rank_to_node_sp_ep(rank, ep_size, gpus_per_node)
        slot = same_expert_slot(rank, ep_size, gpus_per_node)
        replica = slot // sp_size
        return replica * ep_size + ep_position, slot % sp_size

    return [coords(rank) for rank in range(world_size)]


# --- Decoupled intra-node-EP topology (EP axis, a function of ``ep`` only, independent of SP) ----------


def intra_node_ep_groups(world_size: int, ep_size: int, gpus_per_node: int) -> List[List[int]]:
    """Intra-node EP blocks (size ``ep``) for the decoupled topology. Same layout as ``ep_groups``; named
    separately for the decoupled call site. One whole-world group when ``ep == world == gpus_per_node``."""
    return ep_groups(world_size, ep_size, gpus_per_node)


def orthogonal_edp_groups(world_size: int, sp_size: int, ep_size: int) -> List[List[int]]:
    """Expert-DP groups for the orthogonal mesh: the replicas of one expert shard, over both the
    sequence-parallel and the data-parallel axis.

    EP is laid out over the data-parallel world with SP treated as model parallel, so a rank's expert-shard
    position is ``(rank // sp) % ep`` and each shard has ``world // ep`` replicas. Those replicas are what
    hold the same experts and what must all-reduce their partial expert grads. The set contains each rank's
    whole SP group, and equals it exactly when ``world // ep == sp``, which is to say when the data-parallel
    degree equals the expert-parallel degree.
    """
    return [
        [rank for rank in range(world_size) if (rank // sp_size) % ep_size == position]
        for position in range(ep_size)
    ]


def intra_node_ep_replica_groups(world_size: int, ep_size: int) -> List[List[int]]:
    """Expert-DP groups for consecutive or intra-node EP blocks: decoupled, node-local, and host-complete EP.

    The replicas of one expert shard, strided by ``ep`` (``rank % ep == ep_position``). Their partial expert
    grads all-reduce over this group and the expert optimizer/master state shards over it. Singletons when
    ``ep == world`` (no replication).

    Intra-node ``ep``-blocks and host-complete consecutive blocks both make ``rank % ep`` name the shard.
    ``sp`` never enters, so the same groups serve all three.
    """
    return [
        [rank for rank in range(world_size) if rank % ep_size == ep_position]
        for ep_position in range(ep_size)
    ]
