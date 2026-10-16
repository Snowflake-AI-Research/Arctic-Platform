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
"""Sequence-parallel process groups, including expert groups when sequence parallelism is above one.

Sequence parallelism of one registers replicated expert groups only. Above one, this module builds the
DeepSpeed sequence-parallel model-parallel unit and the expert groups for the orthogonal, node-local,
decoupled, and host-complete layouts. The caller supplies ``gpus_per_node``; this module does not read
placement from the environment. The Hugging Face and Liger attention adapter stays with the caller.
"""

from __future__ import annotations

import logging
from typing import Any
from typing import NamedTuple

import deepspeed.runtime.sequence_parallel.parallel_state_sp as sp_mpu
import deepspeed.utils.groups as ds_groups
import torch.distributed as dist

from arctic_platform.model.parallel import topology
from arctic_platform.model.parallel.expert_parallel import ep_group_name
from arctic_platform.model.parallel.expert_parallel import setup_replicated_expert_groups

logger = logging.getLogger(__name__)


class SPGroups(NamedTuple):
    """Result of sequence-parallel group setup (also unpacks positionally).

    With SP disabled, the group fields are ``None`` and the size/rank are ``1``/``0``.
    """

    mpu: Any
    sp_group: Any
    sp_world_size: int
    sp_rank: int


def sp_result(sp_mpu) -> "SPGroups":
    return SPGroups(
        sp_mpu,
        sp_mpu.get_sequence_parallel_group(),
        sp_mpu.get_sequence_parallel_world_size(),
        sp_mpu.get_sequence_parallel_rank(),
    )


def setup_parallel_groups(
    sp_size: int,
    ep_size: int,
    world_size=None,
    gpus_per_node=None,
    ep_group_layout: str = "auto",
):
    """Create the orthogonal SP and DeepEP expert process groups.

    Returns ``(mpu, sp_group, sp_world_size, sp_rank)`` (``mpu`` is ``None`` when ``sp_size == 1``).
    Dispatches over five topologies; each expert shard's replicas all-reduce their partial grads:

      0. ``sp_size == 1`` -- no SP group; the original E+D expert topology (or nothing when ``ep == 1``).
      1. node-local EP -- multi-node, orthogonal EP would leave the node: EP inside a node while SP spans nodes.
      2. decoupled intra-node EP -- full-world SP, general dp>1 shapes the mesh can't place, and ``ep > sp``.
      3. host-complete EP -- ``ep`` is whole nodes and does not fit in one: consecutive EP blocks fill every
         host they span (``ep=16`` on 8-GPU nodes is 8+8). Preferred over the orthogonal mesh, whose SP
         stride would leave a partial NVLink domain on each host.
      4. orthogonal mesh -- EP a strided factor of the DP world (``(world/sp) % ep == 0``); the default.

    Cases 2-4 share the contiguous SP + seq-DP groups from ``initialize_sequence_parallel`` and differ
    only in EP topology. ``ds_groups.mpu`` is registered last because the orthogonal builder reads the
    data-parallel world size from the global mpu and needs it unset while it runs.
    """
    if sp_size == 1:
        setup_replicated_expert_groups(ep_size)
        return SPGroups(None, None, 1, 0)

    if world_size is None:
        world_size = dist.get_world_size()
    if not gpus_per_node:
        raise ValueError("sequence parallelism above 1 requires gpus_per_node")
    resolved_world_size, resolved_gpus_per_node = world_size, int(gpus_per_node)
    ep_topology = topology.resolve_ep_group_topology(
        resolved_world_size,
        sp_size,
        ep_size,
        resolved_gpus_per_node,
        ep_group_layout,
    )

    if ep_topology == "node-local":
        return setup_node_local_groups(sp_size, ep_size, resolved_world_size, resolved_gpus_per_node)

    # Cases 2-4 share these contiguous SP + seq-DP groups ({0..sp-1},{sp..2sp-1},...).
    sp_mpu.initialize_sequence_parallel(sp_size)

    if ep_topology == "decoupled":
        setup_decoupled_expert_groups(ep_size, resolved_world_size, resolved_gpus_per_node)
    elif ep_topology == "host-complete":
        setup_host_complete_expert_groups(ep_size, resolved_world_size)
    elif ep_topology == "orthogonal":
        setup_orthogonal_expert_groups(sp_mpu, ep_size)

    ds_groups.mpu = sp_mpu
    return sp_result(sp_mpu)


def setup_orthogonal_expert_groups(sp_mpu, ep_size: int) -> None:
    """Wire EP for the orthogonal-mesh topology (``(world/sp) % ep == 0``) and apply the expert-DP fix.

    ``_create_expert_data_and_model_parallel`` treats SP as model parallel, so it derives a singleton
    expert-DP group ([[0],[2],...]) -- wrong for SP, where SP siblings hold a full model replica over
    different sequence shards and their expert grads must be summed (and the expert optimizer/master state
    sharded). The ranks owning the same experts are the replicas of one expert-shard position over both the
    sequence-parallel and the data-parallel axis, ``world // ep`` of them, which is what
    ``topology.orthogonal_edp_groups`` builds. Registering the SP group instead reduces over ``sp`` ranks,
    which is the same set only when the data-parallel degree equals the expert-parallel degree; at any other
    shape it sums a fraction of the replicas and the expert gradients come out scaled by the ratio. Must run
    while the global mpu is unset (it reads the DP world size from it). ``dist.new_group`` is collective, so
    every rank walks the same list and keeps only the group it belongs to.
    """
    ds_groups._create_expert_data_and_model_parallel(ep_size, mpu=sp_mpu)
    ep_name = ep_group_name(ep_size)
    rank = dist.get_rank()
    for ranks in topology.orthogonal_edp_groups(
        dist.get_world_size(), sp_mpu.get_sequence_parallel_world_size(), ep_size
    ):
        group = dist.new_group(ranks)
        if rank in ranks:
            ds_groups._EXPERT_DATA_PARALLEL_GROUP[ep_name] = group
            ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[ep_name] = list(ranks)
    assert_expert_data_parallel_group_holds_every_replica(ep_name)


def assert_expert_data_parallel_group_holds_every_replica(ep_name: str) -> None:
    """Refuse a registered expert-DP group that is not the set of ranks holding this rank's expert shard.

    The set is derived here from the expert-parallel groups rather than from ``topology.orthogonal_edp_groups``,
    so an edit to that function cannot satisfy this check by construction -- both would have to be changed in
    agreement, which is a deliberate act rather than an oversight. The same check runs after node-local and
    decoupled registration, which use ``intra_node_ep_replica_groups`` rather than the orthogonal formula.

    The derivation: an expert-parallel group holds one shard per member, so a member's position in it names
    which shard it holds, and the ranks occupying that position across all such groups are that shard's
    replicas. At ``sp`` 2, ``ep`` 2 on 8 ranks DeepSpeed builds expert-parallel groups ``[0,2] [4,6] [1,3]
    [5,7]``, whose position-0 members are ``{0,1,4,5}``; at ``ep`` 4 they are ``[0,2,4,6] [1,3,5,7]``, whose
    position-0 members are ``{0,1}``.

    That second case is why the check runs at every shape and not only where a mismatch is known to arise. The
    replica set equals the sequence-parallel group exactly when ``world // ep == sp``, which is to say when the
    data-parallel degree equals the expert-parallel degree, and at that shape a construction that returns the
    sequence-parallel group is right by coincidence. A topology change that breaks the coincidence would
    otherwise turn a silent-and-correct group into a silent-and-wrong one, with expert gradients scaled by
    ``ep / dp`` and nothing raised.

    Collective on the default group: every rank contributes its position and every rank reaches the same verdict.
    """
    ep_ranks = list(ds_groups._EXPERT_PARALLEL_GROUP_RANKS[ep_name])
    rank = dist.get_rank()
    position = ep_ranks.index(rank)
    positions: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(positions, position)
    replicas = sorted(peer for peer, peer_position in enumerate(positions) if peer_position == position)
    registered = sorted(ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[ep_name])
    if registered != replicas:
        raise RuntimeError(
            f"rank {rank}: the expert-data-parallel group {registered} is not the set of ranks that hold this "
            f"rank's expert shard, {replicas} (read off the expert-parallel groups, this rank at position "
            f"{position} of {ep_ranks}); an expert gradient would all-reduce over {len(registered)} of "
            f"{len(replicas)} replicas and come out scaled by {len(registered)}/{len(replicas)}"
        )


def _register_independent_expert_groups(ep_size: int, world_size: int, ep_rank_groups) -> None:
    """Register EP groups from ``ep_rank_groups`` and expert-DP from ``intra_node_ep_replica_groups``.

    Used by decoupled intra-node EP and host-complete EP: both overlay consecutive-or-block EP on already
    built contiguous SP, and both name a shard by ``rank % ep``. ``dist.new_group`` is collective, so every
    rank walks the same lists and keeps only the groups it belongs to.
    """
    rank = dist.get_rank()
    ep_name = ep_group_name(ep_size)

    for ranks in ep_rank_groups:
        group = dist.new_group(ranks)
        if rank in ranks:
            ds_groups._EXPERT_PARALLEL_GROUP[ep_name] = group
            ds_groups._EXPERT_PARALLEL_GROUP_RANKS[ep_name] = list(ranks)

    expert_dp_group = None
    expert_dp_ranks: list[int] = []
    for ranks in topology.intra_node_ep_replica_groups(world_size, ep_size):
        group = dist.new_group(ranks)
        if rank in ranks:
            expert_dp_group = group
            expert_dp_ranks = list(ranks)
    ds_groups._EXPERT_DATA_PARALLEL_GROUP[ep_name] = expert_dp_group
    ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[ep_name] = expert_dp_ranks
    assert_expert_data_parallel_group_holds_every_replica(ep_name)


def setup_decoupled_expert_groups(ep_size: int, world_size: int, gpus_per_node: int) -> None:
    """Wire EP for the decoupled intra-node-EP topology (``topology.use_intra_node_ep_decoupled``).

    SP groups are already built; this lays out EP purely as a function of ``ep`` (never ``sp``), which is why
    the two are independent. EP groups are intra-node NVLink ep-blocks (``topology.intra_node_ep_groups``),
    each a full set of experts sharded ``ep``-way and replicated ``world//ep`` times. The expert-DP groups
    (``topology.intra_node_ep_replica_groups``, replicas of one shard strided by ``ep``) all-reduce those replicas'
    partial grads and shard the expert optimizer/master state. Reduces to one whole-world EP group +
    singleton expert-DP groups when ``ep == world == gpus_per_node``.
    """
    _register_independent_expert_groups(
        ep_size,
        world_size,
        topology.intra_node_ep_groups(world_size, ep_size, gpus_per_node),
    )


def setup_host_complete_expert_groups(ep_size: int, world_size: int) -> None:
    """Wire EP for host-complete packing (``topology.use_host_complete_ep``).

    SP groups are already built. EP groups are consecutive blocks of ``ep`` ranks, each an integer number of
    full nodes, so every host an EP group spans contributes a complete NVLink domain. Expert-DP is the same
    strided replica set as decoupled/node-local (``rank % ep``).
    """
    _register_independent_expert_groups(
        ep_size,
        world_size,
        topology.host_complete_ep_groups(world_size, ep_size),
    )


def setup_node_local_groups(sp_size: int, ep_size: int, world_size: int, gpus_per_node: int):
    """Build the node-local-EP topology (2D-SP): EP inside a node (NVLink blocks) while SP spans nodes.

    Every group is built by hand because the default builders lay EP out strided by SP, which would put EP
    peers on different nodes here. ``dist.new_group`` is collective, so every rank walks the same lists in
    the same order. SP-group member order matches each rank's ``sp_rank`` from ``topology.rank_to_dp_sp``.
    """
    rank = dist.get_rank()

    sp_group = None
    for ranks in topology.sp_groups(world_size, ep_size, gpus_per_node, sp_size=sp_size):
        group = dist.new_group(ranks)
        if rank in ranks:
            sp_group = group

    # Seq-DP group == whole world (ZeRO reduces dense grads over every replica and sequence shard).
    world_group = dist.new_group(list(range(world_size)))

    if sp_group is None:
        raise RuntimeError(f"rank {rank} not placed in any SP group (node-local-EP)")

    # Register SP state directly (initialize_sequence_parallel would build its own contiguous groups).
    sp_mpu._SEQUENCE_PARALLEL_GROUP = sp_group
    sp_mpu._SEQUENCE_DATA_PARALLEL_GROUP = world_group

    ep_name = ep_group_name(ep_size)
    for ranks in topology.ep_groups(world_size, ep_size, gpus_per_node):
        group = dist.new_group(ranks)
        if rank in ranks:
            ds_groups._EXPERT_PARALLEL_GROUP[ep_name] = group
            ds_groups._EXPERT_PARALLEL_GROUP_RANKS[ep_name] = list(ranks)

    # Same-expert ranks are every replica of this ep_position (equals the SP group when world == sp*ep).
    expert_dp_group = None
    expert_dp_ranks: list[int] = []
    for ranks in topology.intra_node_ep_replica_groups(world_size, ep_size):
        group = dist.new_group(ranks)
        if rank in ranks:
            expert_dp_group = group
            expert_dp_ranks = list(ranks)
    ds_groups._EXPERT_DATA_PARALLEL_GROUP[ep_name] = expert_dp_group
    ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[ep_name] = expert_dp_ranks
    assert_expert_data_parallel_group_holds_every_replica(ep_name)

    ds_groups.mpu = sp_mpu

    return SPGroups(
        sp_mpu,
        sp_group,
        sp_mpu.get_sequence_parallel_world_size(),
        sp_mpu.get_sequence_parallel_rank(),
    )


def log_topology(
    rank: int,
    sp_world_size: int,
    sp_rank: int,
    sp_group,
    ep_size: int,
    ep_group_layout: str = "auto",
    gpus_per_node=None,
) -> None:
    """Log this rank's SP / EP / expert-DP group membership so replica coverage can be verified.

    ``edp_ranks`` is the registered expert-DP group: every replica of this rank's expert shard (orthogonal:
    ``orthogonal_edp_groups``; node-local, decoupled, and host-complete: ``intra_node_ep_replica_groups``).
    That equals the SP group only when ``dp == ep`` on the orthogonal mesh, or when ``world == sp*ep`` on
    node-local EP.
    ``sp_group`` is ``None`` on the ``sp_size == 1`` path.
    """
    sp_ranks = list(dist.get_process_group_ranks(sp_group)) if sp_group is not None else [rank]
    if ep_size > 1:
        ep_name = ep_group_name(ep_size)
        ep_ranks = list(ds_groups._get_expert_parallel_group_ranks(ep_name))
        expert_dp_ranks = list(ds_groups._get_expert_data_parallel_group_ranks(ep_name))
    else:
        ep_ranks = [rank]
        expert_dp_ranks = [rank]
    if not gpus_per_node:
        raise ValueError("sequence parallelism above 1 requires gpus_per_node")
    resolved_gpus_per_node = int(gpus_per_node)
    ep_topology = topology.resolve_ep_group_topology(
        dist.get_world_size(),
        sp_world_size,
        ep_size,
        resolved_gpus_per_node,
        ep_group_layout,
    )
    logger.info(
        "rank=%d parallel topology ep_group_layout=%s ep_topology=%s sp_size=%d sp_rank=%d "
        "ep_ranks=%s edp_ranks=%s sp_ranks=%s",
        rank,
        ep_group_layout,
        ep_topology,
        sp_world_size,
        sp_rank,
        ep_ranks,
        expert_dp_ranks,
        sp_ranks,
    )
