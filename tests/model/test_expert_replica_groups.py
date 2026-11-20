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

"""The expert-DP group must be every rank holding that expert shard, on every topology and at every shape.

An expert gradient all-reduces over the group registered as expert-data-parallel, so that group has to be the
full set of ranks holding the shard. The orthogonal mesh is covered first and node-local EP at the bottom of
the file; both are laid out so that the sequence-parallel group is a subset of the replica set which happens
to be the whole of it at one shape, and both are therefore exposed to the same regression.

On the orthogonal mesh a rank's shard position is ``(rank // sp) % ep`` and each position has ``world // ep``
replicas, spanning both the sequence-parallel and the data-parallel axis. The sequence-parallel group is always
a subset of that set -- its members share ``rank // sp`` and therefore share the position -- so the two coincide
exactly when the sizes match, ``world // ep == sp``, which is to say when ``dp == ep``. Below that shape a
construction returning the sequence-parallel group covers ``ep / dp`` of the partials and the expert gradients
come out scaled by that ratio.

The dangerous half of this is the shape where the two sets agree: a topology change moving a job off ``dp == ep``
turns a silent-and-correct group into a silent-and-wrong one. So the coincidence itself is pinned here, at a
shape where it holds and a shape where it does not, and the replica set is derived from a partition of the world
rather than from the expression under test.

Two layers are covered, and only the second constrains what a training job reduces over. The first is arithmetic:
which rank lists ``topology.orthogonal_edp_groups`` returns. The second is registration: which rank list
``setup_orthogonal_expert_groups`` leaves in DeepSpeed's ``_EXPERT_DATA_PARALLEL_GROUP_RANKS``, since a correct
helper that the setup never consults leaves the wrong group registered and the arithmetic tests still pass.
Registration is driven against a substitute world (``SubstituteWorld``, and ``NodeLocalSubstituteWorld`` for the
other topology) instead of a process group, so it runs in-process on CPU at world sizes no test machine has.

These are pure integer topology assertions -- exact set equality, no tolerance and no GPU.
"""

import itertools
from types import SimpleNamespace

import pytest

import arctic_platform.model.parallel.groups as parallel_groups
from arctic_platform.model.parallel import topology
from arctic_platform.model.parallel.groups import ep_group_name


def sequence_parallel_group(rank: int, sp_size: int) -> list[int]:
    """The contiguous sequence-parallel block holding ``rank``, which is what the mesh's SP groups are."""
    block = rank // sp_size
    return list(range(block * sp_size, (block + 1) * sp_size))


def replica_set_by_partition(world_size: int, sp_size: int, ep_size: int, rank: int) -> list[int]:
    """The ranks holding ``rank``'s expert shard, derived by walking the mesh rather than by indexing it.

    The world is laid out as ``dp`` sequence-parallel blocks of ``sp`` ranks; expert parallelism cycles over
    those blocks with period ``ep``, so block ``b`` holds shard ``b % ep`` and a shard's replicas are every rank
    of every block whose index is congruent to it. Built by enumerating the blocks, so it does not share the
    ``(rank // sp) % ep`` expression with the function under test.
    """
    blocks = [list(range(start, start + sp_size)) for start in range(0, world_size, sp_size)]
    my_block = rank // sp_size
    return sorted(
        peer for index, block in enumerate(blocks) if index % ep_size == my_block % ep_size for peer in block
    )


SHAPES = [
    (world_size, sp_size, ep_size)
    for world_size, sp_size, ep_size in itertools.product([2, 4, 8, 16, 32], [1, 2, 4, 8], [1, 2, 4, 8])
    if world_size % sp_size == 0 and (world_size // sp_size) % ep_size == 0
]


@pytest.mark.parametrize("world_size,sp_size,ep_size", SHAPES)
def test_registered_group_is_the_whole_replica_set(world_size, sp_size, ep_size):
    """Every rank's group is exactly the ranks holding its shard, and the groups partition the world."""
    groups = topology.orthogonal_edp_groups(world_size, sp_size, ep_size)

    assert sorted(rank for group in groups for rank in group) == list(range(world_size))
    assert all(len(group) == world_size // ep_size for group in groups)
    for rank in range(world_size):
        holding = [group for group in groups if rank in group]
        assert len(holding) == 1
        assert sorted(holding[0]) == replica_set_by_partition(world_size, sp_size, ep_size, rank)


@pytest.mark.parametrize("world_size,sp_size,ep_size", SHAPES)
def test_the_sequence_parallel_group_is_a_strict_subset_unless_dp_equals_ep(world_size, sp_size, ep_size):
    """Pins the coincidence: reusing the SP group is right only at ``dp == ep``, and short by ``dp / ep`` else.

    The shape where it is right is the one a future topology change can walk off, so it is asserted rather than
    left as a remark. ``dp > ep`` is the trigger for a group that under-reduces; ``dp < ep`` cannot occur here
    because the orthogonal mesh requires ``dp % ep == 0``.
    """
    groups = topology.orthogonal_edp_groups(world_size, sp_size, ep_size)
    data_parallel_size = world_size // sp_size

    assert data_parallel_size % ep_size == 0
    for rank in range(world_size):
        replicas = sorted(next(group for group in groups if rank in group))
        sp_group = sequence_parallel_group(rank, sp_size)

        assert set(sp_group) <= set(replicas)
        if data_parallel_size == ep_size:
            assert sp_group == replicas
        else:
            assert sp_group != replicas
            assert len(replicas) == len(sp_group) * (data_parallel_size // ep_size)


def test_world_8_sp_2_is_wrong_at_ep_2_and_accidentally_right_at_ep_4():
    """The two measured points, written out: 8 ranks, sp 2, dp 4, where the trigger is ``dp > ep``.

    At ``ep`` 2 rank 0's shard is held by ``{0,1,4,5}`` while the sequence-parallel group is ``{0,1}``, so
    reusing that group sums half the partials. At ``ep`` 4 the shard is held by ``{0,1}`` and the same reuse is
    correct, which is why a wrong construction survived at this world size.
    """
    assert topology.orthogonal_edp_groups(8, 2, 2) == [[0, 1, 4, 5], [2, 3, 6, 7]]
    assert topology.orthogonal_edp_groups(8, 2, 4) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert sequence_parallel_group(0, 2) == [0, 1]


# ``setup_orthogonal_expert_groups`` runs only under sequence parallelism and only when experts are sharded,
# so the shapes it can be driven at are the ones above with ``sp > 1`` and ``ep > 1``.
REGISTRATION_SHAPES = [
    (world_size, sp_size, ep_size) for world_size, sp_size, ep_size in SHAPES if sp_size > 1 and ep_size > 1
]


def deepspeed_expert_groups(world_size: int, sp_size: int, ep_size: int):
    """The expert-parallel and expert-data-parallel rank lists DeepSpeed builds for this mesh.

    ``parallel_state_sp`` aliases ``get_model_parallel_world_size`` to the sequence-parallel world size, so
    ``_create_expert_data_and_model_parallel`` reads ``sp`` as the model-parallel degree: its data-parallel
    groups are the ranks strided by ``sp``, each one is cut into consecutive ``ep``-sized expert-parallel
    blocks, and zipping those blocks by position gives the expert-data-parallel groups it registers before
    the mesh's own groups are applied over them. Pipeline parallelism is absent, so its stride drops out.
    """
    data_parallel_groups = [list(range(offset, world_size, sp_size)) for offset in range(sp_size)]
    expert_parallel_groups = []
    expert_data_parallel_groups = []
    for dp_ranks in data_parallel_groups:
        blocks = [dp_ranks[start : start + ep_size] for start in range(0, len(dp_ranks), ep_size)]
        expert_parallel_groups.extend(blocks)
        expert_data_parallel_groups.extend(list(column) for column in zip(*blocks))
    return expert_parallel_groups, expert_data_parallel_groups


class RecordedGroup:
    """Stand-in for a process group, carrying the rank list it was created from.

    Registration is checked by object and not only by the rank list recorded beside it: an all-reduce goes to
    the group object, so recording the right ranks next to a group built over different ones is still wrong.
    """

    def __init__(self, ranks):
        self.ranks = list(ranks)

    def __repr__(self):
        return f"RecordedGroup({self.ranks})"


class SubstituteCollectives:
    """The distributed calls a group setup makes, answered for a whole simulated world at once.

    A setup reaches the outside through ``dist`` and ``ds_groups`` in its own module, so replacing those two
    runs the real function. Answering collectives for the whole world is exact because every rank's view here
    is a function of the shape alone: ``new_group`` returns a group for any rank list, and
    ``all_gather_object`` returns every rank's expert-shard position. That position map is supplied by the
    subclass, since the two topologies lay expert-parallel groups out differently.
    """

    def __init__(self, world_size: int, rank: int, positions: dict[int, int]):
        self.world_size = world_size
        self.rank = rank
        self.positions = positions
        self.new_group_calls: list[list[int]] = []
        self.ds_groups = SimpleNamespace(
            _EXPERT_PARALLEL_GROUP={},
            _EXPERT_PARALLEL_GROUP_RANKS={},
            _EXPERT_DATA_PARALLEL_GROUP={},
            _EXPERT_DATA_PARALLEL_GROUP_RANKS={},
        )
        self.dist = SimpleNamespace(
            get_rank=lambda: self.rank,
            get_world_size=lambda: self.world_size,
            new_group=self.new_group,
            get_process_group_ranks=lambda group: list(group.ranks),
            all_gather_object=self.all_gather_object,
        )

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(parallel_groups, "dist", self.dist)
        monkeypatch.setattr(parallel_groups, "ds_groups", self.ds_groups)

    def new_group(self, ranks):
        self.new_group_calls.append(list(ranks))
        return RecordedGroup(ranks)

    def all_gather_object(self, output, obj) -> None:
        assert (
            obj == self.positions[self.rank]
        ), f"rank {self.rank} contributed {obj!r}, not its expert-shard position {self.positions[self.rank]}"
        output[:] = [self.positions[peer] for peer in range(self.world_size)]


class SubstituteWorld(SubstituteCollectives):
    """One rank of an orthogonal ``(world_size, sp_size, ep_size)`` mesh, driven in-process.

    ``setup_orthogonal_expert_groups`` reaches the outside through one name the base does not cover, the
    sequence-parallel mpu it is handed, so this adds it. The substitute answers every call either
    construction makes: ``get_sequence_parallel_group`` and ``get_process_group_ranks`` as well as
    ``new_group`` and ``all_gather_object``. A substitute only one of them could run would prove nothing,
    because the other would fail on the substitute rather than on its result.
    """

    def __init__(self, world_size: int, sp_size: int, ep_size: int, rank: int):
        self.sp_size = sp_size
        self.ep_size = ep_size
        self.expert_parallel_groups, self.expert_data_parallel_groups = deepspeed_expert_groups(
            world_size, sp_size, ep_size
        )
        super().__init__(
            world_size,
            rank,
            {peer: position for group in self.expert_parallel_groups for position, peer in enumerate(group)},
        )
        self.ds_groups._create_expert_data_and_model_parallel = self.create_expert_data_and_model_parallel
        self.sp_mpu = SimpleNamespace(
            get_sequence_parallel_group=lambda: RecordedGroup(sequence_parallel_group(self.rank, self.sp_size)),
            get_sequence_parallel_world_size=lambda: self.sp_size,
        )

    def create_expert_data_and_model_parallel(self, ep_size, mpu) -> None:
        """The state DeepSpeed leaves behind, which the mesh's expert-DP groups are then applied over."""
        assert ep_size == self.ep_size
        assert mpu is self.sp_mpu
        name = ep_group_name(ep_size)
        for ranks in self.expert_parallel_groups:
            group = self.new_group(ranks)
            if self.rank in ranks:
                self.ds_groups._EXPERT_PARALLEL_GROUP[name] = group
                self.ds_groups._EXPERT_PARALLEL_GROUP_RANKS[name] = list(ranks)
        for ranks in self.expert_data_parallel_groups:
            group = self.new_group(ranks)
            if self.rank in ranks:
                self.ds_groups._EXPERT_DATA_PARALLEL_GROUP[name] = group
                self.ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[name] = list(ranks)


def register(world_size: int, sp_size: int, ep_size: int, rank: int, monkeypatch) -> SubstituteWorld:
    """Run the real ``setup_orthogonal_expert_groups`` for one rank and hand back what it registered."""
    world = SubstituteWorld(world_size, sp_size, ep_size, rank)
    with monkeypatch.context() as patched:
        world.install(patched)
        parallel_groups.setup_orthogonal_expert_groups(world.sp_mpu, ep_size)
    return world


def test_the_substitute_worlds_expert_group_layout_is_deepspeeds():
    """Pin the substitute against DeepSpeed's own rank builder rather than against a docstring.

    A rank's expert shard is named by its position in the expert-parallel group, so a substitute laying those
    groups out differently would be checking registration on a mesh DeepSpeed does not build, and both
    versions of the setup would be judged against the wrong replica sets.
    """
    from deepspeed.utils.groups import _get_expert_parallel_ranks

    for world_size, sp_size, ep_size in REGISTRATION_SHAPES:
        assert deepspeed_expert_groups(world_size, sp_size, ep_size) == _get_expert_parallel_ranks(
            world_size, sp_size, ep_size
        )

    assert deepspeed_expert_groups(8, 2, 2)[0] == [[0, 2], [4, 6], [1, 3], [5, 7]]
    assert deepspeed_expert_groups(8, 2, 4)[0] == [[0, 2, 4, 6], [1, 3, 5, 7]]


def test_expert_shard_positions_and_the_partition_name_the_same_replicas():
    """Tie the two derivations of "who holds my shard" together, so the registration tests can use either.

    One reads a rank's position off DeepSpeed's expert-parallel groups and collects every rank at the same
    position; the other walks the sequence-parallel blocks of the mesh. Neither goes through
    ``orthogonal_edp_groups``, so agreement between them is not a property of the function under test.
    """
    for world_size, sp_size, ep_size in REGISTRATION_SHAPES:
        expert_parallel_groups, _ = deepspeed_expert_groups(world_size, sp_size, ep_size)
        positions = {peer: position for group in expert_parallel_groups for position, peer in enumerate(group)}
        for rank in range(world_size):
            replicas = sorted(peer for peer in range(world_size) if positions[peer] == positions[rank])
            assert replicas == replica_set_by_partition(world_size, sp_size, ep_size, rank)


@pytest.mark.parametrize("world_size,sp_size,ep_size", REGISTRATION_SHAPES)
def test_setup_registers_the_whole_replica_set_as_the_expert_data_parallel_group(
    world_size, sp_size, ep_size, monkeypatch
):
    """What ``setup_orthogonal_expert_groups`` leaves registered, rank by rank.

    This is the property an expert all-reduce depends on, and it is not implied by the arithmetic tests
    above: those hold whatever the setup does with the groups the helper returns. The rank list and the group
    object are asserted separately because they fail apart -- an all-reduce goes to the object, so a group
    spanning the wrong ranks is wrong however the list beside it was recorded. The rank lists handed to
    ``dist.new_group`` are then compared across ranks, since that call is collective and every rank has to
    walk the same lists in the same order for the groups to match up.
    """
    name = ep_group_name(ep_size)
    calls_per_rank = []
    for rank in range(world_size):
        world = register(world_size, sp_size, ep_size, rank, monkeypatch)
        replicas = replica_set_by_partition(world_size, sp_size, ep_size, rank)

        assert world.ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[name] == replicas
        assert world.ds_groups._EXPERT_DATA_PARALLEL_GROUP[name].ranks == replicas
        calls_per_rank.append(world.new_group_calls)

    assert calls_per_rank.count(calls_per_rank[0]) == world_size


@pytest.mark.parametrize("world_size,sp_size,ep_size", REGISTRATION_SHAPES)
def test_the_registered_group_adds_the_replicas_the_sequence_parallel_group_omits(
    world_size, sp_size, ep_size, monkeypatch
):
    """The registered group measured against the sequence-parallel group, the other candidate for the role.

    At ``dp == ep`` the two are the same set and nothing here can tell them apart. At every other shape the
    registered group must be strictly larger, by exactly ``dp / ep`` -- the factor an expert gradient reduced
    over the sequence-parallel group alone comes out scaled by.
    """
    name = ep_group_name(ep_size)
    data_parallel_size = world_size // sp_size
    for rank in range(world_size):
        world = register(world_size, sp_size, ep_size, rank, monkeypatch)
        registered = world.ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[name]
        sp_ranks = sequence_parallel_group(rank, sp_size)

        assert set(sp_ranks) <= set(registered)
        if data_parallel_size == ep_size:
            assert registered == sp_ranks
        else:
            assert set(sp_ranks) < set(registered)
            assert len(registered) == len(sp_ranks) * (data_parallel_size // ep_size)


def test_the_refusal_reports_the_factor_the_expert_gradient_would_be_scaled_by(monkeypatch):
    """The guard on its own: hand it the sequence-parallel group at a shape where that group is short.

    8 ranks at ``sp`` 2 and ``ep`` 2 gives ``dp`` 4, so rank 0's shard is held by ``{0, 1, 4, 5}`` while its
    sequence-parallel group is ``{0, 1}``, and an all-reduce over the latter sums 2 of the 4 partials.
    """
    world = SubstituteWorld(8, 2, 2, 0)
    name = ep_group_name(2)
    with monkeypatch.context() as patched:
        world.install(patched)
        parallel_groups.setup_orthogonal_expert_groups(world.sp_mpu, 2)
        world.ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[name] = sequence_parallel_group(0, 2)
        with pytest.raises(RuntimeError) as refusal:
            parallel_groups.assert_expert_data_parallel_group_holds_every_replica(name)

    assert "[0, 1] is not the set of ranks" in str(refusal.value)
    assert "[0, 1, 4, 5]" in str(refusal.value)
    assert "scaled by 2/4" in str(refusal.value)


# --- Node-local EP ------------------------------------------------------------------------------------
#
# The same property on the other custom topology. Node-local EP puts each expert-parallel group inside one
# node and lets sequence parallelism cross the network, so a rank's shard is named by its position in an
# intra-node block rather than by ``(rank // sp) % ep``. The replica set is the whole world divided by
# ``ep``, and the sequence-parallel group is only ``sp`` of it, so the two coincide at ``world == sp*ep``
# and nowhere else -- the same coincidence as ``dp == ep`` above, and the same hazard when a job moves off
# it by adding data-parallel replicas.


def intra_node_expert_parallel_groups(world_size: int, ep_size: int, gpus_per_node: int):
    """The intra-node EP blocks node-local EP builds, enumerated node by node.

    DeepEP needs every expert-parallel group inside one NVLink domain, so each node's ranks are cut into
    consecutive ``ep``-sized blocks. Written as slices of each node rather than through
    ``topology.ep_groups``, so the replica sets below do not come from the code under test.
    """
    return [
        list(range(node * gpus_per_node + start, node * gpus_per_node + start + ep_size))
        for node in range(world_size // gpus_per_node)
        for start in range(0, gpus_per_node, ep_size)
    ]


def node_local_replica_set(world_size: int, ep_size: int, gpus_per_node: int, rank: int) -> list[int]:
    """The ranks holding ``rank``'s expert shard: the same position in every intra-node EP block."""
    blocks = intra_node_expert_parallel_groups(world_size, ep_size, gpus_per_node)
    position = next(block.index(rank) for block in blocks if rank in block)
    return sorted(block[position] for block in blocks)


class SubstituteSequenceParallelMpu:
    """Stands in for ``parallel_state_sp``, which node-local setup writes its groups onto and reads back.

    The real module derives the size and rank from the registered group, so the substitute does too rather
    than being told them: a setup that registered the wrong group would otherwise still report the right
    numbers here.
    """

    def __init__(self, rank: int):
        self.rank = rank
        self._SEQUENCE_PARALLEL_GROUP = None
        self._SEQUENCE_DATA_PARALLEL_GROUP = None

    def get_sequence_parallel_world_size(self) -> int:
        return len(self._SEQUENCE_PARALLEL_GROUP.ranks)

    def get_sequence_parallel_rank(self) -> int:
        return self._SEQUENCE_PARALLEL_GROUP.ranks.index(self.rank)


class NodeLocalSubstituteWorld(SubstituteCollectives):
    """One rank of a node-local-EP layout, driven in-process the same way as the mesh above.

    ``setup_node_local_groups`` builds every group itself instead of calling a DeepSpeed builder, so beyond
    the collectives the only thing to stand in for is the sequence-parallel module it writes its groups onto.
    Expert-shard positions come from the intra-node blocks, enumerated independently of ``topology``.
    """

    def __init__(self, world_size: int, sp_size: int, ep_size: int, gpus_per_node: int, rank: int):
        self.sp_size = sp_size
        self.ep_size = ep_size
        self.gpus_per_node = gpus_per_node
        self.expert_parallel_groups = intra_node_expert_parallel_groups(world_size, ep_size, gpus_per_node)
        super().__init__(
            world_size,
            rank,
            {peer: position for group in self.expert_parallel_groups for position, peer in enumerate(group)},
        )
        self.sp_mpu = SubstituteSequenceParallelMpu(rank)

    def install(self, monkeypatch) -> None:
        super().install(monkeypatch)
        monkeypatch.setattr(parallel_groups, "sp_mpu", self.sp_mpu)


def register_node_local(
    world_size: int, sp_size: int, ep_size: int, gpus_per_node: int, rank: int, monkeypatch
) -> NodeLocalSubstituteWorld:
    """Run the real ``setup_node_local_groups`` for one rank and hand back what it registered."""
    world = NodeLocalSubstituteWorld(world_size, sp_size, ep_size, gpus_per_node, rank)
    with monkeypatch.context() as patched:
        world.install(patched)
        world.result = parallel_groups.setup_node_local_groups(sp_size, ep_size, world_size, gpus_per_node)
    return world


# ``world == sp*ep`` is the shape the old construction was written for and the only one where it was right,
# so it is kept alongside the extra-DP shapes that broke it. ``sp=6`` on 8-GPU nodes is the shape whose
# cross-node EP group does not contain rank 0.
NODE_LOCAL_SHAPES = [
    (16, 8, 2, 8),
    (16, 4, 4, 8),
    (32, 16, 2, 8),
    (32, 8, 4, 8),
    (32, 8, 2, 8),
    (32, 4, 4, 8),
    (64, 8, 8, 8),
    (128, 8, 8, 8),
    (24, 6, 2, 8),
]


@pytest.mark.parametrize("world_size,sp_size,ep_size,gpus_per_node", NODE_LOCAL_SHAPES)
def test_every_node_local_shape_takes_the_node_local_path(world_size, sp_size, ep_size, gpus_per_node):
    """The shapes below are only meaningful if setup would actually route them here."""
    assert topology.use_node_local_ep(world_size, sp_size, ep_size, gpus_per_node) is True


@pytest.mark.parametrize("world_size,sp_size,ep_size,gpus_per_node", NODE_LOCAL_SHAPES)
def test_node_local_setup_registers_the_whole_replica_set(world_size, sp_size, ep_size, gpus_per_node, monkeypatch):
    """What ``setup_node_local_groups`` leaves registered, rank by rank.

    The expert-parallel groups it builds are checked first, because a rank's shard is named by its position
    in one: registering the right replica set against blocks laid out differently would be right about the
    wrong thing. The rank list and the group object are then asserted separately because they fail apart --
    an all-reduce goes to the object. The rank lists handed to ``dist.new_group`` are compared across ranks
    last, since that call is collective and every rank has to walk the same lists in the same order.
    """
    name = ep_group_name(ep_size)
    calls_per_rank = []
    for rank in range(world_size):
        world = register_node_local(world_size, sp_size, ep_size, gpus_per_node, rank, monkeypatch)
        expert_parallel_ranks = world.ds_groups._EXPERT_PARALLEL_GROUP_RANKS[name]
        replicas = node_local_replica_set(world_size, ep_size, gpus_per_node, rank)

        assert expert_parallel_ranks == next(block for block in world.expert_parallel_groups if rank in block)
        assert len({peer // gpus_per_node for peer in expert_parallel_ranks}) == 1
        assert world.ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[name] == replicas
        assert world.ds_groups._EXPERT_DATA_PARALLEL_GROUP[name].ranks == replicas
        calls_per_rank.append(world.new_group_calls)

    assert calls_per_rank.count(calls_per_rank[0]) == world_size


@pytest.mark.parametrize("world_size,sp_size,ep_size,gpus_per_node", NODE_LOCAL_SHAPES)
def test_the_node_local_group_adds_the_replicas_the_sequence_parallel_group_omits(
    world_size, sp_size, ep_size, gpus_per_node, monkeypatch
):
    """The registered group measured against the sequence-parallel group, the other candidate for the role.

    At ``world == sp*ep`` there is one sequence-parallel group per expert shard and the two are the same set,
    which is why registering the sequence-parallel group survived at that shape. Adding data-parallel
    replicas cuts the same-expert ranks into ``world // (sp*ep)`` sequence-parallel groups, and reducing over
    one of them is the factor an expert gradient would come out scaled by.
    """
    name = ep_group_name(ep_size)
    for rank in range(world_size):
        world = register_node_local(world_size, sp_size, ep_size, gpus_per_node, rank, monkeypatch)
        registered = world.ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[name]
        sp_ranks = sorted(world.result.sp_group.ranks)

        assert set(sp_ranks) <= set(registered)
        if world_size == sp_size * ep_size:
            assert registered == sp_ranks
        else:
            assert set(sp_ranks) < set(registered)
            assert len(registered) == len(sp_ranks) * (world_size // (sp_size * ep_size))


def test_node_local_refuses_the_sequence_parallel_group_once_extra_dp_makes_it_short(monkeypatch):
    """The regression the guard has to catch, at the shape the 32-GPU repro runs.

    32 GPUs at ``sp`` 8 and ``ep`` 2 gives two sequence-parallel groups per expert shard, so rank 0's shard
    is held by all 16 even ranks while its sequence-parallel group is 8 of them. Registering that group is
    what node-local setup used to do, and it is correct at ``world == sp*ep``, so nothing but this check
    stands between a topology change and expert gradients reduced over half the replicas.
    """
    world = NodeLocalSubstituteWorld(32, 8, 2, 8, 0)
    name = ep_group_name(2)
    with monkeypatch.context() as patched:
        world.install(patched)
        result = parallel_groups.setup_node_local_groups(8, 2, 32, 8)
        world.ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[name] = sorted(result.sp_group.ranks)
        with pytest.raises(RuntimeError) as refusal:
            parallel_groups.assert_expert_data_parallel_group_holds_every_replica(name)

    assert "scaled by 8/16" in str(refusal.value)


HOST_COMPLETE_SHAPES = [
    (32, 2, 16, 8),
    (16, 2, 16, 8),
    (32, 4, 16, 8),
    (32, 16, 16, 8),
]


class HostCompleteSubstituteWorld(SubstituteCollectives):
    """One rank of a host-complete EP overlay, driven the same way as node-local registration tests."""

    def __init__(self, world_size: int, sp_size: int, ep_size: int, rank: int):
        self.sp_size = sp_size
        self.ep_size = ep_size
        self.expert_parallel_groups = topology.host_complete_ep_groups(world_size, ep_size)
        super().__init__(
            world_size,
            rank,
            {peer: position for group in self.expert_parallel_groups for position, peer in enumerate(group)},
        )


def register_host_complete(
    world_size: int, sp_size: int, ep_size: int, rank: int, monkeypatch
) -> HostCompleteSubstituteWorld:
    world = HostCompleteSubstituteWorld(world_size, sp_size, ep_size, rank)
    with monkeypatch.context() as patched:
        world.install(patched)
        parallel_groups.setup_host_complete_expert_groups(ep_size, world_size)
    return world


@pytest.mark.parametrize("world_size,sp_size,ep_size,gpus_per_node", HOST_COMPLETE_SHAPES)
def test_every_host_complete_shape_takes_the_host_complete_path(world_size, sp_size, ep_size, gpus_per_node):
    assert topology.use_host_complete_ep(world_size, sp_size, ep_size, gpus_per_node) is True
    assert topology.use_node_local_ep(world_size, sp_size, ep_size, gpus_per_node) is False


@pytest.mark.parametrize("world_size,sp_size,ep_size,gpus_per_node", HOST_COMPLETE_SHAPES)
def test_host_complete_setup_registers_full_node_ep_groups(world_size, sp_size, ep_size, gpus_per_node, monkeypatch):
    name = ep_group_name(ep_size)
    calls_per_rank = []
    expected_blocks = topology.host_complete_ep_groups(world_size, ep_size)
    for rank in range(world_size):
        world = register_host_complete(world_size, sp_size, ep_size, rank, monkeypatch)
        expert_parallel_ranks = world.ds_groups._EXPERT_PARALLEL_GROUP_RANKS[name]
        replicas = [peer for peer in range(world_size) if peer % ep_size == rank % ep_size]
        assert expert_parallel_ranks == next(block for block in expected_blocks if rank in block)
        by_node = {}
        for peer in expert_parallel_ranks:
            by_node.setdefault(peer // gpus_per_node, []).append(peer)
        assert all(len(ranks) == gpus_per_node for ranks in by_node.values())
        assert world.ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[name] == replicas
        assert world.ds_groups._EXPERT_DATA_PARALLEL_GROUP[name].ranks == replicas
        calls_per_rank.append(world.new_group_calls)
    assert calls_per_rank.count(calls_per_rank[0]) == world_size
