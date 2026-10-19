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

import gc
import weakref
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.model.implementations.gpu.sp import collectives
from arctic_platform.model.implementations.layers import cp_mamba
from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import reserve_free_port
from arctic_platform.testing_utils import torch_assert_equal


@pytest.fixture(autouse=True)
def clean_groups(monkeypatch):
    monkeypatch.setattr(collectives, "_barrier_groups", weakref.WeakKeyDictionary())
    monkeypatch.delenv("ARCTIC_SP_ALLTOALL_BARRIER", raising=False)


class Group:
    pass


@pytest.fixture
def group_runtime(monkeypatch):
    ranks = {}
    created = []
    destroyed = []
    world = Group()
    ranks[world] = [0, 1]
    monkeypatch.setattr(collectives.dist, "group", SimpleNamespace(WORLD=world))
    monkeypatch.setattr(collectives.dist, "is_initialized", lambda: world in ranks)
    monkeypatch.setattr(collectives.dist, "get_process_group_ranks", lambda group: ranks[group])

    def new_group(**kwargs):
        group = Group()
        ranks[group] = kwargs["ranks"]
        created.append(kwargs)
        return group

    def destroy(group):
        destroyed.append(group)
        del ranks[group]

    monkeypatch.setattr(collectives.dist, "new_group", new_group)
    monkeypatch.setattr(collectives.dist, "destroy_process_group", destroy)
    return SimpleNamespace(ranks=ranks, created=created, destroyed=destroyed, world=world)


@pytest.mark.parametrize("value", [None, "", "0", "true", "on"])
def test_disabled_values_do_not_create_barrier_group(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("ARCTIC_SP_ALLTOALL_BARRIER", value)
    monkeypatch.setattr(collectives.dist, "new_group", lambda **kw: pytest.fail("unexpected group"))
    assert collectives._exchange_barrier_group(object()) is None


def test_group_is_cached_for_the_exact_sp_group(monkeypatch, group_runtime):
    monkeypatch.setenv("ARCTIC_SP_ALLTOALL_BARRIER", "1")
    source = Group()
    group_runtime.ranks[source] = [8, 9]
    assert collectives._exchange_barrier_group(source) is collectives._exchange_barrier_group(source)
    assert group_runtime.created == [{"ranks": [8, 9], "backend": "gloo", "use_local_synchronization": True}]
    other = Group()
    group_runtime.ranks[other] = [8, 9]
    collectives._exchange_barrier_group(other)
    assert len(group_runtime.created) == 2


def test_destroy_releases_only_the_requested_group_and_all_is_idempotent(monkeypatch, group_runtime):
    monkeypatch.setenv("ARCTIC_SP_ALLTOALL_BARRIER", "1")
    source = Group()
    group_runtime.ranks[source] = [8, 9]
    first = collectives._exchange_barrier_group(source)
    second = collectives._exchange_barrier_group(None)
    collectives.destroy_exchange_barrier_groups(source)
    assert group_runtime.destroyed == [first]
    assert source not in collectives._barrier_groups
    assert group_runtime.world in collectives._barrier_groups
    collectives.destroy_exchange_barrier_groups()
    collectives.destroy_exchange_barrier_groups()
    assert group_runtime.destroyed == [first, second]
    assert not collectives._barrier_groups


@pytest.mark.parametrize("destroy_source", [False, True])
def test_destroyed_groups_are_not_reused(monkeypatch, group_runtime, destroy_source):
    monkeypatch.setenv("ARCTIC_SP_ALLTOALL_BARRIER", "1")
    source = Group()
    group_runtime.ranks[source] = [8, 9]
    barrier = collectives._exchange_barrier_group(source)
    if destroy_source:
        del group_runtime.ranks[source]
        new_source = Group()
        group_runtime.ranks[new_source] = [8, 9]
    else:
        del group_runtime.ranks[barrier]
        new_source = source
    new_barrier = collectives._exchange_barrier_group(new_source)
    assert new_barrier is not barrier
    assert barrier not in group_runtime.ranks
    assert len(collectives._barrier_groups) == 1


def test_cache_does_not_retain_groups_after_global_teardown(monkeypatch, group_runtime):
    monkeypatch.setenv("ARCTIC_SP_ALLTOALL_BARRIER", "1")
    source = Group()
    group_runtime.ranks[source] = [8, 9]
    barrier = collectives._exchange_barrier_group(source)
    source_ref = weakref.ref(source)
    barrier_ref = weakref.ref(barrier)
    group_runtime.ranks.clear()
    del source, barrier
    gc.collect()
    assert source_ref() is None
    assert barrier_ref() is None
    assert not collectives._barrier_groups
    collectives.destroy_exchange_barrier_groups()
    assert group_runtime.destroyed == []


def test_default_group_reinitialization_uses_the_new_world(monkeypatch, group_runtime):
    monkeypatch.setenv("ARCTIC_SP_ALLTOALL_BARRIER", "1")
    old = collectives._exchange_barrier_group(None)
    group_runtime.ranks.clear()
    world = Group()
    group_runtime.ranks[world] = [0, 1]
    monkeypatch.setattr(collectives.dist.group, "WORLD", world)
    monkeypatch.setattr(collectives.dist, "is_initialized", lambda: True)
    new = collectives._exchange_barrier_group(None)
    assert new is not old
    assert len(collectives._barrier_groups) == 1
    assert group_runtime.destroyed == []


def test_unregistered_source_is_rejected_without_creating_a_group(monkeypatch, group_runtime):
    monkeypatch.setenv("ARCTIC_SP_ALLTOALL_BARRIER", "1")
    with pytest.raises(ValueError, match="not registered"):
        collectives._exchange_barrier_group(Group())
    assert group_runtime.created == []


@pytest.mark.parametrize("sender_fails", [False, True])
@pytest.mark.parametrize("barrier_fails", [False, True])
def test_worker_destroy_releases_barriers_before_process_groups(monkeypatch, sender_fails, barrier_fails):
    pytest.importorskip("ray")
    pytest.importorskip("deepspeed")
    from arctic_platform.common.deepspeed_worker import DeepSpeedWorker

    events = []

    def sender_destroy():
        events.append("sender")
        if sender_fails:
            raise RuntimeError("sender failed")

    def barrier_destroy():
        events.append("barriers")
        if barrier_fails:
            raise RuntimeError("barrier cleanup failed")

    monkeypatch.setattr(collectives, "destroy_exchange_barrier_groups", barrier_destroy)
    monkeypatch.setattr(collectives.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(collectives.dist, "destroy_process_group", lambda: events.append("process-groups"))
    cls = DeepSpeedWorker.__ray_metadata__.modified_class
    worker = object.__new__(cls)
    worker._weight_sender = SimpleNamespace(destroy=sender_destroy)
    worker.engine = object()
    if sender_fails or barrier_fails:
        expected = "barrier cleanup failed" if barrier_fails else "sender failed"
        with pytest.raises(RuntimeError, match=expected):
            worker.destroy()
    else:
        assert worker.destroy()
    assert events == ["sender", "barriers", "process-groups"]
    assert worker._weight_sender is None
    assert worker.engine is None


def _gloo_lifecycle_worker(rank, port):
    import os

    os.environ["ARCTIC_SP_ALLTOALL_BARRIER"] = "1"
    for _ in range(2):
        collectives.dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
        )
        try:
            source = collectives.dist.new_group(ranks=[0, 1], backend="gloo")
            barrier = collectives._exchange_barrier_group(source)
            assert collectives._exchange_barrier_group(source) is barrier
            collectives.dist.barrier(group=barrier)
            collectives.destroy_exchange_barrier_groups(source)
            assert not collectives._barrier_groups
            replacement = collectives._exchange_barrier_group(source)
            assert replacement is not barrier
            collectives.dist.barrier(group=replacement)
            default_barrier = collectives._exchange_barrier_group(None)
            collectives.dist.barrier(group=default_barrier)
        finally:
            collectives.dist.destroy_process_group()
        del source, barrier, replacement, default_barrier
        gc.collect()
        assert not collectives._barrier_groups


@pytest.mark.timeout(90)
def test_real_gloo_groups_survive_teardown_and_reinitialization():
    if not collectives.dist.is_gloo_available():
        pytest.skip("PyTorch was built without Gloo")
    port = reserve_free_port(get_unique_port_number() + 1, span=7)
    torch.multiprocessing.spawn(_gloo_lifecycle_worker, args=(port,), nprocs=2, join=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_barrier_follows_forward_and_backward_exchange(monkeypatch, enabled):
    events = []
    barrier = object() if enabled else None
    group = object()
    monkeypatch.setattr(collectives.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(collectives, "_exchange_barrier_group", lambda group: barrier)
    monkeypatch.setattr(
        collectives.torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(synchronize=lambda: events.append("cuda")),
    )
    monkeypatch.setattr(collectives.dist, "barrier", lambda **kw: events.append(("gloo", kw["group"])))

    def exchange(outputs, inputs, **kwargs):
        assert kwargs["group"] is group
        events.append("exchange")
        for output, source in zip(outputs, inputs, strict=True):
            output.copy_(source)

    monkeypatch.setattr(collectives.dist, "all_to_all", exchange)
    value = torch.arange(16.0).reshape(1, 4, 4).requires_grad_()
    result = collectives.sequence_head_all_to_all(group, value, scatter_dim=2, gather_dim=1)
    torch_assert_equal(result, torch.cat(value.chunk(2, dim=2), dim=1))
    result.sum().backward()
    torch_assert_equal(value.grad, torch.ones_like(value))
    expected = ["exchange", "cuda", ("gloo", barrier)] if enabled else ["exchange"]
    assert events == expected * 2


def test_single_rank_does_not_create_a_barrier(monkeypatch):
    monkeypatch.setattr(collectives.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(collectives, "_exchange_barrier_group", lambda group: pytest.fail("unexpected group"))
    value = torch.ones(1, 2, 2, requires_grad=True)
    collectives.sequence_head_all_to_all(object(), value, scatter_dim=2, gather_dim=1).sum().backward()
    torch_assert_equal(value.grad, torch.ones_like(value))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("direction", ["seq_to_head_parallel", "head_to_seq_parallel"])
def test_mamba_barrier_follows_forward_and_backward_exchange(monkeypatch, enabled, direction):
    events = []
    barrier = object() if enabled else None
    group = object()
    monkeypatch.setattr(cp_mamba, "_exchange_barrier_group", lambda group: barrier)
    monkeypatch.setattr(
        collectives.torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(synchronize=lambda: events.append("cuda")),
    )
    monkeypatch.setattr(collectives.dist, "barrier", lambda **kw: events.append(("gloo", kw["group"])))

    def exchange(output, value, **kwargs):
        assert kwargs["group"] is group
        events.append("exchange")
        output.copy_(value)

    monkeypatch.setattr(cp_mamba.dist, "all_to_all_single", exchange)
    value = torch.arange(16.0).reshape(1, 4, 4).requires_grad_()
    result = getattr(cp_mamba, direction)(value, group, 2)
    if direction == "seq_to_head_parallel":
        expected = torch.cat(value.chunk(2, dim=2), dim=1)
    else:
        expected = torch.cat(value.chunk(2, dim=1), dim=2)
    torch_assert_equal(result, expected)
    result.sum().backward()
    torch_assert_equal(value.grad, torch.ones_like(value))
    expected_events = ["exchange", "cuda", ("gloo", barrier)] if enabled else ["exchange"]
    assert events == expected_events * 2


@pytest.mark.parametrize("direction", ["seq_to_head_parallel", "head_to_seq_parallel"])
def test_mamba_single_rank_does_not_exchange_or_create_a_barrier(monkeypatch, direction):
    monkeypatch.setattr(cp_mamba, "_exchange_barrier_group", lambda group: pytest.fail("unexpected group"))
    monkeypatch.setattr(cp_mamba.dist, "all_to_all_single", lambda *args, **kwargs: pytest.fail("unexpected exchange"))
    value = torch.ones(1, 2, 2, requires_grad=True)
    result = getattr(cp_mamba, direction)(value, object(), 1)
    torch_assert_equal(result, value)
    result.sum().backward()
    torch_assert_equal(value.grad, torch.ones_like(value))
