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

from types import SimpleNamespace

import pytest
import torch

from arctic_platform.model.implementations.gpu.sp import collectives
from arctic_platform.testing_utils import torch_assert_equal


@pytest.fixture(autouse=True)
def clean_groups(monkeypatch):
    monkeypatch.setattr(collectives, "_barrier_groups", {})
    monkeypatch.delenv("ARCTIC_SP_ALLTOALL_BARRIER", raising=False)


def test_default_does_not_create_barrier_group(monkeypatch):
    monkeypatch.setattr(collectives.dist, "new_group", lambda **kw: pytest.fail("unexpected group"))
    assert collectives._exchange_barrier_group(object()) is None


def test_group_is_cached_for_the_exact_sp_group(monkeypatch):
    monkeypatch.setenv("ARCTIC_SP_ALLTOALL_BARRIER", "1")
    calls = []
    monkeypatch.setattr(collectives.dist, "get_process_group_ranks", lambda group: [8, 9])
    monkeypatch.setattr(collectives.dist, "new_group", lambda **kw: calls.append(kw) or object())
    source = object()
    assert collectives._exchange_barrier_group(source) is collectives._exchange_barrier_group(source)
    assert calls == [{"ranks": [8, 9], "backend": "gloo", "use_local_synchronization": True}]
    collectives._exchange_barrier_group(object())
    assert len(calls) == 2


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
