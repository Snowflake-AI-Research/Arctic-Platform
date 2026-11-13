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
from arctic_platform.model.implementations.layers import cp_mamba


@pytest.fixture(autouse=True)
def forbid_extra_synchronization(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("SP exchange must not create groups or synchronize CUDA/Gloo")

    monkeypatch.setattr(collectives.dist, "new_group", unexpected)
    monkeypatch.setattr(collectives.dist, "barrier", unexpected)
    monkeypatch.setattr(torch.cuda, "current_stream", unexpected)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)


@pytest.mark.parametrize("scatter,gather", [(2, 1), (1, 2)])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_sequence_head_exchange_preserves_values_and_gradients(monkeypatch, scatter, gather, noncontiguous):
    group = object()
    events = []
    monkeypatch.setattr(collectives.dist, "get_world_size", lambda group: 2)

    def exchange(outputs, inputs, **kwargs):
        assert kwargs["group"] is group
        events.append("exchange")
        for output, source in zip(outputs, inputs, strict=True):
            assert source.is_contiguous()
            output.copy_(source)

    monkeypatch.setattr(collectives.dist, "all_to_all", exchange)
    value = torch.arange(32.0).reshape(2, 4, 4).transpose(1, 2)
    if not noncontiguous:
        value = value.contiguous()
    value.requires_grad_()
    result = collectives.sequence_head_all_to_all(group, value, scatter_dim=scatter, gather_dim=gather)
    expected = torch.cat(value.chunk(2, dim=scatter), dim=gather)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.square().sum().backward()
    torch.testing.assert_close(value.grad, 2 * value, rtol=0, atol=0)
    assert events == ["exchange", "exchange"]


def test_sequence_head_single_rank_skips_collective(monkeypatch):
    monkeypatch.setattr(collectives.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(collectives.dist, "all_to_all", lambda *a, **kw: pytest.fail("unexpected exchange"))
    value = torch.arange(12.0).reshape(1, 3, 4).transpose(1, 2).requires_grad_()
    result = collectives.sequence_head_all_to_all(object(), value, scatter_dim=2, gather_dim=1)
    torch.testing.assert_close(result, value, rtol=0, atol=0)
    result.square().sum().backward()
    torch.testing.assert_close(value.grad, 2 * value, rtol=0, atol=0)


def test_sequence_head_rejects_indivisible_scatter_dimension(monkeypatch):
    monkeypatch.setattr(collectives.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(collectives.dist, "all_to_all", lambda *a, **kw: pytest.fail("unexpected exchange"))
    with pytest.raises(ValueError, match="not divisible"):
        collectives.sequence_head_all_to_all(object(), torch.ones(1, 4, 3), scatter_dim=2, gather_dim=1)


@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("direction", ["seq_to_head_parallel", "head_to_seq_parallel"])
def test_mamba_exchange_preserves_contiguous_copy_and_gradients(monkeypatch, size, noncontiguous, direction):
    group = object()
    events = []

    def exchange(output, value, **kwargs):
        assert kwargs["group"] is group
        assert value.is_contiguous()
        events.append("exchange")
        output.copy_(value)

    monkeypatch.setattr(cp_mamba.dist, "all_to_all_single", exchange)
    value = torch.arange(32.0).reshape(2, 4, 4).transpose(1, 2)
    if not noncontiguous:
        value = value.contiguous()
    value.requires_grad_()
    result = getattr(cp_mamba, direction)(value, group, size)
    scatter, gather = (2, 1) if direction == "seq_to_head_parallel" else (1, 2)
    expected = torch.cat(value.chunk(size, dim=scatter), dim=gather)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.is_contiguous()
    assert result.data_ptr() != value.data_ptr()
    result.view(-1).square().sum().backward()
    torch.testing.assert_close(value.grad, 2 * value, rtol=0, atol=0)
    assert events == ["exchange", "exchange"]


@pytest.mark.parametrize("sender_fails", [False, True])
@pytest.mark.parametrize("process_group_fails", [False, True])
def test_worker_destroy_preserves_reserve_until_teardown(monkeypatch, sender_fails, process_group_fails):
    from arctic_platform.common.deepspeed_worker import DeepSpeedWorker

    events = []
    reserve = object()
    cls = DeepSpeedWorker.__ray_metadata__.modified_class
    worker = object.__new__(cls)

    def sender_destroy():
        assert worker._gpu_mem_preallocation is reserve
        events.append("sender")
        if sender_fails:
            raise RuntimeError("sender failed")

    def process_group_destroy():
        assert worker._gpu_mem_preallocation is reserve
        events.append("process-groups")
        if process_group_fails:
            raise RuntimeError("process-group cleanup failed")

    monkeypatch.setattr(collectives.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(collectives.dist, "destroy_process_group", process_group_destroy)
    worker._weight_sender = SimpleNamespace(destroy=sender_destroy)
    worker.engine = object()
    worker._gpu_mem_preallocation = reserve
    if sender_fails or process_group_fails:
        expected = "process-group cleanup failed" if process_group_fails else "sender failed"
        with pytest.raises(RuntimeError, match=expected):
            worker.destroy()
    else:
        assert worker.destroy()
    assert events == ["sender", "process-groups"]
    assert worker._weight_sender is None
    assert worker.engine is None
    assert worker._gpu_mem_preallocation is None
