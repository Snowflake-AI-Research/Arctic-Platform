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
"""The sequence-parallel K/V all-gather sums the ranks' gradients in fp32 in its backward and rounds once.

torch's ``dist_nn.all_gather`` backward reduce-scatters in the gradient's dtype; NCCL's bf16 ring then rounds every
partial sum, so a key's gradient picks up one rounding per contributing rank.
"""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.distributed as dist


def _nccl_like_group(monkeypatch, world_size, rank):
    """An in-process stand-in for an NCCL group whose ranks all hold this rank's tensors: ``all_gather`` copies, and
    ``reduce_scatter`` adds the ranks' chunks around a ring in the inputs' own dtype, rounding after every hop."""
    sent = []

    def all_gather(parts, tensor, group=None):
        for part in parts:
            part.copy_(tensor)

    def reduce_scatter(output, inputs, op=None, group=None):
        sent.append(inputs[rank].dtype)
        total = inputs[rank].clone()
        for _ in range(world_size - 1):
            total += inputs[rank]
        output.copy_(total)

    for name, value in (("get_world_size", lambda group=None: world_size), ("get_rank", lambda group=None: rank),
                        ("get_backend", lambda group=None: dist.Backend.NCCL), ("all_gather", all_gather),
                        ("reduce_scatter", reduce_scatter)):  # fmt: skip
        monkeypatch.setattr(dist, name, value)
    return sent


@pytest.mark.parametrize(
    "module",
    [
        "arctic_platform.model.implementations.glm53.context_parallel",
        "arctic_platform.model.implementations.qwen38.qsa_flex",
    ],
)
def test_sequence_gather_backward_sums_ranks_in_fp32(monkeypatch, module):
    gather_sequence = importlib.import_module(module).gather_sequence
    world_size, rank, local_length = 8, 3, 4
    sent = _nccl_like_group(monkeypatch, world_size, rank)
    torch.manual_seed(0)
    local = torch.randn(1, local_length, 2, 16).to(torch.bfloat16).requires_grad_(True)
    gathered = gather_sequence(local, "sp-group")
    grad = torch.randn(gathered.shape).to(torch.bfloat16)
    gathered.backward(grad)

    torch.testing.assert_close(gathered, local.detach().repeat(1, world_size, 1, 1), rtol=0, atol=0)
    # Every rank holds the same gradient, so the exact sum is world_size x this rank's chunk (exact in bf16).
    expected = world_size * grad[:, rank * local_length : (rank + 1) * local_length]
    assert sent == [torch.float32]
    assert local.grad.dtype == torch.bfloat16
    torch.testing.assert_close(local.grad, expected, rtol=0, atol=0)
