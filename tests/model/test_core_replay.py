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
"""A checkpointed block's recompute replays its core kernel from the forward, with unchanged gradients."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import set_checkpoint_early_stop

from arctic_platform.model.implementations.gpu.core_replay import install_core_replay
from arctic_platform.model.implementations.gpu.core_replay import keep_core

calls = {"forward": 0, "backward": 0}


class _Scan(torch.autograd.Function):
    """Stands in for a scan kernel: saves its input and an intermediate, returns (out, state)."""

    @staticmethod
    def forward(ctx, x, w):
        calls["forward"] += 1
        h = torch.cumsum(x * w, dim=0)
        ctx.save_for_backward(x, w, h)
        return torch.tanh(h), h[-1].detach()

    @staticmethod
    def backward(ctx, grad_out, _grad_state):
        calls["backward"] += 1
        x, w, h = ctx.saved_tensors
        grad_h = grad_out * (1 - torch.tanh(h) ** 2)
        grad_xw = torch.flip(torch.cumsum(torch.flip(grad_h, [0]), dim=0), [0])
        return grad_xw * w, (grad_xw * x).sum(0)


class _Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(4, 4)
        self.w = nn.Parameter(torch.randn(4))
        self.out = nn.Linear(4, 4)
        self._arl_keeps_core = True

    def forward(self, x):
        scanned, _state = keep_core(getattr(self, "_arl_core_replay", None), _Scan.apply, self.inp(x), self.w)
        return x + self.out(scanned)


def _grads(replay, checkpointed, steps=2):
    torch.manual_seed(0)
    block = _Mixer()
    if replay:
        assert install_core_replay(block) == 1
    x = torch.randn(6, 4, requires_grad=True)
    for _ in range(steps):  # a second step reuses the kept queues
        block.zero_grad()
        x.grad = None
        with set_checkpoint_early_stop(False):
            y = checkpoint(block, x, use_reentrant=False) if checkpointed else block(x)
        (y**2).sum().backward()
    return [x.grad] + [p.grad for p in block.parameters()]


def test_recompute_replays_the_core_and_keeps_the_gradients():
    calls.update(forward=0, backward=0)
    reference = _grads(False, checkpointed=False)
    assert calls == {"forward": 2, "backward": 2}

    calls.update(forward=0, backward=0)
    replayed = _grads(True, checkpointed=True)
    assert calls == {"forward": 2, "backward": 2}  # no forward from the recomputes
    for got, want in zip(replayed, reference):
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    calls.update(forward=0, backward=0)
    _grads(False, checkpointed=True)
    assert calls == {"forward": 4, "backward": 2}  # without replay each recompute scans again
