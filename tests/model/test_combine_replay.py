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
"""A checkpointed block's recompute replays its combine instead of communicating again, with unchanged gradients."""

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import set_checkpoint_early_stop

from arctic_platform.model.implementations.moe.distributed.combine_replay import CombineReplay

calls = {"forward": 0, "backward": 0}


class _Combine(torch.autograd.Function):
    """Stands in for the EP combine: the communication that must not run twice."""

    @staticmethod
    def forward(ctx, x):
        calls["forward"] += 1
        return x * 2

    @staticmethod
    def backward(ctx, grad):
        calls["backward"] += 1
        return grad * 2


class _Replayed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, combined):
        return combined.clone()

    @staticmethod
    def backward(ctx, grad):
        raise AssertionError("the recompute's graph is discarded")


class _Block(nn.Module):
    def __init__(self, replay):
        super().__init__()
        self.up = nn.Linear(4, 4)
        self.down = nn.Linear(4, 4)
        self.replay = replay

    def forward(self, x):
        if self.replay is not None:
            self.replay.start()
        outs = []
        for chunk in self.up(x).chunk(2):  # two combines per forward, replayed in order
            if self.replay is None:
                outs.append(_Combine.apply(chunk))
            else:
                outs.append(self.replay.run(lambda: _Combine.apply(chunk), lambda c: _Replayed.apply(chunk, c)))
        return torch.tanh(self.down(torch.cat(outs)))  # saves tensors after the combine


def _grads(replay, checkpointed, early_stop=False):
    torch.manual_seed(0)
    block = _Block(replay)
    x = torch.randn(6, 4, requires_grad=True)
    for _ in range(2):  # a second step reuses the replay
        block.zero_grad()
        x.grad = None
        with set_checkpoint_early_stop(early_stop):
            y = checkpoint(block, x, use_reentrant=False) if checkpointed else block(x)
        y.sum().backward()
    return [x.grad] + [p.grad for p in block.parameters()]


# GLM's blocks keep torch's early stop; the tensors saved after the combine still make the recompute reach it.
@pytest.mark.parametrize("early_stop", [False, True])
def test_recompute_replays_the_combine_and_keeps_the_gradients(early_stop):
    reference = _grads(None, checkpointed=False)
    calls.update(forward=0, backward=0)
    replayed = _grads(CombineReplay(), checkpointed=True, early_stop=early_stop)
    assert calls == {"forward": 4, "backward": 4}  # 2 steps x 2 chunks, none from the recomputes
    for got, want in zip(replayed, reference):
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    calls.update(forward=0, backward=0)
    _grads(None, checkpointed=True, early_stop=early_stop)
    assert calls == {"forward": 8, "backward": 4}  # without replay each recompute combines again


def test_an_in_place_write_to_the_kept_output_is_refused():
    import torch.utils.checkpoint as ckpt

    replay = CombineReplay()
    x = torch.randn(4, requires_grad=True)
    replay.start()
    out = replay.run(lambda: _Combine.apply(x), lambda c: _Replayed.apply(x, c))
    with torch.no_grad():
        out.add_(1)  # bumps the version the replay recorded
    ckpt._recompute_thread_local.depth = 1  # as inside a checkpoint recompute
    try:
        with pytest.raises(RuntimeError, match="modified in place"):
            replay.run(lambda: _Combine.apply(x), lambda c: _Replayed.apply(x, c))
    finally:
        ckpt._recompute_thread_local.depth = 0
