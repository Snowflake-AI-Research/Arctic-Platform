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

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import set_checkpoint_early_stop

from arctic_platform.model.implementations.gpu.core_replay import CoreReplay
from arctic_platform.model.implementations.gpu.core_replay import install_core_replay
from arctic_platform.model.implementations.gpu.core_replay import keep_core
from arctic_platform.testing_utils import require_torch_gpu

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


def test_two_forwards_before_the_first_backward_replay_their_own_cores():
    torch.manual_seed(0)
    block = _Mixer()
    install_core_replay(block)
    reference = _Mixer()
    reference.load_state_dict(block.state_dict())
    x1 = torch.randn(6, 4, requires_grad=True)
    x2 = torch.randn(6, 4, requires_grad=True)
    calls.update(forward=0, backward=0)
    with set_checkpoint_early_stop(False):
        y1 = checkpoint(block, x1, use_reentrant=False)
        y2 = checkpoint(block, x2, use_reentrant=False)
    (y2**2).sum().backward()
    (y1**2).sum().backward()
    assert calls["forward"] == 2

    x1_ref = x1.detach().requires_grad_(True)
    x2_ref = x2.detach().requires_grad_(True)
    ((reference(x1_ref) ** 2).sum() + (reference(x2_ref) ** 2).sum()).backward()
    torch.testing.assert_close(x1.grad, x1_ref.grad, rtol=0, atol=0)
    torch.testing.assert_close(x2.grad, x2_ref.grad, rtol=0, atol=0)
    for got, want in zip(block.parameters(), reference.parameters()):
        torch.testing.assert_close(got.grad, want.grad, rtol=1e-6, atol=1e-6)


def test_attention_only_core_replay_leaves_the_scans_to_the_recompute():
    block = _Mixer()
    assert install_core_replay(block, scans=False) == 0
    assert getattr(block, "_arl_core_replay", None) is None
    calls.update(forward=0, backward=0)
    x = torch.randn(6, 4, requires_grad=True)
    with set_checkpoint_early_stop(False):
        (checkpoint(block, x, use_reentrant=False) ** 2).sum().backward()
    assert calls == {"forward": 2, "backward": 1}


def test_outside_a_checkpoint_the_core_runs_plainly():
    torch.manual_seed(0)
    block = _Mixer()
    install_core_replay(block)
    calls.update(forward=0, backward=0)
    block(torch.randn(6, 4, requires_grad=True)).sum().backward()
    with torch.no_grad():
        block(torch.randn(6, 4))
    assert calls == {"forward": 2, "backward": 1}
    assert not block._arl_core_replay._by_forward  # nothing kept outside a checkpoint


def test_an_in_place_write_to_a_kept_output_is_refused():
    torch.manual_seed(0)

    class _Writes(_Mixer):
        def forward(self, x):
            scanned, _ = keep_core(self._arl_core_replay, _Scan.apply, self.inp(x), self.w)
            with torch.no_grad():
                scanned.add_(0)  # bumps the version the replay recorded
            return x + self.out(scanned)

    block = _Writes()
    install_core_replay(block)
    x = torch.randn(6, 4, requires_grad=True)
    with set_checkpoint_early_stop(False):
        y = checkpoint(block, x, use_reentrant=False)
    with pytest.raises(RuntimeError, match="modified in place"):
        y.sum().backward()


def test_a_recompute_without_a_kept_output_raises():
    replay = CoreReplay()
    import torch.utils.checkpoint as ckpt

    class _Probe(nn.Module):
        def forward(self, x):
            return keep_core(replay, torch.tanh, x) ** 2  # pow saves its input: the backward recomputes

    x = torch.randn(4, requires_grad=True)
    with set_checkpoint_early_stop(False):
        y = checkpoint(_Probe(), x, use_reentrant=False)
    for queue in list(replay._by_forward.values()):
        queue.clear()  # as if the kept output were lost
    with pytest.raises(RuntimeError, match="found no kept core output"):
        y.sum().backward()
    assert getattr(ckpt._recompute_thread_local, "depth", 0) == 0


def _varlen_inputs(device, heads=4, kv_heads=2, dim=128):
    torch.manual_seed(0)
    lengths = [300, 1, 723]
    total = sum(lengths)
    cu = torch.tensor([0, 300, 301, 1024], dtype=torch.int32, device=device)
    q = torch.randn(total, heads, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total, kv_heads, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(total, kv_heads, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    return q, k, v, cu, max(lengths)


@require_torch_gpu
@pytest.mark.parametrize("window", [(-1, -1), (127, 0)])
def test_kept_flash_attention_3_matches_flash_attention_3(window):
    fa3 = pytest.importorskip("flash_attn_interface")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FlashAttention-3 needs Hopper")

    class _Attn(nn.Module):
        _flash_attn_version = 3

        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(128, 128, bias=False)
            self._flash_attn_call = fa3.flash_attn_varlen_func

        def forward(self, q, k, v, cu, max_len):
            out = self._flash_attn_call(
                self.qkv(q), self.qkv(k), self.qkv(v), cu, cu, max_len, max_len, causal=True, window_size=window
            )
            return out.float().square().sum()

    q, k, v, cu, max_len = _varlen_inputs("cuda")
    reference = _Attn().cuda().bfloat16()
    kept = _Attn().cuda().bfloat16()
    kept.load_state_dict(reference.state_dict())
    assert install_core_replay(kept) == 1

    def grads(module, wrap):
        tensors = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
        with set_checkpoint_early_stop(False):
            loss = (
                checkpoint(module, *tensors, cu, max_len, use_reentrant=False)
                if wrap
                else module(*tensors, cu, max_len)
            )
        loss.backward()
        return [loss.detach()] + [t.grad for t in tensors] + [module.qkv.weight.grad]

    forwards = []
    original = fa3._flash_attn_forward

    def counted(*args, **kwargs):
        forwards.append(1)
        return original(*args, **kwargs)

    fa3._flash_attn_forward = counted
    try:
        want = grads(reference, wrap=False)
        assert len(forwards) == 1
        got = grads(kept, wrap=True)
        assert len(forwards) == 2  # the recompute kept the forward's output
    finally:
        fa3._flash_attn_forward = original
    torch.testing.assert_close(got[0], want[0], rtol=0, atol=0)
    for g, w in zip(got[1:], want[1:]):
        # The same kernels on the same inputs; FlashAttention-3's backward accumulates dQ with atomics.
        torch.testing.assert_close(g, w, rtol=1e-2, atol=1e-2)
