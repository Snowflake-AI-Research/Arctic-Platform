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

"""Unit test: token-tiled dense FFN is invariant to its input's memory layout.

``apply_tiled_mlp`` shards whatever tensor the wrapped FFN was called with along the token dimension, and
DeepSpeed's ``TiledMLP.backward`` flattens that saved tensor with ``x.view(-1, hidden_size)`` before it
narrows each shard's gradient slice. ``view`` is only legal when the flattened dimensions occupy one
contiguous subspace, so a caller that hands the FFN a transposed or permuted view -- the same values, a
different stride order -- gets a ``RuntimeError`` out of the *backward* pass, after a forward that appeared
to succeed. Tiling is a pure memory transform: for the same values the output and every gradient must match
the un-tiled FFN regardless of how the input tensor is laid out.

Runs on CPU in float64 with tiny tensors. Bounds: with identical values in a contiguous layout the tiled and
un-tiled paths differ only by summation order, measured at 1.136868e-13 worst-case over output, input
gradient and parameter gradients for these shapes; the asserted bound is 1.0e-10, three exponent decades
above that floor.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from arctic_platform.model.implementations.gpu.tiled_mlp import apply_tiled_mlp

# Measured floor for these shapes in float64 is 1.136868e-13 (tiled vs un-tiled, contiguous input, summation
# order only). Held three decades above it so a real layout defect cannot hide under numerical slack.
RTOL = 1e-10
ATOL = 1e-10


class SwiGLUFeedForward(nn.Module):
    """Same shape as the Qwen3.5 dense shared-expert FFN: ``w2(silu(w1 x) * w3 x)``."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.w1 = nn.Linear(hidden, intermediate, bias=False)
        self.w3 = nn.Linear(hidden, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


def _mlp_forward(module: SwiGLUFeedForward, hidden_states: torch.Tensor) -> torch.Tensor:
    return module.w2(F.silu(module.w1(hidden_states)) * module.w3(hidden_states))


def _compute_params(module: SwiGLUFeedForward) -> list[torch.Tensor]:
    return [module.w1.weight, module.w2.weight, module.w3.weight]


def _forward_backward(model: SwiGLUFeedForward, x: torch.Tensor, scale: int = 1):
    output = model(x)
    # A non-constant incoming gradient, so a shard that received the wrong slice cannot cancel out.
    incoming = torch.arange(1, output.numel() + 1, dtype=output.dtype).reshape(output.shape) * scale
    (output * incoming).sum().backward()
    param_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    return output.detach().clone(), x.grad.detach().clone(), param_grads


def _patch_tiled(reference: SwiGLUFeedForward, token_chunk_size: int) -> SwiGLUFeedForward:
    tiled = copy.deepcopy(reference)
    patched = apply_tiled_mlp(
        tiled,
        is_target=lambda module: isinstance(module, SwiGLUFeedForward),
        mlp_forward=_mlp_forward,
        compute_params=_compute_params,
        token_chunk_size=token_chunk_size,
    )
    assert patched == 1
    return tiled


def _assert_matches_untiled(reference: SwiGLUFeedForward, tiled: SwiGLUFeedForward, base: torch.Tensor):
    x_ref = base.clone().requires_grad_(True)
    x_tiled = base.clone().requires_grad_(True)

    out_ref, gin_ref, gp_ref = _forward_backward(reference, x_ref)
    out_tiled, gin_tiled, gp_tiled = _forward_backward(tiled, x_tiled)

    torch.testing.assert_close(out_tiled, out_ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(gin_tiled, gin_ref, rtol=RTOL, atol=ATOL)
    assert gp_tiled.keys() == gp_ref.keys()
    for name in gp_ref:
        torch.testing.assert_close(gp_tiled[name], gp_ref[name], rtol=RTOL, atol=ATOL, msg=f"grad {name}")


def test_tiled_matches_untiled_for_contiguous_batched_hidden_states():
    """Control: the same shape in a contiguous layout, so a failure below is about layout and nothing else."""
    torch.manual_seed(2)
    reference = SwiGLUFeedForward(16, 40).double()
    tiled = _patch_tiled(reference, token_chunk_size=3)

    base = torch.randn(2, 7, 16, dtype=torch.float64)
    assert base.is_contiguous()
    _assert_matches_untiled(reference, tiled, base)


def test_tiled_matches_untiled_for_transposed_batched_hidden_states():
    """A ``[batch, tokens, hidden]`` view whose batch stride is smaller than its token stride.

    This is what a caller produces by building activations token-major and transposing into batch-major.
    ``TiledMLP.backward`` flattens batch and tokens with ``view``, which cannot express that stride order.
    """
    torch.manual_seed(3)
    reference = SwiGLUFeedForward(16, 40).double()
    tiled = _patch_tiled(reference, token_chunk_size=3)

    base = torch.randn(7, 2, 16, dtype=torch.float64).transpose(0, 1)
    assert tuple(base.shape) == (2, 7, 16)
    assert not base.is_contiguous()
    _assert_matches_untiled(reference, tiled, base)


def test_tiled_matches_untiled_for_channel_sliced_packed_tokens():
    """A ``[tokens, hidden]`` slice out of a wider buffer -- the MoE-expert input layout."""
    torch.manual_seed(4)
    reference = SwiGLUFeedForward(16, 40).double()
    tiled = _patch_tiled(reference, token_chunk_size=3)

    base = torch.randn(7, 32, dtype=torch.float64)[:, :16]
    assert tuple(base.shape) == (7, 16)
    assert not base.is_contiguous()
    _assert_matches_untiled(reference, tiled, base)


def test_tiled_matches_untiled_for_channel_sliced_single_row_call():
    """A ``[1, tokens, hidden]`` slice out of a wider buffer -- the packed sequence-parallel call shape."""
    torch.manual_seed(5)
    reference = SwiGLUFeedForward(16, 40).double()
    tiled = _patch_tiled(reference, token_chunk_size=3)

    base = torch.randn(1, 7, 32, dtype=torch.float64)[:, :, :16]
    assert tuple(base.shape) == (1, 7, 16)
    assert not base.is_contiguous()
    _assert_matches_untiled(reference, tiled, base)


def test_tiled_gradients_accumulate_across_ragged_token_counts():
    """Two packed calls of different widths accumulate into one parameter gradient.

    ``ceil(tokens / token_chunk_size)`` gives the two calls different shard counts, and the second call's
    final shard is the one that flips ``ds_grad_is_ready`` back on. A shard-count-dependent reset would
    leave the accumulated parameter gradient short by the first call's contribution.
    """
    torch.manual_seed(11)
    reference = SwiGLUFeedForward(8, 21).double()
    tiled = _patch_tiled(reference, token_chunk_size=64)

    bases = [
        torch.randn(257, 8, dtype=torch.float64),
        torch.randn(385, 8, dtype=torch.float64),
    ]
    ref_inputs = [base.clone().requires_grad_(True) for base in bases]
    tiled_inputs = [base.clone().requires_grad_(True) for base in bases]

    for scale, (x_ref, x_tiled) in enumerate(zip(ref_inputs, tiled_inputs), start=1):
        out_ref, _, _ = _forward_backward(reference, x_ref, scale=scale)
        out_tiled, _, _ = _forward_backward(tiled, x_tiled, scale=scale)
        torch.testing.assert_close(out_tiled, out_ref, rtol=RTOL, atol=ATOL)

    for x_tiled, x_ref in zip(tiled_inputs, ref_inputs):
        torch.testing.assert_close(x_tiled.grad, x_ref.grad, rtol=RTOL, atol=ATOL)
    tiled_params = dict(tiled.named_parameters())
    for name, ref_param in reference.named_parameters():
        torch.testing.assert_close(tiled_params[name].grad, ref_param.grad, rtol=RTOL, atol=ATOL, msg=f"grad {name}")


def test_zero2_tiling_does_not_synchronize_the_global_shard_count(monkeypatch):
    """Ordinary parameters use ZeRO-2 semantics and need no per-layer scalar collective or host wait."""
    import torch.distributed as dist

    reference = SwiGLUFeedForward(8, 21).double()
    tiled = _patch_tiled(reference, token_chunk_size=4)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def unexpected_all_reduce(*_args, **_kwargs):
        raise AssertionError("ZeRO-2 tiled MLP synchronized its shard count")

    monkeypatch.setattr(dist, "all_reduce", unexpected_all_reduce)
    _assert_matches_untiled(reference, tiled, torch.randn(9, 8, dtype=torch.float64))


def test_ragged_rank_uses_the_global_maximum_shard_schedule(monkeypatch):
    """A short rank pads into the two-tile schedule selected by its longer ZeRO-3 peer.

    Each tile can trigger a ZeRO-3 parameter all-gather. If one rank takes the one-call fast path while a peer
    executes two tiles, their collective schedules diverge and can deadlock. The mocked all-reduce reports a
    peer that needs two shards and the projection hook counts this rank's matching schedule.
    """
    import torch.distributed as dist

    torch.manual_seed(12)
    reference = SwiGLUFeedForward(8, 21).double()
    tiled = _patch_tiled(reference, token_chunk_size=4)
    for index, parameter in enumerate(tiled.parameters()):
        parameter.ds_id = index
    calls = 0

    def projection_call(*_args):
        nonlocal calls
        calls += 1

    def all_reduce_max(tensor, op=None):
        assert op == dist.ReduceOp.MAX
        tensor.fill_(2)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", all_reduce_max)
    handle = tiled.w1.register_forward_hook(projection_call)
    base = torch.randn(3, 8, dtype=torch.float64)
    try:
        _assert_matches_untiled(reference, tiled, base)
    finally:
        handle.remove()

    # Forward and backward each recompute both padded tiles. The schedule, not the amount of real local data,
    # is synchronized with the peer.
    assert calls == 4


def test_short_rank_pads_to_an_exact_multiple_of_the_global_shard_count(monkeypatch):
    """Five rows under a four-shard peer schedule must execute four tiles, not torch.chunk's three."""
    import torch.distributed as dist

    torch.manual_seed(13)
    reference = SwiGLUFeedForward(8, 21).double()
    tiled = _patch_tiled(reference, token_chunk_size=4)
    for index, parameter in enumerate(tiled.parameters()):
        parameter.ds_id = index
    calls = 0

    def projection_call(*_args):
        nonlocal calls
        calls += 1

    def all_reduce_max(tensor, op=None):
        assert op == dist.ReduceOp.MAX
        tensor.fill_(4)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", all_reduce_max)
    handle = tiled.w1.register_forward_hook(projection_call)
    base = torch.randn(5, 8, dtype=torch.float64)
    try:
        _assert_matches_untiled(reference, tiled, base)
    finally:
        handle.remove()

    assert calls == 8


def test_ragged_multi_rank_schedule_keeps_zero3_collectives_in_lockstep():
    """Two ranks with different token counts complete the same simulated ZeRO-3 all-gather schedule."""
    import subprocess
    import sys
    from pathlib import Path

    driver = Path(__file__).with_name("ragged_zero3_schedule_driver.py")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(driver),
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
