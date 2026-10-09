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

"""Whether a determinism request reaches the attention kernel this model calls, and what happens when it cannot.

``debug.full_determinism`` is a promise that a repeat of a run reproduces its bits. Qwen3.5 calls a FlashAttention
varlen entry point directly instead of going through Hugging Face's attention integration, so the
``FLASH_ATTENTION_DETERMINISTIC`` variable that ``determinism_worker_env`` exports has no reader on this path
unless the model itself reads it. Two independent things then have to hold: the request has to arrive at the
kernel wherever the kernel can honour it, and a configuration whose head dimension the installed kernel refuses
has to fail while the model is still being built, rather than after a training run whose result cannot be
replayed.

The head dimensions below are properties of the installed kernel rather than choices: it accepts a deterministic
backward at 128 and refuses one above 192, and Qwen3.5 uses 256. Each case asks the kernel instead of asserting
that threshold, so a build that moves it changes what the tests measure rather than what they claim.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration]

DETERMINISM_ENV = "FLASH_ATTENTION_DETERMINISTIC"
SUPPORTED_HEAD_DIM = 128
REFUSED_HEAD_DIM = 256
NUM_HEADS = 8
# Long enough that the query gradient is accumulated across splits of the key/value loop, which is the
# accumulation the flag orders. A short call does not split, so its backward already replays exactly and the
# comparison in the last case would hold for a reason that has nothing to do with the flag.
REPLAY_SEQ_LEN = 4096
REPLAYS = 8


@pytest.fixture
def cuda_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the flash-attention kernel and the replay measurement are both GPU-only")
    return torch.device("cuda:0")


def _attention(head_dim: int):
    """One gated flash-attention module at ``head_dim``, built the way the decoder layer builds it."""
    from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedAttentionConfig,
    )
    from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedFlashAttention,
    )

    config = Qwen3_5MoeGatedAttentionConfig(
        hidden_size=NUM_HEADS * head_dim,
        head_dim=head_dim,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        rms_norm_eps=1e-6,
    )
    return Qwen3_5MoeGatedFlashAttention(config, flash_attn_version=3)


def _varlen_inputs(head_dim: int, seq_len: int, device):
    import torch

    generator = torch.Generator(device=device).manual_seed(4242 + head_dim + seq_len)
    shape = (seq_len, NUM_HEADS, head_dim)
    q, k, v = (
        torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    )
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    grad_out = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
    return q, k, v, cu_seqlens, seq_len, grad_out


def _replay_disagreement(module, head_dim: int, device) -> tuple[int, float]:
    """How many of ``REPLAYS`` backwards through ``module`` disagree with the first, and by how much."""
    import torch

    def once():
        q, k, v, cu_seqlens, max_seqlen, grad_out = _varlen_inputs(head_dim, REPLAY_SEQ_LEN, device)
        module._compute_attention(q, k, v, cu_seqlens, max_seqlen).backward(grad_out)
        return q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone()

    reference = once()
    differing = 0
    worst = 0.0
    for _ in range(REPLAYS - 1):
        candidate = once()
        delta = 0.0
        for left, right in zip(reference, candidate):
            scale = torch.maximum(left.float().abs(), right.float().abs()).clamp_min(1e-12)
            delta = max(delta, float(((left.float() - right.float()).abs() / scale).max()))
        if delta != 0.0:
            differing += 1
        worst = max(worst, delta)
    return differing, worst


def test_a_determinism_request_reaches_the_flash_attention_kernel(monkeypatch, cuda_device):
    """Where the kernel can honour the request, the model has to make it."""
    monkeypatch.setenv(DETERMINISM_ENV, "1")
    module = _attention(SUPPORTED_HEAD_DIM)

    captured: dict = {}

    def record(*args, **kwargs):
        captured.update(kwargs)
        return args[0]

    module._flash_attn_call = record
    module._compute_attention(*_varlen_inputs(SUPPORTED_HEAD_DIM, 16, cuda_device)[:5])

    assert captured.get("deterministic") is True, (
        f"the model called the kernel with {sorted(captured)}, so a determinism request stops at the process "
        f"boundary and the kernel takes its own nondeterministic default at head_dim={SUPPORTED_HEAD_DIM}, "
        "where it would have honoured the request"
    )


def test_requesting_determinism_where_the_kernel_refuses_it_fails_while_the_model_is_built(monkeypatch, cuda_device):
    """A run that cannot be replayed must not start, and the refusal has to say what is irreconcilable."""
    monkeypatch.setenv(DETERMINISM_ENV, "1")

    with pytest.raises(RuntimeError) as raised:
        _attention(REFUSED_HEAD_DIM)

    message = str(raised.value)
    for expected in (
        "full_determinism",
        f"head_dim={REFUSED_HEAD_DIM}",
        "flash_attention_3",
        "Deterministic backward not supported",
    ):
        assert expected in message, f"the refusal does not name {expected!r}, so it reads as: {message}"


def test_the_same_configuration_builds_and_stays_off_the_flag_when_determinism_is_not_requested(
    monkeypatch, cuda_device
):
    """The refusal is conditional on the request. Without it this configuration is the product default."""
    monkeypatch.delenv(DETERMINISM_ENV, raising=False)
    module = _attention(REFUSED_HEAD_DIM)

    captured: dict = {}

    def record(*args, **kwargs):
        captured.update(kwargs)
        return args[0]

    module._flash_attn_call = record
    module._compute_attention(*_varlen_inputs(REFUSED_HEAD_DIM, 16, cuda_device)[:5])

    assert captured.get("deterministic") is not True, (
        "a job that did not ask for determinism was given the deterministic backward, which is a different "
        "kernel and a different cost than the one it configured"
    )


def test_the_request_the_model_passes_removes_the_replay_disagreement(monkeypatch, cuda_device):
    """The quantity the flag exists for, measured on the model's own call path rather than on the kernel's."""
    monkeypatch.delenv(DETERMINISM_ENV, raising=False)
    free_count, free_worst = _replay_disagreement(_attention(SUPPORTED_HEAD_DIM), SUPPORTED_HEAD_DIM, cuda_device)

    monkeypatch.setenv(DETERMINISM_ENV, "1")
    pinned_count, pinned_worst = _replay_disagreement(_attention(SUPPORTED_HEAD_DIM), SUPPORTED_HEAD_DIM, cuda_device)

    # Calibration: without this the case would pass on a shape whose backward never disagreed with itself, and
    # would then say nothing about the flag.
    assert free_count > 0, (
        f"{REPLAYS} unpinned replays at seq_len={REPLAY_SEQ_LEN}, head_dim={SUPPORTED_HEAD_DIM} all agreed, so "
        "this shape does not exercise the accumulation the flag orders and the comparison below is vacuous"
    )
    assert (pinned_count, pinned_worst) == (0, 0.0), (
        f"{pinned_count} of {REPLAYS - 1} pinned replays disagree, worst relative {pinned_worst:.3e}, against "
        f"{free_count} of {REPLAYS - 1} and {free_worst:.3e} unpinned: the request reached the kernel or it did "
        "not, and a pinned arm that still moves means it did not"
    )
