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

"""The fused Qwen3 q_norm / k_norm + rotary kernel matches transformers' modules, and the patched attention trains
like the stock one.

The forward is compared with transformers' ``Qwen3RMSNorm`` + ``apply_rotary_pos_emb`` in BF16 (one BF16 rounding apart:
the kernel rounds the rotation once) and, bitwise, with the once-rounded rotation of transformers' rounded norm output;
the gradients with float64 autograd through the same math.
"""

from __future__ import annotations

import copy
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel runs on CUDA")

transformers = pytest.importorskip("transformers")
from transformers.models.qwen3 import modeling_qwen3  # noqa: E402

from arctic_platform.model import LoaderContext  # noqa: E402
from arctic_platform.model.implementations.kernels.fused_qk_norm_rope import fused_norm_rope  # noqa: E402
from arctic_platform.model.patches.fused_qk_norm_rope import apply_fused_qk_norm_rope  # noqa: E402

EPS = 1e-6


def _rotary(batch, seq_len, head_dim, *, per_batch):
    config = transformers.Qwen3Config(head_dim=head_dim, hidden_size=head_dim * 4, num_attention_heads=4)
    rotary = modeling_qwen3.Qwen3RotaryEmbedding(config).cuda()
    positions = torch.arange(seq_len, device="cuda").unsqueeze(0)
    if per_batch:
        positions = torch.cat([positions + 17 * b for b in range(batch)])
    probe = torch.empty(1, device="cuda", dtype=torch.bfloat16)
    return rotary(probe, positions)


def _reference_bf16(x, weight, cos, sin):
    norm = modeling_qwen3.Qwen3RMSNorm(x.shape[-1], eps=EPS).cuda().to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(weight)
    y = norm(x).transpose(1, 2)
    q, _ = modeling_qwen3.apply_rotary_pos_emb(y, y, cos, sin)
    return q.transpose(1, 2)


def _norm_bf16(x, weight):
    norm = modeling_qwen3.Qwen3RMSNorm(x.shape[-1], eps=EPS).cuda().to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(weight)
        return norm(x)


def _rotate_once(y, cos, sin):
    y = y.double().transpose(1, 2)
    q, _ = modeling_qwen3.apply_rotary_pos_emb(y, y, cos.double(), sin.double())
    return q.transpose(1, 2).bfloat16()


def _reference_f64(x, weight, cos, sin):
    x = x.double()
    n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    y = (weight.double() * n).transpose(1, 2)
    q, _ = modeling_qwen3.apply_rotary_pos_emb(y, y, cos.double(), sin.double())
    return q.transpose(1, 2)


@pytest.mark.parametrize("batch,seq_len,heads,per_batch", [(1, 513, 16, False), (2, 300, 8, True), (1, 64, 2, False)])
def test_fused_norm_rope_matches_transformers(batch, seq_len, heads, per_batch):
    head_dim = 128
    torch.manual_seed(0)
    x = (torch.randn(batch, seq_len, heads, head_dim, device="cuda") * 3).bfloat16()
    weight = (1 + 0.3 * torch.randn(head_dim, device="cuda")).bfloat16()
    cos, sin = _rotary(batch, seq_len, head_dim, per_batch=per_batch)
    grad = torch.randn(batch, seq_len, heads, head_dim, device="cuda").bfloat16()

    fx = x.clone().requires_grad_(True)
    fw = weight.clone().requires_grad_(True)
    out = fused_norm_rope(fx, fw, cos, sin, EPS)
    out.backward(grad)

    expected = _reference_bf16(x, weight, cos, sin)
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    # transformers rounds each product and the sum: up to two BF16 roundings (2**-7 relative each) from the kernel.
    torch.testing.assert_close(out, expected, rtol=3.2e-2, atol=3.2e-2)
    # The spec: transformers' rounded norm output, rotated exactly and rounded once.
    once = _rotate_once(_norm_bf16(x, weight), cos, sin)
    exact = (out == once).float().mean().item()
    assert exact > 0.999, f"only {exact:.5f} of the outputs are the once-rounded rotation of the rounded norm"

    rx = x.double().requires_grad_(True)
    rw = weight.double().requires_grad_(True)
    _reference_f64(rx, rw, cos, sin).backward(grad.double())
    torch.testing.assert_close(fx.grad.double(), rx.grad, rtol=2e-2, atol=2e-2 * rx.grad.abs().max().item() / 8)
    torch.testing.assert_close(fw.grad.double(), rw.grad, rtol=2e-2, atol=1e-2 * rw.grad.abs().max().item())


def test_frozen_weight_gets_no_gradient():
    torch.manual_seed(1)
    x = torch.randn(1, 128, 4, 128, device="cuda").bfloat16().requires_grad_(True)
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    cos, sin = _rotary(1, 128, 128, per_batch=False)
    fused_norm_rope(x, weight, cos, sin, EPS).sum().backward()
    assert x.grad is not None and weight.grad is None


def _model(attn_implementation):
    config = transformers.Qwen3Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=1024,
    )
    config._attn_implementation = attn_implementation
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(config).cuda().to(torch.bfloat16).train()


def _attention_implementations():
    yield "sdpa"
    try:
        import flash_attn_interface  # noqa: F401
    except ImportError:
        return
    yield "flash_attention_3"


@pytest.mark.parametrize("attn_implementation", list(_attention_implementations()))
def test_patched_attention_trains_like_stock(attn_implementation):
    stock = _model(attn_implementation)
    with torch.no_grad():
        for module in stock.modules():
            if isinstance(module, modeling_qwen3.Qwen3RMSNorm):
                module.weight.normal_(1.0, 0.2)
    fused = copy.deepcopy(stock)
    ctx = LoaderContext(spec=types.SimpleNamespace(patches=types.SimpleNamespace(fused_qk_norm_rope=True)))
    apply_fused_qk_norm_rope(fused, ctx)
    torch.manual_seed(1)
    input_ids = torch.randint(0, 256, (1, 512), device="cuda")

    losses, grads = [], []
    for model in (stock, fused):
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        losses.append(loss.detach().float())
        grads.append({name: p.grad.float() for name, p in model.named_parameters()})
    torch.testing.assert_close(losses[1], losses[0], rtol=2e-3, atol=2e-3)
    for name, grad in grads[0].items():
        scale = grad.abs().max().item()
        torch.testing.assert_close(grads[1][name], grad, rtol=5e-2, atol=3e-2 * scale, msg=name)


def test_patch_refuses_a_model_without_qwen3_attention():
    ctx = LoaderContext(spec=types.SimpleNamespace(patches=types.SimpleNamespace(fused_qk_norm_rope=True)))
    with pytest.raises(ValueError, match="no transformers Qwen3Attention"):
        apply_fused_qk_norm_rope(torch.nn.Linear(2, 2), ctx)
