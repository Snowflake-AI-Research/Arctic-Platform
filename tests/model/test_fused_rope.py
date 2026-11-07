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
"""FlashAttention's fused rotary embedding (QuACK) against the PyTorch composition it replaces."""

import pytest
import torch

from arctic_platform.model.implementations.layers import attn
from arctic_platform.model.implementations.layers.attn import AttentionConfig
from arctic_platform.model.implementations.layers.attn import FlashAttention
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed


def test_cpu_keeps_the_pytorch_rotary():
    assert not attn._fused_rope_applies(torch.zeros(1, 4, 2, 128), torch.zeros(1, 4, 128))


@require_torch_gpu
def test_fused_rotary_matches_the_pytorch_projections(monkeypatch):
    pytest.importorskip("quack")
    set_seed(0)
    config = AttentionConfig(
        hidden_size=512,
        head_dim=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        is_causal=True,
        attention_bias=False,
        use_qk_norm=True,
        rms_norm_eps=1e-6,
    )
    module = FlashAttention(config, flash_attn_version=2).cuda().bfloat16()
    seq = 3000
    position_ids = torch.cat([torch.arange(1000), torch.arange(2000)]).cuda().float()  # two packed rows
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, 128, 2, device="cuda").float() / 128))
    freqs = position_ids[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)[None]  # HF's [batch, seq, head_dim] cos / sin, as RotaryEmbedding
    cos, sin = emb.cos().bfloat16(), emb.sin().bfloat16()
    hidden = torch.randn(1, seq, 512, device="cuda", dtype=torch.bfloat16)
    grads = [torch.randn(1, seq, h, 128, device="cuda", dtype=torch.bfloat16) for h in (8, 2, 2)]

    def run(fused):
        monkeypatch.setattr(attn, "_fused_rope_applies", lambda *args: fused)
        x = hidden.clone().requires_grad_(True)
        module.zero_grad(set_to_none=True)
        outs = module.attn_projections(x, (cos, sin))
        torch.autograd.backward(outs, grads)
        return [o.detach() for o in outs] + [x.grad, module.q_proj.weight.grad, module.k_proj.weight.grad]

    want, got = run(False), run(True)

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    for g, w in zip(got, want):
        assert g.shape == w.shape
        assert rel(g, w) < 1e-2
