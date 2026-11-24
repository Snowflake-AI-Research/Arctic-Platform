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

"""``patches.fused_qk_norm_rope``: Qwen3's attention with the fused q_norm / k_norm + rotary kernel trains like the
stock one, on FlashAttention 3 in BF16 as the catalog trains."""

from __future__ import annotations

import copy
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel runs on CUDA")

transformers = pytest.importorskip("transformers")
from transformers.models.qwen3 import modeling_qwen3  # noqa: E402

from arctic_platform.model import LoaderContext  # noqa: E402
from arctic_platform.model.patches.fused_qk_norm_rope import apply_fused_qk_norm_rope  # noqa: E402


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


def test_patched_attention_trains_like_stock():
    pytest.importorskip("flash_attn_interface")
    stock = _model("flash_attention_3")
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
