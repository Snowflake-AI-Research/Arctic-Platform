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

"""``patches.fused_lora_linear``: PEFT LoRA linears add their update inside the lora_B GEMM, and a two-layer Qwen3
with LoRA on every projection trains to PEFT's loss and gradients in BF16 on CUDA (the fused update rounds once)."""

from __future__ import annotations

import copy
import types

import pytest
import torch

from arctic_platform.model import LoaderContext
from arctic_platform.model.patches.fused_lora_linear import apply_fused_lora_linear
from arctic_platform.model.patches.fused_lora_linear import fused_lora_linear_forward

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

CTX = LoaderContext(spec=types.SimpleNamespace(patches=types.SimpleNamespace(fused_lora_linear=True)))
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _lora_model(device, dtype, **lora_kwargs):
    config = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config)
    lora = peft.LoraConfig(r=8, lora_alpha=16, target_modules=TARGETS, **lora_kwargs)
    model = peft.get_peft_model(model, lora, autocast_adapter_dtype=False).to(device=device, dtype=dtype).train()
    with torch.no_grad():  # PEFT starts lora_B at zero, which would hide the update
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0.0, 0.05)
    return model


def _step(model, input_ids):
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    grads = {name: p.grad.detach().float() for name, p in model.named_parameters() if p.grad is not None}
    return loss.detach().float(), grads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the catalog trains LoRA in BF16 on CUDA")
def test_fused_lora_linear_trains_like_peft():
    device, tol = "cuda", 3e-2
    stock = _lora_model(device, torch.bfloat16)
    fused = copy.deepcopy(stock)
    apply_fused_lora_linear(fused, CTX)
    assert all(
        getattr(m.forward, "__func__", None) is fused_lora_linear_forward
        for m in fused.modules()
        if type(m) is peft.tuners.lora.layer.Linear
    )
    torch.manual_seed(1)
    input_ids = torch.randint(0, 128, (2, 48), device=device)
    stock_loss, stock_grads = _step(stock, input_ids)
    fused_loss, fused_grads = _step(fused, input_ids)
    torch.testing.assert_close(fused_loss, stock_loss, rtol=tol, atol=tol)
    assert stock_grads.keys() == fused_grads.keys() and any("lora_A" in n for n in stock_grads)
    for name, grad in stock_grads.items():
        scale = grad.abs().max().item()
        torch.testing.assert_close(fused_grads[name], grad, rtol=tol, atol=tol * scale, msg=name)
