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
"""All-linear LoRA on DeepSeek-V4.1: every adapter of the config reaches the forward and gets a gradient."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

ATTENTION_AND_DENSE = {
    "peft_type": "LORA",
    "r": 4,
    "lora_alpha": 4,
    "target_modules": ["q_a_proj", "q_b_proj", "kv_proj", "o_b_proj", "wkv", "wgate"],
    "target_parameters": ["shared_expert.w1", "shared_expert.w2", "shared_expert.w3", "o_a_proj.weight"],
}


def _model():
    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41ForCausalLM
    from tests.model.test_deepseek_v41_streaming import _config

    torch.manual_seed(0)
    config = _config()
    config.hidden_size, config.moe_intermediate_size = 64, 32
    config.engram_layer_ids, config.engram_num_embeddings, config.vision_n_layers = [], [], 0
    model = DeepseekV41ForCausalLM(config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dtype in (torch.float32, torch.bfloat16) and param.numel():
                param.fill_(1.0) if "norm" in name else param.normal_(0, 0.05)
    return model, config


def _logits(model, ids):
    return model(input_ids=ids, seq_lens=torch.tensor([ids.shape[1]]), position_ids=torch.arange(ids.shape[1])[None])[
        "logits"
    ].float()


def test_every_attention_and_shared_expert_adapter_trains():
    """Including the ratio-2 compressor, which pools in FP32 from its weights rather than calling its linears."""
    from arctic_platform.model.patches.peft import apply_peft

    model, config = _model()
    assert config.compress_ratios[1] == 2
    wrapped = apply_peft(model, dict(ATTENTION_AND_DENSE))
    ids = torch.randint(3, config.vocab_size, (1, 16))
    base = _logits(wrapped, ids)
    with torch.no_grad():
        for name, param in wrapped.named_parameters():
            if ".lora_B" in name:
                param.normal_(0, 0.1)
    out = _logits(wrapped, ids)
    assert (out - base).abs().max() > 1e-3
    out.square().mean().backward()
    adapters = [name for name, _ in wrapped.named_parameters() if ".lora_" in name]
    assert any("layers.1.self_attn.compressor.wgate" in name for name in adapters)
    assert any("o_a_proj" in name for name in adapters) and any("shared_expert" in name for name in adapters)
    untrained = [
        name for name, p in wrapped.named_parameters() if ".lora_" in name and (p.grad is None or not p.grad.any())
    ]
    assert not untrained, untrained


def test_compressor_fp32_linear_without_adapters_is_unchanged():
    from arctic_platform.model.implementations.deepseek_v41.attention import _fp32_linear

    linear = nn.Linear(64, 32, bias=False)
    linear.weight = nn.Parameter(linear.weight.detach().float())
    hidden = torch.randn(3, 5, 64, dtype=torch.bfloat16)
    assert torch.equal(_fp32_linear(linear, hidden), F.linear(hidden.float(), linear.weight.float()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="packed MXFP4 experts with LoRA run on CUDA")
def test_routed_expert_lora_trains_through_the_v41_moe():
    """Expert LoRA wraps ``mlp.experts`` in PEFT's ParamWrapper; the V4.1 MoE must reach the experts through it."""
    from arctic_platform.model.implementations.deepseek_v41.moe import DeepseekV41MoE
    from arctic_platform.model.patches.peft import apply_peft
    from arctic_platform.model.patches.peft import attach_unfused_expert_lora_factors
    from arctic_platform.model.patches.peft import install_unfused_expert_lora_activation
    from tests.model.test_deepseek_v41_streaming import _config

    torch.manual_seed(0)
    config = _config()
    config.hidden_size, config.moe_intermediate_size = 64, 32
    config.quantization_config = {
        "quant_method": "fp8",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    }

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = DeepseekV41MoE(config, 1)

    block = Block()
    with torch.no_grad():
        for name, param in block.named_parameters():
            if param.dtype == torch.int8:
                param.copy_(torch.randint(-128, 128, param.shape, dtype=torch.int8))
            elif name.endswith("_scale_inv"):
                param.fill_(2.0**-4)
            elif param.dtype == torch.float32 and param.numel():
                param.normal_(0, 0.05)
    block = block.cuda()
    peft = {
        "peft_type": "LORA",
        "r": 4,
        "lora_alpha": 4,
        "target_parameters": ["experts.w1", "experts.w2", "experts.w3"],
    }
    wrapped = apply_peft(block, peft)
    install_unfused_expert_lora_activation()
    assert attach_unfused_expert_lora_factors(wrapped) == 3
    with torch.no_grad():
        for name, param in wrapped.named_parameters():
            if ".lora_B" in name:
                param.normal_(0, 0.1)
    x = torch.randn(1, 64, config.hidden_size, device="cuda")
    out = wrapped.base_model.model.mlp(x)
    out.float().square().mean().backward()
    adapters = {name: p for name, p in wrapped.named_parameters() if ".lora_" in name}
    assert len(adapters) == 6
    assert all(p.grad is not None and p.grad.any() for p in adapters.values()), [
        n for n, p in adapters.items() if p.grad is None
    ]
