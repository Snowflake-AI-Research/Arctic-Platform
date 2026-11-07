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

"""``patches.checkpoint_keep_attention`` on the dense Qwen3.5 forward the huggingface loader builds.

A four-layer Qwen3.5 (three GatedDeltaNet layers, one gated full-attention layer) with the fp32 parity forward and the
fused-projection GatedDeltaNet trains to the bitwise-same loss and gradients with the attention kernel kept as with
plain whole-layer checkpointing, and runs the attention kernel once per attention layer. On CPU the convolution and
the rule are the transformers PyTorch references (SDPA attention); on CUDA they are causal-conv1d and FLA's Triton
kernels with FlashAttention 3, so the selective policy also runs over the Triton recompute.
"""

from __future__ import annotations

import copy
import types

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from arctic_platform.model import LoaderContext
from arctic_platform.model.implementations.gpu.gated_delta_net_fused import apply_fused_gated_delta_net
from arctic_platform.model.implementations.gpu.qwen3_5_dense_parity import apply_fp32_parity
from arctic_platform.model.implementations.gpu.vllm_rounding import round_gated_norms_once
from arctic_platform.model.patches.checkpoint_keep_attention import apply_checkpoint_keep_attention
from arctic_platform.model.patches.checkpoint_keep_attention import attention_kernel_ops
from arctic_platform.model.patches.gradient_checkpointing import apply_gradient_checkpointing
from arctic_platform.testing_utils import set_seed


class _CountKernels(TorchDispatchMode):
    """Counts attention-kernel calls below the checkpoint's own dispatch modes, as test_checkpoint_keep_attention.py."""

    def __init__(self):
        super().__init__()
        self.kernels = attention_kernel_ops()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in self.kernels:
            self.calls += 1
        return func(*args, **(kwargs or {}))


def _ctx(gradient_checkpointing) -> LoaderContext:
    return LoaderContext(
        spec=types.SimpleNamespace(
            patches=types.SimpleNamespace(gradient_checkpointing=gradient_checkpointing),
            parallelism=types.SimpleNamespace(sequence_parallel=1),
        )
    )


def _cases():
    yield pytest.param("cpu", torch.float32, "sdpa", 16, id="cpu-sdpa")
    if torch.cuda.is_available():
        try:
            import causal_conv1d  # noqa: F401
            import fla  # noqa: F401
            import flash_attn_interface  # noqa: F401
        except ImportError:
            pass
        else:
            yield pytest.param("cuda", torch.bfloat16, "flash_attention_3", 128, id="cuda-fa3-fla")


@pytest.fixture
def torch_kernels_on_cpu(monkeypatch, request):
    """On CPU, the stock module-level convolution and rule as their PyTorch references."""
    if request.node.callspec.params["device"] == "cpu":
        for name in ("causal_conv1d_fn", "torch_chunk_gated_delta_rule"):
            function = getattr(modeling_qwen3_5, name)
            monkeypatch.setattr(modeling_qwen3_5, name, getattr(function, "__wrapped__", function))


def _model(device, dtype, attn_implementation, head_dim):
    config = Qwen3_5TextConfig(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=256,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=False,
    )
    config._attn_implementation = attn_implementation
    set_seed(0)
    model = modeling_qwen3_5.Qwen3_5ForCausalLM(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.1)
    model = model.to(device=device, dtype=dtype).train()
    round_gated_norms_once(model)
    apply_fp32_parity(model, sequence_parallel=False)
    apply_fused_gated_delta_net(model, fp32_beta=True)
    return model


def _train_step(model, input_ids):
    with _CountKernels() as counter:
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
    grads = {
        name.replace("._checkpoint_wrapped_module", ""): param.grad.detach().clone()
        for name, param in model.named_parameters()
    }
    return loss.detach(), grads, counter.calls


@pytest.mark.parametrize("frequency", [True, 2], ids=["every-layer", "every-2nd-layer"])
@pytest.mark.parametrize("device,dtype,attn_implementation,head_dim", list(_cases()))
def test_keep_attention_matches_whole_layer_recompute_qwen35(
    torch_kernels_on_cpu, device, dtype, attn_implementation, head_dim, frequency
):
    base = _model(device, dtype, attn_implementation, head_dim)
    input_ids = torch.randint(0, 128, (1, 256), device=device, generator=torch.Generator(device).manual_seed(1))

    plain = copy.deepcopy(base)
    apply_gradient_checkpointing(plain, _ctx(frequency))
    kept = copy.deepcopy(base)
    apply_gradient_checkpointing(kept, _ctx(frequency))
    apply_checkpoint_keep_attention(kept, _ctx(frequency))

    plain_loss, plain_grads, plain_calls = _train_step(plain, input_ids)
    kept_loss, kept_grads, kept_calls = _train_step(kept, input_ids)

    # One full-attention layer (index 3): checkpointed with every layer, not with every 2nd layer (0 and 2).
    checkpointed_attention = 1 if frequency is True else 0
    assert plain_calls == 1 + checkpointed_attention
    assert kept_calls == 1
    torch.testing.assert_close(kept_loss, plain_loss, rtol=0, atol=0)
    assert kept_grads.keys() == plain_grads.keys()
    for name, grad in plain_grads.items():
        torch.testing.assert_close(kept_grads[name], grad, rtol=0, atol=0, msg=name)
