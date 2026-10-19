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

from unittest.mock import patch

import pytest
import torch

from arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla import sparse_mla_flashmla_apply
from arctic_platform.model.implementations.glm53 import context_parallel
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal


def _assert_bf16_kernel_close(actual, expected, *, atol=1e-6):
    # Allow one BF16 epsilon per element, with a fixed floor for cancellation near zero.
    torch_assert_close(actual, expected, rtol=torch.finfo(torch.bfloat16).eps, atol=atol)


def _tiny_glm_sparse_model(*, kv_lora_rank=512, qk_rope_head_dim=0):
    configuration = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    modeling = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    config = configuration.Glm5NextTextConfig(
        vocab_size=64,
        pad_token_id=0,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=8,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=4,
        qk_nope_head_dim=4,
        layer_types=["deepseek_sparse_attention"],
        indexer_types=["full"],
        mlp_layer_types=["dense"],
        index_topk=4,
        index_head_dim=4,
        index_n_heads=2,
        index_kpool=2,
        hc_mult=1,
        max_position_embeddings=32,
    )
    model = torch.nn.Module()
    model.model = modeling.Glm5NextTextModel(config)
    context_parallel.apply_context_parallelism(model, 1, None)
    return model


@pytest.mark.parametrize("magnitude", [1.0, 1e-2, 1e-5])
def test_bf16_tolerance_is_per_element(magnitude):
    expected = torch.tensor([1.0, magnitude], dtype=torch.float32)
    close = expected + expected.abs() * torch.finfo(torch.bfloat16).eps / 2
    _assert_bf16_kernel_close(close, expected)
    incorrect = expected.clone()
    incorrect[1] += max(magnitude * 0.1, 2e-6)
    with pytest.raises(AssertionError):
        _assert_bf16_kernel_close(incorrect, expected)


def test_bf16_tolerance_has_a_fixed_floor_for_zero():
    expected = torch.zeros(2)
    _assert_bf16_kernel_close(expected + 5e-7, expected)
    with pytest.raises(AssertionError):
        _assert_bf16_kernel_close(expected + 2e-6, expected)


def test_bf16_forward_tolerance_does_not_scale_with_other_elements():
    expected = torch.tensor([100.0, 1e-2])
    actual = expected.clone()
    actual[1] = 2e-2
    with pytest.raises(AssertionError):
        _assert_bf16_kernel_close(actual, expected, atol=1e-4)


def test_zero_padded_channels_preserve_attention_and_gradients():
    set_seed(42)
    query = torch.randn(1, 6, 2, 512, requires_grad=True)
    kv = torch.randn(1, 9, 1, 512, requires_grad=True)
    indices = torch.tensor([[[0, 1, -1]]]).expand(1, 6, -1)
    kwargs = {"scale": 0.03125, "value_dim": 512}
    expected = context_parallel._reference_sparse_mla(query, kv, indices, **kwargs)
    actual = context_parallel._reference_sparse_mla(
        torch.nn.functional.pad(query, (0, 64)),
        torch.nn.functional.pad(kv, (0, 64)),
        indices,
        **kwargs,
    )
    # Padding can change the FP32 matmul reduction order, not the dot products.
    torch_assert_close(actual, expected, rtol=0, atol=1e-6)
    grad = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, (query, kv), grad, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, (query, kv), grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch_assert_close(actual_grad, expected_grad, rtol=0, atol=1e-6)


@require_torch_gpu
def test_real_flashmla_padded_width_matches_original_reference():
    set_seed(42)
    query = (torch.randn(1, 8, 2, 512, device="cuda", dtype=torch.bfloat16) * 0.05).requires_grad_()
    kv = (torch.randn(1, 17, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.05).requires_grad_()
    indices = torch.arange(16, device="cuda").expand(1, 8, -1).clone()
    indices[:, 0] = -1
    indices = context_parallel._pad_sparse_indices(indices)
    scale = 0.03125
    # FlashMLA accumulates softmax in FP32. The BF16 fallback rounds probabilities
    # before multiplying values; use the independent FP32 reference for kernel parity.
    expected = context_parallel._reference_sparse_mla(
        query.float(),
        kv.float(),
        indices,
        scale=scale,
        value_dim=512,
    ).to(query.dtype)
    padded_query = torch.nn.functional.pad(query, (0, 64))
    padded_kv = torch.nn.functional.pad(kv, (0, 64))
    saved = []

    def pack(value):
        saved.append((value.data_ptr(), value.numel() * value.element_size()))
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        actual = sparse_mla_flashmla_apply(padded_query, padded_kv, indices.unsqueeze(2), scale)
    # FlashMLA retains the padded inputs for reference backward, including 64 zero channels.
    input_bytes = sum(value.numel() * value.element_size() for value in (query, kv))
    retained_bytes = sum(size for ptr, size in saved if ptr in (padded_query.data_ptr(), padded_kv.data_ptr()))
    assert retained_bytes == input_bytes * 576 // 512
    assert {ptr for ptr, _ in saved}.issuperset({padded_query.data_ptr(), padded_kv.data_ptr()})
    # BF16 kernel accumulation leaves cancellation residuals near zero. Use a fixed
    # forward floor, not a budget scaled by the largest output; gradients keep 1e-6.
    _assert_bf16_kernel_close(actual, expected, atol=1e-4)
    torch_assert_equal(actual[:, 0], torch.zeros_like(actual[:, 0]))
    grad = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, (query, kv), grad, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, (query, kv), grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.isfinite(actual_grad).all()
        _assert_bf16_kernel_close(actual_grad, expected_grad)


@require_torch_gpu
@pytest.mark.parametrize(
    "kv_lora_rank,rope_width,use_flashmla",
    [
        pytest.param(512, 0, True, id="rope-free-512-pads-to-576"),
        pytest.param(512, 64, True, id="native-576"),
        pytest.param(448, 64, False, id="query-512-values-not-512"),
        pytest.param(448, 128, False, id="query-576-values-not-512"),
        pytest.param(512, 32, False, id="unsupported-query-544"),
        pytest.param(256, 0, False, id="unsupported-query-256"),
    ],
)
def test_sparse_attention_dispatches_only_supported_flashmla_layouts(kv_lora_rank, rope_width, use_flashmla):
    set_seed(42)
    # Exercise the adapter independently: current GLM configs reject nonzero RoPE widths.
    attention = torch.nn.Module()
    attention.num_heads = 2
    attention.kv_lora_rank = kv_lora_rank
    attention.qk_nope_head_dim = 4
    attention.qk_rope_head_dim = rope_width
    attention.qk_head_dim = 4 + rope_width
    attention.v_head_dim = 4
    attention.q_a_proj = torch.nn.Linear(16, 8, bias=False)
    attention.q_a_layernorm = torch.nn.LayerNorm(8)
    attention.q_b_proj = torch.nn.Linear(8, 2 * attention.qk_head_dim, bias=False)
    attention.kv_a_proj_with_mqa = torch.nn.Linear(16, kv_lora_rank + rope_width, bias=False)
    attention.kv_a_layernorm = torch.nn.LayerNorm(kv_lora_rank)
    attention.kv_b_proj = torch.nn.Linear(kv_lora_rank, 2 * 8, bias=False)
    attention.o_proj = torch.nn.Linear(8, 16, bias=False)
    attention.indexer = None
    attention.next_skip_topk = False
    attention.scaling = 0.03125
    attention._cp_group = None
    attention = attention.to(device="cuda", dtype=torch.bfloat16)
    hidden_states = torch.randn(1, 4, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    indices = torch.arange(4, device="cuda").expand(1, 4, -1)
    with patch.object(context_parallel, "sparse_mla_flashmla_apply", wraps=sparse_mla_flashmla_apply) as flash:
        with patch.object(
            context_parallel, "_reference_sparse_mla", wraps=context_parallel._reference_sparse_mla
        ) as ref:
            hidden, _, _ = context_parallel._sparse_attention_forward(
                attention, hidden_states, None, prev_topk_indices=indices
            )
            assert torch.isfinite(hidden).all()
            hidden.float().square().mean().backward()
            torch.cuda.synchronize()
    assert flash.call_count == int(use_flashmla)
    assert ref.call_count == int(not use_flashmla)
    if use_flashmla:
        assert flash.call_args.args[0].shape[-1] == 576
        assert flash.call_args.args[1].shape[-1] == 576
    else:
        assert ref.call_args.args[0].shape[-1] == kv_lora_rank + rope_width
        assert ref.call_args.kwargs["value_dim"] == kv_lora_rank
    grads = [parameter.grad for parameter in attention.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(gradient).all() for gradient in grads)
    assert torch.isfinite(hidden_states.grad).all()


@require_torch_gpu
def test_glm53_rope_free_model_executes_flashmla():
    set_seed(42)
    model = _tiny_glm_sparse_model().to(device="cuda", dtype=torch.bfloat16).train()
    sparse_layers = sum(layer.block_type == "deepseek_sparse_attention" for layer in model.model.layers)
    assert sparse_layers > 0
    tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
    with patch.object(context_parallel, "sparse_mla_flashmla_apply", wraps=sparse_mla_flashmla_apply) as flash:
        with patch.object(context_parallel, "_reference_sparse_mla", side_effect=AssertionError("Reference fallback")):
            hidden = model.model(input_ids=tokens, use_cache=False).last_hidden_state
            assert torch.isfinite(hidden).all()
            hidden.float().square().mean().backward()
            torch.cuda.synchronize()
    assert flash.call_count == sparse_layers
    assert flash.call_args.args[0].shape[-1] == 576
    assert flash.call_args.args[1].shape[-1] == 576
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(gradient).all() for gradient in grads)
