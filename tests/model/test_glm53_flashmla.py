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

import torch

# The shared fixture checks optional GLM support before importing AP model modules.
from test_glm53_packed_rows import _tiny_glm_sparse_model

from arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla import sparse_mla_flashmla_apply
from arctic_platform.model.implementations.glm53 import context_parallel
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal


def _assert_bf16_kernel_close(actual, expected):
    # Independent kernel/reference reductions round differently in BF16; 1e-4 is smaller
    # than one rounding unit even at magnitude 1/64. Bound absolute error by
    # one BF16 rounding unit at the reference's largest magnitude, with rtol=0.
    rounding_unit = torch.finfo(torch.bfloat16).eps * expected.abs().max().item()
    torch_assert_close(actual, expected, rtol=0, atol=rounding_unit)


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
    actual = sparse_mla_flashmla_apply(
        torch.nn.functional.pad(query, (0, 64)),
        torch.nn.functional.pad(kv, (0, 64)),
        indices.unsqueeze(2),
        scale,
    )
    _assert_bf16_kernel_close(actual, expected)
    torch_assert_equal(actual[:, 0], torch.zeros_like(actual[:, 0]))
    grad = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, (query, kv), grad, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, (query, kv), grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.isfinite(actual_grad).all()
        _assert_bf16_kernel_close(actual_grad, expected_grad)


@require_torch_gpu
def test_glm53_rope_free_model_executes_flashmla():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    set_seed(42)
    model = _tiny_glm_sparse_model()
    config = model.model.config.to_dict()
    config.update(kv_lora_rank=512, qk_rope_head_dim=0)
    model.model = Glm5NextTextModel(Glm5NextTextConfig.from_dict(config))
    context_parallel.apply_context_parallelism(model, 1, None)
    model = model.to(device="cuda", dtype=torch.bfloat16).train()
    tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
    with patch.object(context_parallel, "sparse_mla_flashmla_apply", wraps=sparse_mla_flashmla_apply) as flash:
        with patch.object(context_parallel, "_reference_sparse_mla", side_effect=AssertionError("Reference fallback")):
            hidden = model(input_ids=tokens, use_cache=False).last_hidden_state
            assert torch.isfinite(hidden).all()
            hidden.float().square().mean().backward()
            torch.cuda.synchronize()
    assert flash.call_count == 1
    assert flash.call_args.args[0].shape[-1] == 576
    assert flash.call_args.args[1].shape[-1] == 576
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(gradient).all() for gradient in grads)
