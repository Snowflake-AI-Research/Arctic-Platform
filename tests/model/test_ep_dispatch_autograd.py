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

"""Exercise dispatch callbacks through PyTorch custom-op autograd on CPU."""

import itertools

import pytest
import test_ep_dispatch_score_gradients
import torch

dispatch = test_ep_dispatch_score_gradients.dispatch
_NAMESPACES = itertools.count()


@pytest.mark.parametrize("hidden_dtype", [torch.bfloat16, torch.float32])
def test_dispatch_custom_op_preserves_router_derivative(dispatch, hidden_dtype):
    namespace = f"ap_dispatch_dtype_test_{next(_NAMESPACES)}"
    library = torch.library.Library(namespace, "DEF")
    library.define(
        "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
        "Tensor num_tokens_per_rank, Tensor num_tokens_per_rdma_rank, "
        "Tensor is_token_in_rank, Tensor num_tokens_per_expert) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
    )

    def forward(x, topk_idx, topk_weights, *_):
        return x.clone(), topk_idx.clone(), topk_weights.float().clone(), torch.zeros(1), torch.tensor([1])

    library.impl("dispatch", forward, "CPU")
    torch.library.register_autograd(
        f"{namespace}::dispatch", dispatch.backward, setup_context=dispatch.setup, lib=library
    )
    operation = getattr(getattr(torch.ops, namespace), "dispatch")
    hidden = torch.ones(1, 3, dtype=hidden_dtype, requires_grad=True)
    logits = torch.tensor([[0.1, 0.3]], requires_grad=True)
    scores = logits.softmax(-1)
    upstream = torch.tensor([[1.0001, 1.0002]])
    expected = torch.autograd.grad(scores, logits, upstream, retain_graph=True)[0]
    metadata = torch.zeros(1, dtype=torch.int64)

    recv_hidden, _, recv_scores, _, _ = operation(
        hidden, torch.zeros(1, 2, dtype=torch.int64), scores, metadata, metadata, metadata, metadata
    )
    (recv_hidden.sum() + (recv_scores * upstream).sum()).backward()

    torch.testing.assert_close(logits.grad, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden), rtol=0, atol=0)
    assert expected.abs().max() > 0
    assert dispatch.waits == [True]
