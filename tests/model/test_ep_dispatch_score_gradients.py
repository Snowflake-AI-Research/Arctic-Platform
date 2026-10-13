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

"""Exercise production dispatch autograd without requiring a native transport."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(params=["deepep", "ucclep"])
def dispatch(request):
    source_root = Path(
        os.environ.get(
            "EP_DISPATCH_SOURCE_ROOT",
            Path(__file__).resolve().parents[2] / "arctic_platform/model/implementations/moe/distributed",
        )
    )
    source = source_root / f"{request.param}.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    names = {"_dispatch_setup_context", "_dispatch_backward", "_finish_backward_event"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {"_dispatch_setup_context", "_dispatch_backward"} <= {node.name for node in functions}
    calls, waits = [], []
    event = SimpleNamespace(current_stream_wait=lambda: waits.append(True))

    def combine(**kwargs):
        calls.append(kwargs)
        return kwargs["x"], kwargs["topk_weights"], event

    namespace = {
        "torch": torch,
        "EventOverlap": SimpleNamespace,
        "_buffer": SimpleNamespace(combine=combine),
        "_handle_cache": {1: object()},
        "_new_event_overlap": object,
        "_backward_barrier_group": None,
    }
    # Execute the unchanged dtype/autograd functions; only native transport is stubbed.
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return SimpleNamespace(
        setup=namespace["_dispatch_setup_context"],
        backward=namespace["_dispatch_backward"],
        calls=calls,
        waits=waits,
    )


def context(dispatch, hidden_dtype, score_dtype):
    ctx = SimpleNamespace()
    hidden = torch.ones(2, 3, dtype=hidden_dtype)
    scores = torch.ones(2, 2, dtype=score_dtype)
    dispatch.setup(
        ctx,
        (hidden, torch.zeros(2, 2, dtype=torch.int64), scores, None, None, None, None),
        (hidden, None, scores, None, torch.tensor([1])),
    )
    return ctx


@pytest.mark.parametrize(
    "hidden_dtype,score_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
    ],
)
def test_dispatch_backward_restores_each_input_dtype(dispatch, hidden_dtype, score_dtype):
    ctx = context(dispatch, hidden_dtype, score_dtype)
    hidden_gradient = torch.full((2, 3), 0.123456, dtype=torch.float32)
    score_gradient = torch.tensor([[1.0001, 0.123456], [-1.0001, -0.123456]], dtype=torch.float32)

    result = dispatch.backward(ctx, hidden_gradient, None, score_gradient, None, None)

    assert len(result) == 7
    assert result[0].dtype == hidden_dtype
    assert result[2].dtype == score_dtype
    torch.testing.assert_close(result[0], hidden_gradient.to(hidden_dtype), rtol=0, atol=0)
    torch.testing.assert_close(result[2], score_gradient.to(score_dtype), rtol=0, atol=0)
    assert all(result[index] is None for index in (1, 3, 4, 5, 6))
    assert dispatch.calls[0]["handle"] is ctx.saved_handle
    assert dispatch.calls[0]["topk_weights"].dtype == torch.float32
    assert dispatch.calls[0]["async_finish"] is True
    assert dispatch.calls[0]["allocate_on_comm_stream"] is True
    assert dispatch.waits == [True]


def test_dispatch_backward_preserves_fp32_router_derivative(dispatch):
    ctx = context(dispatch, torch.bfloat16, torch.float32)
    logits = torch.tensor([[0.1, 0.3]], requires_grad=True)
    scores = logits.softmax(-1)
    upstream = torch.tensor([[1.0001, 1.0002]])
    expected = torch.autograd.grad(scores, logits, upstream, retain_graph=True)[0]
    result = dispatch.backward(ctx, torch.ones(1, 3, dtype=torch.bfloat16), None, upstream, None, None)
    actual = torch.autograd.grad(scores, logits, result[2])[0]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert expected.abs().max() > 0


def test_dispatch_backward_without_score_gradient(dispatch):
    ctx = context(dispatch, torch.bfloat16, torch.float32)
    hidden_gradient = torch.ones(2, 3, dtype=torch.bfloat16)

    result = dispatch.backward(ctx, hidden_gradient, None, None, None, None)

    assert result[0] is hidden_gradient
    assert result[2] is None
    assert dispatch.calls[0]["topk_weights"] is None
    assert dispatch.waits == [True]


def test_dispatch_backward_without_hidden_gradient_keeps_existing_behavior(dispatch):
    ctx = context(dispatch, torch.bfloat16, torch.float32)

    assert dispatch.backward(ctx, None, None, None, None, None) == (None,) * 7
    assert dispatch.calls == []
    assert dispatch.waits == []
