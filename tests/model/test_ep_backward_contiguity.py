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

"""Native EP transport requires contiguous tensors even for strided autograd derivatives."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(params=["deepep", "ucclep"])
def transport(request):
    path = (
        Path(__file__).resolve().parents[2]
        / "arctic_platform/model/implementations/moe/distributed"
        / f"{request.param}.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {"_dispatch_backward", "_finish_backward_event"}
    classes = {"_DeepEPCombine", "_UcclEPCombine"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
        or isinstance(node, ast.ClassDef)
        and node.name in classes
    ]
    calls, waits = [], []
    event = SimpleNamespace(current_stream_wait=lambda: waits.append(True))
    events = []

    def combine(**kwargs):
        calls.append(kwargs)
        assert kwargs["x"].is_contiguous(), "native combine requires contiguous input"
        scores = kwargs.get("topk_weights")
        assert scores is None or scores.is_contiguous(), "native combine requires contiguous scores"
        return kwargs["x"], scores, event

    def dispatch(**kwargs):
        calls.append(kwargs)
        assert kwargs["x"].is_contiguous(), "native dispatch requires contiguous input"
        return kwargs["x"], None, None, None, None, event

    def previous_event():
        events.append(True)
        return object()

    namespace = {
        "torch": torch,
        "EventOverlap": SimpleNamespace,
        "_buffer": SimpleNamespace(combine=combine, dispatch=dispatch),
        "_new_event_overlap": previous_event,
        "_backward_barrier_group": None,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    name = "_DeepEPCombine" if request.param == "deepep" else "_UcclEPCombine"
    return SimpleNamespace(
        dispatch_backward=namespace["_dispatch_backward"],
        combine=namespace[name],
        calls=calls,
        waits=waits,
        events=events,
    )


def derivative(layout, rows, columns):
    if layout == "expanded":
        return torch.tensor(0.125).expand(rows, columns)
    if layout == "transposed":
        return torch.arange(rows * columns, dtype=torch.float32).view(columns, rows).t()
    return torch.arange(rows * columns, dtype=torch.float32).view(rows, columns)


@pytest.mark.parametrize("layout", ["expanded", "transposed", "contiguous"])
def test_combine_autograd_accepts_strided_derivatives(transport, layout):
    inputs = torch.ones(2, 3, requires_grad=True)
    handle = object()

    class Combine(transport.combine):
        @staticmethod
        def forward(ctx, inputs):
            ctx.handle = handle
            return inputs.clone()

    upstream = derivative(layout, 2, 3)
    result = Combine.apply(inputs)
    if layout == "expanded":
        # Scalar reductions produce stride-zero derivatives through real autograd.
        (result.sum() * 0.125).backward()
    else:
        result.backward(upstream)
    torch.testing.assert_close(inputs.grad, upstream, atol=0, rtol=0)
    assert transport.calls[0]["handle"] is handle
    assert transport.waits == [True]


@pytest.mark.parametrize("layout", ["expanded", "transposed", "contiguous"])
@pytest.mark.parametrize("strided_inputs", ["hidden", "scores", "both"])
def test_dispatch_backward_accepts_strided_hidden_and_score_derivatives(transport, layout, strided_inputs):
    hidden = derivative(layout if strided_inputs != "scores" else "contiguous", 2, 3)
    scores = derivative(layout if strided_inputs != "hidden" else "contiguous", 2, 2)
    ctx = SimpleNamespace(saved_handle=object(), input_dtype=hidden.dtype, topk_weights_dtype=scores.dtype)

    actual = transport.dispatch_backward(ctx, hidden, None, scores, None, None)

    torch.testing.assert_close(actual[0], hidden, atol=0, rtol=0)
    torch.testing.assert_close(actual[2], scores, atol=0, rtol=0)
    assert transport.calls[0]["handle"] is ctx.saved_handle
    assert transport.waits == [True]


@pytest.mark.parametrize("operation", ["dispatch_backward", "combine_backward"])
def test_backward_materializes_before_recording_communication_event(transport, monkeypatch, operation):
    original = torch.Tensor.contiguous
    calls = []

    def contiguous(tensor, *args, **kwargs):
        assert not transport.events, "communication event recorded before gradient materialization"
        calls.append(tensor)
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", contiguous)
    hidden = derivative("expanded", 2, 3)
    ctx = SimpleNamespace(
        handle=object(), saved_handle=object(), input_dtype=hidden.dtype, topk_weights_dtype=hidden.dtype
    )
    if operation == "dispatch_backward":
        transport.dispatch_backward(ctx, hidden, None, derivative("expanded", 2, 2), None, None)
        assert len(calls) == 2
    else:
        transport.combine.backward(ctx, hidden)
        assert len(calls) == 1
    assert transport.events == [True]
