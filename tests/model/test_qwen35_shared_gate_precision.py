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

"""Preserve the shared gate's projection precision and activation-dtype sigmoid."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F


def shared_gate(*, functional=F, torch_api=torch):
    path = (
        Path(__file__).resolve().parents[2]
        / "arctic_platform/model/implementations/qwen35/models/qwen3_5_moe/modeling_qwen3_5_moe.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_shared_expert_gate"
    )
    namespace = {"torch": torch_api, "nn": torch.nn, "F": functional}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_shared_expert_gate"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("bias", [False, True])
def test_projection_keeps_fp32_before_activation_dtype_sigmoid(dtype, bias):
    events = []

    def linear(hidden, weight, offset):
        events.append(("linear", hidden.dtype, weight.dtype, None if offset is None else offset.dtype))
        assert not torch.is_autocast_enabled("cpu")
        return F.linear(hidden, weight, offset)

    def sigmoid(logits):
        events.append(("sigmoid", logits.dtype))
        assert not torch.is_autocast_enabled("cpu")
        return torch.sigmoid(logits)

    function = shared_gate(
        functional=SimpleNamespace(linear=linear),
        torch_api=SimpleNamespace(Tensor=torch.Tensor, autocast=torch.autocast, sigmoid=sigmoid),
    )
    hidden = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=dtype, requires_grad=True)
    gate = torch.nn.Linear(2, 1, bias=bias, dtype=dtype)
    with torch.no_grad():
        gate.weight.fill_(0.125)
        if bias:
            gate.bias.fill_(0.0625)
    parameters = (hidden, gate.weight) if not bias else (hidden, gate.weight, gate.bias)
    before = [tensor.detach().clone() for tensor in parameters]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = function(hidden, gate)
    assert events == [
        ("linear", torch.float32, torch.float32, torch.float32 if bias else None),
        ("sigmoid", dtype),
    ]
    assert actual.dtype == dtype
    actual.float().sum().backward()
    for tensor, expected in zip(parameters, before):
        assert tensor.grad.dtype == dtype
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_exact_gate_values_and_derivatives_survive_row_splitting(dtype):
    function = shared_gate()
    hidden = torch.tensor([[1.0, 0.5], [0.5, 1.0]] * 4, dtype=dtype, requires_grad=True)
    gate = torch.nn.Linear(2, 1, bias=True, dtype=dtype)
    with torch.no_grad():
        gate.weight.fill_(0.125)
        # Exact zero logits isolate row layout from gradient-summation roundoff.
        gate.bias.fill_(-0.1875)
    full = function(hidden, gate)
    split = torch.cat([function(rows, gate) for rows in hidden.split(2)])
    torch.testing.assert_close(full, split, rtol=0, atol=0)
    parameters = (hidden, gate.weight, gate.bias)
    full_derivatives = torch.autograd.grad(full.sum(), parameters, retain_graph=True)
    split_derivatives = torch.autograd.grad(split.sum(), parameters)
    for actual, expected in zip(split_derivatives, full_derivatives):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
