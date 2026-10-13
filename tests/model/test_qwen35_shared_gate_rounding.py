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

"""Compare the actual shared gate with the configured activation-dtype sigmoid."""

import ast
import importlib.util
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def shared_gate():
    root = Path(
        os.environ.get(
            "QWEN35_SOURCE_ROOT",
            Path(__file__).resolve().parents[2] / "arctic_platform/model/implementations",
        )
    )
    path = root / "qwen35/models/qwen3_5_moe/modeling_qwen3_5_moe.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_shared_expert_gate"
    )
    namespace = {"torch": torch, "nn": torch.nn, "F": F}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    candidate_path = os.environ.get("QWEN35_GATE_CANDIDATE")
    if candidate_path:
        spec = importlib.util.spec_from_file_location("qwen35_gate_candidate", candidate_path)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        namespace["_shared_expert_gate"].__code__ = candidate.corrected_gate_code(path)
    return namespace["_shared_expert_gate"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias", [False, True])
def test_shared_gate_rounds_logits_before_sigmoid(shared_gate, dtype, bias):
    hidden = torch.tensor([[1.0, 1.0]], dtype=dtype, requires_grad=True)
    gate = torch.nn.Linear(2, 1, bias=bias, dtype=dtype)
    with torch.no_grad():
        gate.weight.copy_(torch.tensor([[0.015625, 0.054931640625]], dtype=dtype))
        if bias:
            gate.bias.zero_()
    expected = torch.sigmoid(F.linear(hidden, gate.weight, gate.bias))
    expected_grads = torch.autograd.grad(expected.sum(), (hidden, gate.weight), retain_graph=True)

    actual = shared_gate(hidden, gate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(actual.sum(), (hidden, gate.weight))
    for left, right in zip(actual_grads, expected_grads):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_shared_gate_preserves_input_and_parameter_storage(shared_gate):
    hidden = torch.randn(2, 8)
    gate = torch.nn.Linear(8, 1)
    original = [value.clone() for value in (hidden, gate.weight, gate.bias)]
    result = shared_gate(hidden, gate)
    assert result.dtype == hidden.dtype and result.shape == (2, 1)
    for value, expected in zip((hidden, gate.weight, gate.bias), original):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
