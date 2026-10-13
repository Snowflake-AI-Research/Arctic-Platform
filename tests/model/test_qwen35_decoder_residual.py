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

"""Exercise the actual decoder forward with isolated attention/expert branches."""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def decoder_forward():
    root = Path(
        os.environ.get(
            "QWEN35_SOURCE_ROOT",
            Path(__file__).resolve().parents[2] / "arctic_platform/model/implementations",
        )
    )
    path = root / "qwen35/models/qwen3_5_moe/modeling_qwen3_5_moe.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    layer = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5MoeDecoderLayer"
    )
    forward = next(node for node in layer.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    namespace = {
        "torch": torch,
        "Optional": __import__("typing").Optional,
        "_shared_expert_gate": lambda hidden, gate: torch.full_like(hidden[..., :1], 0.5),
    }
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(path), "exec"), namespace)
    candidate_path = os.environ.get("QWEN35_DECODER_CANDIDATE")
    if candidate_path:
        spec = importlib.util.spec_from_file_location("decoder_candidate", candidate_path)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        namespace["forward"].__code__ = candidate.corrected_forward_code(path)
    return namespace["forward"]


def branches(dtype, mixer):
    shared = torch.tensor([[2.0]], dtype=dtype, requires_grad=True)
    routed = torch.tensor([[[1.0]]], dtype=dtype, requires_grad=True)

    def attention(*args, **kwargs):
        return torch.zeros_like(args[0] if args else kwargs["hidden_states"])

    layer = SimpleNamespace(
        layer_type=mixer,
        input_layernorm=lambda x: x,
        post_attention_layernorm=lambda x: x,
        linear_attn=attention,
        self_attn=lambda **kwargs: (attention(**kwargs), None),
        mlp=lambda hidden, **kwargs: routed.expand_as(hidden),
        shared_expert=lambda hidden: shared.expand_as(hidden),
        shared_expert_gate=None,
    )
    return layer, routed, shared


@pytest.mark.parametrize("mixer", ["linear_attention", "full_attention"])
@pytest.mark.parametrize("dtype,residual", [(torch.bfloat16, 256.0), (torch.float16, 2048.0), (torch.float32, 256.0)])
def test_decoder_adds_combined_expert_output_to_residual(decoder_forward, dtype, residual, mixer):
    layer, routed, shared = branches(dtype, mixer)
    hidden = torch.tensor([[[residual]]], dtype=dtype, requires_grad=True)

    actual = decoder_forward(layer, hidden)
    expected = hidden + (routed + 0.5 * shared.view_as(hidden))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().sum().backward()
    assert hidden.grad.item() == 2 * expected.item()
    assert routed.grad.item() == 2 * expected.item()
    assert shared.grad.item() == expected.item()
