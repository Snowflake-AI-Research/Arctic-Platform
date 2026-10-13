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

"""Cover optional fused GDN normalization and the model-specific fallback."""

import ast
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version
from torch import nn
from torch.nn import functional as F

from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import torch_assert_equal


class _FusedSentinel(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))


@pytest.fixture
def model_classes():
    path = (
        Path(__file__).resolve().parents[2]
        / "arctic_platform/model/implementations/qwen35/models/qwen3_5_moe/modeling_qwen3_5_moe.py"
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Qwen3_5MoeConfig": SimpleNamespace,
        "FusedRMSNormGated": _FusedSentinel,
        "causal_conv1d_fn": None,
        "chunk_gated_delta_rule": None,
        "torch_chunk_gated_delta_rule": lambda *args, **kwargs: None,
    }
    tree = ast.parse(path.read_text(), filename=str(path))
    for name in ("Qwen3_5MoeRMSNormGated", "Qwen3_5MoeGatedDeltaNet"):
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        if name == "Qwen3_5MoeGatedDeltaNet":
            node.body = [
                member for member in node.body if isinstance(member, ast.FunctionDef) and member.name == "__init__"
            ]
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec", dont_inherit=True), namespace)
    return namespace


def config():
    return SimpleNamespace(
        hidden_size=32,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )


@pytest.mark.parametrize("fused_available", [True, False])
def test_gdn_selects_optional_fused_normalization_or_model_fallback(model_classes, fused_available):
    model_classes["FusedRMSNormGated"] = _FusedSentinel if fused_available else None
    layer = model_classes["Qwen3_5MoeGatedDeltaNet"](config())
    expected_class = _FusedSentinel if fused_available else model_classes["Qwen3_5MoeRMSNormGated"]
    assert type(layer.norm) is expected_class
    assert (layer.norm.eps if fused_available else layer.norm.variance_epsilon) == config().rms_norm_eps
    assert list(layer.norm.state_dict()) == ["weight"]
    assert layer.norm.weight.shape == (config().linear_value_head_dim,)


def reference_norm(hidden, gate, weight, eps):
    dtype = hidden.dtype
    hidden = hidden.float()
    normalized = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps)
    weighted = weight * normalized.to(dtype)
    return (weighted * F.silu(gate.float())).to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_local_normalization_matches_hf_rounding_and_all_derivatives(model_classes, dtype, weight_dtype):
    norm = model_classes["Qwen3_5MoeRMSNormGated"](128).to(weight_dtype)
    generator = torch.Generator().manual_seed(12345)
    hidden, gate, upstream = [torch.randn(2, 3, 128, generator=generator).to(dtype) for _ in range(3)]
    hidden.requires_grad_()
    gate.requires_grad_()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.7, 1.3, 128))
    expected = reference_norm(hidden, gate, norm.weight, norm.variance_epsilon)
    actual = norm(hidden, gate)
    torch_assert_equal(actual, expected)
    parameters = (hidden, gate, norm.weight)
    golden = torch.autograd.grad(expected, parameters, upstream, retain_graph=True)
    observed = torch.autograd.grad(actual, parameters, upstream)
    for left, right in zip(observed, golden):
        torch_assert_equal(left, right)


@require_torch_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_local_fallback_matches_installed_supported_hf(model_classes, dtype):
    if not Version("5.17") <= Version(version("transformers")) < Version("5.18"):
        pytest.skip("Requires the supported Transformers 5.17 model implementation")
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNormGated

    norm = model_classes["Qwen3_5MoeRMSNormGated"](128, eps=config().rms_norm_eps).to(device="cuda", dtype=dtype)
    reference = Qwen3_5MoeRMSNormGated(128, eps=config().rms_norm_eps)
    reference = reference.to(device="cuda", dtype=dtype)
    reference.load_state_dict(norm.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(12345)
    hidden, gate, upstream = [
        torch.randn(256, 128, generator=generator).to(device="cuda", dtype=dtype) for _ in range(3)
    ]
    actual_hidden, actual_gate = [tensor.detach().clone().requires_grad_() for tensor in (hidden, gate)]
    ref_hidden, ref_gate = [tensor.detach().clone().requires_grad_() for tensor in (hidden, gate)]
    actual, expected = norm(actual_hidden, actual_gate), reference(ref_hidden, ref_gate)
    observed = torch.autograd.grad(actual, (actual_hidden, actual_gate, norm.weight), upstream)
    golden = torch.autograd.grad(expected, (ref_hidden, ref_gate, reference.weight), upstream)
    torch_assert_equal(actual, expected)
    for left, right in zip(observed, golden):
        torch_assert_equal(left, right)
