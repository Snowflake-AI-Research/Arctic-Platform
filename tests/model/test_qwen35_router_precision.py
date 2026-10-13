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

"""Keep Qwen3.5 route-weight precision separate from the shared router default."""

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from torch import nn
from torch.nn import functional as F


class _UnusedBranch(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def set_ep_comm_backend(self, backend):
        pass


@pytest.fixture
def model_classes():
    root = Path(
        os.environ.get(
            "QWEN35_SOURCE_ROOT",
            Path(__file__).resolve().parents[2] / "arctic_platform/model/implementations",
        )
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Literal": Literal,
        "dataclass": dataclass,
        "ROUTER_REPLAY_FRESH": -1,
        "GroupedExperts": _UnusedBranch,
        "TokenReorderer": _UnusedBranch,
        "BCFeedForward": _UnusedBranch,
        "FeedForward": _UnusedBranch,
        "GradientCheckpointingLayer": nn.Module,
        "Qwen3_5MoeConfig": SimpleNamespace,
        "Qwen3_5MoeGatedDeltaNet": _UnusedBranch,
        "Qwen3_5MoeRMSNorm": _UnusedBranch,
        "_get_gated_attention": lambda config: _UnusedBranch(),
    }
    sources = (
        (root / "moe/layers/moe.py", ("MoEArgs", "TokenChoiceTopKRouter", "MoE")),
        (
            root / "qwen35/models/qwen3_5_moe/modeling_qwen3_5_moe.py",
            ("Qwen3_5MoeDecoderLayer",),
        ),
    )
    for path, names in sources:
        tree = ast.parse(path.read_text(), filename=str(path))
        for name in names:
            node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
            if name in ("MoE", "Qwen3_5MoeDecoderLayer"):
                node.body = [
                    member for member in node.body if isinstance(member, ast.FunctionDef) and member.name == "__init__"
                ]
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec", dont_inherit=True), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture(autouse=True)
def cpu_histogram(monkeypatch):
    histogram = torch.histc
    # CUDA accepts integer routing indices; CPU requires an otherwise equivalent float histogram.
    monkeypatch.setattr(torch, "histc", lambda values, **kwargs: histogram(values.float(), **kwargs))


def qwen_router(classes, dtype, mixer):
    config = SimpleNamespace(
        hidden_size=3,
        layer_types=[mixer],
        num_experts=4,
        num_experts_per_tok=2,
        use_grouped_mm=False,
        load_balance_coeff=None,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        rms_norm_eps=1e-6,
    )
    router = classes.Qwen3_5MoeDecoderLayer(config, 0).mlp.router.to(dtype)
    with torch.no_grad():
        router.gate.weight.copy_(torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3], [-1, -1, -1]], dtype=dtype))
    return router


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("mixer", ["linear_attention", "full_attention"])
@pytest.mark.parametrize("routing", ["own", "replay", "fresh"])
def test_qwen_route_weights_match_activation_dtype_and_derivatives(model_classes, dtype, mixer, routing):
    router = qwen_router(model_classes, dtype, mixer)
    hidden = torch.tensor([[0.3, 0.2, 0.1], [0.1, 0.4, 0.2]], dtype=dtype, requires_grad=True)
    probabilities = F.softmax(router.gate(hidden).float(), dim=-1)
    own_indices = probabilities.topk(router.top_k, dim=-1).indices
    replayed = None
    expected_indices = own_indices
    if routing != "own":
        replayed = torch.tensor([[3, 1], [2, 0]])
        expected_indices = replayed
        if routing == "fresh":
            router.enable_fresh_replay_routing()
            replayed = replayed.clone()
            replayed[1] = model_classes.ROUTER_REPLAY_FRESH
            expected_indices = torch.stack((replayed[0], own_indices[1]))
    expected = probabilities.gather(1, expected_indices)
    expected = (expected / expected.sum(-1, keepdim=True)).to(dtype)
    actual, indices, counts = router(hidden, routed_experts=replayed)

    assert torch.equal(indices, expected_indices)
    assert counts.sum().item() == hidden.shape[0] * router.top_k
    assert actual.dtype == hidden.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = (hidden, router.gate.weight)
    expected_grads = torch.autograd.grad(expected[:, 0].sum(), parameters, retain_graph=True)
    actual_grads = torch.autograd.grad(actual[:, 0].sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("score_func", ["softmax", "sigmoid"])
def test_shared_router_keeps_fp32_route_weights_by_default(model_classes, dtype, score_func):
    args = model_classes.MoEArgs(
        num_experts=4,
        num_shared_experts=0,
        top_k=2,
        score_func=score_func,
        load_balance_coeff=None,
    )
    router = model_classes.MoE(args, dim=3, hidden_dim=4).router.to(dtype)
    hidden = torch.ones(2, 3, dtype=dtype, requires_grad=True)
    scores, indices, _ = router(hidden)
    logits = router.gate(hidden).float()
    probabilities = logits.softmax(-1) if score_func == "softmax" else logits.sigmoid()
    assert scores.dtype == torch.float32
    torch.testing.assert_close(scores, probabilities.gather(1, indices), rtol=0, atol=0)
    assert scores.grad_fn is not None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("score_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("normalize", [False, True])
def test_opt_in_rounds_only_final_scaled_weights(model_classes, dtype, score_func, normalize):
    router = model_classes.TokenChoiceTopKRouter(3, 4, 2, score_func, normalize, 1.7, route_weights_dtype="input").to(
        dtype
    )
    with torch.no_grad():
        router.gate.weight.copy_(torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3], [-1, -1, -1]], dtype=dtype))
    hidden = torch.tensor([[0.3, 0.2, 0.1], [0.1, 0.4, 0.2]], dtype=dtype, requires_grad=True)
    bias = torch.tensor([-10.0, -10.0, 10.0, 10.0], requires_grad=True)
    logits = router.gate(hidden).float()
    probabilities = logits.softmax(-1) if score_func == "softmax" else logits.sigmoid()
    expected_indices = (probabilities + bias).topk(2, dim=-1).indices
    expected = probabilities.gather(1, expected_indices)
    if normalize:
        expected = expected / (expected.sum(-1, keepdim=True) + 1e-20)
    expected = (expected * router.route_scale).to(dtype)

    actual, indices, counts = router(hidden, expert_bias=bias)
    assert actual.dtype == dtype
    assert torch.equal(indices, expected_indices)
    assert counts.sum().item() == 4
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = (hidden, router.gate.weight, bias)
    expected_grads = torch.autograd.grad(expected[:, 0].sum(), parameters, retain_graph=True, allow_unused=True)
    actual_grads = torch.autograd.grad(actual[:, 0].sum(), parameters, allow_unused=True)
    for actual_grad, expected_grad in zip(actual_grads[:2], expected_grads[:2]):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert actual_grads[-1] is expected_grads[-1] is None


def test_unknown_route_weight_precision_is_rejected(model_classes):
    with pytest.raises(ValueError, match="Unknown route weights dtype"):
        model_classes.TokenChoiceTopKRouter(3, 4, 2, "softmax", True, 1, route_weights_dtype="float16")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mixer", ["linear_attention", "full_attention"])
@pytest.mark.parametrize("routing", ["own", "replay", "fresh"])
def test_qwen_autocast_rounds_to_router_logits_not_fp32_hidden(model_classes, dtype, mixer, routing):
    router = qwen_router(model_classes, torch.float32, mixer)
    hidden = torch.tensor([[0.3, 0.2, 0.1], [0.1, 0.4, 0.2]], requires_grad=True)
    replayed = None
    with torch.autocast("cpu", dtype=dtype):
        logits = router.gate(hidden)
        assert logits.dtype == dtype and hidden.dtype == torch.float32
        probabilities = F.softmax(logits.float(), dim=-1)
        indices = probabilities.topk(router.top_k, dim=-1).indices
        if routing != "own":
            replayed = torch.tensor([[3, 1], [2, 0]])
            indices = replayed
            if routing == "fresh":
                router.enable_fresh_replay_routing()
                replayed = replayed.clone()
                replayed[1] = model_classes.ROUTER_REPLAY_FRESH
                indices = torch.stack((replayed[0], probabilities[1].topk(router.top_k).indices))
        expected = probabilities.gather(1, indices)
        expected = (expected / expected.sum(-1, keepdim=True)).to(logits.dtype)
        actual, actual_indices, counts = router(hidden, routed_experts=replayed)
    assert actual.dtype == logits.dtype
    assert torch.equal(actual_indices, indices)
    assert counts.sum().item() == 4
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = (hidden, router.gate.weight)
    golden = torch.autograd.grad(expected[:, 0].sum(), parameters, retain_graph=True)
    observed = torch.autograd.grad(actual[:, 0].sum(), parameters)
    for left, right in zip(observed, golden):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_shared_router_still_uses_fp32_weights_under_autocast(model_classes, dtype):
    router = model_classes.TokenChoiceTopKRouter(3, 4, 2, "softmax", True, 1)
    hidden = torch.tensor([[0.3, 0.2, 0.1]], requires_grad=True)
    with torch.autocast("cpu", dtype=dtype):
        assert router.gate(hidden).dtype == dtype
        scores, indices, _ = router(hidden)
    assert scores.dtype == torch.float32
    assert indices.shape == (1, 2)
    assert scores.grad_fn is not None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("score_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("normalize", [False, True])
def test_logits_option_rounds_after_normalizing_and_scaling(model_classes, dtype, score_func, normalize):
    router = model_classes.TokenChoiceTopKRouter(3, 4, 2, score_func, normalize, 1.7, route_weights_dtype="logits")
    hidden = torch.tensor([[0.3, 0.2, 0.1], [0.1, 0.4, 0.2]], requires_grad=True)
    with torch.autocast("cpu", dtype=dtype):
        logits = router.gate(hidden)
        probabilities = logits.float().softmax(-1) if score_func == "softmax" else logits.float().sigmoid()
        expected, indices = probabilities.topk(2, dim=-1)
        if normalize:
            expected = expected / (expected.sum(-1, keepdim=True) + 1e-20)
        expected = (expected * 1.7).to(logits.dtype)
        actual, actual_indices, _ = router(hidden)
    assert actual.dtype == dtype
    assert torch.equal(actual_indices, indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = (hidden, router.gate.weight)
    golden = torch.autograd.grad(expected[:, 0].sum(), parameters, retain_graph=True)
    observed = torch.autograd.grad(actual[:, 0].sum(), parameters)
    for left, right in zip(observed, golden):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
