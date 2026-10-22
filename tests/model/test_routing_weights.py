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

import gc
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.model.implementations.moe.routing_weights import apply_routing_weights
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal


@pytest.fixture(autouse=True)
def _seed():
    set_seed(42)


def _reference(hidden, scores):
    return (hidden.float() * scores.float().reshape(-1, 1)).to(hidden.dtype)


def _require_cuda_memory(required_bytes):
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < required_bytes:
        pytest.skip(
            f"CUDA memory regression needs {required_bytes / 1024**3:g} GiB free; has {free_bytes / 1024**3:g}"
        )


@pytest.mark.parametrize("free_gib", [16, 20, 22 - 1 / 1024**3, 22, 24])
def test_cuda_memory_capacity_guard(monkeypatch, free_gib):
    required_bytes = 22 * 1024**3
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (int(free_gib * 1024**3), 24 * 1024**3))
    if free_gib < 22:
        with pytest.raises(pytest.skip.Exception, match="needs 22 GiB free"):
            _require_cuda_memory(required_bytes)
    else:
        _require_cuda_memory(required_bytes)


@pytest.mark.parametrize("hidden_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("score_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("requires_grad", [(True, True), (True, False), (False, True)])
def test_routing_weights_forward_and_gradients(hidden_dtype, score_dtype, requires_grad):
    hidden = torch.randn(11, 8, dtype=hidden_dtype)[:, ::2].requires_grad_(requires_grad[0])
    scores = torch.randn(11, dtype=score_dtype).requires_grad_(requires_grad[1])
    reference_hidden = hidden.detach().clone().requires_grad_(requires_grad[0])
    reference_scores = scores.detach().clone().requires_grad_(requires_grad[1])
    actual = apply_routing_weights(hidden, scores)
    expected = _reference(reference_hidden, reference_scores)
    torch_assert_equal(actual, expected)
    gradient = torch.randn(actual.shape, dtype=hidden_dtype)
    actual.backward(gradient)
    expected.backward(gradient)
    for value, reference in [(hidden, reference_hidden), (scores, reference_scores)]:
        if value.requires_grad:
            torch_assert_equal(value.grad, reference.grad)


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_routing_weights_explicit_output_dtype(out_dtype):
    hidden = torch.randn(6, 8, dtype=torch.bfloat16)[:, ::2].requires_grad_()
    scores = torch.randn(2, 3, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_scores = scores.detach().clone().requires_grad_()
    expected = (reference_hidden.float() * reference_scores.reshape(-1, 1)).to(out_dtype)
    actual = apply_routing_weights(hidden, scores, out_dtype=out_dtype)
    assert actual.dtype == out_dtype
    torch_assert_equal(actual, expected)
    if out_dtype == torch.float32:
        assert not torch.equal(expected, expected.to(hidden.dtype).float())
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch_assert_equal(hidden.grad, reference_hidden.grad)
    torch_assert_equal(scores.grad, reference_scores.grad)


def test_routing_weights_preserves_score_shape():
    hidden = torch.randn(6, 4, requires_grad=True)
    scores = torch.randn(2, 3, requires_grad=True)
    output = apply_routing_weights(hidden, scores)
    output.sum().backward()
    assert scores.grad.shape == scores.shape
    torch_assert_equal(scores.grad, hidden.sum(dim=1).reshape(scores.shape))


@pytest.mark.parametrize("out_dtype", [None, torch.float32])
def test_routing_weights_saves_original_precision(out_dtype):
    hidden = torch.randn(16, 32, dtype=torch.bfloat16, requires_grad=True)
    scores = torch.randn(16, requires_grad=True)
    captured = []

    def pack(value):
        captured.append((value.dtype, value.numel() * value.element_size(), value.data_ptr()))
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        output = apply_routing_weights(hidden, scores, out_dtype=out_dtype)
    assert captured == [
        (hidden.dtype, hidden.numel() * hidden.element_size(), hidden.data_ptr()),
        (scores.dtype, scores.numel() * scores.element_size(), scores.data_ptr()),
    ]
    output.sum().backward()


def test_routing_weights_second_derivatives():
    hidden = torch.randn(5, 3, requires_grad=True)
    scores = torch.randn(5, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_scores = scores.detach().clone().requires_grad_()
    derivatives = []
    for function, values in [
        (apply_routing_weights, (hidden, scores)),
        (_reference, (reference_hidden, reference_scores)),
    ]:
        loss = function(*values).square().sum()
        first = torch.autograd.grad(loss, values, create_graph=True)
        second = torch.autograd.grad(sum(value.sum() for value in first), values)
        derivatives.append((first, second))
    for actual, expected in zip(*derivatives, strict=True):
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            # Algebraically equal second derivatives may use different FP32 reduction orders.
            torch_assert_close(actual_tensor, expected_tensor, rtol=0, atol=1e-6)


@pytest.mark.parametrize("score_before_experts", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("backend", ["ucclep", "deepep"])
def test_ep_weighting_sites_preserve_forward_and_gradients(monkeypatch, score_before_experts, dtype, backend):
    ep = pytest.importorskip(f"arctic_platform.model.implementations.moe.distributed.{backend}")
    hidden = torch.randn(6, 4, dtype=dtype, requires_grad=True)
    scores = torch.randn(6, 2, requires_grad=True)
    indices = torch.tensor([[0, 1], [1, 0], [0, -1], [1, 0], [-1, 1], [0, 1]])
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_scores = scores.detach().clone().requires_grad_()
    handle = torch.tensor([123])
    calls = []

    def recorded_weighting(x, weights):
        assert weights.dtype == torch.float32
        assert weights.ndim == 1 and len(weights) == len(x)
        calls.append(score_before_experts)
        return apply_routing_weights(x, weights)

    class LocalCombine:
        @staticmethod
        def apply(x, handle_id):
            assert handle_id is handle
            return x

    monkeypatch.setattr(ep, "apply_routing_weights", recorded_weighting)
    monkeypatch.setattr(ep, "_sync_dispatch", lambda _: None)
    monkeypatch.setattr(ep, "_UcclEPCombine" if backend == "ucclep" else "_DeepEPCombine", LocalCombine)
    pending = ep._PendingDispatchState(
        hidden,
        indices,
        scores,
        torch.tensor([5, 5]),
        handle,
        score_before_experts,
    )
    dispatched, _, state = ep.finalize_dispatch_tokens(pending)
    actual = ep.combine_tokens(dispatched, state)
    permute = ep._permute_tokens if backend == "ucclep" else ep.permute_tokens
    unpermute = ep._unpermute_tokens if backend == "ucclep" else ep.unpermute_tokens
    if backend == "ucclep":
        rows, weights, row_indices = permute(reference_hidden, indices, reference_scores)
    else:
        rows, weights, row_indices, _ = permute(reference_hidden, indices, reference_scores, 2)
    expected = unpermute(_reference(rows, weights), row_indices, len(hidden))
    assert calls == [score_before_experts]
    torch_assert_equal(actual, expected)
    gradient = torch.randn(actual.shape, dtype=dtype)
    actual.backward(gradient)
    expected.backward(gradient)
    torch_assert_equal(hidden.grad, reference_hidden.grad)
    torch_assert_equal(scores.grad, reference_scores.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("site", ["before", "after", "latent"])
def test_local_moe_weighting_sites_preserve_forward_and_gradients(monkeypatch, dtype, site):
    from arctic_platform.model.implementations.moe.layers import moe

    hidden = torch.randn(6, 4, dtype=dtype, requires_grad=True)
    scores = torch.randn(10, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_scores = scores.detach().clone().requires_grad_()
    rows = torch.tensor([0, 1, 2, 3, 4, 0, 1, 3, 4, 5])
    counts = torch.tensor([5, 5])
    calls = []

    def recorded_weighting(x, weights, *, out_dtype=None):
        assert weights.dtype == torch.float32
        assert weights.shape == (len(x),)
        calls.append((x.dtype, weights.shape, out_dtype))
        return apply_routing_weights(x, weights, out_dtype=out_dtype)

    def experts(x, num_tokens_per_expert):
        torch_assert_equal(num_tokens_per_expert, counts)
        return x.square()

    monkeypatch.setattr(moe, "apply_routing_weights", recorded_weighting)
    layer = SimpleNamespace(
        experts=experts,
        score_before_experts=site == "before",
        fc1_latent_proj=lambda x: x * 0.5,
        fc2_latent_proj=lambda x: x * 0.25,
        routed_scaling_factor=1.5,
    )
    cls = moe.LatentMoE if site == "latent" else moe.MoE
    actual = cls._run_routed_experts(layer, hidden, rows, counts, scores)
    gathered = reference_hidden.index_select(0, rows)
    if site == "before":
        expected = _reference(gathered, reference_scores).square()
    elif site == "after":
        expected = _reference(gathered.square(), reference_scores)
    else:
        expected = _reference((gathered * 0.5).square(), reference_scores) * 1.5 * 0.25
    assert calls == [(dtype, scores.shape, dtype if site == "after" else None)]
    torch_assert_equal(actual, expected)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch_assert_equal(hidden.grad, reference_hidden.grad)
    torch_assert_equal(scores.grad, reference_scores.grad)


def test_local_moe_after_experts_preserves_fp32_output():
    from arctic_platform.model.implementations.moe.layers import moe

    hidden = torch.randn(6, 4, dtype=torch.float32, requires_grad=True)
    scores = torch.randn(10, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_scores = scores.detach().clone().requires_grad_()
    rows = torch.tensor([0, 1, 2, 3, 4, 0, 1, 3, 4, 5])
    counts = torch.tensor([5, 5])

    def experts(x, num_tokens_per_expert):
        torch_assert_equal(num_tokens_per_expert, counts)
        return x.square().to(torch.bfloat16)

    layer = SimpleNamespace(experts=experts, score_before_experts=False)
    expert_output = experts(reference_hidden.index_select(0, rows), counts)
    expected = (expert_output.float() * reference_scores.reshape(-1, 1)).to(reference_hidden.dtype)
    actual = moe.MoE._run_routed_experts(layer, hidden, rows, counts, scores)
    assert actual.dtype == torch.float32
    assert not torch.equal(expected, expected.to(torch.bfloat16).float())
    torch_assert_equal(actual, expected)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch_assert_equal(hidden.grad, reference_hidden.grad)
    torch_assert_equal(scores.grad, reference_scores.grad)


@require_torch_gpu
def test_routing_weights_cuda_retained_memory():
    reserve_bytes = 20 * 1024**3
    # Keep the tuning reserve, but admit only devices with room for both measured workloads.
    _require_cuda_memory(reserve_bytes + 2 * 1024**3)
    reserve = torch.empty(reserve_bytes, dtype=torch.uint8, device="cuda")
    hidden = torch.randn(8192, 1024, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    scores = torch.randn(8192, device="cuda", requires_grad=True)

    def measure(function):
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        outputs = [function(hidden, scores) for _ in range(8)]
        torch.cuda.synchronize()
        retained = torch.cuda.memory_allocated() - baseline
        peak = torch.cuda.max_memory_allocated() - baseline
        torch.stack(outputs).sum().backward()
        gradients = (hidden.grad.clone(), scores.grad.clone())
        hidden.grad = scores.grad = None
        del outputs
        gc.collect()
        torch.cuda.synchronize()
        return retained, peak, gradients

    reference_retained, reference_peak, reference_gradients = measure(_reference)
    retained, peak, gradients = measure(apply_routing_weights)
    assert reserve.numel() == reserve_bytes
    assert reference_retained - retained >= 8 * hidden.numel() * 4
    assert peak < reference_peak
    for actual, expected in zip(gradients, reference_gradients, strict=True):
        torch_assert_equal(actual, expected)
    print(
        f"CUDA_ROUTING_WEIGHTS_MEMORY reserve_gib=20 retained_saving_bytes={reference_retained - retained} "
        f"reference_peak_bytes={reference_peak} optimized_peak_bytes={peak}"
    )
