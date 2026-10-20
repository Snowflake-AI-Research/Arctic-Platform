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

"""Observational boundary hooks must not change the product computation."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from arctic_platform.correctness.diagnostics.b200_gradient_boundary import install_capture
from arctic_platform.correctness.diagnostics.b200_gradient_control import install_norm_backward_sync
from arctic_platform.correctness.diagnostics.b200_post_metric_capture import capture_after_metrics
from arctic_platform.correctness.diagnostics.b200_reduction_boundary import install_reduction_input_capture
from arctic_platform.correctness.diagnostics.b200_reference_capture import capture_reference_subprocess
from arctic_platform.correctness.diagnostics.b200_reference_capture import capturing_gradient_norms
from arctic_platform.testing_utils import torch_assert_equal


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))

    def forward(self, hidden):
        return hidden * (1 + self.weight)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 2)
        self.language_model = nn.Module()
        self.language_model.norm = Qwen3_5RMSNorm()
        self.language_model.layers = nn.ModuleList([nn.Module()])
        self.language_model.layers[0].linear_attn = nn.Module()
        self.language_model.layers[0].linear_attn.out_proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.embedding.weight.fill_(0.5)
            self.language_model.layers[0].linear_attn.out_proj.weight.fill_(0.25)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, ids):
        hidden = self.embedding(ids)
        hidden = self.language_model.layers[0].linear_attn.out_proj(hidden)
        return self.language_model.norm(hidden)


def test_gradient_boundary_capture_is_observational(tmp_path):
    expected, actual = _Model(), _Model()
    local, handles = install_capture(actual, tmp_path / "capture")
    ids = torch.tensor([[1, 2]])
    expected_output, actual_output = expected(ids), actual(ids)
    torch_assert_equal(actual_output, expected_output)
    expected_output.sum().backward()
    actual_output.sum().backward()
    for (_, left), (_, right) in zip(actual.named_parameters(), expected.named_parameters()):
        torch_assert_equal(left.grad, right.grad)
    assert len(local) == 2
    for name, gradient in local.items():
        torch_assert_equal(gradient, dict(actual.named_parameters())[name].grad)
    values = torch.load(tmp_path / "capture" / "norm-call-0000.pt", weights_only=True)
    torch_assert_equal(values["ids"], ids)
    torch_assert_equal(values["output_gradient"], torch.ones_like(actual_output))
    torch_assert_equal(values["output"], actual_output.detach())
    for handle in handles:
        handle.remove()


def test_gradient_boundary_sync_only_is_observational(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    expected, actual = _Model(), _Model()
    handle = install_norm_backward_sync(actual)
    ids = torch.tensor([[1, 2]])
    expected_output, actual_output = expected(ids), actual(ids)
    torch_assert_equal(actual_output, expected_output)
    assert not calls
    expected_output.sum().backward()
    actual_output.sum().backward()
    assert calls == [actual_output.device]
    for (_, left), (_, right) in zip(actual.named_parameters(), expected.named_parameters()):
        torch_assert_equal(left.grad, right.grad)
    handle.remove()
    actual(ids).sum().backward()
    assert len(calls) == 1


def test_reduction_input_capture_preserves_values_and_restores_hooks():
    expected, actual = _Model(), _Model()

    class Optimizer:
        ipg_buckets = {torch.float32: SimpleNamespace(buffer=[torch.empty(1, dtype=torch.float32)], elements=0)}

        def get_gradient_for_reduction(self, parameter):
            return parameter.grad_accum.to(torch.bfloat16)

        def get_param_gradient_attribute(self, parameter):
            return parameter.grad_accum

        def get_param_comm_dtype(self, parameter):
            return torch.float32

    optimizer = Optimizer()
    original = optimizer.get_gradient_for_reduction
    captures, remove = install_reduction_input_capture(actual, optimizer)
    ids = torch.tensor([[1, 2]])
    expected_output, actual_output = expected(ids), actual(ids)
    torch_assert_equal(actual_output, expected_output)
    expected_output.sum().backward()
    actual_output.sum().backward()
    for (name, parameter), (_, reference_parameter) in zip(actual.named_parameters(), expected.named_parameters()):
        torch_assert_equal(parameter.grad, reference_parameter.grad)
        if name not in captures:
            continue
        parameter.grad_accum = parameter.grad.float().clone()
        result = optimizer.get_gradient_for_reduction(parameter)
        torch_assert_equal(result, original(parameter))
        entry = captures[name]["reduction_inputs"][0]
        assert entry["communication_dtype"] == "torch.float32"
        assert entry["bucket_dtypes"] == ["torch.float32"]
        torch_assert_equal(entry["accumulated"], parameter.grad_accum)
        torch_assert_equal(entry["returned"], result)
        torch_assert_equal(captures[name]["local"][0], parameter.grad)
        result.zero_()
        parameter.grad_accum.zero_()
        parameter.grad.zero_()
        torch_assert_equal(entry["accumulated"], reference_parameter.grad)
        torch_assert_equal(entry["returned"], reference_parameter.grad.bfloat16())
        torch_assert_equal(captures[name]["local"][0], reference_parameter.grad)
    remove()
    assert optimizer.get_gradient_for_reduction == original
    actual(ids).sum().backward()
    assert all(len(values["local"]) == 1 for values in captures.values())


def test_reference_capture_preserves_norm_result_and_gradient_values(tmp_path):
    model = _Model()
    model(torch.tensor([[1, 2]])).sum().backward()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    expected = {name: tensor.clone() for name, tensor in gradients.items()}
    result = {name: float(tensor.norm()) for name, tensor in gradients.items()}
    calls = []

    def original(values, device):
        assert values is gradients
        assert device == "cpu"
        calls.append(True)
        return result

    wrapper = capturing_gradient_norms(original, tmp_path / "reference")
    assert wrapper(gradients, "cpu") is result
    assert calls == [True]
    for name, tensor in gradients.items():
        torch_assert_equal(tensor, expected[name])
    saved = torch.load(tmp_path / "reference" / "gradients.pt", weights_only=True)
    assert len(saved) == 2
    for name, tensor in saved.items():
        torch_assert_equal(tensor, expected[name])


def test_bucket_capture_records_offsets_and_restores_reduction():
    model = _Model()
    parameters = list(model.parameters())
    bucket = SimpleNamespace(
        buffer=[torch.empty(1)],
        elements=sum(parameter.numel() for parameter in parameters),
        index=0,
        params=[(0, index, index) for index in range(len(parameters))],
    )
    calls = []
    sentinel = object()

    class Optimizer:
        ipg_buckets = {torch.float32: bucket}
        bit16_groups = [parameters]

        def get_gradient_for_reduction(self, parameter):
            return parameter.grad

        def reduce_ipg_grads(self, comm_dtype=None):
            calls.append(comm_dtype)
            return sentinel

    optimizer = Optimizer()
    original_getter, original_reduce = optimizer.get_gradient_for_reduction, optimizer.reduce_ipg_grads
    captures, remove = install_reduction_input_capture(model, optimizer)
    assert optimizer.reduce_ipg_grads(comm_dtype=torch.float32) is sentinel
    assert calls == [torch.float32]
    events = next(iter(captures.values()))["bucket_events"]
    assert len(events) == 1
    assert events[0]["parameter_names"] == [name for name, _ in model.named_parameters()]
    assert events[0]["elements"] == bucket.elements
    offset = 0
    for name, parameter in model.named_parameters():
        if name in captures:
            assert events[0]["selected_offsets"][name] == offset
        offset += parameter.numel()
    remove()
    assert optimizer.get_gradient_for_reduction == original_getter
    assert optimizer.reduce_ipg_grads == original_reduce
    optimizer.reduce_ipg_grads()
    assert len(events) == 1


@pytest.mark.parametrize("raise_inside", [False, True])
def test_reference_subprocess_capture_forwards_and_restores(monkeypatch, tmp_path, raise_inside):
    from arctic_platform.correctness import harness

    calls = []
    sentinel = object()

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return sentinel

    original = SimpleNamespace(run=run)
    runner = SimpleNamespace(subprocess=original)
    monkeypatch.setattr(harness, "runner", runner, raising=False)
    command = ["python", "-m", "arctic_platform.correctness.reference", "--batch", "batch.json"]
    environment = {"CUDA_VISIBLE_DEVICES": "3"}
    try:
        with capture_reference_subprocess(tmp_path):
            assert runner.subprocess.run(command, env=environment, capture_output=True, text=True) is sentinel
            assert calls[0] == (
                [
                    "python",
                    "-m",
                    "arctic_platform.correctness.diagnostics.b200_reference_capture",
                    "--batch",
                    "batch.json",
                ],
                {
                    "env": {**environment, "B200_REFERENCE_CAPTURE_DIR": str(tmp_path)},
                    "capture_output": True,
                    "text": True,
                },
            )
            assert command[2] == "arctic_platform.correctness.reference"
            assert environment == {"CUDA_VISIBLE_DEVICES": "3"}
            other = ["python", "-m", "another.module"]
            assert runner.subprocess.run(other, env=environment) is sentinel
            assert calls[1][0] is other
            assert calls[1][1]["env"] is environment
            if raise_inside:
                raise RuntimeError("test restoration")
    except RuntimeError:
        assert raise_inside
    assert runner.subprocess is original


def test_post_metric_capture_runs_only_after_original_and_preserves_values(tmp_path):
    model = _Model()
    model(torch.tensor([[1, 2]])).sum().backward()
    expected = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    result = {"gradient_norms_per_param": {name: float(gradient.norm()) for name, gradient in expected.items()}}
    calls = []

    def original():
        calls.append("metrics")
        return result

    def full_gradient(parameter):
        assert calls[0] == "metrics"
        calls.append("gradient")
        return parameter.grad

    directory = tmp_path / "post-metric"
    assert capture_after_metrics(original, model, full_gradient, directory) is result
    assert calls == ["metrics", "gradient", "gradient"]
    saved = torch.load(directory / "gradients.pt", weights_only=True)
    assert len(saved) == 2
    for name, parameter in model.named_parameters():
        torch_assert_equal(parameter.grad, expected[name])
        if name in saved:
            parameter.grad.zero_()
            torch_assert_equal(saved[name], expected[name])


@pytest.mark.parametrize("case", [None, "gas4"])
def test_gradient_control_preserves_selected_frozen_case(monkeypatch, tmp_path, case):
    from arctic_platform import correctness
    from arctic_platform.correctness.diagnostics import b200_gradient_control

    calls = []
    monkeypatch.setattr(
        correctness, "__main__", SimpleNamespace(main=lambda args: calls.append(args) or 0), raising=False
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA B200")
    arguments = ["gradient-control", "--mode", "control", "--out", str(tmp_path / "control")]
    if case is not None:
        arguments += ["--case", case]
    monkeypatch.setattr(sys, "argv", arguments)
    assert b200_gradient_control.main() == 0
    assert calls == [
        [
            "run",
            "--config",
            str(b200_gradient_control.CONFIG),
            "--test",
            "single-step-grads",
            "--case",
            case or "gas1",
            "--out",
            str(tmp_path / "control" / "report"),
        ]
    ]
