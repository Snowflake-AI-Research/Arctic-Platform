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

"""Post-metric source inspection must preserve metrics and gradient buffers."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from arctic_platform.correctness.diagnostics.dp_gradient_sources import observe_gradient_sources
from arctic_platform.correctness.diagnostics.dp_gradient_sources import optimizer_accumulated_gradient
from arctic_platform.testing_utils import torch_assert_equal


def _model():
    model = nn.Module()
    model.model = nn.Module()
    model.model.norm = nn.Linear(2, 1, bias=False)
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].linear_attn = nn.Module()
    model.model.layers[0].linear_attn.out_proj = nn.Linear(2, 2, bias=False)
    model.unselected = nn.Linear(2, 2, bias=False)
    return model


@pytest.mark.parametrize("local_present", [True, False])
def test_sources_are_observed_after_metrics_without_mutating_buffers(tmp_path, local_present):
    model = _model()
    calls = []
    originals = {}
    parameters = list(model.parameters())
    buffers = [torch.full_like(parameter, 0.5) for parameter in parameters]
    optimizer = SimpleNamespace(bf16_groups=[parameters], fp32_groups_gradients=[buffers])
    for name, parameter in model.named_parameters():
        accumulated = optimizer_accumulated_gradient(optimizer, parameter)
        parameter.grad = torch.ones_like(parameter) if local_present else None
        originals[name] = (
            None if parameter.grad is None else parameter.grad.clone(),
            accumulated.clone(),
        )
    result = {"gradient_norms_per_param": {"sentinel": 9.0}}

    def original():
        assert not calls
        calls.append(("metrics", None))
        return result

    def full_gradient(parameter):
        assert calls[0] == ("metrics", None)
        calls.append(
            (
                "ordinary",
                next(name for name, candidate in model.named_parameters() if candidate is parameter),
            )
        )
        return parameter.grad if parameter.grad is not None else optimizer_accumulated_gradient(optimizer, parameter)

    destination = tmp_path / "gas1" / "rank-0.json"
    assert observe_gradient_sources(original, model, full_gradient, destination, optimizer=optimizer) is result
    data = json.loads(destination.read_text())
    assert data["metrics"] == result
    assert len(data["observations"]) == 2
    assert all("unselected" not in name for _, name in calls if name is not None)
    for item in data["observations"]:
        assert item["local_gradient_present"] is local_present
        assert item["ordinary_is_local"] is local_present
        assert item["ordinary_accumulated_delta"] > 0 if local_present else item["ordinary_accumulated_delta"] == 0
    for name, parameter in model.named_parameters():
        local, accumulated = originals[name]
        if local_present:
            torch_assert_equal(parameter.grad, local)
        else:
            assert parameter.grad is None
        torch_assert_equal(optimizer_accumulated_gradient(optimizer, parameter), accumulated)


def test_failed_original_metrics_prevent_source_reads_and_artifacts(tmp_path):
    def original():
        raise RuntimeError("metric failed")

    def getter(parameter):
        pytest.fail("gradient getter must not precede successful ordinary metrics")

    destination = tmp_path / "rank-0.json"
    with pytest.raises(RuntimeError, match="metric failed"):
        observe_gradient_sources(original, _model(), getter, destination)
    assert not destination.exists()


@pytest.mark.parametrize("mapping_present", [True, False])
def test_optimizer_buffer_observation_does_not_call_mapped_collectives(tmp_path, mapping_present):
    model = _model()
    parameters = list(model.parameters())
    accumulated = [torch.full_like(parameter, 0.5).flatten() for parameter in parameters]
    optimizer = SimpleNamespace(bf16_groups=[parameters], fp32_groups_gradients=[accumulated])
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
        parameter._hp_mapping = object() if mapping_present else None

        def mapped_getter():
            pytest.fail("observation must not issue mapped-gradient collectives")

        parameter.get_full_hp_grad = mapped_getter
    result = {"gradient_norms_per_param": {"sentinel": 9.0}}
    destination = tmp_path / "rank-0.json"
    observed = observe_gradient_sources(
        lambda: result,
        model,
        lambda parameter: parameter.grad,
        destination,
        optimizer=optimizer,
    )
    assert observed is result
    data = json.loads(destination.read_text())
    for item in data["observations"]:
        assert item["accumulated_dtype"] == "torch.float32"
        assert item["ordinary_accumulated_delta"] > 0
    for parameter, buffer in zip(parameters, accumulated, strict=True):
        torch_assert_equal(parameter.grad, torch.ones_like(parameter))
        torch_assert_equal(buffer, torch.full_like(parameter, 0.5).flatten())


def test_optimizer_buffer_lookup_uses_identity_across_groups():
    first = nn.Parameter(torch.ones(2, 2))
    second = nn.Parameter(torch.ones(2, 2))
    first_buffer = torch.full((4,), 0.25)
    second_buffer = torch.full((4,), 0.5)
    optimizer = SimpleNamespace(
        bf16_groups=[[first], [second]],
        fp32_groups_gradients=[[first_buffer], [second_buffer]],
    )
    assert optimizer_accumulated_gradient(optimizer, first).data_ptr() == first_buffer.data_ptr()
    assert optimizer_accumulated_gradient(optimizer, second).data_ptr() == second_buffer.data_ptr()
    assert optimizer_accumulated_gradient(optimizer, nn.Parameter(torch.ones(2, 2))) is None
    assert optimizer_accumulated_gradient(object(), first) is None


def test_observer_selects_checkpoint_wrapped_language_parameters_not_visual_norm(tmp_path):
    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = _model().model
    layer = model.model.language_model.layers[0]
    wrapper = nn.Module()
    wrapper._checkpoint_wrapped_module = layer
    model.model.language_model.layers[0] = wrapper
    model.model.visual = nn.Module()
    model.model.visual.merger = nn.Module()
    model.model.visual.merger.norm = nn.Linear(2, 1, bias=False)
    parameters = list(model.parameters())
    optimizer = SimpleNamespace(
        bf16_groups=[parameters],
        fp32_groups_gradients=[[torch.ones_like(parameter) for parameter in parameters]],
    )
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    destination = tmp_path / "rank-0.json"
    observe_gradient_sources(
        lambda: {},
        model,
        lambda parameter: parameter.grad,
        destination,
        optimizer=optimizer,
    )
    names = {item["parameter"] for item in json.loads(destination.read_text())["observations"]}
    assert names == {
        "model.language_model.layers.0._checkpoint_wrapped_module.linear_attn.out_proj.weight",
        "model.language_model.norm.weight",
    }
