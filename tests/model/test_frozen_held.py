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
"""The weight export and state dict of frozen weights a model holds outside their parameters (moe/frozen_held.py), on
CPU with a holding that keeps each layer's frozen weight in a side table."""

import contextlib

import pytest
import torch
from torch import nn

from arctic_platform.model.implementations.moe import frozen_held
from arctic_platform.model.weight_export import WeightExportContract
from arctic_platform.model.weight_export import export_named_parameters
from arctic_platform.model.weight_export import materialized_parameters
from arctic_platform.model.weight_export import register_weight_export
from arctic_platform.model.weight_export import save_hf_pretrained
from arctic_platform.model.weight_export import weight_export_contract


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for i in range(3):
            layer = nn.Module()
            layer.frozen = nn.Parameter(torch.full((4,), float(i + 1)), requires_grad=False)
            layer.lora = nn.Parameter(torch.full((2,), -float(i + 1)))
            self.model.layers.append(layer)
        self.norm = nn.Parameter(torch.ones(4), requires_grad=False)


class _SideTable:
    """Holds each layer's ``frozen`` in a table, its parameter empty, and logs what it materializes."""

    def __init__(self, model: _Model, log: list) -> None:
        self.layers = list(model.model.layers)
        self.table = [layer.frozen.data.clone() for layer in self.layers]
        self.whole = 0
        self.log = log
        self.empty_all()

    def empty_all(self):
        for layer in self.layers:
            layer.frozen.data = torch.empty(0)

    @property
    def held(self):
        return not self.whole

    @contextlib.contextmanager
    def materialize(self, tensors):
        layers = [i for i, layer in enumerate(self.layers) if any(t is layer.frozen for t in tensors)]
        self.log.append(("materialize", tuple(layers)))
        for i in layers:
            self.layers[i].frozen.data = self.table[i].clone()
        try:
            yield
        finally:
            for i in layers:
                self.layers[i].frozen.data = torch.empty(0)

    @contextlib.contextmanager
    def whole_on_host(self):
        for layer, values in zip(self.layers, self.table):
            layer.frozen.data = values
        self.whole += 1
        try:
            yield
        finally:
            self.whole -= 1
            self.empty_all()


def _layer_tensors(named: list, i: int) -> list:
    return [tensor for name, tensor in named if name.startswith(f"model.layers.{i}.")]


def test_the_export_lists_the_models_own_names_and_materializes_the_held_ones():
    model, log = _Model(), []
    want = {name: p.detach().clone() for name, p in model.named_parameters()}
    frozen_held.hold_frozen_weights(model, _SideTable(model, log))
    named = list(export_named_parameters(model))
    assert [name for name, _ in named] == list(want), "the model's own names and order"
    for i in range(3):  # one export unit per decoder layer, as the MoE families' builders export
        with materialized_parameters(model, _layer_tensors(named, i)):
            for name, tensor in named:
                if name.startswith(f"model.layers.{i}."):
                    assert torch.equal(tensor, want[name]), name
    assert log == [("materialize", (0,)), ("materialize", (1,)), ("materialize", (2,))]
    assert all(layer.frozen.numel() == 0 for layer in model.model.layers), "held again after the export"


class _Logger:
    """A second holding of the same model that holds nothing and logs its calls."""

    held = False

    def __init__(self, log: list) -> None:
        self.log = log

    def materialize(self, tensors):
        self.log.append(("second", len(tensors)))
        return contextlib.nullcontext()

    def whole_on_host(self):
        self.log.append(("second", "whole"))
        return contextlib.nullcontext()


def test_two_holdings_of_one_model_materialize_together_in_registration_order():
    model, log = _Model(), []
    frozen_held.hold_frozen_weights(model, _SideTable(model, log))
    frozen_held.hold_frozen_weights(model, _Logger(log))
    named = list(export_named_parameters(model))
    for i in range(3):
        with materialized_parameters(model, _layer_tensors(named, i)):
            assert model.model.layers[i].frozen.numel() == 4
    assert log == [("materialize", (0,)), ("second", 2), ("materialize", (1,)), ("second", 2)] + [
        ("materialize", (2,)),
        ("second", 2),
    ]
    with frozen_held.whole_frozen_weights(model):
        assert model.model.layers[0].frozen.numel() == 4
    assert log[-1] == ("second", "whole")


def test_the_state_dict_is_refused_outside_whole_frozen_weights_and_whole_inside():
    model, reference = _Model(), _Model()
    table = _SideTable(model, [])
    frozen_held.hold_frozen_weights(model, table)
    with pytest.raises(RuntimeError, match="holds frozen weights"):
        model.state_dict()
    with pytest.raises(RuntimeError, match="holds frozen weights"):
        model.load_state_dict(reference.state_dict())
    wrapper = nn.Sequential(model)  # a wrapper (PEFT's) of the model the holding is registered with
    with frozen_held.whole_frozen_weights(wrapper):
        state = wrapper.state_dict()
        assert all(torch.equal(state[f"0.{k}"], v) for k, v in reference.state_dict().items())
        changed = {k: v + 1 for k, v in state.items()}
        wrapper.load_state_dict(changed)
    assert model.model.layers[1].frozen.numel() == 0
    assert torch.equal(table.table[1], reference.model.layers[1].frozen + 1), "the load wrote the holding"


def test_a_family_builder_that_reads_parameters_directly_is_refused():
    from arctic_platform.model.implementations.moe.deepspeed_integration import build_iter_full_hf_weights

    def own_builder(model):
        return lambda **kwargs: iter(())

    model = _Model()
    register_weight_export(model, WeightExportContract(hf=build_iter_full_hf_weights, vllm=own_builder))
    frozen_held.hold_frozen_weights(model, _SideTable(model, []))
    contract = weight_export_contract(model)
    assert contract.hf is build_iter_full_hf_weights
    with pytest.raises(NotImplementedError, match="own_builder reads the parameters directly"):
        contract.vllm(model)


def test_a_model_holding_other_parameters_and_a_one_rank_hf_save_are_refused(tmp_path):
    from arctic_platform.model.weight_export import HeldParameters
    from arctic_platform.model.weight_export import register_held_parameters

    other = _Model()
    register_held_parameters(
        other, HeldParameters(named_parameters=lambda m: m.named_parameters(), materialize=contextlib.nullcontext)
    )
    with pytest.raises(RuntimeError, match="already holds parameters"):
        frozen_held.hold_frozen_weights(other, _SideTable(other, []))
    model = _Model()
    frozen_held.hold_frozen_weights(model, _SideTable(model, []))
    with pytest.raises(NotImplementedError, match="save_hf_pretrained writes on one rank"):
        save_hf_pretrained(model, str(tmp_path))
