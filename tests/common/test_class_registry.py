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

"""Class-loss registration, fallback, and callback contract tests."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

import pytest
import torch

from arctic_platform.common.registry import LOSS_CAPABILITIES_ATTR
from arctic_platform.common.registry import LOSS_FNS
from arctic_platform.common.registry import PACKED_LOSS_REDUCTION_ATTR
from arctic_platform.registry import RegistryMeta
from arctic_platform.registry import RegistryValidationError
from arctic_platform.registry import get_registered_class
from arctic_platform.rl.processors import BaseLoss
from arctic_platform.rl.processors import PackedLossReduction
from arctic_platform.rl.processors import resolve_loss
from arctic_platform.rl.processors import resolve_packed_loss_reduction
from arctic_platform.rl.processors.pipeline import run_pipeline


@pytest.fixture(autouse=True)
def _preserve_class_registry():
    original = RegistryMeta._registry
    RegistryMeta._registry = {family: dict(entries) for family, entries in original.items()}
    yield
    RegistryMeta._registry = original


def test_registry_auto_registers_subclasses_and_lists_available_names():
    class ExampleBase(ABC, metaclass=RegistryMeta):
        @classmethod
        def _validate_subclass(cls):
            if cls.__abstractmethods__:
                raise RegistryValidationError("registered examples must be concrete")

        @abstractmethod
        def execute(self):
            raise NotImplementedError

    class FirstExample(ExampleBase):
        name = "first"

        def execute(self):
            return 1

    assert get_registered_class("ExampleBase", "first") is FirstExample
    with pytest.raises(LookupError, match=r"Available registered classes: \['first'\]"):
        get_registered_class("ExampleBase", "missing")


def test_registry_rejects_missing_duplicate_and_abstract_subclasses():
    class ExampleBase(ABC, metaclass=RegistryMeta):
        @classmethod
        def _validate_subclass(cls):
            if cls.__abstractmethods__:
                raise RegistryValidationError("registered examples must be concrete")

        @abstractmethod
        def execute(self):
            raise NotImplementedError

    with pytest.raises(RegistryValidationError, match="must define a 'name'"):

        class MissingName(ExampleBase):
            def execute(self):
                return None

    class FirstExample(ExampleBase):
        name = "duplicate"

        def execute(self):
            return None

    with pytest.raises(RegistryValidationError, match="already registered"):

        class DuplicateExample(ExampleBase):
            name = "duplicate"

            def execute(self):
                return None

    with pytest.raises(RegistryValidationError, match="must be concrete"):

        class AbstractExample(ExampleBase):
            name = "abstract"


def test_loss_resolver_preserves_legacy_fallback_metadata(monkeypatch):
    calls = []

    def function_loss(model_outputs, batch, meta, config, device):
        calls.append((model_outputs, batch, meta, config, device))
        return torch.tensor(2.0), {"legacy": 1}

    reduction = object()

    def reduction_callback(microbatches, config, loss_name):
        assert loss_name == "_legacy_test"
        return reduction

    setattr(function_loss, PACKED_LOSS_REDUCTION_ATTR, reduction_callback)
    setattr(function_loss, LOSS_CAPABILITIES_ATTR, frozenset({"needs_test_output"}))
    monkeypatch.setitem(LOSS_FNS, "_legacy_test", function_loss)

    loss_object = resolve_loss("_legacy_test")
    outputs = {"logits": torch.ones(1), "logprobs": torch.zeros(1)}
    assert loss_object.loss(outputs, {"batch": 1}, {"meta": 2}, {"value": 3}, "cpu")[1] == {"legacy": 1}
    assert calls and loss_object.packed_reduction_callback([{}], {}, "_legacy_test") is reduction
    assert loss_object.has_capability("needs_test_output")
    assert loss_object.is_legacy_adapter_for(function_loss)
    loss_object.output_callback(outputs)
    assert set(outputs) == {"logprobs"}


def test_legacy_adapter_canonicalizes_structurally_compatible_foreign_reduction(monkeypatch):
    @dataclass(frozen=True)
    class ForeignPackedLossReduction:
        loss_scales: tuple[float, ...]
        reporting_weights: tuple[float, ...]
        loss_is_additive: bool

    foreign = ForeignPackedLossReduction(
        loss_scales=(0.25, 0.75),
        reporting_weights=(1.0, 3.0),
        loss_is_additive=False,
    )

    calls = []

    def function_loss(model_outputs, context, config, device):
        calls.append((model_outputs, context, config, device))
        return torch.tensor(2.0), {"four_arg": True}

    def reduction_callback(microbatches, config, loss_name):
        assert len(microbatches) == 2
        assert loss_name == "_foreign_legacy_reduction"
        return foreign

    setattr(function_loss, PACKED_LOSS_REDUCTION_ATTR, reduction_callback)
    monkeypatch.setitem(LOSS_FNS, "_foreign_legacy_reduction", function_loss)

    loss_object = resolve_loss("_foreign_legacy_reduction")
    reduction = resolve_packed_loss_reduction(
        {"loss_fn": "_foreign_legacy_reduction", "config": {}},
        [{}, {}],
        loss_object=loss_object,
    )
    outputs = {"logprobs": torch.zeros(1)}
    loss, metrics = loss_object.loss(
        outputs,
        {"batch": 1, "shared": "batch"},
        {"meta": 2, "shared": "meta"},
        {"value": 3},
        "cpu",
    )

    assert type(reduction) is PackedLossReduction
    assert reduction == PackedLossReduction(
        loss_scales=(0.25, 0.75),
        reporting_weights=(1.0, 3.0),
        loss_is_additive=False,
    )
    assert loss.item() == 2.0
    assert metrics == {"four_arg": True}
    assert calls == [
        (
            outputs,
            {"meta": 2, "shared": "batch", "batch": 1},
            {"value": 3},
            "cpu",
        )
    ]


def test_native_loss_rejects_structurally_compatible_foreign_reduction():
    @dataclass(frozen=True)
    class ForeignPackedLossReduction:
        loss_scales: tuple[float, ...]
        reporting_weights: tuple[float, ...]
        loss_is_additive: bool

    class NativeLoss(BaseLoss):
        name = "_native_foreign_reduction"

        def packed_reduction_callback(self, microbatches, config, loss_fn_name):
            return ForeignPackedLossReduction(
                loss_scales=(1.0,),
                reporting_weights=(1.0,),
                loss_is_additive=False,
            )

        def loss(self, model_outputs, batch, meta, config, device):
            return torch.tensor(0.0), {}

    with pytest.raises(TypeError, match="must return PackedLossReduction"):
        resolve_packed_loss_reduction(
            {"loss_fn": NativeLoss.name, "config": {}},
            [{}],
            loss_object=NativeLoss(),
        )


def test_registered_class_precedes_same_named_function_without_replacing_it(monkeypatch):
    def legacy(*_args):
        return torch.tensor(1.0), {"legacy": True}

    monkeypatch.setitem(LOSS_FNS, "_class_precedence", legacy)

    class ClassLoss(BaseLoss):
        name = "_class_precedence"

        def loss(self, model_outputs, batch, meta, config, device):
            return torch.tensor(2.0), {"class": True}

    resolved = resolve_loss("_class_precedence")
    assert isinstance(resolved, ClassLoss)
    assert LOSS_FNS["_class_precedence"] is legacy
    assert resolved.loss({}, {}, {}, {}, "cpu")[1] == {"class": True}


def test_base_loss_callbacks_are_no_ops_by_default():
    class NoOpLoss(BaseLoss):
        name = "_no_op_callbacks"

        def loss(self, model_outputs, batch, meta, config, device):
            return torch.tensor(0.0), {}

    loss_object = NoOpLoss()
    request = {"value": 1}
    context = {"value": 2}
    config = {"value": 3}
    kwargs = {"value": 4}
    output_keys = ["logits"]
    metrics = {"value": 5}
    outputs = {"logits": torch.ones(1)}

    assert loss_object.batching_callback(request) is None
    assert loss_object.has_capability("missing") is False
    assert loss_object.is_legacy_adapter_for(lambda: None) is False
    assert loss_object.validation_callback(context, config) is None
    assert loss_object.model_forward_callback(kwargs, context, config, output_keys) is None
    assert loss_object.packed_reduction_callback([context], config, loss_object.name) is None
    assert loss_object.metrics_callback([metrics], metrics) is None
    assert loss_object.output_callback(outputs) is None
    assert request == {"value": 1}
    assert kwargs == {"value": 4}
    assert output_keys == ["logits"]
    assert set(outputs) == {"logits"}


def test_pipeline_invokes_class_callbacks_in_execution_order():
    events = []

    class OrderedLoss(BaseLoss):
        name = "_ordered_callbacks"

        def validation_callback(self, context, config):
            events.append("validate")

        def model_forward_callback(self, model_kwargs, context, config, output_keys):
            events.append("model_callback")
            model_kwargs["objective_flag"] = True
            output_keys.append("objective_output")

        def output_callback(self, model_outputs):
            events.append("output_callback")
            model_outputs.pop("logits", None)
            model_outputs.pop("objective_output", None)

        def loss(self, model_outputs, batch, meta, config, device):
            events.append("loss")
            return model_outputs["objective_output"].sum(), {}

    class Engine:
        global_rank = 0

        def train(self):
            pass

        def __call__(self, **kwargs):
            events.append("model")
            assert kwargs["objective_flag"] is True
            return {
                "logits": torch.ones(1, 2, 3),
                "objective_output": torch.ones(1, requires_grad=True),
            }

        def backward(self, loss, scale_wrt_gas=False):
            events.append("backward")
            assert scale_wrt_gas is False

    result = run_pipeline(
        Engine(),
        (),
        {"input_ids": torch.ones(1, 2, dtype=torch.long)},
        {"cu_seqlens": torch.tensor([0, 2], dtype=torch.int32)},
        {"loss_fn": "_ordered_callbacks", "post": [], "config": {}},
        "cpu",
        backward=True,
        pack=False,
    )

    assert events == [
        "validate",
        "model_callback",
        "model",
        "loss",
        "output_callback",
        "backward",
    ]
    assert result == {"batch": {}, "metrics": {}, "avg_loss": 1.0}


def test_packed_pipeline_reuses_one_stateful_loss_for_reduction_and_callbacks():
    from arctic_platform.rl.processors.packed_reduction import local_mean_packed_loss_reduction

    events = []
    instances = []

    class StatefulPackedLoss(BaseLoss):
        name = "_stateful_packed_callbacks"

        def __init__(self):
            instances.append(self)
            self.ready = False

        def packed_reduction_callback(self, microbatches, config, loss_fn_name):
            assert loss_fn_name == self.name
            self.ready = True
            events.append("reduction")
            return local_mean_packed_loss_reduction(
                [float(microbatch["loss_mask"].sum()) for microbatch in microbatches]
            )

        def validation_callback(self, context, config):
            assert self.ready
            events.append("validation")

        def model_forward_callback(self, model_kwargs, context, config, output_keys):
            assert self.ready
            events.append("model_callback")
            output_keys.append("objective_output")

        def loss(self, model_outputs, batch, meta, config, device):
            assert self.ready
            events.append("loss")
            return -model_outputs["objective_output"].mean(), {}

        def output_callback(self, model_outputs):
            assert self.ready
            events.append("output")
            model_outputs.pop("objective_output")

    class Engine:
        global_rank = 0

        def __init__(self):
            self.parameter = torch.tensor(-2.0, requires_grad=True)

        def set_gradient_accumulation_boundary(self, _boundary):
            pass

        def train(self):
            pass

        def __call__(self, input_ids, **_kwargs):
            events.append("model")
            return {"objective_output": self.parameter.expand_as(input_ids)}

        def backward(self, loss, scale_wrt_gas=False):
            assert scale_wrt_gas is False
            events.append("backward")
            loss.backward()

    engine = Engine()
    result = run_pipeline(
        engine,
        (),
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.ones(2, 2, dtype=torch.long),
            "loss_mask": torch.ones(2, 2),
        },
        {"pad_token_id": 0},
        {"loss_fn": StatefulPackedLoss.name, "post": [], "config": {}},
        "cpu",
        backward=True,
        pack=True,
        max_tokens_per_mb=2,
    )

    assert len(instances) == 1
    assert result["avg_loss"] == 2.0
    assert engine.parameter.grad.item() == -1.0
    assert events == [
        "reduction",
        "validation",
        "model_callback",
        "model",
        "loss",
        "output",
        "backward",
        "validation",
        "model_callback",
        "model",
        "loss",
        "output",
        "backward",
    ]
