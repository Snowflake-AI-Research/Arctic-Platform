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

"""Non-master ranks finish each EP export collective before queuing the next (CPU, faked collectives)."""

import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from arctic_platform.model import iter_lora_weights
from arctic_platform.model import weight_export
from arctic_platform.model.implementations.moe import deepspeed_integration
from arctic_platform.model.implementations.moe import vllm_weights

_GROUP = object()


class _ExportableModel(nn.Module):
    @staticmethod
    def convert_layer_to_hf(layer_state, layer_index):
        del layer_index
        return layer_state


class Qwen3_5MoeForExportTest(_ExportableModel):
    pass


class _LoraModel(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self._parameters_for_test = list(parameters)
        self.peft_config = {"default": SimpleNamespace(r=2)}

    def named_parameters(self, *args, **kwargs):
        del args, kwargs
        return iter(self._parameters_for_test)


def _model(*shapes, cls=_ExportableModel):
    model = cls()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    experts = nn.Module()
    for index, shape in enumerate(shapes):
        parameter = nn.Parameter(torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape))
        parameter.group_name = "ep_size_2"
        parameter.allreduce = False
        experts.register_parameter(f"w{index}", parameter)
    model.model.layers[0].mlp = nn.Module()
    model.model.layers[0].mlp.experts = experts
    return model


def _ep_sharded(shape):
    return SimpleNamespace(
        data=torch.ones(shape), requires_grad=True, group_name="ep_size_2", allreduce=False, numel=lambda: 1
    )


@pytest.fixture
def events(monkeypatch):
    recorded: list[tuple[str, tuple[int, ...]]] = []
    groups_module = ModuleType("deepspeed.utils.groups")
    groups_module._get_expert_parallel_group = lambda _name: _GROUP
    utils_module = ModuleType("deepspeed.utils")
    utils_module.__path__ = []
    utils_module.groups = groups_module
    deepspeed_module = ModuleType("deepspeed")
    deepspeed_module.__path__ = []
    deepspeed_module.utils = utils_module
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed_module)
    monkeypatch.setitem(sys.modules, "deepspeed.utils", utils_module)
    monkeypatch.setitem(sys.modules, "deepspeed.utils.groups", groups_module)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1])

    def gather(tensor, gather_list=None, dst=0, group=None):
        assert group is _GROUP and dst == 0
        recorded.append(("gather", tuple(tensor.shape)))
        if gather_list is not None:
            for output in gather_list:
                output.copy_(tensor)

    def all_gather(outputs, tensor, group=None):
        assert group is _GROUP
        recorded.append(("all_gather", tuple(tensor.shape)))
        for output in outputs:
            output.copy_(tensor)

    monkeypatch.setattr(torch.distributed, "gather", gather)
    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    monkeypatch.setattr(
        weight_export, "wait_for_export_collective", lambda tensor: recorded.append(("wait", tuple(tensor.shape)))
    )
    return recorded


def _set_rank(monkeypatch, rank):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: rank)


def test_hf_export_non_master_waits_after_every_gather(monkeypatch, events):
    _set_rank(monkeypatch, 1)

    assert list(deepspeed_integration.build_iter_full_hf_weights(_model((4, 2), (3, 2)))()) == []

    # HF export gathers bounded flat byte chunks into CPU storage.
    assert events == [("all_gather", (32,)), ("wait", (32,)), ("all_gather", (24,)), ("wait", (24,))]


def test_hf_export_non_master_waits_after_every_row_slice(monkeypatch, events):
    _set_rank(monkeypatch, 1)
    monkeypatch.setattr(deepspeed_integration, "_EXPORT_GATHER_BYTES", 16)

    assert list(deepspeed_integration.build_iter_full_hf_weights(_model((3, 2)))()) == []

    assert events == [("all_gather", (8,)), ("wait", (8,))] * 3


@pytest.mark.parametrize("gather_bytes", (1 << 30, 16))
def test_hf_export_master_does_not_wait_and_still_receives_every_shard(monkeypatch, events, gather_bytes):
    _set_rank(monkeypatch, 0)
    monkeypatch.setattr(deepspeed_integration, "_EXPORT_GATHER_BYTES", gather_bytes)
    model = _model((3, 2))

    exported = dict(deepspeed_integration.build_iter_full_hf_weights(model)())

    assert all(kind == "all_gather" for kind, _ in events)
    expected = model.model.layers[0].mlp.experts.w0.detach()
    torch.testing.assert_close(exported["model.layers.0.mlp.experts.w0"], torch.cat([expected, expected]))


def test_vllm_export_non_master_waits_after_every_all_gather(monkeypatch, events):
    _set_rank(monkeypatch, 1)

    assert list(vllm_weights.build_iter_full_vllm_weights(_model((4, 2), (3, 2)))()) == []

    assert events == [("all_gather", (4, 2)), ("wait", (4, 2)), ("all_gather", (3, 2)), ("wait", (3, 2))]


def test_qwen3_5_moe_vllm_export_non_master_waits_after_every_all_gather(monkeypatch, events):
    qwen35 = pytest.importorskip("arctic_platform.model.implementations.qwen35.deepspeed_integration")
    _set_rank(monkeypatch, 1)

    model = _model((4, 2), (3, 2), cls=Qwen3_5MoeForExportTest)
    assert list(qwen35._build_iter_full_vllm_weights(model)()) == []

    assert events == [("all_gather", (4, 2)), ("wait", (4, 2)), ("all_gather", (3, 2)), ("wait", (3, 2))]


def test_lora_export_non_master_waits_after_every_ep_all_gather(monkeypatch, events):
    _set_rank(monkeypatch, 1)

    def expert_loras(_model):
        yield "model.layers.0.mlp.experts", "w1", "A", _ep_sharded((4, 2)), 2

    monkeypatch.setattr(weight_export, "_iter_param_wrapper_expert_loras", expert_loras)
    model = _LoraModel([("base_model.model.layers.0.q_proj.lora_A.default.weight", _ep_sharded((3, 2)))])

    assert list(iter_lora_weights(model, is_master=False, is_zero3=False)) == []

    assert [kind for kind, _ in events] == ["all_gather", "wait", "all_gather", "wait"]


def test_gather_ep_shards_waits_for_its_all_gather(events):
    shards = weight_export._gather_ep_shards(_ep_sharded((3, 2)))

    assert len(shards) == 2
    assert events == [("all_gather", (3, 2)), ("wait", (3, 2))]


def test_wait_for_export_collective_synchronizes_the_cuda_stream(monkeypatch):
    synchronized = []

    class _Stream:
        def __init__(self, device):
            self.device = device

        def synchronize(self):
            synchronized.append(self.device)

    monkeypatch.setattr(torch.cuda, "current_stream", _Stream)

    weight_export.wait_for_export_collective(SimpleNamespace(is_cuda=True, device="cuda:3"))

    assert synchronized == ["cuda:3"]


def test_wait_for_export_collective_is_a_no_op_off_gpu(monkeypatch):
    def unexpected(_device):
        raise AssertionError("a CPU tensor must not touch a CUDA stream")

    monkeypatch.setattr(torch.cuda, "current_stream", unexpected)

    weight_export.wait_for_export_collective(torch.ones(2))
