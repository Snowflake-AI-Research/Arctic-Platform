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

"""CPU-only coverage for module-declared persistent buffers in HF and vLLM exports."""

import pytest
import torch
from torch import nn

from arctic_platform.model.implementations.moe.vllm_weights import named_weight_sync_tensors

_BUFFER_NAMES = (
    "layer_multipliers",
    "ngram_heads_vocab_sizes",
    "ngram_heads_offsets",
)


def _ngram_module(*, persistent=True):
    module = nn.Module()
    module._dss_weight_export_buffers = frozenset(_BUFFER_NAMES)
    module.weight = nn.Parameter(torch.ones(2, 2))
    for index, name in enumerate(_BUFFER_NAMES):
        module.register_buffer(
            name,
            torch.tensor([2**53 + index, 2**53 + index + 1], dtype=torch.int64),
            persistent=persistent,
        )
    return module


def _assert_same_entries(actual, expected):
    assert [name for name, _ in actual] == [name for name, _ in expected]
    for (_, actual_tensor), (_, expected_tensor) in zip(actual, expected, strict=True):
        assert actual_tensor is expected_tensor
        assert actual_tensor.dtype == expected_tensor.dtype
        assert actual_tensor.device == expected_tensor.device


def test_complete_persistent_ngram_buffers_preserve_identity_dtype_and_order():
    module = _ngram_module()
    original = {name: tensor.clone() for name, tensor in module.named_buffers()}

    exported = list(named_weight_sync_tensors(module))

    _assert_same_entries(exported, list(module.named_parameters()) + list(module.named_buffers()))
    for name, tensor in exported[1:]:
        assert tensor.dtype == torch.int64
        assert tensor.tolist() == original[name].tolist()
        assert tensor.data_ptr() == getattr(module, name).data_ptr()


def test_nested_checkpoint_wrapped_module_names_are_unchanged():
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0]._checkpoint_wrapped_module = nn.Module()
    model.model.layers[0]._checkpoint_wrapped_module.ngram_embedding = _ngram_module()
    model.model.layers[0].register_buffer("expert_bias", torch.zeros(2))

    exported = list(named_weight_sync_tensors(model))

    _assert_same_entries(exported, list(model.named_parameters()) + list(model.named_buffers()))
    assert exported[1][0] == "model.layers.0.expert_bias"
    assert exported[2][0] == "model.layers.0._checkpoint_wrapped_module.ngram_embedding.layer_multipliers"


@pytest.mark.parametrize("buffer_name", _BUFFER_NAMES)
def test_nonpersistent_ngram_buffer_is_not_exported(buffer_name):
    module = _ngram_module()
    module.register_buffer(buffer_name, getattr(module, buffer_name), persistent=False)

    exported = list(named_weight_sync_tensors(module))

    assert buffer_name not in dict(exported)
    _assert_same_entries(
        exported,
        list(module.named_parameters())
        + [(name, buffer) for name, buffer in module.named_buffers() if name != buffer_name],
    )


def test_persistence_is_checked_on_each_owning_module():
    model = nn.Module()
    model.persistent = _ngram_module()
    model.transient = _ngram_module(persistent=False)

    exported = list(named_weight_sync_tensors(model))

    _assert_same_entries(
        exported,
        list(model.named_parameters())
        + [(f"persistent.{name}", buffer) for name, buffer in model.persistent.named_buffers()],
    )


def test_only_exact_registered_buffer_names_are_added():
    module = nn.Module()
    module._dss_weight_export_buffers = frozenset((*_BUFFER_NAMES, "not_registered"))
    module.layer_multipliers = nn.Parameter(torch.ones(1))
    module.not_registered = torch.ones(1, dtype=torch.int64)
    module.register_buffer("other_layer_multipliers", torch.ones(1, dtype=torch.int64))
    module.register_buffer("ngram_heads_offsets_extra", torch.ones(1, dtype=torch.int64))
    module.register_buffer("tokens_per_expert", torch.ones(1, dtype=torch.int64))
    module.register_buffer("ngram_heads_vocab_sizes", None)
    module.child = _ngram_module()

    exported = list(named_weight_sync_tensors(module))

    _assert_same_entries(
        exported,
        list(module.named_parameters()) + [(f"child.{name}", tensor) for name, tensor in module.child.named_buffers()],
    )
    assert [name for name, _ in exported].count("layer_multipliers") == 1


def test_other_family_buffers_and_existing_expert_bias_behavior_are_unchanged():
    model = nn.Module()
    model.register_buffer("expert_bias", torch.ones(2))
    model.mlp = nn.Module()
    model.mlp.weight = nn.Parameter(torch.ones(2))
    model.mlp.register_buffer("expert_bias", torch.ones(2), persistent=False)
    model.mlp.register_buffer("tokens_per_expert", torch.ones(2))
    model.mlp.register_buffer("inv_freq", torch.ones(2))

    _assert_same_entries(
        list(named_weight_sync_tensors(model)),
        list(model.named_parameters()) + [("mlp.expert_bias", model.mlp.expert_bias)],
    )


def test_shared_modules_and_buffers_keep_named_buffers_deduplication():
    model = nn.Module()
    model.first = _ngram_module()
    model.alias = model.first
    model.other = nn.Module()
    model.other.register_buffer("layer_multipliers", model.first.layer_multipliers)

    _assert_same_entries(
        list(named_weight_sync_tensors(model)),
        list(model.named_parameters()) + list(model.named_buffers()),
    )


@pytest.mark.parametrize("nested", (False, True))
def test_same_named_buffers_without_module_declaration_are_not_exported(nested):
    module = _ngram_module()
    del module._dss_weight_export_buffers
    model = nn.Module()
    model.child = module
    if not nested:
        model = module

    _assert_same_entries(list(named_weight_sync_tensors(model)), list(model.named_parameters()))


@pytest.mark.parametrize("persistent", (False, True))
def test_other_modules_can_declare_their_own_persistent_buffer_names(persistent):
    module = nn.Module()
    module._dss_weight_export_buffers = frozenset({"custom_buffer"})
    module.register_buffer("custom_buffer", torch.tensor([7], dtype=torch.int64), persistent=persistent)
    module.register_buffer("unlisted_buffer", torch.ones(1))

    expected = [("custom_buffer", module.custom_buffer)] if persistent else []
    _assert_same_entries(list(named_weight_sync_tensors(module)), expected)


def test_parent_declaration_does_not_opt_in_child_buffers():
    model = nn.Module()
    model._dss_weight_export_buffers = frozenset(_BUFFER_NAMES)
    model.child = _ngram_module()
    del model.child._dss_weight_export_buffers

    _assert_same_entries(list(named_weight_sync_tensors(model)), list(model.named_parameters()))


def test_qwen_ngram_module_declares_all_persistent_export_buffers():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import Qwen4ExpTextNGramEmbeddingPrimeRL

    config = Qwen4ExpTextConfig(
        vocab_size=32,
        ngram_size=3,
        heads_per_ngram=1,
        ngram_vocab_size_base=11,
        make_ngram_vocab_size_divisible_by=4,
        eos_token_id=1,
    )
    module = Qwen4ExpTextNGramEmbeddingPrimeRL(config, embedding_dim=8, layer_idx=0)

    assert module._dss_weight_export_buffers == frozenset(_BUFFER_NAMES)
    assert {name for name, _ in module.named_buffers()} == set(_BUFFER_NAMES)
    _assert_same_entries(
        list(named_weight_sync_tensors(module)),
        list(module.named_parameters()) + list(module.named_buffers()),
    )


@pytest.mark.parametrize(("weight_format", "checkpoint_export"), (("hf", True), ("hf", False), ("vllm", False)))
def test_checkpoint_export_keeps_persistent_buffers_and_weight_sync_omits_them(
    monkeypatch, weight_format, checkpoint_export
):
    from arctic_platform.model.implementations.moe.deepspeed_integration import build_iter_full_hf_weights
    from arctic_platform.model.implementations.moe.vllm_weights import build_iter_full_vllm_weights

    class ExportableModel(nn.Module):
        @staticmethod
        def convert_layer_to_hf(layer_state, layer_index):
            del layer_index
            return layer_state

    model = ExportableModel()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    layer = model.model.layers[0]
    layer._checkpoint_wrapped_module = nn.Module()
    layer._checkpoint_wrapped_module.ngram_embedding = _ngram_module()
    layer.transient = _ngram_module(persistent=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    build_iterator = build_iter_full_hf_weights if weight_format == "hf" else build_iter_full_vllm_weights

    iterator = build_iterator(model)
    exported = list(iterator(include_skipped=True) if checkpoint_export else iterator())
    expected = list(model.named_parameters()) + [
        (name, tensor) for name, tensor in model.named_buffers() if checkpoint_export and ".transient." not in name
    ]

    assert [name for name, _ in exported] == [name.replace("._checkpoint_wrapped_module", "") for name, _ in expected]
    for (_, actual_tensor), (_, expected_tensor) in zip(exported, expected, strict=True):
        assert actual_tensor.dtype == expected_tensor.dtype
        assert actual_tensor.device == expected_tensor.device
        assert actual_tensor.data_ptr() == expected_tensor.data_ptr()
        assert actual_tensor.tolist() == expected_tensor.tolist()
