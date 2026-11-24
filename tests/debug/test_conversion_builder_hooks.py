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
"""CPU schema/routing controls; native tiny DCP reads, not capacity or GPU admission."""

import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import save_file
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader

from arctic_platform.debug import lora
from arctic_platform.debug.training import ReducedFixtureMTP
from arctic_platform.model.implementations.qwen35.deepspeed_integration import _build_model_config
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions
from arctic_platform.testing_utils import torch_assert_equal


def _descriptor():
    files = [
        {"path": "config.json", "size_bytes": 0, "sha256": "0" * 64},
        {"path": "model.safetensors.index.json", "size_bytes": 0, "sha256": "1" * 64},
    ]
    return {
        "schema_version": 1,
        "config_id": "cpu-debug-contract",
        "fixture_locator": "cpu-debug-contract",
        "reference_path": "/data-fast/debug-contract-reference",
        "reviewed_hash": "a" * 16,
        "fixture_sha256": (
            hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        ),
        "files": files,
        "source_checkpoint": "test/source",
        "num_hidden_layers": 1,
        "vision_depth": None,
        "materialization": "pretrained-prefix-reduction",
    }


def test_reduced_descriptor_passes_native_options_to_runtime_without_coercion():
    supplied = _descriptor()
    options = Qwen3_5MoeOptions.model_validate({"debug": {"reduced_fixture_mtp": supplied}})
    assert type(options.debug.reduced_fixture_mtp) is ReducedFixtureMTP
    restored = Qwen3_5MoeOptions.model_validate_json(options.model_dump_json())
    runtime = _build_model_config("test/source", 2, 1, "float32", "eager", restored)
    assert runtime.debug.reduced_fixture_mtp == supplied
    assert not runtime.debug.random_init


@pytest.mark.parametrize("change", [{"schema_version": True}, {"unexpected": 1}])
def test_reduced_descriptor_rejects_invalid_native_loader_fields(change):
    supplied = {**_descriptor(), **change}
    with pytest.raises(ValidationError):
        Qwen3_5MoeOptions.model_validate({"debug": {"reduced_fixture_mtp": supplied}})


def test_disabled_construction_observers_return_original_reader_and_do_not_touch_model(monkeypatch):
    monkeypatch.delenv(lora.MODEL_CONSTRUCTION_ENV, raising=False)
    monkeypatch.delenv(lora.MODEL_CONSTRUCTION_SHA_ENV, raising=False)
    monkeypatch.delenv(lora.VERIFY_ENV, raising=False)
    reader, model = object(), object()
    assert lora.maybe_observe_checkpoint_reader(reader, model, {}, None, None) is reader
    assert lora.maybe_observe_model_materialization(model, None, None, None, None) is None


@pytest.mark.parametrize("family", ["qwen35", "glm52"])
def test_native_cpu_dcp_reader_hook_preserves_exact_weights_and_original_source(tmp_path, monkeypatch, family):
    builder = importlib.import_module(f"arctic_platform.model.implementations.{family}.model_builder")
    monkeypatch.delenv(lora.MODEL_CONSTRUCTION_ENV, raising=False)
    monkeypatch.delenv(lora.MODEL_CONSTRUCTION_SHA_ENV, raising=False)
    monkeypatch.delenv(lora.VERIFY_ENV, raising=False)
    source = tmp_path / "native"
    source.mkdir()
    expected = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3), "bias": torch.tensor([7.0, 9.0])}
    save_file(expected, source / "model.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict.fromkeys(expected, "model.safetensors")})
    )
    model = torch.nn.Linear(3, 2, device="meta")
    model.config = SimpleNamespace(tie_word_embeddings=False)
    config = builder.ModelConfig(name=str(source), fsdp_cpu_offload=True)
    dims = object()
    events = []

    def observe(reader, actual_model, state_dict, actual_config, actual_dims):
        assert type(reader) is HuggingFaceStorageReader
        assert actual_model is model and actual_config is config and actual_dims is dims
        assert set(state_dict) == set(expected)
        events.append("reader")
        assert lora.maybe_observe_checkpoint_reader(reader, model, state_dict, config, dims) is reader
        return reader

    monkeypatch.setattr(builder, "maybe_observe_checkpoint_reader", observe)
    if family == "qwen35":
        capture = builder.maybe_capture_reduced_fixture_mtp

        def observe_capture(actual_model, original_source, actual_config):
            assert actual_model is model and original_source == source and actual_config is config
            events.append("capture")
            return capture(actual_model, original_source, actual_config)

        monkeypatch.setattr(builder, "maybe_capture_reduced_fixture_mtp", observe_capture)
    builder.load_dcp_from_hf(model, config, dims)
    assert events == (["capture", "reader"] if family == "qwen35" else ["reader"])
    for name, actual in model.state_dict().items():
        torch_assert_equal(actual, expected[name])


def test_shared_materialization_hook_runs_after_model_configuration_before_allocation(monkeypatch):
    from arctic_platform.model.implementations.moe import deepspeed_integration as shared

    events = []
    model = torch.nn.Linear(3, 2, device="meta")
    config = SimpleNamespace(
        optimization_dtype="float32",
        fused_lm_head_token_chunk_size="disabled",
        fp32_lm_head=False,
        ac=object(),
    )
    dims = SimpleNamespace(ep_enabled=False)
    mesh = object()

    def load(actual_model, actual_config, actual_dims):
        assert actual_model is model and actual_config is config and actual_dims is dims
        events.append("load")
        model.to_empty(device="cpu")

    adapter = SimpleNamespace(
        dtype_map={"float32": torch.float32},
        get_model=lambda *args, **kwargs: model,
        configure_moe_ep_backend=lambda *args: events.append("ep-config"),
        configure_family_backend=lambda *args: events.append("family-config"),
        inject_lm_head=lambda *args, **kwargs: events.append("lm-head"),
        apply_sequence_parallelism=lambda *args: events.append("sp"),
        apply_ac=lambda *args: events.append("ac"),
        load_dcp_from_hf=load,
        reset_runtime_moe_buffers=lambda *args: None,
        shared_expert_type=torch.nn.Linear,
        shared_expert_forward=lambda *args: None,
    )

    def observe(actual_model, actual_adapter, actual_config, actual_dims, actual_mesh):
        assert actual_model is model and actual_adapter is adapter and actual_config is config
        assert actual_dims is dims and actual_mesh is mesh
        assert all(parameter.is_meta for parameter in model.parameters())
        events.append("observe")

    monkeypatch.setattr(shared, "maybe_observe_model_materialization", observe)
    assert shared.setup_model_local_no_train(adapter, config, dims, mesh) is model
    assert events == ["ep-config", "family-config", "lm-head", "sp", "ac", "observe", "load"]
