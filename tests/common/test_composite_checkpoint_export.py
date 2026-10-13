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

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from arctic_platform.common.deepspeed_worker import DeepSpeedWorker
from arctic_platform.common.deepspeed_worker import _canonical_hf_export_state_dict
from arctic_platform.common.deepspeed_worker import _copy_source_sidecars
from arctic_platform.common.deepspeed_worker import _model_full_hf_export_state_dict
from arctic_platform.common.deepspeed_worker import restore_source_weight_layout
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal


def _write_model(path: Path, tensors: dict[str, torch.Tensor], config: dict) -> None:
    path.mkdir(parents=True)
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    (path / "config.json").write_text(json.dumps(config))


def test_composite_export_restores_source_layout_and_assets(tmp_path: Path) -> None:
    source = tmp_path / "source"
    exported = tmp_path / "exported"
    source_tensors = {
        "model.language_model.layer.weight": torch.zeros(2),
        "model.visual.layer.weight": torch.arange(3.0),
        "model.visual.other.weight": torch.arange(4.0),
    }
    _write_model(source, source_tensors, {"model_type": "composite", "text_config": {"model_type": "text"}})
    (source / "tokenizer.json").write_text("{}")
    trained = {
        "model.language_model.layer.weight": torch.ones(2),
        "model.visual.layer.weight": torch.full((3,), 9.0),
        "visual.other.weight": torch.full((4,), 9.0),
    }
    _write_model(
        exported,
        trained,
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "text_config": {"model_type": "qwen3_5_moe_text"},
        },
    )

    assert restore_source_weight_layout(str(source), str(exported))
    _copy_source_sidecars(str(source), str(exported))

    index = json.loads((exported / "model.safetensors.index.json").read_text())
    tensors = {}
    for name, shard in index["weight_map"].items():
        with safe_open(exported / shard, framework="pt") as handle:
            tensors[name] = handle.get_tensor(name)
    assert set(tensors) == {"model.layer.weight"}
    assert torch.equal(tensors["model.layer.weight"], trained["model.language_model.layer.weight"])
    assert "model.visual.layer.weight" not in tensors
    assert "model.visual.other.weight" not in tensors
    exported_config = json.loads((exported / "config.json").read_text())
    assert exported_config == {
        "model_type": "qwen3_5_moe_text",
        "architectures": ["Qwen3_5MoeForCausalLM"],
    }
    assert (exported / "tokenizer.json").is_file()


def test_model_full_hf_export_iterator_runs_on_nonzero_rank() -> None:
    calls = []

    def iter_weights():
        calls.append(True)
        yield "model.layer.weight", torch.ones(2)

    model = SimpleNamespace(_iter_full_hf_weights=iter_weights)

    assert _model_full_hf_export_state_dict(model, rank=1) is None
    assert calls == [True]


def test_model_full_hf_export_iterator_returns_rank_zero_weights() -> None:
    tensor = torch.ones(2)
    model = SimpleNamespace(_iter_full_hf_weights=lambda: iter((("model.layer.weight", tensor),)))

    assert _model_full_hf_export_state_dict(model, rank=0) == {"model.layer.weight": tensor}


def test_nonzero_export_rank_returns_without_waiting_on_a_barrier(tmp_path: Path) -> None:
    model = SimpleNamespace(named_parameters=lambda: [])
    worker = SimpleNamespace(rank=1, engine=SimpleNamespace(module=model))
    with patch("arctic_platform.common.deepspeed_worker.torch.distributed.barrier") as barrier:
        worker_class = DeepSpeedWorker.__ray_metadata__.modified_class
        result = worker_class.export_hf_checkpoint(worker, str(tmp_path))

    assert result is None
    barrier.assert_not_called()


def test_hf_model_export_only_canonicalizes_checkpoint_wrapper_names() -> None:
    tensor = torch.ones(2)
    state_dict = {"model._checkpoint_wrapped_module.layer.weight": tensor}

    exported = _canonical_hf_export_state_dict(SimpleNamespace(), state_dict)

    assert exported == {"model.layer.weight": tensor}


@pytest.mark.parametrize("checkpoint_wrapped", [False, True])
def test_moe_peft_export_preserves_adapter_tensors_and_reload(tmp_path: Path, monkeypatch, checkpoint_wrapped) -> None:
    from peft import LoraConfig
    from peft import PeftModel
    from peft import get_peft_model
    from peft import get_peft_model_state_dict
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from transformers import Qwen3Config
    from transformers import Qwen3ForCausalLM

    from arctic_platform.model.implementations.moe.deepspeed_integration import build_iter_full_hf_weights

    set_seed(41)
    base = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            use_cache=False,
            attn_implementation="eager",
        )
    )
    reload_base = copy.deepcopy(base)
    if checkpoint_wrapped:
        base.model.layers[0] = checkpoint_wrapper(base.model.layers[0])
    monkeypatch.setattr(type(base), "convert_layer_to_hf", staticmethod(lambda state, index: state), raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    base._iter_full_hf_weights = build_iter_full_hf_weights(base)
    model = get_peft_model(
        base, LoraConfig(task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"], r=2, lora_alpha=4)
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.copy_(torch.arange(parameter.numel()).reshape_as(parameter) / 100)
    expected = {
        name.replace("._checkpoint_wrapped_module", ""): tensor.clone()
        for name, tensor in get_peft_model_state_dict(model, state_dict=dict(model.named_parameters())).items()
    }
    assert len(expected) == 4
    worker = SimpleNamespace(rank=0, engine=SimpleNamespace(module=model), _source_model_dir=None)
    worker_class = DeepSpeedWorker.__ray_metadata__.modified_class

    exported = worker_class.export_hf_checkpoint(worker, str(tmp_path))

    assert exported == str(tmp_path / "hf")
    with safe_open(tmp_path / "hf" / "adapter_model.safetensors", framework="pt") as handle:
        assert set(handle.keys()) == set(expected)
        for name, tensor in expected.items():
            torch_assert_equal(handle.get_tensor(name), tensor)
    reloaded = PeftModel.from_pretrained(reload_base, exported).eval()
    for name, tensor in get_peft_model_state_dict(reloaded).items():
        torch_assert_equal(tensor, expected[name])
    model.eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch_assert_equal(reloaded(input_ids=tokens).logits, model(input_ids=tokens).logits)


def test_peft_export_bypasses_base_iterator_on_nonzero_rank(tmp_path: Path) -> None:
    def base_iterator():
        raise AssertionError("PEFT adapter export must not use the base-only HF iterator")

    model = SimpleNamespace(
        peft_config={"default": object()}, _iter_full_hf_weights=base_iterator, named_parameters=lambda: []
    )
    worker = SimpleNamespace(rank=1, engine=SimpleNamespace(module=model))
    worker_class = DeepSpeedWorker.__ray_metadata__.modified_class

    assert worker_class.export_hf_checkpoint(worker, str(tmp_path)) is None
    assert not (tmp_path / "hf").exists()


@pytest.mark.parametrize("invalid", ["collision", "empty", "missing"])
def test_peft_adapter_canonicalization_rejects_invalid_files(tmp_path: Path, invalid) -> None:
    from arctic_platform.common import deepspeed_worker

    checkpoint = tmp_path / "adapter_model.safetensors"
    if invalid != "missing":
        tensors = (
            {}
            if invalid == "empty"
            else {
                "layer.q_proj.lora_A.weight": torch.ones(2),
                "layer._checkpoint_wrapped_module.q_proj.lora_A.weight": torch.zeros(2),
            }
        )
        save_file(tensors, checkpoint, metadata={"format": "pt"})
        original = checkpoint.read_bytes()
    with pytest.raises((ValueError, FileNotFoundError)):
        deepspeed_worker._canonicalize_peft_adapter_checkpoint(str(tmp_path), ["default"])
    if invalid != "missing":
        assert checkpoint.read_bytes() == original


def test_peft_adapter_canonicalization_preserves_named_adapters_and_metadata(tmp_path: Path) -> None:
    from arctic_platform.common import deepspeed_worker

    for adapter, value in [("default", 2.0), ("research", 3.0)]:
        directory = tmp_path / ("" if adapter == "default" else adapter)
        directory.mkdir(exist_ok=True)
        save_file(
            {"layer._checkpoint_wrapped_module.q_proj.lora_A.weight": torch.full((2,), value)},
            directory / "adapter_model.safetensors",
            metadata={"format": "pt", "adapter": adapter},
        )
    deepspeed_worker._canonicalize_peft_adapter_checkpoint(str(tmp_path), ["default", "research"])
    for adapter, value in [("default", 2.0), ("research", 3.0)]:
        checkpoint = tmp_path / ("" if adapter == "default" else adapter) / "adapter_model.safetensors"
        with safe_open(checkpoint, framework="pt") as handle:
            assert list(handle.keys()) == ["layer.q_proj.lora_A.weight"]
            assert handle.metadata() == {"format": "pt", "adapter": adapter}
            torch_assert_equal(handle.get_tensor(handle.keys()[0]), torch.full((2,), value))


def test_qwen3_5_text_export_loads_as_qwen3_5_config(tmp_path: Path) -> None:
    from transformers import Qwen3_5Config
    from transformers import Qwen3_5TextConfig

    vllm_config = pytest.importorskip("vllm.transformers_utils.config")
    vllm_qwen3_5 = pytest.importorskip("vllm.transformers_utils.configs.qwen3_5")

    from arctic_platform.common.deepspeed_worker import replace_exported_qwen3_5_text_config

    text_cfg = Qwen3_5TextConfig(num_hidden_layers=4)
    source_cfg = Qwen3_5Config(text_config=text_cfg.to_dict(), vision_config={"depth": 2})
    source = tmp_path / "source"
    exported = tmp_path / "exported"
    source.mkdir()
    exported.mkdir()
    (source / "config.json").write_text(json.dumps(source_cfg.to_dict()))
    saved = text_cfg.to_dict()
    saved["model_type"] = "qwen3_5_text"
    saved["architectures"] = ["Qwen3_5ForCausalLM"]
    (exported / "config.json").write_text(json.dumps(saved))

    assert replace_exported_qwen3_5_text_config(str(source), str(exported))
    loaded = vllm_config.get_config(str(exported), trust_remote_code=False)
    assert isinstance(loaded, vllm_qwen3_5.Qwen3_5Config)
    assert loaded.model_type == "qwen3_5"
    assert loaded.architectures == ["Qwen3_5ForCausalLM"]
    assert loaded.text_config.num_hidden_layers == 4
    assert loaded.vision_config.depth == 2


def test_qwen3_5_moe_text_export_is_not_rewritten(tmp_path: Path) -> None:
    from arctic_platform.common.deepspeed_worker import replace_exported_qwen3_5_text_config

    exported = tmp_path / "exported"
    exported.mkdir()
    moe = {"model_type": "qwen3_5_moe_text", "architectures": ["Qwen3_5MoeForCausalLM"]}
    (exported / "config.json").write_text(json.dumps(moe))

    assert not replace_exported_qwen3_5_text_config(None, str(exported))
    assert json.loads((exported / "config.json").read_text()) == moe


def test_qwen3_5_export_keeps_engine_config_class(tmp_path: Path) -> None:
    from transformers import AutoConfig
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    vllm_qwen3_5 = pytest.importorskip("vllm.transformers_utils.configs.qwen3_5")

    parent = AutoConfig.for_model("qwen3_5").to_dict()
    parent["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    text = AutoConfig.for_model("qwen3_5_text").to_dict()
    text["architectures"] = ["Qwen3_5ForCausalLM"]
    language = {"model.language_model.layer.weight": torch.ones(2)}
    visual = {"model.visual.layer.weight": torch.zeros(1)}

    saved_text = tmp_path / "saved-text"
    source_text = tmp_path / "source-text"
    _write_model(source_text, {**language, **visual}, parent)
    _write_model(saved_text, {"model.layer.weight": torch.ones(2)}, text)
    assert restore_source_weight_layout(str(source_text), str(saved_text))
    loaded_text = AutoConfig.from_pretrained(saved_text)
    assert type(loaded_text) is vllm_qwen3_5.Qwen3_5Config
    assert not isinstance(loaded_text, Qwen3_5TextConfig)

    saved_parent = tmp_path / "saved-parent"
    source_parent = tmp_path / "source-parent"
    nested = dict(parent)
    nested["architectures"] = ["Qwen3_5ForCausalLM"]
    _write_model(source_parent, {**language, **visual}, parent)
    _write_model(saved_parent, language, nested)
    assert restore_source_weight_layout(str(source_parent), str(saved_parent))
    loaded_parent = AutoConfig.from_pretrained(saved_parent)
    assert type(loaded_parent) is vllm_qwen3_5.Qwen3_5Config
    assert not isinstance(loaded_parent, Qwen3_5TextConfig)
