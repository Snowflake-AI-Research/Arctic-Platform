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

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from arctic_platform.common.deepspeed_worker import DeepSpeedWorker
from arctic_platform.common.deepspeed_worker import _canonical_hf_export_state_dict
from arctic_platform.common.deepspeed_worker import _copy_source_sidecars
from arctic_platform.common.deepspeed_worker import _model_full_hf_export_state_dict
from arctic_platform.common.deepspeed_worker import restore_source_weight_layout


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


def test_qwen3_5_text_export_loads_as_qwen3_5_config(tmp_path: Path) -> None:
    from transformers import Qwen3_5Config
    from transformers import Qwen3_5TextConfig
    from vllm.transformers_utils.config import get_config
    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config as VllmQwen3_5Config

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
    loaded = get_config(str(exported), trust_remote_code=False)
    assert isinstance(loaded, VllmQwen3_5Config)
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
    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config as EngineQwen3_5Config

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
    assert type(loaded_text) is EngineQwen3_5Config
    assert not isinstance(loaded_text, Qwen3_5TextConfig)

    saved_parent = tmp_path / "saved-parent"
    source_parent = tmp_path / "source-parent"
    nested = dict(parent)
    nested["architectures"] = ["Qwen3_5ForCausalLM"]
    _write_model(source_parent, {**language, **visual}, parent)
    _write_model(saved_parent, language, nested)
    assert restore_source_weight_layout(str(source_parent), str(saved_parent))
    loaded_parent = AutoConfig.from_pretrained(saved_parent)
    assert type(loaded_parent) is EngineQwen3_5Config
    assert not isinstance(loaded_parent, Qwen3_5TextConfig)
