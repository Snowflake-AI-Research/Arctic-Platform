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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Local export validation against tiny real safetensors, without training or sampling."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.utils import hub

from arctic_platform.correctness.checks import inference_checkpoint_loss as check
from arctic_platform.correctness.harness.inference_export import FAMILIES


@pytest.fixture(params=["qwen3_5", "qwen3_5_moe"])
def projection_case(tmp_path, monkeypatch, request):
    family = request.param
    composite_class, text_type, text_class = FAMILIES[family]
    text = {
        "model_type": text_type,
        "num_hidden_layers": 2,
        "hidden_size": 4,
        "intermediate_size": 8,
        "vocab_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "use_cache": True,
        "layer_types": ["linear_attention", "full_attention"],
        "rope_parameters": {"rope_theta": 10000000},
    }
    if family.endswith("_moe"):
        text.update(num_experts=4, num_experts_per_tok=2)
    source_config = {
        "model_type": family,
        "architectures": [composite_class],
        "text_config": text,
        "vision_config": {"depth": 1},
    }
    exported_config = {
        **copy.deepcopy(text),
        "architectures": [text_class],
        "use_cache": False,
        "transformers_version": "synthetic-test",
    }
    tensors = {
        "model.language_model.embed_tokens.weight": torch.ones(8, 4, dtype=torch.bfloat16),
        "lm_head.weight": torch.ones(8, 4, dtype=torch.bfloat16),
        "model.language_model.layers.0.norm.weight": torch.ones(4, dtype=torch.float32),
        "model.language_model.layers.1.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        "model.visual.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        "mtp.fc.weight": torch.ones(4, 8, dtype=torch.bfloat16),
    }
    if family.endswith("_moe"):
        tensors["model.language_model.layers.0.mlp.experts.gate_up_proj"] = torch.ones(4, 16, 4, dtype=torch.bfloat16)
    exported_tensors = {
        name: tensor.to(torch.bfloat16)
        for name, tensor in tensors.items()
        if name.startswith("model.language_model.") or name == "lm_head.weight"
    }
    source, exported = tmp_path / "source", tmp_path / "export"
    source.mkdir()
    exported.mkdir()

    def write(root, config, values, *, original=False):
        (root / "config.json").write_text(json.dumps(config))
        save_file(values, str(root / "model.safetensors"))
        total = sum(tensor.numel() * tensor.element_size() for tensor in values.values())
        index = {
            "metadata": {"total_size": float(total) if original else total},
            "weight_map": {name: "model.safetensors" for name in values},
        }
        (root / "model.safetensors.index.json").write_text(json.dumps(index))

    write(source, source_config, tensors, original=True)
    write(exported, exported_config, exported_tensors)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(**source_config))

    def cached_file(model_name, filename, **kwargs):
        assert model_name == "Qwen/synthetic-composite"
        assert kwargs == {"local_files_only": True}
        return str(source / filename)

    monkeypatch.setattr(hub, "cached_file", cached_file)
    ctx = SimpleNamespace(cfg=SimpleNamespace(model_name="Qwen/synthetic-composite"))
    return SimpleNamespace(
        ctx=ctx,
        source=source,
        exported=exported,
        source_config=source_config,
        config=exported_config,
        tensors=exported_tensors,
        write=write,
    )


def test_local_full_text_export_is_validated_against_original_headers(projection_case):
    case = projection_case
    original = (case.source / "config.json").read_bytes()
    assert check.validate_local_inference_export(case.ctx, case.exported) == case.exported
    assert (case.source / "config.json").read_bytes() == original


@pytest.mark.parametrize("field", ["num_hidden_layers", "hidden_size", "rope_parameters", "dtype", "unknown"])
def test_local_text_export_rejects_model_configuration_drift(projection_case, field):
    case = projection_case
    config = copy.deepcopy(case.config)
    if field == "rope_parameters":
        config[field]["rope_theta"] = 10000
    elif field == "dtype":
        config[field] = "float32"
    else:
        config[field] = 1
    case.write(case.exported, config, case.tensors)
    with pytest.raises(ValueError):
        check.validate_local_inference_export(case.ctx, case.exported)


@pytest.mark.parametrize("failure", ["missing", "extra", "vision", "mtp", "shape", "rename", "dtype"])
def test_local_text_export_rejects_incomplete_or_altered_weights(projection_case, failure):
    case = projection_case
    tensors = dict(case.tensors)
    name = "model.language_model.layers.1.norm.weight"
    if failure == "missing":
        tensors.pop(name)
    elif failure in {"extra", "vision", "mtp"}:
        key = {
            "extra": "model.language_model.unexpected.weight",
            "vision": "model.visual.norm.weight",
            "mtp": "mtp.fc.weight",
        }[failure]
        tensors[key] = torch.ones(4, dtype=torch.bfloat16)
    elif failure == "shape":
        tensors[name] = tensors[name].reshape(2, 2)
    elif failure == "rename":
        tensors["model.layers.1.norm.weight"] = tensors.pop(name)
    else:
        tensors[name] = tensors[name].float()
    case.write(case.exported, case.config, tensors)
    with pytest.raises(ValueError):
        check.validate_local_inference_export(case.ctx, case.exported)


def test_local_text_export_requires_one_original_snapshot(projection_case, tmp_path, monkeypatch):
    case = projection_case
    other = tmp_path / "other-snapshot"
    other.mkdir()
    index = other / "model.safetensors.index.json"
    index.write_bytes((case.source / index.name).read_bytes())
    monkeypatch.setattr(
        hub,
        "cached_file",
        lambda model, filename, **kwargs: str(index if filename == index.name else case.source / filename),
    )
    with pytest.raises(RuntimeError, match="one snapshot"):
        check.validate_local_inference_export(case.ctx, case.exported)
