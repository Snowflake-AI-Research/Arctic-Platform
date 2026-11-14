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

"""``synth_model.materialize_tiny`` keeps an original's fields, tensor names, dtypes and FP8 layout at reduced size."""

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from arctic_platform.correctness.harness.config import hash_directory
from arctic_platform.correctness.onboarding import synth_model

L, S = "linear_attention", "deepseek_sparse_attention"
SIDECARS = {
    "tokenizer_config.json": "{}\n",
    "vocab.json": "{}\n",
    "generation_config.json": '{"eos_token_id": 1}\n',
    "chat_template.jinja": "{{ messages }}\n",
}
# The GLM-5.3 original keeps its hyper-connection bases and scales, KDA decay parameters and routing bias in FP32,
# and block-quantizes the MLA and MLP projections.
GLM_FP32 = re.compile(r"(hc_\w+_(base|scale)|A_log|dt_bias|e_score_correction_bias)$")
GLM_FP8 = re.compile(
    r"layers\.\d+\.(self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)"
    r"|mlp\.(experts\.\d+\.|shared_experts\.)?(gate|up|down)_proj)\.weight$"
)


def _resolve(path, **_):
    return Path(path).resolve()


def _glm_raw_config():
    return {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": {
            "model_type": "glm5_next_text",
            "hidden_size": 64,
            "num_hidden_layers": 9,
            "num_nextn_predict_layers": 1,
            "layer_types": [L, L, L, S, L, L, L, S, L],
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 6,
            "linear_attn_config": {"num_heads": 4, "kda_layers": [0, 1, 2, 4, 5, 6, 8], "full_attn_layers": [3, 7]},
            "eos_token_id": [1, 2],
        },
        "vision_config": {"depth": 4, "out_hidden_size": 64},
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [
                "lm_head",
                "model.layers.0.hc_attn_base",
                "model.layers.7.mlp.gate",
                "model.layers.9.eh_proj",
                "visual.blocks.0.attn.qkv",
                "visual.blocks.3.attn.qkv",
                "visual.merger.proj",
            ],
        },
    }


def test_tiny_config_cuts_per_layer_entries_and_renumbers_quantization_modules():
    original = _glm_raw_config()
    tiny = synth_model.tiny_config_dict(original, 5, {"hidden_size": 32}, {"out_hidden_size": 32}, vision_depth=1)

    text = tiny["text_config"]
    assert text["num_hidden_layers"] == 5
    assert text["hidden_size"] == 32
    assert text["layer_types"] == [L, L, L, S, L]
    assert text["mlp_layer_types"] == ["dense"] * 3 + ["sparse"] * 2
    assert text["linear_attn_config"] == {"num_heads": 4, "kda_layers": [0, 1, 2, 4], "full_attn_layers": [3]}
    assert text["eos_token_id"] == [1, 2]
    assert tiny["vision_config"] == {"depth": 1, "out_hidden_size": 32}
    # The prediction layer after the last of nine decoder layers follows the last of five; layer 7 and block 3 go.
    assert tiny["quantization_config"]["modules_to_not_convert"] == [
        "lm_head",
        "model.layers.0.hc_attn_base",
        "model.layers.5.eh_proj",
        "visual.blocks.0.attn.qkv",
        "visual.merger.proj",
    ]
    assert original == _glm_raw_config()
    assert {entry["key"] for entry in synth_model.config_diff(original, tiny)} == {
        "text_config.hidden_size",
        "text_config.layer_types",
        "text_config.linear_attn_config.full_attn_layers",
        "text_config.linear_attn_config.kda_layers",
        "text_config.mlp_layer_types",
        "text_config.num_hidden_layers",
        "vision_config.depth",
        "vision_config.out_hidden_size",
        "quantization_config.modules_to_not_convert",
    }


def test_tiny_config_refuses_fields_the_original_lacks_and_dropped_layer_kinds():
    with pytest.raises(ValueError, match="no 'head_dim' to override"):
        synth_model.tiny_config_dict(_glm_raw_config(), 5, {"head_dim": 8})
    with pytest.raises(ValueError, match="no 'patch_size' to override"):
        synth_model.tiny_config_dict(_glm_raw_config(), 5, {}, {"patch_size": 8})
    with pytest.raises(ValueError, match="deepseek_sparse_attention"):
        synth_model.tiny_config_dict(_glm_raw_config(), 3, {})
    with pytest.raises(ValueError, match="num_layers"):
        synth_model.tiny_config_dict(_glm_raw_config(), 10, {})


def test_copy_sidecars_default_patterns_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(synth_model, "require_data_fast_path", _resolve)
    source = tmp_path / "source"
    source.mkdir()
    for name, body in {**SIDECARS, "merges.txt": "a b\n", "config.json": "{}\n"}.items():
        (source / name).write_text(body)
    (tmp_path / "reduced").mkdir()
    (tmp_path / "tiny").mkdir()

    reduced = synth_model.copy_sidecars(str(source), tmp_path / "reduced")
    tiny = synth_model.copy_sidecars(str(source), tmp_path / "tiny", synth_model.TINY_SIDECAR_PATTERNS)

    assert {path.name for path in reduced} == {"tokenizer_config.json", "merges.txt", "chat_template.jinja"}
    assert {path.name for path in tiny} == {path.name for path in reduced} | {"vocab.json", "generation_config.json"}


def _write_original(
    root: Path, config: dict, complete, *, fp32=None, fp8=None, modules_to_not_convert=(), fused=False
):
    """Write ``config`` built from seed 1 and completed by ``complete`` as an original checkpoint with sidecars.

    ``fused`` keeps the experts in the layout the model holds in memory, as Qwen3.8 checkpoints store them, instead of
    one tensor per expert as GLM-5.3 checkpoints store them.
    """
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.conversion_mapping import get_model_conversion_mapping

    from arctic_platform.model.implementations.fp8 import quantize_to_fp8_blockwise

    build = root.parent / f"{root.name}-build"
    build.mkdir()
    (build / "config.json").write_text(json.dumps(config))
    cfg = AutoConfig.from_pretrained(build)
    torch.manual_seed(1)
    model = getattr(transformers, cfg.architectures[0])(cfg)
    model._weight_conversions = [] if fused else get_model_conversion_mapping(model, add_legacy=False)
    synth_model._save_reduced_pretrained(model, build / "saved")
    tensors = {}
    for shard in sorted((build / "saved").glob("*.safetensors")):
        tensors.update(load_file(shard))
    complete(tensors)
    for name, tensor in tensors.items():
        if tensor.is_floating_point():
            tensors[name] = tensor.to(torch.float32 if fp32 is not None and fp32.search(name) else torch.bfloat16)
    written = copy.deepcopy(config)
    if fp8 is not None:
        for name in [name for name in tensors if fp8.search(name)]:
            tensors[name], tensors[f"{name}_scale_inv"] = quantize_to_fp8_blockwise(tensors[name], 128)
        written["quantization_config"] = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": list(modules_to_not_convert),
        }
    root.mkdir()
    synth_model._write_checkpoint(tensors, root)
    (root / "config.json").write_text(json.dumps(written))
    for name, body in SIDECARS.items():
        (root / name).write_text(body)


def _load(root: Path) -> dict:
    from safetensors.torch import load_file

    tensors = {}
    for shard in sorted(root.glob("*.safetensors")):
        tensors.update(load_file(shard))
    return tensors


def _check_manifest(manifest: dict, sources: Path) -> Path:
    tiny = Path(manifest["path"])
    assert manifest["content_hash"] == hash_directory(tiny)
    assert manifest["prediction_layers"] == 1
    assert json.loads(Path(f"{tiny}.manifest.json").read_text()) == manifest
    for name, entry in manifest["files"].items():
        assert entry["sha256"] == hashlib.sha256((tiny / name).read_bytes()).hexdigest()
    for name, body in SIDECARS.items():
        assert (tiny / name).read_text() == (sources / name).read_text() == body
    return tiny


def _glm_original_config():
    from arctic_platform import testing_utils

    config = testing_utils.tiny_glm5_next_config().to_dict()
    config["architectures"] = ["Glm5NextForConditionalGeneration"]
    config["text_config"].update(
        num_hidden_layers=6,
        layer_types=[L, L, L, S, L, L],
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 3,
        indexer_types=["full"] * 6,
        num_nextn_predict_layers=1,
        linear_attn_config={"kda_layers": [0, 1, 2, 4, 5], "full_attn_layers": [3]},
    )
    config["vision_config"]["depth"] = 2
    return config


def _glm_prediction_layer(tensors):
    """GLM-5.3's next-token layer: a sparse decoder layer without hyper-connection mixers, plus its projections."""
    import torch

    for name in [name for name in tensors if name.startswith("model.language_model.layers.3.")]:
        suffix = name.removeprefix("model.language_model.layers.3.")
        if not suffix.startswith("hc_"):
            tensors[f"model.language_model.layers.6.{suffix}"] = tensors[name].clone()
    tensors["model.language_model.layers.6.eh_proj.weight"] = torch.randn(16, 32)
    for norm in ("enorm", "hnorm", "shared_head.norm"):
        tensors[f"model.language_model.layers.6.{norm}.weight"] = torch.ones(16)


def _glm_tiny_sources(original: dict) -> dict:
    """Tiny tensor name to original tensor name: five decoder layers, the prediction layer, two experts, one block."""
    sources = {}
    for name in original:
        tiny = name
        if match := re.match(r"model\.language_model\.layers\.(\d+)\.(.+)$", name):
            if int(match[1]) == 5:
                continue
            tiny = f"model.language_model.layers.{5 if int(match[1]) == 6 else match[1]}.{match[2]}"
        if (match := re.search(r"\.experts\.(\d+)\.", name)) and int(match[1]) >= 2:
            continue
        if (match := re.match(r"model\.visual\.blocks\.(\d+)\.", name)) and int(match[1]) >= 1:
            continue
        sources[tiny] = name
    return sources


def test_tiny_glm_keeps_names_dtypes_fp8_layout_and_prediction_layer(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.glm5_next")
    from transformers import Glm5NextForConditionalGeneration

    from arctic_platform.model.implementations.fp8 import quantize_to_fp8_blockwise

    monkeypatch.setattr(synth_model, "require_data_fast_path", _resolve)
    config = _glm_original_config()
    manifests = {}
    for kind in ("bf16", "fp8"):
        _write_original(
            tmp_path / f"original-{kind}",
            config,
            _glm_prediction_layer,
            fp32=GLM_FP32,
            fp8=GLM_FP8 if kind == "fp8" else None,
            modules_to_not_convert=[
                "lm_head",
                "model.layers.6.eh_proj",
                "visual.blocks.0.attn.qkv",
                "visual.blocks.1.attn.qkv",
            ],
        )
        manifests[kind] = synth_model.materialize_tiny(
            str(tmp_path / f"original-{kind}"),
            str(tmp_path / f"tiny-{kind}"),
            num_layers=5,
            text_overrides={"hidden_size": 8, "n_routed_experts": 2},
            vision_overrides={"out_hidden_size": 8},
            vision_depth=1,
        )

    for kind, manifest in manifests.items():
        tiny = _check_manifest(manifest, tmp_path / f"original-{kind}")
        original = synth_model.checkpoint_schema(tmp_path / f"original-{kind}")
        stored = synth_model.checkpoint_schema(tiny)
        sources = _glm_tiny_sources(original)
        assert set(stored) == set(sources)
        assert {name: dtype for name, (dtype, _) in stored.items()} == {
            name: original[source][0] for name, source in sources.items()
        }
        assert stored["model.language_model.layers.5.eh_proj.weight"][1] == (8, 16)
        assert {"key": "text_config.hidden_size", "original": 16, "tiny": 8} in manifest["config_diff"]
        written = json.loads((tiny / "config.json").read_text())
        assert written["text_config"]["layer_types"] == [L, L, L, S, L]
        assert written["text_config"]["linear_attn_config"] == {"kda_layers": [0, 1, 2, 4], "full_attn_layers": [3]}
        assert written["vision_config"]["depth"] == 1
        assert ("quantization_config" in written) is (kind == "fp8")
    assert json.loads((tmp_path / "tiny-fp8" / "config.json").read_text())["quantization_config"][
        "modules_to_not_convert"
    ] == ["lm_head", "model.layers.5.eh_proj", "visual.blocks.0.attn.qkv"]

    bf16, fp8 = _load(tmp_path / "tiny-bf16"), _load(tmp_path / "tiny-fp8")
    assert any(f"{name}_scale_inv" in fp8 for name in bf16)
    for name, tensor in bf16.items():
        if f"{name}_scale_inv" in fp8:
            weight, scale = quantize_to_fp8_blockwise(tensor, 128)
            assert torch.equal(fp8[name].view(torch.uint8), weight.view(torch.uint8)), name
            assert torch.equal(fp8[f"{name}_scale_inv"], scale), name
        else:
            assert torch.equal(fp8[name], tensor), name
    assert torch.equal(bf16["model.language_model.layers.5.enorm.weight"], torch.ones(8, dtype=torch.bfloat16))

    _, info = Glm5NextForConditionalGeneration.from_pretrained(tmp_path / "tiny-bf16", output_loading_info=True)
    assert not info["missing_keys"] and not info["mismatched_keys"]
    assert all(".layers.5." in key for key in info["unexpected_keys"])


# More n-gram shards than fused expert tensors, as in Qwen3.8 (128 shards), so that a layout chosen by shard names
# alone would be the wrong one.
QWEN_NGRAM_PARTS = 8


def _qwen_original_config():
    from arctic_platform import testing_utils

    config = testing_utils.tiny_qwen4_exp_config().to_dict()
    config["architectures"] = ["Qwen4ExpForConditionalGeneration"]
    config["text_config"].update(
        num_hidden_layers=3,
        layer_types=["linear_attention", "full_attention", "full_attention"],
        mtp={"hybrid": True, "layer_types": ["full_attention"], "num_hidden_layers": 1},
        mtp_num_hidden_layers=1,
        split_ngram_parts=QWEN_NGRAM_PARTS,
    )
    return config


def _qwen_mtp(tensors):
    """Qwen3.8's MTP: a full-attention decoder layer and its own mixer and projections under ``mtp``."""
    import torch

    for name in [name for name in tensors if name.startswith("model.language_model.layers.2.")]:
        tensors[f"mtp.layers.0.{name.removeprefix('model.language_model.layers.2.')}"] = tensors[name].clone()
    for name in [name for name in tensors if name.startswith("model.language_model.hyper_connection_mixer.")]:
        tensors[f"mtp.{name.removeprefix('model.language_model.')}"] = tensors[name].clone()
    for projection in ("fc_embedding", "fc_hidden"):
        tensors[f"mtp.{projection}.weight"] = torch.randn(16, 16)
    tensors["mtp.pre_fc_norm_embedding.weight"] = torch.zeros(16)
    tensors["mtp.pre_fc_norm_hidden.weight"] = torch.zeros(32)
    # The original stores each n-gram table as ``split_ngram_parts`` row blocks.
    for name in [name for name in tensors if name.endswith(".ngram_embedding.weight")]:
        for index, block in enumerate(tensors.pop(name).tensor_split(QWEN_NGRAM_PARTS, dim=0)):
            tensors[f"{name.removesuffix('.weight')}.shard_{index}.weight"] = block.clone()


def test_tiny_qwen_keeps_names_dtypes_mtp_and_ngram_shards(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen4_exp")
    from transformers import Qwen4ExpForConditionalGeneration

    monkeypatch.setattr(synth_model, "require_data_fast_path", _resolve)
    _write_original(tmp_path / "original", _qwen_original_config(), _qwen_mtp, fused=True)
    manifest = synth_model.materialize_tiny(
        str(tmp_path / "original"),
        str(tmp_path / "tiny"),
        num_layers=2,
        text_overrides={"hidden_size": 8, "num_experts": 2},
        vision_overrides={"out_hidden_size": 8},
    )

    tiny = _check_manifest(manifest, tmp_path / "original")
    original = synth_model.checkpoint_schema(tmp_path / "original")
    stored = synth_model.checkpoint_schema(tiny)
    assert set(stored) == {name for name in original if not name.startswith("model.language_model.layers.2.")}
    assert all(stored[name][0] == original[name][0] for name in stored)
    assert sum(".ngram_embedding.shard_" in name for name in stored) == QWEN_NGRAM_PARTS
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" in stored
    assert stored["mtp.pre_fc_norm_hidden.weight"][1] == (16,)
    assert stored["mtp.hyper_connection_mixer.hc_norm.weight"][1] == (16,)
    tensors = _load(tiny)
    assert torch.count_nonzero(tensors["mtp.pre_fc_norm_hidden.weight"]) == 0

    _, info = Qwen4ExpForConditionalGeneration.from_pretrained(tiny, output_loading_info=True)
    assert not info["missing_keys"] and not info["mismatched_keys"]
    assert all(key.startswith("mtp.") for key in info["unexpected_keys"])
