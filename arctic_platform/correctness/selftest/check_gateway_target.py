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

"""Gateway target body rewriting, LoRA adapter export, gating, and the FP8 fixture cut."""


import json
from pathlib import Path
from types import SimpleNamespace

import peft
import pytest
import torch
from safetensors.torch import load_file
from safetensors.torch import save_file

from arctic_platform.correctness.gateway import compare
from arctic_platform.correctness.gateway import lora_export
from arctic_platform.correctness.gateway import target
from arctic_platform.correctness.onboarding import fp8_fixture
from arctic_platform.correctness.onboarding import layer_slice
from arctic_platform.correctness.reference import deepseek_v4_reference

LORA = {"peft_type": "LORA", "r": 2, "lora_alpha": 2, "target_modules": ["q_proj"], "task_type": "CAUSAL_LM"}


def _catalog(peft_config=None, mb=262144) -> dict:
    tc = {
        "n_gpus": 8,
        "sp_size": 8,
        "ep_size": 8,
        "max_seq_len": 262144,
        "mb_spec": {"max_tokens_per_mb": mb},
        "prime_rl": {"ep_comm_backend": "deepep"},
        "debug": {"full_determinism": True},
    }
    if peft_config:
        tc["peft_config"] = peft_config
    return {
        "sub_job_configs": [
            {"job_type": "training", "model_name": "org/model", "training_config": tc},
            {"job_type": "sampling", "model_name": "org/model"},
        ]
    }


def test_job_body_keeps_config_microbatch_and_enables_grad_norms() -> None:
    catalog = _catalog()
    body = target.job_body(catalog, "/data-fast/fixture", 4, 65536)
    assert [s["job_type"] for s in body["sub_job_configs"]] == ["training"]
    sub = body["sub_job_configs"][0]
    tc = sub["training_config"]
    assert sub["model_name"] == "/data-fast/fixture"
    assert (tc["n_gpus"], tc["sp_size"], tc["ep_size"], tc["max_seq_len"]) == (4, 4, 4, 65536)
    assert tc["mb_spec"] == {"max_tokens_per_mb": 262144}
    assert tc["prime_rl"] == {"ep_comm_backend": "deepep", "fused_cross_entropy": "liger"}
    assert tc["debug"] == {"full_determinism": True, "gradient_norms_per_param": True, "gradient_sample_max_numel": 0}
    assert catalog == _catalog(), "the catalog body must not be mutated"
    assert target.training_config(target.job_body(catalog, "f", 4, 65536, max_tokens_per_mb=8192))["mb_spec"] == {
        "max_tokens_per_mb": 8192
    }


def test_grad_norms_fold_expert_stacks_and_take_the_owning_rank() -> None:
    stepped = {
        "metrics": {
            "gradient_norms_per_param": [{}, {"a.weight": 3.0}],
            "gradient_norms_per_expert": [{"experts.w": [3.0, 4.0]}],
        }
    }
    norms, per_expert = target.grad_norms(stepped)
    assert norms == {"a.weight": 3.0, "experts.w": 5.0}
    assert per_expert == {"experts.w": [3.0, 4.0]}


def test_run_target_requires_lora_export_inputs() -> None:
    with pytest.raises(ValueError, match="LoRA target needs"):
        target.run_target(object(), _catalog(LORA), batch=None)


def test_gate_is_four_spreads_rounded_up_with_a_floor() -> None:
    refs = [{"grad_norms": {"a": 1.0, "b": 2.0}}, {"grad_norms": {"a": 1.0002, "b": 2.0001}}]
    spread, gate = compare.gate(refs)
    assert spread == pytest.approx(2e-4)
    assert gate == 1e-3
    refs.append({"grad_norms": {"a": 1.0, "b": 2.0004}})
    assert compare.gate(refs)[1] == 2e-3
    assert compare.gate(refs[:1]) == (0.0, 1e-3)


def test_lora_norms_share_one_name_across_engines() -> None:
    target_name = (
        "module.base_model.model.model.layers.0._checkpoint_wrapped_module.self_attn.q_proj.lora_A.default.weight"
    )
    reference_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    expected = "model.layers.0.self_attn.q_proj.lora_A.weight"
    for name in (target_name, reference_name):
        record = compare.canonical_lora_norms({"grad_norms": {name: 1.0, "model.embed_tokens.weight": 2.0}})
        assert record["grad_norms"] == {expected: 1.0}
    full = {"grad_norms": {"model.embed_tokens.weight": 2.0}}
    assert compare.canonical_lora_norms(full) is full
    with pytest.raises(ValueError, match="two LoRA tensors"):
        compare.canonical_lora_norms({"grad_norms": {target_name: 1.0, reference_name: 1.0}})


def test_head_to_head_reports_change_and_reference_distance() -> None:
    a = {"norms": {"x": [1.01, 1.0], "y": [2.0, 2.0]}}
    b = {"norms": {"x": [1.0, 1.0], "y": [2.0, 2.0]}}
    out = compare.head_to_head(a, b, repeat={"norms": {"x": [1.01, 1.0], "y": [2.0, 2.0]}})
    assert out["tensors"] == 2
    assert out["max"] == (pytest.approx(0.01), "x")
    assert out["a_further_from_ref"] == 1
    assert out["repeat_max"] == 0.0


def test_reference_paths_prefer_per_label_references(tmp_path) -> None:
    for name in ("ref-full-gas1-r0.json", "ref-lora-pr-gas1-r0.json", "ref-lora-pr-gas1-r1.json"):
        (tmp_path / name).write_text("{}")
    assert [p.name for p in compare.reference_paths(tmp_path, "lora", "pr", "gas1")] == [
        "ref-lora-pr-gas1-r0.json",
        "ref-lora-pr-gas1-r1.json",
    ]
    assert [p.name for p in compare.reference_paths(tmp_path, "full", "pr", "gas1")] == ["ref-full-gas1-r0.json"]


def test_lora_factors_use_peft_names_in_float32() -> None:
    state = {
        "base_model.model.model.layers.0._checkpoint_wrapped_module.self_attn.q_proj.lora_A.default.weight": (
            torch.ones(2, 4, dtype=torch.bfloat16)
        ),
        "base_model.model.model.layers.0._checkpoint_wrapped_module.self_attn.q_proj.lora_B.default.weight": (
            torch.zeros(4, 2, dtype=torch.bfloat16)
        ),
        "base_model.model.model.layers.0._checkpoint_wrapped_module.self_attn.q_proj.base_layer.weight": torch.ones(
            4, 4
        ),
    }
    factors = lora_export.lora_factors(state)
    assert sorted(factors) == [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
    ]
    assert all(t.dtype == torch.float32 for t in factors.values())
    with pytest.raises(ValueError, match="no LoRA factors"):
        lora_export.lora_factors({"model.embed_tokens.weight": torch.ones(1)})


def test_write_adapter_round_trips_through_peft(tmp_path) -> None:
    factors = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 4)}
    dest = tmp_path / "adapter"
    lora_export.write_adapter(dest, factors, LORA)
    config = peft.PeftConfig.from_pretrained(str(dest))
    assert (config.r, config.lora_alpha, sorted(config.target_modules)) == (2, 2, ["q_proj"])
    assert torch.equal(load_file(str(dest / "adapter_model.safetensors"))[next(iter(factors))], torch.ones(2, 4))


def test_widen_adapter_rewrites_bfloat16_in_place(tmp_path) -> None:
    save_file({"a.lora_A.weight": torch.ones(2, 2, dtype=torch.bfloat16)}, str(tmp_path / "adapter_model.safetensors"))
    lora_export.widen_adapter(tmp_path)
    assert load_file(str(tmp_path / "adapter_model.safetensors"))["a.lora_A.weight"].dtype == torch.float32
    assert [p.name for p in tmp_path.iterdir()] == ["adapter_model.safetensors"]


def test_zone_url_is_the_newest_announced_port(tmp_path) -> None:
    log = tmp_path / "zone.log"
    log.write_text("Uvicorn running on http://0.0.0.0:41001\n...\nUvicorn running on http://0.0.0.0:41002\n")
    assert lora_export.zone_url_from_log(log) == "http://127.0.0.1:41002"
    log.write_text("starting\n")
    with pytest.raises(RuntimeError, match="no zone server port"):
        lora_export.zone_url_from_log(log)


def test_fp8_fixture_keeps_scales_quantization_and_sidecars(tmp_path) -> None:
    source, like, dest = tmp_path / "source", tmp_path / "like", tmp_path / "dest"
    source.mkdir()
    like.mkdir()
    fp8 = torch.ones(4, 4).to(torch.float8_e4m3fn)
    save_file(
        {
            "model.layers.0.mlp.weight": fp8,
            "model.layers.0.mlp.weight_scale_inv": torch.ones(1, 1),
            "model.layers.0.norm.weight": torch.ones(4, dtype=torch.bfloat16),
            "model.layers.9.mlp.weight": fp8.clone(),
        },
        str(source / "model.safetensors"),
    )
    source_names = ["model.layers.0.mlp.weight", "model.layers.0.mlp.weight_scale_inv"]
    source_names += ["model.layers.0.norm.weight", "model.layers.9.mlp.weight"]
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(source_names, "model.safetensors")})
    )
    quantization = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    (source / "config.json").write_text(json.dumps({"quantization_config": quantization}))
    like_map = dict.fromkeys(["model.layers.0.mlp.weight", "model.layers.0.norm.weight"], "model.safetensors")
    (like / "model.safetensors.index.json").write_text(json.dumps({"weight_map": like_map}))
    (like / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))
    (like / "tokenizer_config.json").write_text("{}")

    assert fp8_fixture.build(source, like, dest, shard_bytes=20) == {"tensors": 3, "scales": 1, "shards": 2}
    index = json.loads((dest / "model.safetensors.index.json").read_text())["weight_map"]
    assert sorted(index) == [
        "model.layers.0.mlp.weight",
        "model.layers.0.mlp.weight_scale_inv",
        "model.layers.0.norm.weight",
    ]
    assert (
        load_file(str(dest / index["model.layers.0.mlp.weight"]))["model.layers.0.mlp.weight"].dtype
        == torch.float8_e4m3fn
    )
    assert json.loads((dest / "config.json").read_text()) == {
        "num_hidden_layers": 1,
        "quantization_config": quantization,
    }
    assert (dest / "tokenizer_config.json").exists()
    with pytest.raises(KeyError, match="missing from the FP8 source"):
        fp8_fixture.fp8_keys({}, ["model.layers.0.mlp.weight"])


def test_fp8_keys_take_deepseek_scale_siblings_once() -> None:
    source = dict.fromkeys(["layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale", "layers.0.hc_attn_scale"], "f")
    assert fp8_fixture.fp8_keys(source, ["layers.0.attn.wq_a.weight", "layers.0.hc_attn_scale"]) == [
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.scale",
        "layers.0.hc_attn_scale",
    ]
    assert fp8_fixture.fp8_keys(source, ["layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale"]) == [
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.scale",
    ]


def test_layer_slice_keeps_original_names_and_truncates_padded_lists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(layer_slice, "require_data_fast_path", lambda path, label: Path(path))
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.mkdir()
    tensors = {
        "embed.weight": torch.ones(2),
        "layers.0.attn.wq_a.weight": torch.ones(2),
        "layers.0.attn.wq_a.scale": torch.ones(1),
        "layers.1.attn.wq_a.weight": torch.ones(2),
        "layers.10.attn.wq_a.weight": torch.ones(2),
        "mtp.0.attn.wq_a.weight": torch.ones(2),
    }
    save_file(tensors, str(source / "model.safetensors"))
    weight_map = dict.fromkeys(tensors, "model.safetensors")
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (source / "config.json").write_text(
        json.dumps(
            {"num_hidden_layers": 3, "compress_ratios": [0, 4, 128, 0], "ids": [2], "num_nextn_predict_layers": 1}
        )
    )
    (source / "tokenizer.json").write_text("{}")

    assert layer_slice.slice_layers(source, dest, 1) == {"tensors": 3, "shards": 1}
    index = json.loads((dest / "model.safetensors.index.json").read_text())["weight_map"]
    assert sorted(index) == ["embed.weight", "layers.0.attn.wq_a.scale", "layers.0.attn.wq_a.weight"]
    assert json.loads((dest / "config.json").read_text()) == {
        "num_hidden_layers": 1,
        "compress_ratios": [0],
        "ids": [2],
        "num_nextn_predict_layers": 0,
    }
    assert (dest / "tokenizer.json").exists()


def test_deepseek_v4_kv_fake_quant_tiles_scales_and_passes_gradient() -> None:
    x = torch.zeros(2, 192, dtype=torch.bfloat16)
    x[0, :64] = 1.3
    x[0, 64:128] = 0.75
    x[0, 0] = 448.0 * 2.0**-3
    x[0, 128:] = 3.14159
    x[1, :64] = 1e-9
    x.requires_grad_(True)

    amax_floor = deepseek_v4_reference.fake_quant_kv(x, 64, floor_scale=False)
    scale_floor = deepseek_v4_reference.fake_quant_kv(x, 64, floor_scale=True)

    scale = 2.0**-3
    expected = (torch.full((63,), 1.3) / scale).to(torch.float8_e4m3fn).float() * scale
    assert torch.equal(amax_floor[0, 1:64].float(), expected)
    assert amax_floor[0, 0] == 448.0 * scale
    assert torch.equal(amax_floor[0, 64:128], x[0, 64:128])
    assert torch.equal(amax_floor[:, 128:], x[:, 128:])
    assert amax_floor[1, 0] == 2.0**-30
    assert scale_floor[1, 0] == 0
    amax_floor.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_deepseek_v4_rms_norm_returns_input_dtype() -> None:
    norm = SimpleNamespace(weight=torch.full((4,), 2.0), variance_epsilon=0.0)
    out = deepseek_v4_reference.rms_norm_forward(norm, torch.ones(1, 4, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, torch.full((1, 4), 2.0, dtype=torch.bfloat16))


def test_fp8_fixture_carries_listed_config_keys(tmp_path) -> None:
    source, like, dest = tmp_path / "source", tmp_path / "like", tmp_path / "dest"
    source.mkdir()
    like.mkdir()
    save_file({"w.weight": torch.ones(2, dtype=torch.bfloat16)}, str(source / "model.safetensors"))
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w.weight": "model.safetensors"}}))
    (source / "config.json").write_text(json.dumps({"quantization_config": {"fmt": "e4m3"}, "expert_dtype": "fp4"}))
    (like / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w.weight": "model.safetensors"}}))
    (like / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))
    fp8_fixture.build(source, like, dest, carry=("quantization_config", "expert_dtype"))
    assert json.loads((dest / "config.json").read_text()) == {
        "num_hidden_layers": 1,
        "quantization_config": {"fmt": "e4m3"},
        "expert_dtype": "fp4",
    }
