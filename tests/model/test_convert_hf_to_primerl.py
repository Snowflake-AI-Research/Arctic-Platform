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
import json
from pathlib import Path

from arctic_platform.model.convert_hf_to_primerl import ConverterSpec
from arctic_platform.model.convert_hf_to_primerl import convert_checkpoint
from arctic_platform.model.convert_hf_to_primerl import inspect_checkpoint
from arctic_platform.model.convert_hf_to_primerl import main
from arctic_platform.model.convert_hf_to_primerl import promote_hashed_cache
from arctic_platform.model.convert_hf_to_primerl import supported_model_types
from arctic_platform.model.implementations.moe.conversion_cache import conversion_cache_ready
from arctic_platform.model.implementations.moe.conversion_cache import hashed_conversion_cache_path
from arctic_platform.model.implementations.moe.conversion_cache import sibling_conversion_cache_path


class _FakeConverter:
    @classmethod
    def is_hf_state_dict(cls, state_dict):
        return any("mlp.experts.1.up_proj" in name for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict):
        return any("mlp.experts.w1" in name for name in state_dict)

    @classmethod
    def convert_to_prime(cls, state_dict):
        renamed = {k.replace("mlp.experts.1.up_proj", "mlp.experts.w1"): v for k, v in list(state_dict.items())}
        state_dict.clear()
        state_dict.update(renamed)
        return state_dict


def _write_ready_cache(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model-00001-of-00001.safetensors").write_bytes(b"not-used")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}})
    )


def _hf_checkpoint(tmp_path: Path, model_type: str = "glm_moe_dsa") -> Path:
    checkpoint = tmp_path / "GLM-5.2"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": model_type}))
    return checkpoint


def test_list_models_includes_glm_and_qwen():
    names = supported_model_types()
    assert "glm_moe_dsa" in names
    assert "qwen3_5_moe" in names
    assert "qwen3_moe" in names
    assert "glm4_moe" in names
    assert "glm5_next" in names
    assert "minimax_m2" in names
    assert "nemotron_h" in names
    assert "afmoe" in names
    assert "llama" in names


def test_main_list_models(capsys):
    assert main(["--list-models"]) == 0
    assert "glm_moe_dsa" in capsys.readouterr().out


def test_inspect_hf_checkpoint(monkeypatch, tmp_path):
    checkpoint = _hf_checkpoint(tmp_path)
    monkeypatch.setattr(
        "arctic_platform.model.convert_hf_to_primerl.load_state_dict_keys",
        lambda path: ["model.layers.0.mlp.experts.1.up_proj.weight"],
    )
    monkeypatch.setattr(
        "arctic_platform.model.convert_hf_to_primerl.resolve_converter",
        lambda model_type: ConverterSpec(model_type, _FakeConverter, "arctic-platform"),
    )

    info = inspect_checkpoint(checkpoint)
    assert info["needs_conversion"] is True
    assert info["already_prime"] is False
    assert info["dest"] == str(sibling_conversion_cache_path(checkpoint, "prime"))


def test_inspect_llama_is_noop(monkeypatch, tmp_path):
    checkpoint = _hf_checkpoint(tmp_path, model_type="llama")
    monkeypatch.setattr(
        "arctic_platform.model.convert_hf_to_primerl.load_state_dict_keys",
        lambda path: ["model.layers.0.self_attn.q_proj.weight"],
    )
    info = inspect_checkpoint(checkpoint)
    assert info["already_prime"] is True
    assert info["needs_conversion"] is False


def test_convert_writes_sibling_cache(monkeypatch, tmp_path):
    checkpoint = _hf_checkpoint(tmp_path)
    dest = sibling_conversion_cache_path(checkpoint, "prime")
    monkeypatch.setattr(
        "arctic_platform.model.convert_hf_to_primerl.load_state_dict_keys",
        lambda path: ["model.layers.0.mlp.experts.1.up_proj.weight"],
    )
    monkeypatch.setattr(
        "arctic_platform.model.convert_hf_to_primerl.resolve_converter",
        lambda model_type: ConverterSpec(model_type, _FakeConverter, "arctic-platform"),
    )
    monkeypatch.setattr(
        "arctic_platform.model.convert_hf_to_primerl.load_state_dict",
        lambda path: {"model.layers.0.mlp.experts.1.up_proj.weight": object()},
    )

    def fake_save(state_dict, save_dir):
        assert "mlp.experts.w1" in next(iter(state_dict))
        _write_ready_cache(save_dir)

    monkeypatch.setattr("arctic_platform.model.convert_hf_to_primerl.save_state_dict", fake_save)

    assert convert_checkpoint(checkpoint) == dest.resolve()
    assert conversion_cache_ready(dest) is True

    # Second call must not convert again.
    calls = {"save": 0}

    def boom_save(state_dict, save_dir):
        calls["save"] += 1
        raise AssertionError("should reuse ready sibling cache")

    monkeypatch.setattr("arctic_platform.model.convert_hf_to_primerl.save_state_dict", boom_save)
    convert_checkpoint(checkpoint)
    assert calls["save"] == 0


def test_convert_failure_removes_partial_cache(monkeypatch, tmp_path):
    checkpoint = _hf_checkpoint(tmp_path)
    dest = sibling_conversion_cache_path(checkpoint, "prime")
    monkeypatch.setattr(
        "arctic_platform.model.convert_hf_to_primerl.load_state_dict_keys",
        lambda path: ["model.layers.0.mlp.experts.1.up_proj.weight"],
    )
    monkeypatch.setattr(
        "arctic_platform.model.convert_hf_to_primerl.resolve_converter",
        lambda model_type: ConverterSpec(model_type, _FakeConverter, "arctic-platform"),
    )
    monkeypatch.setattr(
        "arctic_platform.model.convert_hf_to_primerl.load_state_dict",
        lambda path: {"model.layers.0.mlp.experts.1.up_proj.weight": object()},
    )

    def partial_save(state_dict, save_dir):
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "model-00001-of-00001.safetensors").write_bytes(b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr("arctic_platform.model.convert_hf_to_primerl.save_state_dict", partial_save)

    try:
        convert_checkpoint(checkpoint)
        raise AssertionError("expected convert_checkpoint to fail")
    except RuntimeError as exc:
        assert "disk full" in str(exc)

    leftovers = [p.name for p in dest.parent.iterdir() if p.name == dest.name or p.name.startswith(f"{dest.name}.")]
    assert leftovers == []


def test_promote_from_explicit_cache(monkeypatch, tmp_path):
    checkpoint = _hf_checkpoint(tmp_path)
    src = tmp_path / "existing-prime"
    _write_ready_cache(src)
    dest = sibling_conversion_cache_path(checkpoint, "prime")
    monkeypatch.setattr(
        "arctic_platform.model.convert_hf_to_primerl.load_state_dict_keys",
        lambda path: ["model.layers.0.mlp.experts.1.up_proj.weight"],
    )
    monkeypatch.setattr(
        "arctic_platform.model.convert_hf_to_primerl.resolve_converter",
        lambda model_type: ConverterSpec(model_type, _FakeConverter, "arctic-platform"),
    )

    convert_checkpoint(checkpoint, promote_from=src)
    assert conversion_cache_ready(dest) is True
    assert (src / "model-00001-of-00001.safetensors").stat().st_ino == (
        dest / "model-00001-of-00001.safetensors"
    ).stat().st_ino


def test_promote_hashed_cache_to_sibling(tmp_path):
    checkpoint = _hf_checkpoint(tmp_path)
    cache_root = tmp_path / "scratch"
    hashed = hashed_conversion_cache_path(cache_root, checkpoint, "prime")
    _write_ready_cache(hashed)
    dest = sibling_conversion_cache_path(checkpoint, "prime")

    assert promote_hashed_cache(checkpoint, dest, cache_root=cache_root) == dest
    assert conversion_cache_ready(dest) is True
    # Same-filesystem promote should hardlink when possible.
    hashed_shard = hashed / "model-00001-of-00001.safetensors"
    dest_shard = dest / "model-00001-of-00001.safetensors"
    assert hashed_shard.stat().st_ino == dest_shard.stat().st_ino


def test_promote_hashed_cache_explicit_root_beats_environment(monkeypatch, tmp_path):
    checkpoint = _hf_checkpoint(tmp_path)
    env_root = tmp_path / "env-cache"
    explicit_root = tmp_path / "explicit-cache"
    env_cache = hashed_conversion_cache_path(env_root, checkpoint, "prime")
    explicit_cache = hashed_conversion_cache_path(explicit_root, checkpoint, "prime")
    _write_ready_cache(env_cache)
    _write_ready_cache(explicit_cache)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_DIR", str(env_root))

    dest = sibling_conversion_cache_path(checkpoint, "prime")
    assert promote_hashed_cache(checkpoint, dest, cache_root=explicit_root) == dest

    dest_shard = dest / "model-00001-of-00001.safetensors"
    assert dest_shard.stat().st_ino == (explicit_cache / dest_shard.name).stat().st_ino
    assert dest_shard.stat().st_ino != (env_cache / dest_shard.name).stat().st_ino


def test_main_dry_run(monkeypatch, tmp_path, capsys):
    checkpoint = _hf_checkpoint(tmp_path)
    monkeypatch.setattr(
        "arctic_platform.model.convert_hf_to_primerl.load_state_dict_keys",
        lambda path: ["model.layers.0.mlp.experts.1.up_proj.weight"],
    )
    monkeypatch.setattr(
        "arctic_platform.model.convert_hf_to_primerl.resolve_converter",
        lambda model_type: ConverterSpec(model_type, _FakeConverter, "arctic-platform"),
    )
    assert main(["--dry-run", str(checkpoint)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["needs_conversion"] is True
    assert payload["model_type"] == "glm_moe_dsa"
