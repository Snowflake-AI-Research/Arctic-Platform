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

import builtins
import json
from pathlib import Path


def test_conversion_cache_ready_requires_index_and_referenced_shards(tmp_path):
    from arctic_platform.model import conversion_cache_ready

    cache_dir = tmp_path / "prime"
    cache_dir.mkdir()

    assert conversion_cache_ready(cache_dir) is False

    (cache_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}})
    )
    (cache_dir / "model-00001-of-00002.safetensors").write_bytes(b"not-used")

    assert conversion_cache_ready(cache_dir) is False

    (cache_dir / "model-00002-of-00002.safetensors").write_bytes(b"not-used")

    assert conversion_cache_ready(cache_dir) is True


def test_a_state_dict_too_small_to_shard_still_gets_an_index(tmp_path):
    """A converted checkpoint the cache can read, whatever its size.

    ``save_state_dict`` writes the index last, and ``conversion_cache_ready`` treats its presence as the mark of
    a complete cache. Sharding only kicks in above a size threshold, so without an index for the single-file
    case every checkpoint under it converts successfully and is then rejected as incomplete on the next look,
    which fails the load rather than falling back.
    """
    import torch

    from arctic_platform.model import conversion_cache_ready
    from arctic_platform.model import save_exported_state_dict

    save_dir = tmp_path / "prime"
    save_exported_state_dict({"a": torch.zeros(4), "b": torch.ones(2, 2)}, save_dir)

    index_path = save_dir / "model.safetensors.index.json"
    assert index_path.is_file(), f"no index written; directory holds {sorted(p.name for p in save_dir.iterdir())}"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    assert set(weight_map) == {"a", "b"}
    assert all((save_dir / shard).is_file() for shard in set(weight_map.values()))
    assert conversion_cache_ready(save_dir) is True


def test_conversion_cache_scope_detects_data_fast_and_env_override(monkeypatch):
    from arctic_platform.model import conversion_cache_is_node_local

    assert conversion_cache_is_node_local(Path("/data-fast/prime-rl-weight-cache/model/prime")) is True
    assert conversion_cache_is_node_local(Path("/tmp/prime-rl-weight-cache/model/prime")) is True
    assert conversion_cache_is_node_local(Path("/checkpoint/prime-rl-weight-cache/model/prime")) is False

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "node-local")
    assert conversion_cache_is_node_local(Path("/checkpoint/prime-rl-weight-cache/model/prime")) is True

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "shared")
    assert conversion_cache_is_node_local(Path("/data-fast/prime-rl-weight-cache/model/prime")) is False
    assert conversion_cache_is_node_local(Path("/tmp/prime-rl-weight-cache/model/prime")) is False


def test_node_local_conversion_cache_uses_lock_and_skips_ready_cache(tmp_path):
    from arctic_platform.model.implementations.qwen35 import conversion_cache

    source_path = tmp_path / "source"
    source_path.mkdir()
    cache_path = tmp_path / "cache" / "prime"
    calls = {"load": 0, "convert": 0, "save": 0}

    def fake_load_state_dict(path):
        assert path == source_path
        calls["load"] += 1
        return {"a": object()}

    def fake_convert(state_dict):
        calls["convert"] += 1
        state_dict["b"] = state_dict.pop("a")

    def fake_save_state_dict(state_dict, save_dir):
        calls["save"] += 1
        save_dir.mkdir(parents=True)
        (save_dir / "model-00001-of-00001.safetensors").write_bytes(b"not-used")
        (save_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"b": "model-00001-of-00001.safetensors"}})
        )

    conversion_cache.ensure_node_local_conversion_cache(
        source_path,
        cache_path,
        fake_convert,
        "HF",
        "PrimeRL",
        fake_load_state_dict,
        fake_save_state_dict,
        rank=0,
        local_rank=0,
    )
    conversion_cache.ensure_node_local_conversion_cache(
        source_path,
        cache_path,
        fake_convert,
        "HF",
        "PrimeRL",
        fake_load_state_dict,
        fake_save_state_dict,
        rank=0,
        local_rank=0,
    )

    assert calls == {"load": 1, "convert": 1, "save": 1}
    assert conversion_cache.conversion_cache_ready(cache_path) is True


def test_failed_node_local_conversion_removes_tmp_cache(tmp_path):
    from arctic_platform.model.implementations.qwen35 import conversion_cache

    source_path = tmp_path / "source"
    source_path.mkdir()
    cache_path = tmp_path / "cache" / "prime"

    def fake_save_state_dict(state_dict, save_dir):
        save_dir.mkdir(parents=True)
        (save_dir / "model-00001-of-00001.safetensors").write_bytes(b"partial")
        raise RuntimeError("disk full")

    try:
        conversion_cache.ensure_node_local_conversion_cache(
            source_path,
            cache_path,
            lambda state_dict: None,
            "HF",
            "PrimeRL",
            lambda path: {"a": object()},
            fake_save_state_dict,
            rank=0,
            local_rank=0,
        )
        raise AssertionError("expected conversion to fail")
    except RuntimeError as exc:
        assert "disk full" in str(exc)

    leftovers = [p.name for p in cache_path.parent.iterdir() if p.name == "prime" or p.name.startswith("prime.")]
    assert leftovers == []


def test_forced_node_local_ready_sibling_does_not_write_or_lock(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from arctic_platform.model.implementations.qwen35 import conversion_cache

    source_path = tmp_path / "source"
    source_path.mkdir()
    sibling = conversion_cache.sibling_conversion_cache_path(source_path, "prime")
    _write_ready_cache(sibling)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "node")

    snapshot_path = conversion_cache.resolve_conversion_cache_path(
        SimpleNamespace(weight_conversion_cache_dir=str(tmp_path / "scratch")),
        source_path,
        "prime",
    )
    assert snapshot_path == sibling
    assert conversion_cache.conversion_cache_is_node_local(snapshot_path) is True

    real_open = builtins.open

    def guarded_open(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise AssertionError(f"unexpected write attempt: {path}")
        return real_open(path, mode, *args, **kwargs)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("ready immutable sibling must not require writes or locks")

    monkeypatch.setattr(conversion_cache, "open", guarded_open, raising=False)
    monkeypatch.setattr(Path, "mkdir", unexpected_call)
    monkeypatch.setattr(conversion_cache.fcntl, "flock", unexpected_call)

    conversion_cache.ensure_node_local_conversion_cache(
        source_path,
        snapshot_path,
        unexpected_call,
        "HF",
        "PrimeRL",
        unexpected_call,
        unexpected_call,
        rank=0,
        local_rank=0,
    )
    assert not (source_path / ".prime.lock").exists()


def _write_ready_cache(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "model-00001-of-00001.safetensors").write_bytes(b"not-used")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}})
    )


def test_resolve_conversion_cache_prefers_ready_sibling_over_hashed_scratch(tmp_path):
    from types import SimpleNamespace

    from arctic_platform.model.implementations.moe.conversion_cache import hashed_conversion_cache_path
    from arctic_platform.model.implementations.moe.conversion_cache import resolve_conversion_cache_path
    from arctic_platform.model.implementations.moe.conversion_cache import sibling_conversion_cache_path

    source = tmp_path / "GLM-5.2"
    source.mkdir()
    sibling = sibling_conversion_cache_path(source, "prime")
    _write_ready_cache(sibling)

    config = SimpleNamespace(weight_conversion_cache_dir=str(tmp_path / "scratch"))
    resolved = resolve_conversion_cache_path(config, source, "prime")
    assert resolved == sibling
    assert resolved != hashed_conversion_cache_path(config.weight_conversion_cache_dir, source, "prime")


def test_resolve_conversion_cache_uses_hashed_scratch_when_sibling_missing(tmp_path):
    from types import SimpleNamespace

    from arctic_platform.model.implementations.moe.conversion_cache import hashed_conversion_cache_path
    from arctic_platform.model.implementations.moe.conversion_cache import resolve_conversion_cache_path
    from arctic_platform.model.implementations.moe.conversion_cache import sibling_conversion_cache_path

    source = tmp_path / "GLM-5.2"
    source.mkdir()
    config = SimpleNamespace(weight_conversion_cache_dir=str(tmp_path / "scratch"))
    resolved = resolve_conversion_cache_path(config, source, "prime")
    assert resolved == hashed_conversion_cache_path(config.weight_conversion_cache_dir, source, "prime")
    assert resolved != sibling_conversion_cache_path(source, "prime")


def test_resolve_conversion_cache_falls_back_to_sibling_without_scratch_root(tmp_path):
    from types import SimpleNamespace

    from arctic_platform.model.implementations.moe.conversion_cache import resolve_conversion_cache_path
    from arctic_platform.model.implementations.moe.conversion_cache import sibling_conversion_cache_path

    source = tmp_path / "GLM-5.2"
    source.mkdir()
    config = SimpleNamespace(weight_conversion_cache_dir="")
    assert resolve_conversion_cache_path(config, source, "prime") == sibling_conversion_cache_path(source, "prime")


def test_conversion_cache_root_precedence_matches_runtime(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.glm52.config import ModelConfig as GlmModelConfig
    from arctic_platform.model.implementations.moe.conversion_cache import DEFAULT_WEIGHT_CONVERSION_CACHE_DIR
    from arctic_platform.model.implementations.moe.conversion_cache import hashed_conversion_cache_path
    from arctic_platform.model.implementations.moe.conversion_cache import resolve_conversion_cache_path
    from arctic_platform.model.implementations.moe.conversion_cache import resolve_conversion_cache_root
    from arctic_platform.model.implementations.moe.conversion_cache import sibling_conversion_cache_path
    from arctic_platform.model.implementations.qwen35.config import ModelConfig as QwenModelConfig

    source = tmp_path / "GLM-5.2"
    source.mkdir()
    env_root = tmp_path / "env-cache"
    explicit_root = tmp_path / "config-cache"

    monkeypatch.delenv("DSS_WEIGHT_CONVERSION_CACHE_DIR", raising=False)
    assert resolve_conversion_cache_root() == DEFAULT_WEIGHT_CONVERSION_CACHE_DIR
    for config_type in (QwenModelConfig, GlmModelConfig):
        default_config = config_type()
        assert default_config.weight_conversion_cache_dir is None
        assert resolve_conversion_cache_path(default_config, source, "prime") == hashed_conversion_cache_path(
            DEFAULT_WEIGHT_CONVERSION_CACHE_DIR, source, "prime"
        )

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_DIR", "")
    assert resolve_conversion_cache_root() == DEFAULT_WEIGHT_CONVERSION_CACHE_DIR

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_DIR", str(env_root))
    assert resolve_conversion_cache_root() == str(env_root)
    assert resolve_conversion_cache_root(explicit_root) == explicit_root

    for config_type in (QwenModelConfig, GlmModelConfig):
        default_config = config_type()
        assert resolve_conversion_cache_path(default_config, source, "prime") == hashed_conversion_cache_path(
            env_root, source, "prime"
        )

        explicit_config = config_type(weight_conversion_cache_dir=str(explicit_root))
        assert resolve_conversion_cache_path(explicit_config, source, "prime") == hashed_conversion_cache_path(
            explicit_root, source, "prime"
        )

        sibling_config = config_type(weight_conversion_cache_dir="")
        assert resolve_conversion_cache_path(sibling_config, source, "prime") == sibling_conversion_cache_path(
            source, "prime"
        )
