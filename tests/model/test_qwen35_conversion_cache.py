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

import pytest


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


@pytest.mark.parametrize("scope", ["node-local", "shared"])
def test_failed_node_local_conversion_removes_tmp_cache(tmp_path, monkeypatch, scope):
    from arctic_platform.model.implementations.qwen35 import conversion_cache

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", scope)
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
    assert sorted(leftovers) == ["prime.failed"]
    assert "disk full" in (cache_path.parent / "prime.failed").read_text()
    assert not conversion_cache.conversion_cache_ready(cache_path)


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


@pytest.mark.parametrize("callback_name", ["stream_convert_fn"])
def test_node_local_conversion_uses_checkpoint_callback_and_reuses_ready_cache(tmp_path, callback_name):
    from arctic_platform.model.implementations.moe import conversion_cache

    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "cache" / "prime"
    converted = []

    def unexpected_bulk_call(*args):
        raise AssertionError("checkpoint conversion must not load the entire state dict")

    def convert_checkpoint(source_path, temporary_path):
        assert source_path == source
        assert temporary_path.parent == destination.parent
        assert not destination.exists()
        converted.append(temporary_path)
        _write_ready_cache(temporary_path)

    for _ in range(2):
        conversion_cache.ensure_node_local_conversion_cache(
            source,
            destination,
            unexpected_bulk_call,
            "HF",
            "PrimeRL",
            unexpected_bulk_call,
            unexpected_bulk_call,
            rank=0,
            local_rank=0,
            **{callback_name: convert_checkpoint},
        )

    assert len(converted) == 1
    assert not converted[0].exists()
    assert conversion_cache.conversion_cache_ready(destination)


def test_bulk_conversion_preserves_in_place_contract_when_converter_returns_other_data(tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache

    saved = []
    original = {"original": object()}

    def convert(state):
        assert state is original
        state["converted"] = state.pop("original")
        return {"not_a_full_state_dict": "converter metadata"}

    def save(state, path):
        saved.append(state.copy())
        _write_ready_cache(path)

    conversion_cache.ensure_node_local_conversion_cache(
        tmp_path,
        tmp_path / "prime",
        convert,
        "HF",
        "PrimeRL",
        lambda path: original,
        save,
        rank=0,
        local_rank=0,
    )
    assert saved == [original]
    assert set(saved[0]) == {"converted"}


@pytest.mark.parametrize("failure", ["exception", "missing-index", "missing-shard"])
@pytest.mark.parametrize("callback_name", ["stream_convert_fn"])
def test_checkpoint_callback_failure_never_publishes_cache(tmp_path, failure, callback_name):
    from arctic_platform.model.implementations.moe import conversion_cache

    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "cache" / "prime"

    def unexpected_bulk_call(*args):
        raise AssertionError("checkpoint callback must bypass bulk conversion")

    def convert_checkpoint(source_path, temporary_path):
        assert source_path == source
        temporary_path.mkdir()
        if failure == "exception":
            raise RuntimeError("injected checkpoint conversion failure")
        if failure == "missing-index":
            (temporary_path / "model.safetensors").write_bytes(b"partial")
        else:
            (temporary_path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "missing.safetensors"}})
            )

    with pytest.raises(RuntimeError, match="conversion failure|complete checkpoint"):
        conversion_cache.ensure_node_local_conversion_cache(
            source,
            destination,
            unexpected_bulk_call,
            "HF",
            "PrimeRL",
            unexpected_bulk_call,
            unexpected_bulk_call,
            rank=0,
            local_rank=0,
            **{callback_name: convert_checkpoint},
        )

    assert not destination.exists()
    assert not list(destination.parent.glob("prime.tmp-*"))


def test_preparation_failure_does_not_wait_for_blocked_heartbeat(monkeypatch, tmp_path):
    import threading
    from types import SimpleNamespace

    from arctic_platform.model.implementations.moe import conversion_cache as cache

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    native = cache._write_cache_marker
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", "launch")
    monkeypatch.setattr(cache, "_CACHE_HEARTBEAT_JOIN_TIMEOUT", 0.01)

    def write(path, value):
        if path.name.endswith(".in-progress"):
            entered.set()
            assert release.wait(5)
            finished.set()
        else:
            native(path, value)

    monkeypatch.setattr(cache, "_write_cache_marker", write)

    @cache.record_conversion_preparation_failure
    def prepare(config, rank):
        assert entered.wait(5)
        raise ValueError("original setup failure")

    try:
        with pytest.raises(ValueError, match="original setup failure"):
            prepare(SimpleNamespace(weight_conversion_cache_dir=tmp_path), rank=0)
        failure = cache._read_cache_marker(cache._preparation_marker_path("launch", tmp_path).with_suffix(".failed"))
        assert failure["error"] == "original setup failure"
        assert cache._ACTIVE_ATTEMPT.get() is None
    finally:
        release.set()
        assert finished.wait(5)


def test_conversion_marker_write_failure_removes_only_its_own_temporary_file(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache

    other = tmp_path / "prime.in-progress.other.tmp"
    other.write_bytes(b"another writer")

    def fail(*args):
        raise OSError("marker rename failed")

    monkeypatch.setattr(conversion_cache.os, "replace", fail)
    with pytest.raises(OSError, match="marker rename failed"):
        conversion_cache._write_cache_marker(tmp_path / "prime.in-progress", dict(attempt_id="this-attempt"))
    assert list(tmp_path.iterdir()) == [other]


def test_abandoned_conversion_sweep_removes_stale_owner_and_destination_artifacts(tmp_path):
    import os
    import time

    from arctic_platform.model.implementations.moe import conversion_cache as cache

    destination = tmp_path / "prime"
    old = tmp_path / "prime.tmp-rank0-pid123-old"
    old.mkdir()
    owner = old.with_name(f"{old.name}.owner")
    owner.touch()
    orphan = tmp_path / "prime.tmp-rank0-pid123-orphan.owner"
    orphan.touch()
    stale = tmp_path / "prime.stale-old"
    stale.mkdir()
    recent = tmp_path / "prime.stale-recent"
    recent.mkdir()
    timestamp = time.time() - 61
    for artifact in (owner, orphan, stale):
        os.utime(artifact, (timestamp, timestamp))
    cache._cleanup_abandoned_conversion_workspaces(destination, node_local=True)
    assert not old.exists()
    assert not owner.exists()
    assert not orphan.exists()
    assert not stale.exists()
    assert not recent.exists()


def test_conversion_cache_real_heartbeat_recovers_while_waiter_observes_progress(monkeypatch, tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from arctic_platform.model.implementations.moe import conversion_cache

    cache = tmp_path / "prime"
    writer_started, reader_started, recovered = threading.Event(), threading.Event(), threading.Event()
    original_write = conversion_cache._write_cache_marker
    original_read = conversion_cache._read_cache_marker
    heartbeats, observed = [], []

    def marker(path, value):
        if path.name.endswith(".in-progress"):
            heartbeats.append(value["heartbeat"])
            if value["heartbeat"] == 1:
                raise OSError("transient shared filesystem error")
        original_write(path, value)

    def read(path):
        value = original_read(path)
        if threading.current_thread().name.startswith("cache-reader") and value.get("heartbeat", 0) >= 2:
            observed.append(value["heartbeat"])
            recovered.set()
        return value

    def convert(source, destination):
        writer_started.set()
        assert reader_started.wait(3)
        assert recovered.wait(3), "waiter must observe real heartbeat progress after the failed write"
        _write_ready_cache(destination)

    def wait():
        reader_started.set()
        conversion_cache.wait_for_conversion_cache(
            cache, timeout=3, poll_interval=0.005, writer_idle_timeout=1, attempt_id="heartbeat-retry"
        )

    monkeypatch.setattr(conversion_cache, "_CACHE_HEARTBEAT_INTERVAL", 0.02)
    monkeypatch.setattr(conversion_cache, "_write_cache_marker", marker)
    monkeypatch.setattr(conversion_cache, "_read_cache_marker", read)
    with (
        ThreadPoolExecutor(max_workers=1) as writer_pool,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-reader") as reader_pool,
    ):
        writer = writer_pool.submit(
            conversion_cache.ensure_node_local_conversion_cache,
            tmp_path,
            cache,
            lambda state: None,
            "HF",
            "PrimeRL",
            lambda path: {},
            lambda *args: None,
            rank=0,
            local_rank=0,
            stream_convert_fn=convert,
            attempt_id="heartbeat-retry",
        )
        assert writer_started.wait(3)
        reader = reader_pool.submit(wait)
        writer.result(timeout=5)
        reader.result(timeout=5)
    assert heartbeats[:3] == [0, 1, 2]
    assert len(observed) > 0
    assert conversion_cache.conversion_cache_ready(cache)
    assert not cache.with_name("prime.in-progress").exists()


def test_shared_conversion_sweeps_observed_dead_workspace_during_conversion(monkeypatch, tmp_path):
    import time

    from arctic_platform.model.implementations.moe import conversion_cache as cache

    destination = tmp_path / "prime"
    orphan = destination.with_name("prime.tmp-rank0-pid123-old")
    orphan.mkdir()
    owner = orphan.with_name(f"{orphan.name}.owner")
    cache._write_cache_marker(owner, dict(host="another-node", pid=123))
    monkeypatch.setattr(cache, "_ABANDONED_CACHE_AGE", 0.02)
    monkeypatch.setattr(cache, "_CACHE_HEARTBEAT_INTERVAL", 0.01)

    def convert(source, output):
        deadline = time.monotonic() + 2
        while orphan.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not orphan.exists(), "shared cleanup never made a second liveness observation"
        assert not owner.exists()
        _write_ready_cache(output)

    cache.ensure_node_local_conversion_cache(
        tmp_path,
        destination,
        lambda state: None,
        "HF",
        "PrimeRL",
        lambda path: {},
        lambda *args: None,
        rank=0,
        local_rank=0,
        stream_convert_fn=convert,
        node_local=False,
    )
    assert cache.conversion_cache_ready(destination)


def test_shared_workspace_cleanup_observes_generations_not_server_clock(monkeypatch, tmp_path):
    import os

    from arctic_platform.model.implementations.moe import conversion_cache as cache

    destination = tmp_path / "prime"
    live = tmp_path / "prime.tmp-rank0-pid123-live"
    dead = tmp_path / "prime.tmp-rank1-pid124"
    live.mkdir()
    dead.mkdir()
    owner = live.with_name(f"{live.name}.owner")
    cache._write_cache_marker(owner, dict(host="other-node", pid=123, heartbeat=0))
    os.utime(owner, ns=(1, 1))
    clock = [0]
    monkeypatch.setattr(cache.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cache.time, "time", lambda: pytest.fail("server and client clocks are not comparable"))
    monkeypatch.setattr(cache, "_WORKSPACE_OBSERVATIONS", {})
    cache._cleanup_abandoned_conversion_workspaces(destination)
    assert live.exists() and dead.exists()
    clock[0] = cache._ABANDONED_CACHE_AGE + 1
    cache._write_cache_marker(owner, dict(host="other-node", pid=123, heartbeat=1))
    os.utime(owner, ns=(1, 1))
    cache._cleanup_abandoned_conversion_workspaces(destination)
    assert live.exists()
    assert not dead.exists()


def test_node_local_locked_retry_reclaims_old_format_workspace_immediately(tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache as cache

    destination = tmp_path / "prime"
    workspace = tmp_path / "prime.tmp-rank0-pid123"
    workspace.mkdir()
    (workspace / "partial-shard").touch()
    with cache._conversion_writer_lock(tmp_path / ".prime.lock", node_local=True):
        cache._cleanup_abandoned_conversion_workspaces(destination, node_local=True)
    assert not workspace.exists()


@pytest.mark.parametrize("live", [False, True])
def test_preparation_heartbeat_covers_slow_setup_and_detects_dead_setup(monkeypatch, tmp_path, live):
    from arctic_platform.model.implementations.moe import conversion_cache as cache

    marker = cache._preparation_marker_path("launch", tmp_path).with_suffix(".in-progress")
    destination = tmp_path / "prime"
    clock = [0]
    monkeypatch.setattr(cache.time, "monotonic", lambda: clock[0])

    def sleep(delay):
        clock[0] += delay
        if clock[0] == 1 or live:
            cache._write_cache_marker(marker, dict(attempt_id="launch", heartbeat=clock[0]))
        if live and clock[0] == 6:
            destination.mkdir()
            (destination / "weights.safetensors").touch()
            (destination / "model.safetensors.index.json").write_text(
                json.dumps(dict(weight_map=dict(weight="weights.safetensors")))
            )

    monkeypatch.setattr(cache.time, "sleep", sleep)
    arguments = dict(
        timeout=10,
        poll_interval=1,
        writer_idle_timeout=2,
        writer_start_timeout=2,
        attempt_id="launch",
        preparation_root=tmp_path,
    )
    if live:
        cache.wait_for_conversion_cache(destination, **arguments)
        assert clock[0] == 6
    else:
        with pytest.raises(RuntimeError, match="writer stopped heartbeating"):
            cache.wait_for_conversion_cache(destination, **arguments)
        assert clock[0] == 3


@pytest.mark.parametrize("fresh", [False, True])
def test_sibling_preparation_failure_is_visible_outside_writer_cwd(monkeypatch, tmp_path, fresh):
    from types import SimpleNamespace

    from arctic_platform.model.implementations.moe import conversion_cache as cache

    source, cwd = tmp_path / "source", tmp_path / "cwd"
    source.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", "sibling-launch")
    config = SimpleNamespace(name=str(source), weight_conversion_cache_dir="")

    @cache.record_conversion_preparation_failure
    def prepare(config, rank):
        raise ValueError("source discovery failed")

    with pytest.raises(ValueError, match="source discovery failed"):
        prepare(config, rank=0)
    root = cache.conversion_preparation_root(config, source)
    assert root == source
    assert len(list(source.glob(".preparation-*.failed"))) == 1
    assert list(cwd.iterdir()) == []
    error = RuntimeError if fresh else TimeoutError
    with pytest.raises(error, match="source discovery failed"):
        cache.wait_for_conversion_cache(
            source / "prime", timeout=0, attempt_id="sibling-launch", fresh_attempt=fresh, preparation_root=root
        )
