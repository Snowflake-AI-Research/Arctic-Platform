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


@pytest.mark.parametrize("race", [False, True])
def test_conversion_publication_recovers_incomplete_destination_and_concurrent_winner(monkeypatch, tmp_path, race):
    import errno

    from arctic_platform.model.implementations.moe import conversion_cache as cache

    destination = tmp_path / "prime"
    destination.mkdir()
    (destination / "old-shard.safetensors").touch()
    native_replace = cache.os.replace
    raced = []

    def ready(path, name):
        path.mkdir(exist_ok=True)
        (path / name).touch()
        (path / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map=dict(weight=name))))

    def replace(source, target):
        if race and target == destination and not raced:
            raced.append(True)
            ready(destination, "winner.safetensors")
            raise OSError(errno.ENOTEMPTY, "concurrent publisher")
        return native_replace(source, target)

    monkeypatch.setattr(cache.os, "replace", replace)
    cache._write_conversion_cache(
        tmp_path,
        destination,
        lambda state: None,
        "HF",
        "Prime",
        lambda path: {},
        lambda state, target: ready(target, "candidate.safetensors"),
        rank=0,
        local_rank=0,
    )
    assert cache.conversion_cache_ready(destination)
    assert (destination / ("winner.safetensors" if race else "candidate.safetensors")).exists()
    assert list(tmp_path.glob("prime.stale-*")) == []
    assert list(tmp_path.glob("prime.tmp-*")) == []


@pytest.mark.parametrize("marker", ["failed", "in-progress", "preparation"])
def test_reused_attempt_waiter_ignores_baseline_generations_before_late_writer(monkeypatch, tmp_path, marker):
    from arctic_platform.model.implementations.moe import conversion_cache as cache

    destination = tmp_path / "prime"
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", "reused")
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_DIR", str(tmp_path))
    marker_path = (
        cache._preparation_marker_path("reused").with_suffix(".failed")
        if marker == "preparation"
        else tmp_path / f"prime.{marker}"
    )
    cache._write_cache_marker(marker_path, dict(attempt_id="reused", writer_id="old", heartbeat=0, error="stale"))
    clock = [0]
    monkeypatch.setattr(cache.time, "monotonic", lambda: clock[0])

    def sleep(delay):
        clock[0] += delay
        if clock[0] == 35:
            cache._write_cache_marker(
                tmp_path / "prime.in-progress", dict(attempt_id="reused", writer_id="new", heartbeat=0)
            )
        if clock[0] == 36:
            destination.mkdir()
            (destination / "weights.safetensors").touch()
            (destination / "model.safetensors.index.json").write_text(
                json.dumps(dict(weight_map=dict(weight="weights.safetensors")))
            )

    monkeypatch.setattr(cache.time, "sleep", sleep)
    cache.wait_for_conversion_cache(
        destination,
        timeout=60,
        poll_interval=1,
        writer_idle_timeout=2,
        writer_start_timeout=60,
        attempt_id="reused",
    )
    assert clock[0] == 36


@pytest.mark.parametrize("stale_progress", [False, True])
def test_local_nonzero_rank_can_start_retry_with_reused_failure_id(monkeypatch, tmp_path, stale_progress):
    from arctic_platform.model.implementations.moe import conversion_cache as cache

    destination = tmp_path / "prime"
    cache._write_cache_marker(tmp_path / "prime.failed", dict(attempt_id="reused", writer_id="old", error="old"))
    if stale_progress:
        cache._write_cache_marker(
            tmp_path / "prime.in-progress", dict(attempt_id="reused", writer_id="old", heartbeat=0)
        )

    def save(state, target):
        target.mkdir()
        (target / "weights.safetensors").touch()
        (target / "model.safetensors.index.json").write_text(
            json.dumps(dict(weight_map=dict(weight="weights.safetensors")))
        )

    cache.ensure_node_local_conversion_cache(
        tmp_path,
        destination,
        lambda state: None,
        "HF",
        "Prime",
        lambda path: {},
        save,
        rank=1,
        local_rank=1,
        attempt_id="reused",
        node_local=True,
    )
    assert cache.conversion_cache_ready(destination)


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
    assert sorted(leftovers) == ["prime.failed"]
    assert "disk full" in (cache_path.parent / "prime.failed").read_text()


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


def test_conversion_cache_retry_allows_late_writer_and_clock_skew(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache

    cache = tmp_path / "prime"
    conversion_cache._write_cache_marker(cache.with_name("prime.failed"), dict(attempt_id="old", error="stale"))
    conversion_cache._write_cache_marker(cache.with_name("prime.in-progress"), dict(attempt_id="old", heartbeat=0))
    clock = [0.0]
    monkeypatch.setattr(conversion_cache.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(conversion_cache.time, "time", lambda: pytest.fail("must not compare wall clocks"))
    monkeypatch.setattr(conversion_cache.fcntl, "flock", lambda *args: pytest.fail("must not probe shared flock"))
    monkeypatch.setattr(conversion_cache, "conversion_cache_ready", lambda path: clock[0] >= 36)

    def sleep(delay):
        clock[0] += delay
        if clock[0] == 35:
            conversion_cache._write_cache_marker(
                cache.with_name("prime.in-progress"), dict(attempt_id="retry", heartbeat=0)
            )

    monkeypatch.setattr(conversion_cache.time, "sleep", sleep)
    conversion_cache.wait_for_conversion_cache(cache, timeout=60, poll_interval=1, attempt_id="retry")
    assert clock[0] == 36


def test_conversion_cache_detects_killed_writer_from_heartbeat(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache

    cache = tmp_path / "prime"
    conversion_cache._write_cache_marker(cache.with_name("prime.in-progress"), dict(attempt_id="killed", heartbeat=0))
    clock = [0.0]
    monkeypatch.setattr(conversion_cache.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(conversion_cache.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    with pytest.raises(RuntimeError, match="writer stopped heartbeating"):
        conversion_cache.wait_for_conversion_cache(
            cache, timeout=3600, writer_idle_timeout=2, attempt_id="killed", fresh_attempt=True
        )
    assert clock[0] == 2


def test_conversion_cache_local_failure_is_not_repeated_by_waiting_rank(monkeypatch, tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from arctic_platform.model.implementations.moe import conversion_cache

    monkeypatch.delenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", raising=False)
    cache = tmp_path / "prime"
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    calls = []
    native_flock = conversion_cache.fcntl.flock

    def flock(fd, operation):
        if operation == conversion_cache.fcntl.LOCK_EX and entered.is_set():
            waiting.set()
        return native_flock(fd, operation)

    def fail(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(3)
        raise ValueError("conversion failed \ud800")

    monkeypatch.setattr(conversion_cache.fcntl, "flock", flock)
    monkeypatch.setattr(conversion_cache, "_write_conversion_cache", fail)
    args = (tmp_path, cache, lambda state: None, "HF", "PrimeRL", lambda path: {}, lambda *args: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(conversion_cache.ensure_node_local_conversion_cache, *args, rank=0, local_rank=0)
        assert entered.wait(3)
        reader = pool.submit(conversion_cache.ensure_node_local_conversion_cache, *args, rank=1, local_rank=1)
        assert waiting.wait(3)
        release.set()
        with pytest.raises(ValueError, match="conversion failed"):
            writer.result(timeout=3)
        with pytest.raises(RuntimeError, match="writer failed"):
            reader.result(timeout=3)
    assert calls == [1]


@pytest.mark.parametrize("scope", ["node-local", "shared"])
def test_conversion_cache_late_rank_sees_current_attempt_failure(monkeypatch, tmp_path, scope):
    from arctic_platform.model.implementations.moe import conversion_cache

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", "launch")
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise KeyboardInterrupt("injected interrupt")

    monkeypatch.setattr(conversion_cache, "_write_conversion_cache", fail)
    cache = tmp_path / "prime"
    args = (tmp_path, cache, lambda state: None, "HF", "PrimeRL", lambda path: {}, lambda *args: None)
    with pytest.raises(KeyboardInterrupt, match="injected interrupt"):
        conversion_cache.ensure_node_local_conversion_cache(*args, rank=0, local_rank=0)
    with pytest.raises(RuntimeError, match="writer failed.*injected interrupt"):
        if scope == "shared":
            conversion_cache.wait_for_conversion_cache(cache, timeout=0, fresh_attempt=True)
        else:
            conversion_cache.ensure_node_local_conversion_cache(*args, rank=1, local_rank=1, fresh_attempt=True)
    assert calls == [1]


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


@pytest.mark.parametrize("phase", ["lock", "first-marker"])
def test_conversion_cache_setup_failure_reaches_waiter_without_progress(monkeypatch, tmp_path, phase):
    from arctic_platform.model.implementations.moe import conversion_cache

    cache = tmp_path / "prime"

    if phase == "lock":

        def fail(*args):
            raise OSError("writer setup failed")

        monkeypatch.setattr(conversion_cache.fcntl, "flock", fail)
    else:
        original = conversion_cache._write_cache_marker

        def marker(path, value):
            if path.name.endswith(".in-progress"):
                raise OSError("writer setup failed")
            original(path, value)

        monkeypatch.setattr(conversion_cache, "_write_cache_marker", marker)
    with pytest.raises(OSError, match="writer setup failed"):
        conversion_cache.ensure_node_local_conversion_cache(
            tmp_path,
            cache,
            lambda state: None,
            "HF",
            "PrimeRL",
            lambda path: {},
            lambda *args: None,
            rank=0,
            local_rank=0,
            attempt_id="setup-failure",
        )
    assert not cache.with_name("prime.in-progress").exists()
    with pytest.raises(RuntimeError, match="writer failed.*writer setup failed"):
        conversion_cache.wait_for_conversion_cache(cache, timeout=0, attempt_id="setup-failure", fresh_attempt=True)


def test_conversion_cache_preserves_other_writer_temporary_files_without_coherent_lock(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache

    cache = tmp_path / "prime"
    other = cache.with_name("prime.tmp-rank0-pid123")
    other.mkdir()
    (other / "live-shard").write_bytes(b"other node is still writing")
    failed = cache.with_name("prime.failed")
    conversion_cache._write_cache_marker(failed, dict(attempt_id="other-attempt", error="other failure"))
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "node-local")
    monkeypatch.setattr(conversion_cache.fcntl, "flock", lambda *args: None)

    def convert(source, destination):
        assert destination != other
        assert (other / "live-shard").read_bytes() == b"other node is still writing"
        _write_ready_cache(destination)

    conversion_cache.ensure_node_local_conversion_cache(
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
        attempt_id="this-attempt",
    )
    assert (other / "live-shard").read_bytes() == b"other node is still writing"
    assert not failed.exists()
    assert conversion_cache.conversion_cache_ready(cache)


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


@pytest.mark.parametrize("same_attempt", [False, True])
def test_shared_conversion_retries_without_filesystem_lock_support(monkeypatch, tmp_path, same_attempt):
    import errno

    from arctic_platform.model.implementations.moe import conversion_cache

    destination = tmp_path / "prime"
    conversion_cache._write_cache_marker(
        destination.with_name("prime.failed"),
        dict(attempt_id="retry" if same_attempt else "old", error="previous failure"),
    )

    def unsupported(*args):
        raise OSError(errno.ENOLCK, "shared filesystem has no flock")

    monkeypatch.setattr(conversion_cache.fcntl, "flock", unsupported)
    conversion_cache.ensure_node_local_conversion_cache(
        tmp_path,
        destination,
        lambda state: None,
        "HF",
        "Prime",
        lambda path: {},
        lambda *args: None,
        rank=0,
        local_rank=0,
        attempt_id="retry",
        node_local=False,
        stream_convert_fn=lambda source, target: _write_ready_cache(target),
    )
    assert conversion_cache.conversion_cache_ready(destination)
    assert not destination.with_name("prime.failed").exists()


def test_shared_waiter_observes_failure_before_first_progress_marker(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache

    native = conversion_cache._write_cache_marker

    def marker(path, value):
        if path.name.endswith(".in-progress"):
            raise OSError("startup marker failed")
        native(path, value)

    monkeypatch.setattr(conversion_cache, "_write_cache_marker", marker)
    destination = tmp_path / "prime"
    with pytest.raises(OSError, match="startup marker failed"):
        conversion_cache.ensure_node_local_conversion_cache(
            tmp_path,
            destination,
            lambda state: None,
            "HF",
            "Prime",
            lambda path: {},
            lambda *args: None,
            rank=0,
            local_rank=0,
            attempt_id="setup",
            node_local=False,
        )
    with pytest.raises(RuntimeError, match="writer failed.*startup marker failed"):
        conversion_cache.wait_for_conversion_cache(destination, timeout=0, attempt_id="setup", fresh_attempt=True)


def test_pre_cache_preparation_failure_reaches_shared_waiters(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", "source-failure")

    @conversion_cache.record_conversion_preparation_failure
    def prepare(*, rank):
        raise RuntimeError("checkpoint discovery failed")

    with pytest.raises(RuntimeError, match="checkpoint discovery failed"):
        prepare(rank=0)
    with pytest.raises(RuntimeError, match="Weight preparation writer failed.*checkpoint discovery failed"):
        conversion_cache.wait_for_conversion_cache(
            tmp_path / "prime", timeout=0, attempt_id="source-failure", fresh_attempt=True
        )


def test_shared_waiter_bounds_writer_death_before_startup(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache

    clock = [0.0]
    monkeypatch.setattr(conversion_cache.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(conversion_cache.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    with pytest.raises(TimeoutError, match="startup progress within 40"):
        conversion_cache.wait_for_conversion_cache(
            tmp_path / "prime",
            timeout=3600,
            writer_start_timeout=40,
            poll_interval=1,
            attempt_id="killed-before-start",
        )
    assert clock[0] == 40


def test_cleanup_preserves_live_workspace_and_removes_only_abandoned_workspaces(tmp_path):
    import os

    from arctic_platform.model.implementations.moe import conversion_cache

    destination = tmp_path / "prime"
    old, live, recent = [tmp_path / f"prime.tmp-rank0-pid123-{name}" for name in ("old", "live", "recent")]
    for directory in (old, live, recent):
        directory.mkdir()
    ancient = conversion_cache.time.time() - conversion_cache._ABANDONED_CACHE_AGE - 60
    for directory in (old, live):
        os.utime(directory, (ancient, ancient))
    live.with_name(f"{live.name}.owner").touch()
    conversion_cache._cleanup_abandoned_conversion_workspaces(destination)
    assert not old.exists()
    assert live.exists()
    assert recent.exists()


@pytest.mark.parametrize("family", ["qwen35", "glm52"])
@pytest.mark.parametrize("scope", ["shared", "node-local"])
def test_production_loading_propagates_fresh_attempt_and_early_failure(monkeypatch, tmp_path, family, scope):
    import importlib
    from types import SimpleNamespace

    import torch.distributed as dist

    from arctic_platform.model.implementations.moe import conversion_cache as cache

    builder = importlib.import_module(f"arctic_platform.model.implementations.{family}.model_builder")

    class Model:
        def to_empty(self, *, device):
            return self

        def state_dict(self):
            return {"prime": None}

        def is_hf_state_dict(self, state):
            return "hf" in state

        def is_prime_state_dict(self, state):
            return "prime" in state

        def convert_to_prime(self, state):
            return state

        def stream_convert_to_prime(self, source, destination):
            events.append(("conversion", rank[0]))
            raise OSError("local conversion failed")

    rank, tokens, events = [0], [], []
    monkeypatch.delenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", raising=False)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: rank[0])

    def broadcast(message, src):
        events.append(("broadcast", rank[0]))
        if rank[0] == 0:
            tokens.append(message[0])
        else:
            message[0] = tokens[0]

    def keys(path):
        events.append(("keys", rank[0]))
        if rank[0] == 0:
            raise ValueError("checkpoint discovery failed")
        return ["hf"]

    monkeypatch.setattr(dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(dist, "barrier", lambda: None)
    monkeypatch.setattr(builder, "PreTrainedModelPrimeRL", Model)
    monkeypatch.setattr(builder, "load_state_dict_keys", keys)
    monkeypatch.setattr(
        builder,
        "get_world",
        lambda: SimpleNamespace(rank=rank[0], local_rank=rank[0], is_master=rank[0] == 0),
    )
    monkeypatch.setattr(builder, "resolve_conversion_cache_path", lambda *args: tmp_path / "prime")
    monkeypatch.setattr(builder, "conversion_cache_is_node_local", lambda path: scope == "node-local")
    config = SimpleNamespace(
        name=str(tmp_path),
        fsdp_cpu_offload=True,
        debug=SimpleNamespace(random_init=False),
        dequantize_checkpoint=False,
        weight_conversion_cache_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="checkpoint discovery failed"):
        builder.load_dcp_from_hf(Model(), config, SimpleNamespace())
    assert events[:2] == [("broadcast", 0), ("keys", 0)]
    assert cache._ACTIVE_ATTEMPT.get() is None
    rank[0] = 1
    if scope == "shared":
        with pytest.raises(RuntimeError, match="Weight preparation writer failed.*checkpoint discovery failed"):
            builder.load_dcp_from_hf(Model(), config, SimpleNamespace())
    else:
        with pytest.raises(OSError, match="local conversion failed"):
            builder.load_dcp_from_hf(Model(), config, SimpleNamespace())
        assert ("conversion", 1) in events
        failure = cache._read_cache_marker(tmp_path / "prime.failed")
        assert failure["attempt_id"] == tokens[0]
        assert "local conversion failed" in failure["error"]
    assert tokens and len([event for event in events if event[0] == "broadcast"]) == 2
    assert cache._ACTIVE_ATTEMPT.get() is None


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
    cache._cleanup_abandoned_conversion_workspaces(destination)
    assert not old.exists()
    assert not owner.exists()
    assert not orphan.exists()
    assert not stale.exists()
    assert recent.exists()


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


@pytest.mark.parametrize("family", ["qwen35", "glm52"])
def test_successful_production_load_ignores_preparation_marker_cleanup_error(monkeypatch, tmp_path, family):
    import importlib
    from types import SimpleNamespace

    import torch.distributed as dist

    from arctic_platform.model.implementations.moe import conversion_cache as cache

    builder = importlib.import_module(f"arctic_platform.model.implementations.{family}.model_builder")
    loaded, removed = [], []
    native_unlink = Path.unlink
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", "cleanup-error")
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "barrier", lambda: None)

    class Model:
        config = SimpleNamespace(tie_word_embeddings=False)

        def to_empty(self, *, device):
            return self

        def state_dict(self):
            return {}

    def unlink(path, *args, **kwargs):
        if path.name.startswith(".preparation-") and path.suffix == ".failed":
            removed.append(path)
            raise PermissionError("marker cleanup denied")
        return native_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(builder, "fix_model_post_empty", lambda model: None)
    monkeypatch.setattr(builder, "_move_buffers_to_cuda", lambda *args: None)
    monkeypatch.setattr(builder, "dcp_load", lambda state, *, storage_reader: loaded.append(storage_reader.path))
    config = SimpleNamespace(
        name=str(tmp_path),
        fsdp_cpu_offload=True,
        debug=SimpleNamespace(random_init=False),
        weight_conversion_cache_dir=str(tmp_path),
    )
    builder.load_dcp_from_hf(Model(), config, None)
    assert len(loaded) == 1
    assert len(removed) == 2
    assert not cache._preparation_marker_path("cleanup-error", tmp_path).with_suffix(".failed").exists()
    assert cache._ACTIVE_ATTEMPT.get() is None
