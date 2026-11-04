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

"""The conversion half of the MoE loaders runs without a process group and fills the cache the load then reads."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakePrimeModel:
    def state_dict(self):
        return {"model.layers.0.mlp.experts.w1": None}

    def is_hf_state_dict(self, state_dict):
        return "model.layers.0.mlp.experts.0.gate_proj.weight" in state_dict

    def is_prime_state_dict(self, state_dict):
        return "model.layers.0.mlp.experts.w1" in state_dict

    def convert_to_prime(self, state_dict):
        return state_dict

    def convert_to_hf(self, state_dict):
        return state_dict


@pytest.mark.parametrize("builder_name", ["qwen35", "glm52"])
@pytest.mark.parametrize("scope", ["node-local", "shared"])
@pytest.mark.parametrize("rank", [0, 8])
def test_prepare_weight_conversion_writes_with_the_load_writer_rule(monkeypatch, tmp_path, builder_name, scope, rank):
    import importlib

    builder = importlib.import_module(f"arctic_platform.model.implementations.{builder_name}.model_builder")
    source = tmp_path / "source"
    cache = tmp_path / "cache" / "prime"
    calls = []
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", "test-launch")
    monkeypatch.setattr(builder, "PreTrainedModelPrimeRL", _FakePrimeModel)
    hf_keys = ["model.layers.0.mlp.experts.0.gate_proj.weight"]
    monkeypatch.setattr(builder, "load_state_dict_keys", lambda path: hf_keys)
    monkeypatch.setattr(builder, "resolve_conversion_cache_path", lambda config, path, fmt: cache)
    monkeypatch.setattr(builder, "conversion_cache_is_node_local", lambda path: scope == "node-local")
    monkeypatch.setattr(builder, "conversion_cache_ready", lambda path: False)
    monkeypatch.setattr(builder, "ensure_node_local_conversion_cache", lambda *a, **k: calls.append(("node", k)))
    monkeypatch.setattr(builder, "wait_for_conversion_cache", lambda path, **kwargs: calls.append(("wait", kwargs)))
    config = SimpleNamespace(dequantize_checkpoint=False)

    result = builder.prepare_weight_conversion(_FakePrimeModel(), config, source, rank=rank, local_rank=0)

    assert result == cache
    if scope == "node-local":
        assert [(kind, k["rank"], k["local_rank"]) for kind, k in calls] == [("node", rank, 0)]
    else:
        assert [kind for kind, _ in calls] == (["node"] if rank == 0 else ["wait"])


def test_shared_conversion_cache_waits_for_publication_and_times_out(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache

    cache = tmp_path / "prime"
    checks = iter([False, False, True])
    sleeps = []
    monkeypatch.setattr(conversion_cache, "conversion_cache_ready", lambda path: next(checks))
    monkeypatch.setattr(conversion_cache.time, "sleep", sleeps.append)
    conversion_cache.wait_for_conversion_cache(cache, timeout=10)
    assert sleeps == [0.25, 0.25]
    monkeypatch.setattr(conversion_cache, "conversion_cache_ready", lambda path: False)
    with pytest.raises(TimeoutError, match="did not become ready"):
        conversion_cache.wait_for_conversion_cache(cache, timeout=0)


def test_shared_conversion_cache_propagates_writer_failure_and_allows_retry(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache

    cache = tmp_path / "prime"
    source = tmp_path / "source"
    source.mkdir()
    arguments = (source, cache, lambda state: None, "HF", "PrimeRL", lambda path: {}, lambda *args: None)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "shared")

    def failed(source, destination):
        destination.mkdir(parents=True)
        raise OSError("conversion failed")

    with pytest.raises(OSError, match="conversion failed"):
        conversion_cache.ensure_node_local_conversion_cache(*arguments, rank=0, local_rank=0, stream_convert_fn=failed)
    failure_path = cache.with_name("prime.failed")
    token = json.loads(failure_path.read_text())["attempt_id"]
    with pytest.raises(RuntimeError, match="writer failed.*conversion failed"):
        conversion_cache.wait_for_conversion_cache(cache, attempt_id=token)
    abandoned = cache.with_name("prime.tmp-rank0-pid999999")
    abandoned.mkdir()

    def successful(source, destination):
        assert abandoned.exists(), "a retry must not delete another writer's workspace"
        destination.mkdir(parents=True)
        (destination / "weights.safetensors").write_bytes(b"ready")
        (destination / "model.safetensors.index.json").write_text(
            json.dumps(dict(weight_map=dict(weight="weights.safetensors")))
        )

    conversion_cache.ensure_node_local_conversion_cache(*arguments, rank=0, local_rank=0, stream_convert_fn=successful)
    conversion_cache.wait_for_conversion_cache(cache)
    assert json.loads(failure_path.read_text())["attempt_id"] == token
    assert not cache.with_name("prime.in-progress").exists()


def test_shared_conversion_retry_waiter_starts_before_writer(monkeypatch, tmp_path):
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from arctic_platform.model.implementations.moe import conversion_cache

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "shared")
    cache = tmp_path / "prime"
    failure = cache.with_name("prime.failed")
    failure.write_text("previous attempt failed")
    os.utime(failure, (1, 1))
    waiting = threading.Event()
    original_sleep = conversion_cache.time.sleep

    def sleep(delay):
        waiting.set()
        original_sleep(delay)

    monkeypatch.setattr(conversion_cache.time, "sleep", sleep)

    def convert(source, destination):
        destination.mkdir()
        (destination / "weights.safetensors").write_bytes(b"ready")
        (destination / "model.safetensors.index.json").write_text(
            json.dumps(dict(weight_map=dict(weight="weights.safetensors")))
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        waiter = executor.submit(conversion_cache.wait_for_conversion_cache, cache, timeout=3, poll_interval=0.01)
        assert waiting.wait(2), "waiter must ignore the stale failure before the writer starts"
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
        )
        waiter.result(timeout=3)
    assert conversion_cache.conversion_cache_ready(cache)


def test_conversion_heartbeat_recovers_after_transient_marker_error(monkeypatch, tmp_path):
    import threading

    from arctic_platform.model.implementations.moe import conversion_cache

    recovered = threading.Event()
    original = conversion_cache._write_cache_marker
    heartbeats = []

    def marker(path, value):
        if path.name.endswith(".in-progress"):
            heartbeats.append(value["heartbeat"])
            if value["heartbeat"] == 1:
                raise OSError("transient shared filesystem error")
            if value["heartbeat"] >= 2:
                recovered.set()
        original(path, value)

    def convert(*args, **kwargs):
        assert recovered.wait(3), "a failed marker write must not stop the heartbeat"

    monkeypatch.setattr(conversion_cache, "_CACHE_HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr(conversion_cache, "_write_cache_marker", marker)
    monkeypatch.setattr(conversion_cache, "_write_conversion_cache", convert)
    conversion_cache.ensure_node_local_conversion_cache(
        tmp_path,
        tmp_path / "prime",
        lambda state: None,
        "HF",
        "PrimeRL",
        lambda path: {},
        lambda *args: None,
        rank=0,
        local_rank=0,
        attempt_id="transient",
    )
    assert heartbeats[:3] == [0, 1, 2]


@pytest.mark.parametrize("phase", ["lock", "first-marker"])
def test_shared_waiter_observes_writer_setup_failure_without_progress(monkeypatch, tmp_path, phase):
    from arctic_platform.model.implementations.moe import conversion_cache

    cache = tmp_path / "prime"
    error = OSError("writer setup failed")

    def fail(*args, **kwargs):
        raise error

    if phase == "lock":
        monkeypatch.setattr(conversion_cache.fcntl, "flock", fail)
    else:
        original = conversion_cache._write_cache_marker

        def marker(path, value):
            if path.name.endswith(".in-progress"):
                raise error
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
            attempt_id="setup",
        )
    assert not cache.with_name("prime.in-progress").exists()
    with pytest.raises(RuntimeError, match="writer failed.*writer setup failed"):
        conversion_cache.wait_for_conversion_cache(cache, timeout=0, attempt_id="setup")


def test_shared_conversion_waiter_detects_killed_writer(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache

    cache = tmp_path / "prime"
    conversion_cache._write_cache_marker(cache.with_name("prime.in-progress"), dict(attempt_id="killed", heartbeat=0))
    clock = [0.0]
    monkeypatch.setattr(conversion_cache.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(conversion_cache.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    with pytest.raises(RuntimeError, match="writer stopped heartbeating"):
        conversion_cache.wait_for_conversion_cache(cache, timeout=3600, writer_idle_timeout=2)
    assert clock[0] == 2


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_conversion_failure_marker_cannot_replace_original_exception(monkeypatch, tmp_path, error_type):
    from arctic_platform.model.implementations.moe import conversion_cache

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "shared")
    original = error_type("original conversion failure")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(conversion_cache, "_write_conversion_cache", fail)
    native = conversion_cache._write_cache_marker

    def marker(path, value):
        if path.name.endswith(".failed"):
            raise UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogate")
        native(path, value)

    monkeypatch.setattr(conversion_cache, "_write_cache_marker", marker)
    with pytest.raises(error_type, match="original conversion failure") as error:
        conversion_cache.ensure_node_local_conversion_cache(
            tmp_path,
            tmp_path / "prime",
            lambda state: None,
            "HF",
            "PrimeRL",
            lambda path: {},
            lambda *args: None,
            rank=0,
            local_rank=0,
        )
    assert error.value is original


def test_shared_writer_can_start_late_without_lock_visibility_or_clock_agreement(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache

    cache = tmp_path / "prime"
    conversion_cache._write_cache_marker(cache.with_name("prime.failed"), dict(attempt_id="old", error="stale"))
    conversion_cache._write_cache_marker(cache.with_name("prime.in-progress"), dict(attempt_id="old", heartbeat=0))
    clock = [0.0]
    monkeypatch.setattr(conversion_cache.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(conversion_cache.time, "time", lambda: -1000000)
    monkeypatch.setattr(conversion_cache.fcntl, "flock", lambda *args: pytest.fail("shared waiter must not use flock"))
    monkeypatch.setattr(conversion_cache, "conversion_cache_ready", lambda path: clock[0] >= 36)

    def sleep(delay):
        clock[0] += delay
        if clock[0] == 35:
            conversion_cache._write_cache_marker(
                cache.with_name("prime.in-progress"), dict(attempt_id="new", heartbeat=0)
            )

    monkeypatch.setattr(conversion_cache.time, "sleep", sleep)
    conversion_cache.wait_for_conversion_cache(cache, timeout=60, poll_interval=1)
    assert clock[0] == 36


def test_node_local_concurrent_failure_converts_only_once(monkeypatch, tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from arctic_platform.model.implementations.moe import conversion_cache

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "node-local")
    cache = tmp_path / "prime"
    entered, release = threading.Event(), threading.Event()
    writes = []
    native_flock = conversion_cache.fcntl.flock
    waiting = threading.Event()

    def flock(file, operation):
        if operation == conversion_cache.fcntl.LOCK_EX and entered.is_set():
            waiting.set()
        return native_flock(file, operation)

    def fail(*args, **kwargs):
        writes.append(1)
        entered.set()
        assert release.wait(3)
        raise ValueError("deterministic failure \ud800")

    monkeypatch.setattr(conversion_cache.fcntl, "flock", flock)
    monkeypatch.setattr(conversion_cache, "_write_conversion_cache", fail)
    args = (tmp_path, cache, lambda state: None, "HF", "PrimeRL", lambda path: {}, lambda *args: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(conversion_cache.ensure_node_local_conversion_cache, *args, rank=0, local_rank=0)
        assert entered.wait(3)
        waiter = pool.submit(conversion_cache.ensure_node_local_conversion_cache, *args, rank=1, local_rank=1)
        assert waiting.wait(3)
        release.set()
        with pytest.raises(ValueError, match="deterministic failure"):
            writer.result(timeout=3)
        with pytest.raises(RuntimeError, match="writer failed"):
            waiter.result(timeout=3)
    assert writes == [1]


@pytest.mark.parametrize("scope", ["node-local", "shared"])
def test_late_rank_observes_same_attempt_failure_without_reconversion(monkeypatch, tmp_path, scope):
    from arctic_platform.model.implementations.moe import conversion_cache

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", scope)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", "failed-launch")
    writes = []

    def fail(*args, **kwargs):
        writes.append(1)
        raise OSError("writer exited before the waiter arrived")

    monkeypatch.setattr(conversion_cache, "_write_conversion_cache", fail)
    cache = tmp_path / "prime"
    args = (tmp_path, cache, lambda state: None, "HF", "PrimeRL", lambda path: {}, lambda *args: None)
    with pytest.raises(OSError, match="writer exited"):
        conversion_cache.ensure_node_local_conversion_cache(*args, rank=0, local_rank=0)
    with pytest.raises(RuntimeError, match="writer failed.*writer exited"):
        if scope == "shared":
            conversion_cache.wait_for_conversion_cache(cache, timeout=0)
        else:
            conversion_cache.ensure_node_local_conversion_cache(*args, rank=1, local_rank=1)
    assert writes == [1]


def test_explicit_retry_ignores_old_failure_even_with_skewed_clocks(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache

    cache = tmp_path / "prime"
    conversion_cache._write_cache_marker(cache.with_name("prime.failed"), dict(attempt_id="old", error="old failure"))
    conversion_cache._write_cache_marker(cache.with_name("prime.in-progress"), dict(attempt_id="old", heartbeat=0))
    clock = [0.0]
    monkeypatch.setattr(conversion_cache.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        conversion_cache.time, "time", lambda: pytest.fail("attempt identity must not use wall clocks")
    )
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


def test_distributed_conversion_agrees_on_attempt_and_standalone_shared_requires_one(monkeypatch):
    import torch.distributed as dist

    from arctic_platform.model.implementations.moe import conversion_cache

    monkeypatch.delenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", raising=False)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    rank = [0]
    agreed = []
    monkeypatch.setattr(dist, "get_rank", lambda: rank[0])

    def broadcast(tokens, *, src):
        assert src == 0
        if rank[0] == 0:
            agreed.append(tokens[0])
        else:
            tokens[0] = agreed[0]

    monkeypatch.setattr(dist, "broadcast_object_list", broadcast)
    writer = conversion_cache.conversion_cache_attempt_id(shared=True)
    rank[0] = 1
    assert conversion_cache.conversion_cache_attempt_id(shared=True) == writer
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with pytest.raises(ValueError, match="DSS_WEIGHT_CONVERSION_ATTEMPT_ID"):
        conversion_cache.conversion_cache_attempt_id(shared=True)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", "launcher-retry")
    assert conversion_cache.conversion_cache_attempt_id(shared=True) == "launcher-retry"


@pytest.mark.parametrize("builder_name", ["qwen35", "glm52"])
def test_prepare_weight_conversion_selects_the_checkpoint_streamer(monkeypatch, tmp_path, builder_name):
    import importlib

    builder = importlib.import_module(f"arctic_platform.model.implementations.{builder_name}.model_builder")
    model = _FakePrimeModel()
    stream = lambda source, target: None  # noqa: E731
    model.convert_to_prime_checkpoint = stream
    calls = []
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", "test-launch")
    monkeypatch.setattr(builder, "PreTrainedModelPrimeRL", _FakePrimeModel)
    monkeypatch.setattr(
        builder, "load_state_dict_keys", lambda path: ["model.layers.0.mlp.experts.0.gate_proj.weight"]
    )
    monkeypatch.setattr(builder, "resolve_conversion_cache_path", lambda *a: tmp_path / "prime")
    monkeypatch.setattr(builder, "conversion_cache_is_node_local", lambda path: False)
    monkeypatch.setattr(builder, "ensure_node_local_conversion_cache", lambda *a, **kw: calls.append(kw))
    config = SimpleNamespace(dequantize_checkpoint=False)
    assert builder.prepare_weight_conversion(model, config, tmp_path, rank=0, local_rank=0) == tmp_path / "prime"
    assert calls[0]["stream_convert_fn"] is stream


@pytest.mark.parametrize("builder_name", ["qwen35", "glm52"])
def test_prepare_weight_conversion_skips_a_checkpoint_in_the_model_format(monkeypatch, tmp_path, builder_name):
    import importlib

    builder = importlib.import_module(f"arctic_platform.model.implementations.{builder_name}.model_builder")
    monkeypatch.setattr(builder, "PreTrainedModelPrimeRL", _FakePrimeModel)
    monkeypatch.setattr(builder, "load_state_dict_keys", lambda path: ["model.layers.0.mlp.experts.w1"])
    unexpected = lambda *a, **k: pytest.fail("no conversion applies")  # noqa: E731
    monkeypatch.setattr(builder, "resolve_conversion_cache_path", unexpected)

    for model in (_FakePrimeModel(), object()):
        assert builder.prepare_weight_conversion(model, SimpleNamespace(), tmp_path, rank=0, local_rank=0) is None


def _adapter(events, *, random_init=False):
    from arctic_platform.model.implementations.moe.deepspeed_integration import MoEDeepSpeedAdapter

    def build_model_config(model_name, ep_size, dp_replicate, optimization_dtype, attn, options):
        events.append(("config", model_name, ep_size, dp_replicate, optimization_dtype, attn, options))
        return SimpleNamespace(
            debug=SimpleNamespace(random_init=random_init),
            optimization_dtype=optimization_dtype,
            fused_lm_head_token_chunk_size=None,
            fp32_lm_head=False,
        )

    def get_model(config, *, device, dtype):
        events.append(("model", device.type, dtype))
        return "meta-model"

    def prepare(model, config, source, *, rank, local_rank):
        events.append(("prepare", model, source, rank, local_rank))
        return Path("/cache/prime")

    unused = lambda *a, **k: pytest.fail("the load path is not part of preparation")  # noqa: E731
    return MoEDeepSpeedAdapter(
        dtype_map={"bfloat16": "bf16-dtype"},
        get_model=get_model,
        configure_moe_ep_backend=lambda *args: events.append(("backend",)),
        configure_family_backend=lambda *args: events.append(("family",)),
        inject_lm_head=lambda *args, **kwargs: events.append(("lm_head",)),
        apply_sequence_parallelism=unused,
        apply_ac=unused,
        load_dcp_from_hf=unused,
        reset_runtime_moe_buffers=unused,
        shared_expert_type=object,
        shared_expert_forward=unused,
        build_model_config=build_model_config,
        resolve_weight_source=lambda config: Path("/source"),
        prepare_weight_conversion=prepare,
    )


def _prepare(adapter, **overrides):
    from arctic_platform.model.implementations.moe.deepspeed_integration import prepare_moe_weights

    arguments = dict(
        model_name="org/model",
        optimization_dtype="bfloat16",
        attn_implementation="sdpa",
        ep_size=16,
        world_size=32,
        options="options",
        rank=8,
        local_rank=0,
    )
    return prepare_moe_weights(adapter, **{**arguments, **overrides})


def test_prepare_moe_weights_converts_from_a_meta_model_with_the_load_config():
    events = []

    assert _prepare(_adapter(events)) == Path("/cache/prime")
    assert events == [
        ("config", "org/model", 16, 2, "bfloat16", "sdpa", "options"),
        ("model", "meta", "bf16-dtype"),
        ("backend",),
        ("family",),
        ("lm_head",),
        ("prepare", "meta-model", Path("/source"), 8, 0),
    ]


def test_prepare_moe_weights_skips_random_init_and_rejects_an_indivisible_world():
    events = []

    assert _prepare(_adapter(events, random_init=True)) is None
    assert [event[0] for event in events] == ["config"]
    with pytest.raises(ValueError, match="divisible"):
        _prepare(_adapter([]), world_size=24)


def test_prepare_model_weights_dispatches_to_the_resolved_loader_hook(monkeypatch):
    from arctic_platform.model import loader

    received = []
    monkeypatch.setitem(
        loader._LOADERS,
        "with-hook",
        loader._LoaderEntry(fn=None, matches=None, prepare_weights=lambda spec, **kw: received.append(kw) or "path"),
    )
    monkeypatch.setitem(loader._LOADERS, "without-hook", loader._LoaderEntry(fn=None, matches=None))

    hooked = loader.prepare_model_weights(SimpleNamespace(loader="with-hook"), world_size=32, rank=8, local_rank=0)
    plain = loader.prepare_model_weights(SimpleNamespace(loader="without-hook"), world_size=32, rank=8, local_rank=0)

    assert (hooked, plain) == ("path", None)
    assert received == [{"world_size": 32, "rank": 8, "local_rank": 0}]


@pytest.mark.parametrize(
    "name, tiny_config, implementation, preparer",
    [
        ("glm5_next", "tiny_glm5_next_config", "glm53", "prepare_glm5_next_weights"),
        ("qwen4_exp", "tiny_qwen4_exp_config", "qwen38", "prepare_qwen4_exp_weights"),
    ],
)
def test_flash_moe_weight_hook_takes_the_spec_its_loader_defaulted(
    monkeypatch, tmp_path, name, tiny_config, implementation, preparer
):
    # The spec stores every FlashMoeOptions default, fp32_routed_weighting among them, and the hook revalidates it.
    import importlib

    from arctic_platform.model import loader
    from arctic_platform.model.moe_spec import moe_model_spec

    (tmp_path / "config.json").write_text(json.dumps(dict(model_type=name)))
    received = {}
    module = importlib.import_module(f"arctic_platform.model.implementations.{implementation}.deepspeed_integration")
    monkeypatch.setattr(module, preparer, lambda **kwargs: received.update(kwargs) or tmp_path / "cache")
    spec = moe_model_spec(model_name=str(tmp_path), ep_size=8, sp_size=8)

    path = loader.prepare_model_weights(spec, world_size=16, rank=8, local_rank=0)

    assert spec.loader == name
    assert spec.loader_options["fp32_routed_weighting"] is False
    assert path == tmp_path / "cache"
    assert received["options"].fp32_routed_weighting is False
    assert (received["ep_size"], received["world_size"], received["rank"]) == (8, 16, 8)


@pytest.mark.parametrize("name", ["glm5_next", "qwen4_exp", "qwen3_5_moe", "generic_moe", "glm_moe_dsa"])
def test_every_converting_moe_loader_registers_a_weight_hook(name):
    import arctic_platform.model.loaders  # noqa: F401
    from arctic_platform.model import loader

    assert loader._LOADERS[name].prepare_weights is not None


@pytest.mark.parametrize("streaming", [False, True])
def test_preparation_rejects_incomplete_conversion_before_returning_a_path(monkeypatch, tmp_path, streaming):
    from arctic_platform.model.implementations.qwen35 import model_builder as builder

    model = _FakePrimeModel()

    def incomplete(*args):
        target = args[-1]
        target.mkdir(parents=True)
        (target / "model.safetensors.index.json").write_text(
            json.dumps(dict(weight_map=dict(weight="missing.safetensors")))
        )

    if streaming:
        model.convert_to_prime_checkpoint = incomplete
    monkeypatch.setattr(builder, "PreTrainedModelPrimeRL", _FakePrimeModel)
    monkeypatch.setattr(
        builder, "load_state_dict_keys", lambda path: ["model.layers.0.mlp.experts.0.gate_proj.weight"]
    )
    monkeypatch.setattr(builder, "load_state_dict", lambda path: {})
    monkeypatch.setattr(builder, "save_state_dict", incomplete)
    cache = tmp_path / "prime"
    monkeypatch.setattr(builder, "resolve_conversion_cache_path", lambda *args: cache)
    monkeypatch.setattr(builder, "conversion_cache_is_node_local", lambda path: True)
    with pytest.raises(RuntimeError, match="did not publish a complete checkpoint"):
        builder.prepare_weight_conversion(
            model, SimpleNamespace(dequantize_checkpoint=False), tmp_path, rank=0, local_rank=0
        )
    assert not cache.exists()


def _tiny_preparation_model(family):
    import torch

    if family == "glm5_next":
        from test_glm5_next import _tiny_config

        from arctic_platform.model.implementations.glm53.modeling_glm5_next import (
            Glm5NextForConditionalGenerationPrimeRL,
        )

        config, model_type = _tiny_config(), Glm5NextForConditionalGenerationPrimeRL
    elif family == "qwen4_exp":
        from test_qwen4_exp import _tiny_config

        from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import (
            Qwen4ExpForConditionalGenerationPrimeRL,
        )

        config, model_type = _tiny_config(), Qwen4ExpForConditionalGenerationPrimeRL
    elif family == "qwen3_5_moe":
        from test_qwen35_moe_numerics import _tiny_config

        from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe import Qwen3_5MoeForCausalLM

        config, model_type = _tiny_config(), Qwen3_5MoeForCausalLM
    else:
        if family == "glm_moe_dsa":
            from arctic_platform.model.implementations.glm52.models.glm_moe_dsa import GlmMoeDsaConfig as Config
            from arctic_platform.model.implementations.glm52.models.glm_moe_dsa import (
                GlmMoeDsaForCausalLM as model_type,
            )
        else:
            from arctic_platform.model.implementations.qwen3_moe.configuration_qwen3_moe import (
                Qwen3MoeConfig as Config,
            )
            from arctic_platform.model.implementations.qwen3_moe.modeling_qwen3_moe import (
                Qwen3MoeForCausalLM as model_type,
            )

        config = Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=4,
            n_routed_experts=4,
            num_experts_per_tok=2,
            first_k_dense_replace=0,
            q_lora_rank=8,
            kv_lora_rank=4,
            qk_rope_head_dim=4,
            qk_nope_head_dim=4,
            v_head_dim=4,
            index_n_heads=2,
            index_head_dim=4,
            index_topk=4,
            pad_token_id=0,
            use_grouped_mm=False,
        )
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        return model_type(config)


@pytest.mark.parametrize("family", ["qwen3_5_moe", "generic_moe", "glm_moe_dsa", "glm5_next", "qwen4_exp"])
@pytest.mark.parametrize("dequantize", [False, True])
def test_real_family_preparation_and_loading_choose_the_same_cache(monkeypatch, tmp_path, family, dequantize):
    import torch
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.moe.conversion_cache import hashed_conversion_cache_path
    from arctic_platform.model.implementations.moe.layers.lm_head import inject_prime_lm_head

    builder_name = "glm52" if family == "glm_moe_dsa" else "qwen35"
    builder = importlib.import_module(f"arctic_platform.model.implementations.{builder_name}.model_builder")
    model = _tiny_preparation_model(family)
    hf_keys = dict.fromkeys(
        [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.gate_up_proj",
        ]
    )
    assert model.is_hf_state_dict(hf_keys)
    source, cache_root = tmp_path / "source", tmp_path / "explicit-cache"
    source.mkdir()
    if dequantize:
        hf_keys["model.layers.0.mlp.experts.0.gate_proj.scale"] = None
    save_file({name: torch.ones(1) for name in hf_keys}, source / "model.safetensors")
    monkeypatch.setattr(builder, "resolve_weight_source", lambda config: source)
    monkeypatch.setattr(builder, "conversion_cache_is_node_local", lambda path: True)
    monkeypatch.setattr(builder, "conversion_cache_ready", lambda path: True)
    monkeypatch.setattr(builder, "ensure_node_local_conversion_cache", lambda *args, **kwargs: None)
    config = SimpleNamespace(
        fsdp_cpu_offload=True,
        debug=SimpleNamespace(random_init=False),
        dequantize_checkpoint=dequantize,
        weight_conversion_cache_dir=str(cache_root),
    )
    fmt = "prime-dequantized" if dequantize and builder_name == "qwen35" else "prime"
    cache = hashed_conversion_cache_path(cache_root, source, fmt)
    calls = []
    monkeypatch.setattr(builder, "ensure_node_local_conversion_cache", lambda *args, **kwargs: calls.append(args[1]))
    prepared = builder.prepare_weight_conversion(model, config, source, rank=0, local_rank=0)
    inject_prime_lm_head(model, chunk_size=2, fused_cross_entropy=False, fp32_lm_head=False)
    loaded = []
    monkeypatch.setattr(builder, "get_world", lambda: SimpleNamespace(rank=0, local_rank=0))
    monkeypatch.setattr(builder.torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(builder, "dcp_load", lambda state, *, storage_reader: loaded.append(Path(storage_reader.path)))
    monkeypatch.setattr(builder, "_move_buffers_to_cuda", lambda *args: None)
    monkeypatch.setattr(builder, "_init_buffers_post_meta", lambda *args: None, raising=False)
    monkeypatch.setattr(type(model), "init_buffers_post_meta", lambda self: None, raising=False)
    builder.load_dcp_from_hf(model, config, None)
    assert prepared == cache
    assert loaded == [prepared]
    assert calls == [cache, cache]


@pytest.mark.parametrize("family", ["qwen3_5_moe", "generic_moe", "glm_moe_dsa", "glm5_next", "qwen4_exp"])
@pytest.mark.parametrize("dequantize", [False, True])
@pytest.mark.parametrize("scope", ["node-local", "shared"])
def test_real_family_written_cache_is_reused_by_loading(monkeypatch, tmp_path, family, dequantize, scope):
    import torch

    from arctic_platform.model.implementations.moe import conversion_cache
    from arctic_platform.model.implementations.moe.layers.lm_head import inject_prime_lm_head
    from arctic_platform.model.implementations.moe.weights import load_state_dict
    from arctic_platform.model.implementations.moe.weights import save_state_dict
    from arctic_platform.testing_utils import torch_assert_equal

    builder_name = "glm52" if family == "glm_moe_dsa" else "qwen35"
    builder = importlib.import_module(f"arctic_platform.model.implementations.{builder_name}.model_builder")
    model = _tiny_preparation_model(family)
    expected = {
        name: torch.arange(weight.numel(), dtype=weight.dtype).reshape(weight.shape)
        for name, weight in model.state_dict().items()
    }
    hf = model.convert_to_hf({name: weight.clone() for name, weight in expected.items()})
    source, cache_root = tmp_path / "source", tmp_path / "cache"
    save_state_dict(hf, source)
    config = SimpleNamespace(
        fsdp_cpu_offload=True,
        debug=SimpleNamespace(random_init=False),
        dequantize_checkpoint=dequantize,
        weight_conversion_cache_dir=str(cache_root),
    )
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", scope)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", "test-launch")
    monkeypatch.setattr(builder, "resolve_weight_source", lambda config: source)
    original_write = conversion_cache._write_conversion_cache
    writes = []

    def write(*args, **kwargs):
        writes.append(args[1])
        return original_write(*args, **kwargs)

    monkeypatch.setattr(conversion_cache, "_write_conversion_cache", write)
    prepared = builder.prepare_weight_conversion(model, config, source, rank=0, local_rank=0)
    assert conversion_cache.conversion_cache_ready(prepared)
    converted = load_state_dict(prepared)
    assert set(converted) == set(expected)
    for name, weight in expected.items():
        torch_assert_equal(converted[name], weight)

    inject_prime_lm_head(model, chunk_size=2, fused_cross_entropy=False, fp32_lm_head=False)
    loaded = []
    monkeypatch.setattr(builder, "get_world", lambda: SimpleNamespace(rank=0, local_rank=0, is_master=True))
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(builder, "dcp_load", lambda state, *, storage_reader: loaded.append(Path(storage_reader.path)))
    monkeypatch.setattr(builder, "_move_buffers_to_cuda", lambda *args: None)
    monkeypatch.setattr(builder, "_init_buffers_post_meta", lambda *args: None, raising=False)
    monkeypatch.setattr(type(model), "init_buffers_post_meta", lambda self: None, raising=False)
    builder.load_dcp_from_hf(model, config, None)
    assert loaded == [prepared]
    assert writes == [prepared], "load must reuse the actually published preparation cache"


@pytest.mark.parametrize("family", ["qwen3_5_moe", "generic_moe", "glm_moe_dsa", "glm5_next", "qwen4_exp"])
@pytest.mark.parametrize("dequantize", [False, True])
@pytest.mark.parametrize("scope", ["node-local", "shared"])
def test_adapter_preparation_and_runtime_load_reuse_the_written_cache(
    monkeypatch, tmp_path, family, dequantize, scope
):
    from dataclasses import replace

    import torch

    from arctic_platform.model.implementations.moe import conversion_cache
    from arctic_platform.model.implementations.moe import deepspeed_integration as integration
    from arctic_platform.model.implementations.moe.weights import load_state_dict
    from arctic_platform.model.implementations.moe.weights import save_state_dict
    from arctic_platform.model.loaders.glm_moe_dsa import GlmMoeDsaOptions
    from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions
    from arctic_platform.testing_utils import torch_assert_equal

    implementation = dict(
        qwen3_5_moe="qwen35", generic_moe="qwen35", glm_moe_dsa="glm52", glm5_next="glm53", qwen4_exp="qwen38"
    )[family]
    module = importlib.import_module(f"arctic_platform.model.implementations.{implementation}.deepspeed_integration")
    adapter = module._generic_adapter() if family == "generic_moe" else module._adapter()
    builder_name = "glm52" if family == "glm_moe_dsa" else "qwen35"
    builder = importlib.import_module(f"arctic_platform.model.implementations.{builder_name}.model_builder")
    original = _tiny_preparation_model(family)
    source, cache_root = tmp_path / "source", tmp_path / "cache"
    expected = {
        name: torch.arange(value.numel(), dtype=torch.float32).reshape(value.shape).to(value.dtype)
        for name, value in original.state_dict().items()
    }
    hf = original.convert_to_hf({name: value.clone() for name, value in expected.items()})
    if dequantize and builder_name == "qwen35":
        # Exercise the scale-sensitive cache identity, independently of packed-weight arithmetic.
        hf["conversion_probe.scale"] = torch.ones(1)
    save_state_dict(hf, source)
    original.config.save_pretrained(source)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", scope)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", "adapter-launch")
    attn = dict(glm5_next="sparse_mla", qwen4_exp="qsa_flex").get(family, "sdpa")
    options_type = GlmMoeDsaOptions if family == "glm_moe_dsa" else Qwen3_5MoeOptions
    options = options_type(
        weight_conversion_cache_dir=str(cache_root),
        moe_use_grouped_mm=False,
        fused_cross_entropy=False,
        fused_lm_head_token_chunk_size=2,
    )
    build_config = adapter.build_model_config

    def configured(*args):
        config = replace(build_config(*args), fsdp_cpu_offload=True)
        if builder_name == "qwen35":
            config.dequantize_checkpoint = dequantize
        return config

    adapter = replace(adapter, build_model_config=configured)
    writes, loaded, classifications = [], [], []
    native_write = conversion_cache._write_conversion_cache
    native_prepare = adapter.prepare_weight_conversion

    def write(*args, **kwargs):
        writes.append(args[1])
        return native_write(*args, **kwargs)

    def classify(model, config, source, **kwargs):
        path = native_prepare(model, config, source, **kwargs)
        classifications.append((model.is_prime_state_dict(model.state_dict()), path))
        return path

    adapter = replace(adapter, prepare_weight_conversion=classify)
    monkeypatch.setattr(conversion_cache, "_write_conversion_cache", write)
    monkeypatch.setattr(builder, "prepare_weight_conversion", classify)
    monkeypatch.setattr(builder, "get_world", lambda: SimpleNamespace(rank=0, local_rank=0, is_master=True))
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(builder, "_move_buffers_to_cuda", lambda *args: None)
    monkeypatch.setattr(builder, "_init_buffers_post_meta", lambda *args: None, raising=False)
    monkeypatch.setattr(type(original), "init_buffers_post_meta", lambda self: None, raising=False)

    def load(state, *, storage_reader):
        path = Path(storage_reader.path)
        loaded.append(path)
        converted = load_state_dict(path)
        for name, value in expected.items():
            torch_assert_equal(converted[name], value)

    monkeypatch.setattr(builder, "dcp_load", load)
    prepared = integration.prepare_moe_weights(
        adapter,
        model_name=str(source),
        optimization_dtype="float32",
        attn_implementation=attn,
        ep_size=1,
        world_size=1,
        options=options,
        rank=0,
        local_rank=0,
    )
    config = configured(str(source), 1, 1, "float32", attn, options)
    integration.setup_model_local_no_train(adapter, config, SimpleNamespace(ep_enabled=False), None)
    assert prepared.name == ("prime-dequantized" if dequantize and builder_name == "qwen35" else "prime")
    assert classifications == [(True, prepared), (True, prepared)]
    assert loaded == [prepared]
    assert writes == [prepared], "the runtime adapter must not convert again after standalone preparation"
