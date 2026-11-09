"""Unit tests for ``SEMIP_DUMP_PUBLISH``: a miss dumps and publishes only on request.

Loads ``server/semip_engine.py`` from its file the way ``test_image_cache_key``
does, and drives ``restore_and_wrap`` with every GPU-, CRIU- and cluster-facing
step replaced, so what is tested is the decision: refuse the miss, or dump,
restore and then publish.

    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_dump_publish.py -v
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_INFERENCE = os.path.dirname(os.path.dirname(_HERE))
_ENGINE = os.path.join(_INFERENCE, "server", "semip_engine.py")


def _load_semip_engine():
    pkg = types.ModuleType("arctic_platform.inference")
    pkg.__path__ = []
    semip = types.ModuleType("arctic_platform.inference.semi_persistence")
    semip.Instance = object
    sys.modules.setdefault("arctic_platform.inference", pkg)
    sys.modules.setdefault("arctic_platform.inference.semi_persistence", semip)
    spec = importlib.util.spec_from_file_location("semip_engine_dump_publish",
                                                  _ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


se = _load_semip_engine()

CONFIG = {"model": "/mnt/neutrino/base-models/Qwen/Qwen3.8-27B",
          "tensor_parallel_size": 1}


def _paths(tmp_path, *, slot=0, count=1, mirror=True):
    key_dir = str(tmp_path / "cache" / "aaaaaaaaaaaa_bbbbbbbbbbbb")
    model_dir = (os.path.join(key_dir, f"replica{slot}") if count > 1
                 else key_dir)
    root = str(tmp_path / "mirror" / "Qwen3.8-27B")
    return se._ImagePaths(
        model_dir, "aaaaaaaaaaaa_bbbbbbbbbbbb",
        os.path.join(root, "skeleton") if mirror else None,
        os.path.join(root, "weight") if mirror else None,
        "img@sha256:00", "580.0", slot, count, key_dir)


@pytest.fixture
def publish_dir(tmp_path, monkeypatch):
    path = str(tmp_path / "publish")
    monkeypatch.setattr(se, "_PUBLISH_DIR", path)
    monkeypatch.setattr(se, "_PUBLISH_POLL_S", 0.01)
    monkeypatch.setattr(se, "_is_publishing_pod", lambda: True)
    return path


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """``restore_and_wrap`` with everything outside the decision recorded."""
    calls = {"dump": 0, "restore": 0, "publish": []}
    paths = _paths(tmp_path)
    sys.modules["semip_logging"] = types.SimpleNamespace(
        attach_pod_log=lambda names: None)
    monkeypatch.setattr(se, "_vllm_config_from_engine_kwargs",
                        lambda kwargs: dict(CONFIG))
    monkeypatch.setattr(se, "_resolve_model_dir", lambda cfg: paths)
    monkeypatch.setattr(se, "_resolve_physical_gpus", lambda: [0])
    monkeypatch.setattr(se, "_check_tp_matches_gpus", lambda *a: None)
    monkeypatch.setattr(se, "_resolve_published_skeleton",
                        lambda p: (None, None))
    monkeypatch.setattr(se, "_materialize_from_source",
                        lambda *a, **k: False)
    monkeypatch.setattr(se, "_weights_dir_for_restore", lambda *a: None)

    def dump(cfg, model_dir, gpus, **kwargs):
        calls["dump"] += 1

    def restore(model_dir, engine_kwargs, gpus, **kwargs):
        calls["restore"] += 1
        return "engine"

    monkeypatch.setattr(se, "_dump", dump)
    monkeypatch.setattr(se, "_restore_with_port_retry", restore)
    monkeypatch.setattr(se, "_start_publish",
                        lambda p, skel, world: calls["publish"].append(
                            (p, skel, world)))
    monkeypatch.delenv(se._DUMP_PUBLISH_ENV, raising=False)
    return calls, paths


# --------------------------------------------------------------------------
# The flag and the refusal
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("YES", True), (" 1 ", True),
    ("", False), ("0", False), ("false", False), (None, False)])
def test_dump_publish_requested(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(se._DUMP_PUBLISH_ENV, raising=False)
    else:
        monkeypatch.setenv(se._DUMP_PUBLISH_ENV, value)
    assert se._dump_publish_requested() is expected


def test_miss_error_names_both_ways_out():
    text = str(se._miss_error("/data-fast/x/key", None))
    assert "inference_config.semi_p to false" in text
    assert "resubmit" in text
    assert f"{se._DUMP_PUBLISH_ENV}=1" in text
    assert "/data-fast/x/key" in text
    assert "published image exists" not in text


def test_miss_error_points_at_a_published_image_it_could_not_use():
    text = str(se._miss_error("/k", "/mirror/skeleton/k_wt"))
    assert "/mirror/skeleton/k_wt" in text


# --------------------------------------------------------------------------
# restore_and_wrap's miss path
# --------------------------------------------------------------------------


def test_miss_without_the_flag_fails_before_dumping(engine):
    calls, _ = engine
    with pytest.raises(RuntimeError, match="semi_p to false"):
        se.restore_and_wrap({})
    assert calls == {"dump": 0, "restore": 0, "publish": []}


def test_miss_with_the_flag_dumps_restores_then_publishes(engine, monkeypatch):
    calls, paths = engine
    monkeypatch.setenv(se._DUMP_PUBLISH_ENV, "1")
    assert se.restore_and_wrap({}) == "engine"
    assert calls["dump"] == 1 and calls["restore"] == 1
    assert calls["publish"] == [(paths, None, 1)]


def test_a_local_hit_ignores_the_flag(engine, monkeypatch):
    calls, paths = engine
    image = os.path.join(paths.model_dir, se._IMAGE_DIR)
    os.makedirs(image)
    with open(os.path.join(image, "meta.json"), "w") as handle:
        handle.write("{}")
    monkeypatch.setenv(se._DUMP_PUBLISH_ENV, "1")
    assert se.restore_and_wrap({}) == "engine"
    assert calls == {"dump": 0, "restore": 1, "publish": []}


def test_a_materialized_copy_needs_no_flag_and_publishes_nothing(engine,
                                                                 monkeypatch):
    calls, _ = engine
    monkeypatch.setattr(se, "_materialize_from_source", lambda *a, **k: True)
    assert se.restore_and_wrap({}) == "engine"
    assert calls == {"dump": 0, "restore": 1, "publish": []}


def test_a_failed_restore_after_the_dump_publishes_nothing(engine, monkeypatch):
    calls, _ = engine
    monkeypatch.setenv(se._DUMP_PUBLISH_ENV, "1")

    def broken(*a, **k):
        raise RuntimeError("restore failed")

    monkeypatch.setattr(se, "_restore_with_port_retry", broken)
    with pytest.raises(RuntimeError, match="restore failed"):
        se.restore_and_wrap({})
    assert calls["dump"] == 1 and calls["publish"] == []


# --------------------------------------------------------------------------
# What gets published, and when
# --------------------------------------------------------------------------


def test_publish_is_skipped_with_the_mirror_off(tmp_path):
    reason = se._publish_skip_reason(_paths(tmp_path, mirror=False), None)
    assert se._IMAGE_SOURCE_ENV in reason


def test_publish_is_skipped_for_a_key_already_published(tmp_path):
    reason = se._publish_skip_reason(_paths(tmp_path), "/mirror/skeleton/k_wt")
    assert "/mirror/skeleton/k_wt" in reason


def test_publish_goes_ahead_for_a_new_key(tmp_path):
    assert se._publish_skip_reason(_paths(tmp_path), None) is None


@pytest.mark.parametrize("count,world,published", [
    (1, 1, True),    # lone TP=1: the flat tp1 key
    (8, 1, True),    # tp1x8
    (3, 1, True),    # tp1x3: its own key, on any 3 GPUs
    (2, 1, True),
    (4, 2, True),    # tp2x4
    (2, 4, True),    # tp4x2
    (1, 8, True),    # tp8
    (1, 2, False),   # TP=2 on a 2-GPU pod: bound to devices others lack
    (2, 2, False),
])
def test_only_a_whole_node_or_a_tp1_pod_is_published(tmp_path, count, world,
                                                     published):
    reason = se._publish_skip_reason(_paths(tmp_path, count=count), None, world)
    assert (reason is None) is published
    if not published:
        assert "whole node" in reason

def test_publish_command_names_the_script_and_the_bucket():
    command = se._publish_command("/data-fast/image-cache_neutrino/k")
    assert os.path.isfile(command[1])
    assert command[2] == "/data-fast/image-cache_neutrino/k"
    assert command[command.index("--bucket") + 1] == se._PUBLISH_BUCKET


def _record_publishes(monkeypatch):
    published = []
    done = threading.Event()

    def run(key_dir):
        published.append(key_dir)
        done.set()
        return 0

    monkeypatch.setattr(se, "_run_publisher", run)
    return published, done


def test_a_lone_replica_publishes_its_key(tmp_path, publish_dir, monkeypatch):
    published, done = _record_publishes(monkeypatch)
    paths = _paths(tmp_path)
    se._start_publish(paths, None, 1)
    assert done.wait(5)
    assert published == [paths.key_dir]


def test_slot_zero_waits_for_every_replica_of_the_pod(tmp_path, publish_dir,
                                                      monkeypatch):
    published, done = _record_publishes(monkeypatch)
    count = 4   # TP=2 x 4 on a whole node
    for slot in (0, 2, 3):
        se._start_publish(_paths(tmp_path, slot=slot, count=count), None, 2)
    time.sleep(0.1)
    assert published == []
    se._start_publish(_paths(tmp_path, slot=1, count=count), None, 2)
    assert done.wait(5)
    assert published == [_paths(tmp_path).key_dir]


def test_a_partial_tp1_pod_publishes_its_own_key(tmp_path, publish_dir,
                                                 monkeypatch):
    published, done = _record_publishes(monkeypatch)
    for slot in (1, 2):
        se._start_publish(_paths(tmp_path, slot=slot, count=3), None, 1)
    time.sleep(0.1)
    assert published == []
    se._start_publish(_paths(tmp_path, slot=0, count=3), None, 1)
    assert done.wait(5)
    assert published == [_paths(tmp_path).key_dir]


def test_a_partial_tp2_pod_publishes_nothing(tmp_path, publish_dir,
                                             monkeypatch):
    published, _ = _record_publishes(monkeypatch)
    for slot in range(3):
        se._start_publish(_paths(tmp_path, slot=slot, count=3), None, 2)
    time.sleep(0.1)
    assert published == []
    assert not os.path.exists(se._publish_ready_marker(
        _paths(tmp_path).key_dir, 0))


def test_only_slot_zero_publishes(tmp_path, publish_dir, monkeypatch):
    published, _ = _record_publishes(monkeypatch)
    se._start_publish(_paths(tmp_path, slot=1, count=2), None, 4)
    time.sleep(0.1)
    assert published == []
    assert os.path.exists(se._publish_ready_marker(
        _paths(tmp_path).key_dir, 1))


def test_a_replica_that_never_restores_blocks_the_publish(tmp_path, publish_dir,
                                                          monkeypatch):
    published, _ = _record_publishes(monkeypatch)
    paths = _paths(tmp_path, slot=0, count=2)
    os.makedirs(os.path.dirname(se._publish_ready_marker(paths.key_dir, 0)))
    open(se._publish_ready_marker(paths.key_dir, 0), "w").close()
    se._publish_when_ready(paths.key_dir, 2, timeout_s=0.05)
    assert published == []


def test_only_one_pod_of_a_multi_pod_job_publishes(tmp_path, publish_dir,
                                                  monkeypatch):
    published, _ = _record_publishes(monkeypatch)
    monkeypatch.setattr(se, "_is_publishing_pod", lambda: False)
    se._start_publish(_paths(tmp_path), None, 1)
    time.sleep(0.1)
    assert published == []


def _fake_ray(monkeypatch, mine, nodes):
    ray = types.ModuleType("ray")
    ray.nodes = lambda: nodes
    ray.util = types.SimpleNamespace(get_node_ip_address=lambda: mine)
    monkeypatch.setitem(sys.modules, "ray", ray)


def _node(ip, gpus=8, alive=True):
    return {"NodeManagerAddress": ip, "Alive": alive,
            "Resources": {"GPU": gpus} if gpus else {"CPU": 8}}


def test_the_publishing_pod_is_the_lowest_gpu_pod(monkeypatch):
    nodes = [_node("10.0.0.9", gpus=0), _node("10.0.0.5"), _node("10.0.0.7"),
             _node("10.0.0.1", alive=False)]
    _fake_ray(monkeypatch, "10.0.0.5", nodes)
    assert se._is_publishing_pod() is True
    _fake_ray(monkeypatch, "10.0.0.7", nodes)
    assert se._is_publishing_pod() is False


def test_a_pod_that_cannot_read_the_cluster_does_not_publish(monkeypatch):
    def broken():
        raise RuntimeError("ray not initialized")

    _fake_ray(monkeypatch, "10.0.0.5", [])
    sys.modules["ray"].nodes = broken
    assert se._is_publishing_pod() is False


def test_a_skipped_publish_writes_no_marker(tmp_path, publish_dir, monkeypatch):
    published, _ = _record_publishes(monkeypatch)
    paths = _paths(tmp_path, mirror=False)
    se._start_publish(paths, None, 1)
    assert not os.path.exists(se._publish_ready_marker(paths.key_dir, 0))
    assert published == []


def test_a_pod_spanning_engine_publishes_on_every_pod(tmp_path, monkeypatch):
    published, done = _record_publishes(monkeypatch)
    asked = []

    class Agent:
        class publish:  # noqa: N801 - mirrors a Ray actor method handle
            @staticmethod
            def remote(key_dir):
                asked.append(key_dir)

    paths = _paths(tmp_path)
    se._start_publish_nodes(paths, None, [Agent(), Agent()])
    assert done.wait(5)
    assert asked == [paths.key_dir, paths.key_dir]
    assert published == [paths.key_dir]


def test_a_pod_spanning_engine_skips_an_already_published_key(tmp_path,
                                                              monkeypatch):
    published, _ = _record_publishes(monkeypatch)
    asked = []

    class Agent:
        class publish:  # noqa: N801
            @staticmethod
            def remote(key_dir):
                asked.append(key_dir)

    se._start_publish_nodes(_paths(tmp_path), "/mirror/skeleton/k_wt",
                            [Agent()])
    time.sleep(0.1)
    assert asked == [] and published == []


def test_run_publisher_logs_the_tail_and_returns_the_exit_code(
        tmp_path, publish_dir, monkeypatch):
    script = tmp_path / "fake_publish.py"
    script.write_text("import sys; print('published by node0'); sys.exit(3)\n")
    monkeypatch.setattr(se, "_publish_command",
                        lambda key_dir: [sys.executable, str(script)])
    rc = se._run_publisher(str(tmp_path / "key"))
    assert rc == 3
    with open(os.path.join(publish_dir, "key.log")) as handle:
        assert "published by node0" in handle.read()
