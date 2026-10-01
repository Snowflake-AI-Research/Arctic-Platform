import os

from arctic_inference.server.router_replay import shm
from arctic_inference.server.worker import InferenceWorker, WorkerLifecycleState


def _configure_tmp_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCTIC_ROUTER_REPLAY_SHM_REGISTRY_DIR", str(tmp_path / "registry"))
    monkeypatch.setenv("ARCTIC_ROUTER_REPLAY_SHM_DIR", str(tmp_path / "shm"))
    monkeypatch.setenv("ARCTIC_ROUTER_REPLAY_SHM_LOCK_DIR", str(tmp_path / "locks"))
    (tmp_path / "shm").mkdir()
    (tmp_path / "locks").mkdir()


def test_scoped_cleanup_removes_only_dead_pid_entries(monkeypatch, tmp_path):
    _configure_tmp_dirs(monkeypatch, tmp_path)

    dead = shm.register_expected_buffer(
        scope="zone-a",
        model_id="job-2",
        instance_id="dead-instance",
        dp_rank=0,
        pid=999_999_999,
    )
    alive = shm.register_expected_buffer(
        scope="zone-a",
        model_id="job-2",
        instance_id="alive-instance",
        dp_rank=0,
        pid=os.getpid(),
    )
    for entry in (dead, alive):
        open(entry["shm_path"], "wb").close()
        open(entry["lock_file"], "wb").close()

    result = shm.cleanup_scope(scope="zone-a", model_id="job-2", stale_only=True)

    assert result["entries"] == 2
    assert result["removed"] == 3
    assert not os.path.exists(dead["shm_path"])
    assert not os.path.exists(dead["lock_file"])
    assert not os.path.exists(dead["registry_path"])
    assert os.path.exists(alive["shm_path"])
    assert os.path.exists(alive["lock_file"])
    assert os.path.exists(alive["registry_path"])


def test_cleanup_scope_is_scoped_by_zone_and_model(monkeypatch, tmp_path):
    _configure_tmp_dirs(monkeypatch, tmp_path)

    target = shm.register_expected_buffer(
        scope="zone-a",
        model_id="job-2",
        instance_id="target",
        dp_rank=0,
        pid=999_999_999,
    )
    other = shm.register_expected_buffer(
        scope="zone-b",
        model_id="job-2",
        instance_id="other",
        dp_rank=0,
        pid=999_999_999,
    )
    for entry in (target, other):
        open(entry["shm_path"], "wb").close()
        open(entry["lock_file"], "wb").close()

    shm.cleanup_scope(scope="zone-a", model_id="job-2", stale_only=True)

    assert not os.path.exists(target["shm_path"])
    assert os.path.exists(other["shm_path"])
    assert os.path.exists(other["registry_path"])


def test_inference_worker_shutdown_cleans_registered_shm(monkeypatch, tmp_path):
    _configure_tmp_dirs(monkeypatch, tmp_path)
    worker_cls = InferenceWorker.__ray_metadata__.modified_class
    worker = worker_cls()
    entry = shm.register_expected_buffer(
        scope="zone-a",
        model_id="job-2",
        instance_id="shutdown-instance",
        dp_rank=0,
        pid=os.getpid(),
    )
    open(entry["shm_path"], "wb").close()
    open(entry["lock_file"], "wb").close()
    worker._router_replay_shm_entry = entry
    worker._router_replay_shm_scope = "zone-a"
    worker.state = WorkerLifecycleState.READY

    class FakeLLM:
        pass

    worker.llm = FakeLLM()

    worker.shutdown()

    assert not os.path.exists(entry["shm_path"])
    assert not os.path.exists(entry["lock_file"])
    assert not os.path.exists(entry["registry_path"])
    assert worker.state == WorkerLifecycleState.UNINITIALIZED
