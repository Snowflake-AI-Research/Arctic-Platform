"""Shutdown kills each worker's process tree and waits for the GCS to report it dead.

Ray is mocked at the module's seams: the GCS reads, the per-node process-tree
kill task, and ``ray.kill``. ``_kill_process_trees`` itself runs for real
against a local process tree.
"""

import asyncio
import logging
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest
import ray

from arctic_platform.inference.server import replica_pool as replica_pool_mod
from arctic_platform.inference.server.replica_pool import ReplicaPool

_LOGGER = "arctic_platform.inference.server"


class _RemoteMethod:
    def __init__(self, func):
        self._func = func

    def remote(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class _Actor:
    def __init__(self, index: int, *, hang_shutdown: bool = False, fail_init: bool = False) -> None:
        self.index = index
        self.actor_id = f"actor{index}"
        self._ray_actor_id = SimpleNamespace(hex=lambda: self.actor_id)
        self._hang_shutdown = hang_shutdown
        self._fail_init = fail_init
        self.initialize = _RemoteMethod(self._initialize)
        self.shutdown = _RemoteMethod(self._shutdown)

    async def _initialize(self, *args, **kwargs):
        if self._fail_init:
            raise RuntimeError("worker init failed")

    async def _shutdown(self):
        if self._hang_shutdown:
            await asyncio.Event().wait()
        return {"status": "ok"}


class _WorkerClass:
    def __init__(self, fail_init_index: int) -> None:
        self.actors: list[_Actor] = []
        self._fail_init_index = fail_init_index

    def options(self, **kwargs):
        return self

    def remote(self):
        actor = _Actor(len(self.actors), fail_init=len(self.actors) == self._fail_init_index)
        self.actors.append(actor)
        return actor


class _FakeGcs:
    """Actor table whose rows go DEAD a few reads after the actor is killed."""

    def __init__(self, *, dies_after_reads: int | None = 2) -> None:
        self.rows: dict[str, dict] = {}
        self._reads_left: dict[str, int] = {}
        self._dies_after_reads = dies_after_reads
        self.observed_dead: set[str] = set()

    def add(self, actor_id: str, *, pid: int, node_id: str = "node-a", ip: str = "10.0.0.1", cls: str = "X") -> None:
        self.rows[actor_id] = dict(
            ActorID=actor_id,
            ActorClassName=cls,
            State="ALIVE",
            Pid=pid,
            StartTime=0,
            Address=dict(NodeID=node_id, IPAddress=ip),
        )

    def kill(self, actor_id: str) -> None:
        if self._dies_after_reads is not None and actor_id not in self._reads_left:
            self._reads_left[actor_id] = self._dies_after_reads

    def read(self, actor_ids):
        records = []
        for actor_id in actor_ids:
            if actor_id is None or actor_id not in self.rows:
                records.append(None)
                continue
            row = self.rows[actor_id]
            if actor_id in self._reads_left:
                self._reads_left[actor_id] -= 1
                if self._reads_left[actor_id] <= 0:
                    row["State"] = "DEAD"
            if row["State"] == "DEAD":
                self.observed_dead.add(actor_id)
            records.append(dict(row))
        return records


def _install(monkeypatch, gcs: _FakeGcs, *, ranks: list[dict] | None = None):
    tree_kills: list[tuple[str, list]] = []
    killed: list[str] = []
    by_pid = {}

    async def kill_trees_on_node(node_id, roots, timeout_s):
        tree_kills.append((node_id, list(roots)))
        pids = [pid for pid, _ in roots]
        for pid in pids:
            gcs.kill(by_pid[pid])
        return dict(killed=pids + [pid + 10_000 for pid in pids], alive=[])

    def ray_kill(actor):
        killed.append(actor.actor_id)
        gcs.kill(actor.actor_id)

    def read_records(actor_ids):
        for row in gcs.rows.values():
            by_pid[row["Pid"]] = row["ActorID"]
        return gcs.read(actor_ids)

    monkeypatch.setattr(replica_pool_mod, "_read_actor_records", read_records)
    monkeypatch.setattr(replica_pool_mod, "_kill_trees_on_node", kill_trees_on_node)
    monkeypatch.setattr(replica_pool_mod, "_driver_actor_records", lambda drivers, started: list(ranks or []))
    monkeypatch.setattr(replica_pool_mod.ray, "kill", ray_kill)
    monkeypatch.setattr(replica_pool_mod, "_KILL_POLL_INTERVAL_S", 0.01)
    return tree_kills, killed


def test_hung_shutdown_hard_kills_trees_and_returns_after_death(monkeypatch, caplog):
    gcs = _FakeGcs(dies_after_reads=3)
    workers = [_Actor(0, hang_shutdown=True), _Actor(1)]
    for worker in workers:
        gcs.add(worker.actor_id, pid=100 + worker.index)
    tree_kills, killed = _install(monkeypatch, gcs)
    monkeypatch.setenv("ARCTIC_WORKER_SHUTDOWN_TIMEOUT_S", "0.05")
    caplog.set_level(logging.INFO, logger=_LOGGER)

    pool = ReplicaPool()
    pool._workers = list(workers)
    asyncio.run(pool.shutdown())

    assert tree_kills == [("node-a", [(100, 0.0), (101, 0.0)])]
    assert sorted(killed) == ["actor0", "actor1"]
    # Returned only after the GCS was seen reporting both actors DEAD.
    assert gcs.observed_dead == {"actor0", "actor1"}
    assert "shutdown RPCs still running for workers [0]" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert pool._workers == []


def test_tp_rank_actors_of_killed_engines_are_killed_and_confirmed(monkeypatch):
    gcs = _FakeGcs()
    worker = _Actor(0)
    gcs.add(worker.actor_id, pid=100)
    gcs.add("rank0", pid=200, node_id="node-b", ip="10.0.0.2", cls="RayWorkerWrapper")
    tree_kills, _ = _install(monkeypatch, gcs, ranks=[dict(gcs.rows["rank0"])])

    pool = ReplicaPool()
    pool._workers = [worker]
    asyncio.run(pool.shutdown())

    assert ("node-b", [(200, 0.0)]) in tree_kills
    assert gcs.observed_dead == {"actor0", "rank0"}


def test_confirm_timeout_logs_error_and_returns(monkeypatch, caplog):
    gcs = _FakeGcs(dies_after_reads=None)
    workers = [_Actor(0), _Actor(1)]
    for worker in workers:
        gcs.add(worker.actor_id, pid=100 + worker.index)
    _install(monkeypatch, gcs)
    monkeypatch.setenv("ARCTIC_WORKER_KILL_CONFIRM_TIMEOUT_S", "0.2")
    caplog.set_level(logging.WARNING, logger=_LOGGER)

    pool = ReplicaPool()
    pool._workers = list(workers)
    start = time.monotonic()
    asyncio.run(pool.shutdown())
    elapsed = time.monotonic() - start

    assert 0.2 <= elapsed < 2.0
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "['worker 0', 'worker 1']" in errors[0]
    assert pool._workers == []


def test_probe_confirms_death_when_gcs_is_unreadable(monkeypatch, caplog):
    gcs = _FakeGcs()
    _install(monkeypatch, gcs)
    calls = []

    async def ready():
        calls.append(1)
        if len(calls) < 3:
            await asyncio.sleep(1.0)
        raise ray.exceptions.ActorDiedError()

    worker = _Actor(0)
    worker.__ray_ready__ = _RemoteMethod(ready)
    caplog.set_level(logging.ERROR, logger=_LOGGER)

    pool = ReplicaPool()
    pool._workers = [worker]
    asyncio.run(pool.shutdown())

    assert len(calls) == 3
    assert not caplog.records


def test_init_error_propagates_when_hard_kill_raises(monkeypatch):
    def read_records(actor_ids):
        raise RuntimeError("gcs unreachable")

    killed = []
    monkeypatch.setattr(replica_pool_mod, "_read_actor_records", read_records)
    monkeypatch.setattr(replica_pool_mod.ray, "kill", lambda actor: killed.append(actor.index))

    worker_cls = _WorkerClass(fail_init_index=1)
    pool = ReplicaPool(worker_cls=worker_cls)
    config = SimpleNamespace(
        tensor_parallel_size=1,
        ray_num_gpus=None,
        extra_env={},
        clear_cache_on_weight_sync=True,
        to_engine_kwargs=lambda: {"model": "test-model"},
    )

    with pytest.raises(RuntimeError, match="worker init failed"):
        asyncio.run(pool.initialize(config, model_id="job-1", num_replicas=2))

    # ray.kill still reaches every worker when the GCS cannot be read.
    assert killed == [0, 1]
    assert pool._workers == []
    assert pool._config is None


def test_init_error_propagates_when_cleanup_raises(monkeypatch):
    async def shutdown_workers(workers):
        raise RuntimeError("cleanup failed")

    worker_cls = _WorkerClass(fail_init_index=0)
    pool = ReplicaPool(worker_cls=worker_cls)
    monkeypatch.setattr(pool, "_shutdown_workers", shutdown_workers)
    config = SimpleNamespace(
        tensor_parallel_size=1,
        ray_num_gpus=None,
        extra_env={},
        clear_cache_on_weight_sync=True,
        to_engine_kwargs=lambda: {"model": "test-model"},
    )

    with pytest.raises(RuntimeError, match="worker init failed"):
        asyncio.run(pool.initialize(config, model_id="job-1", num_replicas=1))

    assert pool._workers == []
    assert pool._config is None


def test_driver_actor_records_matches_engine_core_drivers(monkeypatch):
    from ray._private import state

    jobs = [
        dict(JobID="01000000", DriverIPAddress="10.0.0.1", DriverPid=500, StartTime=10_000),
        # Same pid, earlier job: a previous driver, not this engine's.
        dict(JobID="02000000", DriverIPAddress="10.0.0.1", DriverPid=500, StartTime=1_000),
        dict(JobID="03000000", DriverIPAddress="10.0.0.2", DriverPid=500, StartTime=10_000),
    ]
    actors = {
        "01000000": {
            "r0": dict(ActorID="r0", ActorClassName="RayWorkerWrapper", State="ALIVE"),
            "r1": dict(ActorID="r1", ActorClassName="RayWorkerWrapper", State="DEAD"),
            "o0": dict(ActorID="o0", ActorClassName="Other", State="ALIVE"),
        },
        "02000000": {"r2": dict(ActorID="r2", ActorClassName="RayWorkerWrapper", State="ALIVE")},
        "03000000": {"r3": dict(ActorID="r3", ActorClassName="RayWorkerWrapper", State="ALIVE")},
    }
    monkeypatch.setattr(state, "jobs", lambda: jobs)
    monkeypatch.setattr(state, "actors", lambda job_id: actors[job_id.hex()])

    records = replica_pool_mod._driver_actor_records({"10.0.0.1": {500, 501}}, started_after_s=5.0)

    assert [r["ActorID"] for r in records] == ["r0"]
    assert replica_pool_mod._driver_actor_records({}, started_after_s=0.0) == []


_TREE = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print(child.pid, flush=True)
time.sleep(60)
"""


def test_kill_process_trees_kills_descendants():
    root = subprocess.Popen([sys.executable, "-c", _TREE], stdout=subprocess.PIPE, text=True)
    try:
        child_pid = int(root.stdout.readline())
        started = psutil.Process(root.pid).create_time()

        result = replica_pool_mod._kill_process_trees([(root.pid, started)], wait_s=10.0)

        assert result == dict(killed=sorted([root.pid, child_pid]), alive=[])
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
    finally:
        root.kill()
        root.wait()


def test_kill_process_trees_skips_reused_pid():
    root = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        # The actor started well before this process was created, so the pid
        # now belongs to something else.
        started = psutil.Process(root.pid).create_time() - 60.0

        result = replica_pool_mod._kill_process_trees([(root.pid, started)], wait_s=1.0)

        assert result == dict(killed=[], alive=[])
        assert root.poll() is None
    finally:
        root.kill()
        root.wait()
