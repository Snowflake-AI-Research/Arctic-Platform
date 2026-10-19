"""Shutdown kills each worker's process tree and waits for the GCS to report it dead.

Ray is mocked at the module's seams: the GCS reads, the per-node process-tree
kill task, and ``ray.kill``. ``_kill_process_trees`` itself runs for real
against local process trees.
"""

import asyncio
import logging
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import psutil
import pytest
import ray

from arctic_platform.inference.server import replica_pool as replica_pool_mod
from arctic_platform.inference.server.replica_pool import ReplicaPool

_LOGGER = "arctic_platform.inference.server"
# Actor start time in the fake GCS, in ms, as the GCS reports it. Non-zero, so
# the pid-reuse and job-start checks run.
_START_MS = 1_700_000_000_000


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
        self.unreadable_reads: dict[str, int] = {}
        self._reads_left: dict[str, int] = {}
        self._dies_after_reads = dies_after_reads
        self.observed_dead: set[str] = set()

    def add(self, actor_id: str, *, pid: int, node_id: str = "node-a", ip: str = "10.0.0.1", cls: str = "X") -> None:
        self.rows[actor_id] = dict(
            ActorID=actor_id,
            ActorClassName=cls,
            State="ALIVE",
            Pid=pid,
            StartTime=_START_MS,
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
            if self.unreadable_reads.get(actor_id, 0) > 0:
                self.unreadable_reads[actor_id] -= 1
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
    driver_calls: list[tuple] = []

    async def kill_trees_on_node(node_id, roots, timeout_s):
        tree_kills.append((node_id, list(roots)))
        by_pid = {row["Pid"]: actor_id for actor_id, row in gcs.rows.items()}
        for pid, _, kill_root in roots:
            if kill_root:
                gcs.kill(by_pid[pid])
        return dict(killed=[pid + 10_000 for pid, _, _ in roots], alive=[], denied=[])

    def ray_kill(actor):
        killed.append(actor.actor_id)
        gcs.kill(actor.actor_id)

    def driver_actor_records(drivers, started_after_s, killed_at_s):
        driver_calls.append((drivers, started_after_s, killed_at_s))
        return list(ranks or [])

    monkeypatch.setattr(replica_pool_mod, "_read_actor_records", gcs.read)
    monkeypatch.setattr(replica_pool_mod, "_kill_trees_on_node", kill_trees_on_node)
    monkeypatch.setattr(replica_pool_mod, "_driver_actor_records", driver_actor_records)
    monkeypatch.setattr(replica_pool_mod.ray, "kill", ray_kill)
    monkeypatch.setattr(replica_pool_mod, "_KILL_POLL_INTERVAL_S", 0.01)
    return SimpleNamespace(tree_kills=tree_kills, killed=killed, driver_calls=driver_calls)


def _config():
    return SimpleNamespace(
        tensor_parallel_size=1,
        ray_num_gpus=None,
        extra_env={},
        clear_cache_on_weight_sync=True,
        to_engine_kwargs=lambda: {"model": "test-model"},
    )


def _errors(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def test_hung_shutdown_hard_kills_trees_and_returns_after_death(monkeypatch, caplog):
    gcs = _FakeGcs(dies_after_reads=3)
    workers = [_Actor(0, hang_shutdown=True), _Actor(1)]
    for worker in workers:
        gcs.add(worker.actor_id, pid=100 + worker.index)
    seen = _install(monkeypatch, gcs)
    monkeypatch.setenv("ARCTIC_WORKER_SHUTDOWN_TIMEOUT_S", "0.05")
    caplog.set_level(logging.INFO, logger=_LOGGER)

    pool = ReplicaPool()
    pool._workers = list(workers)
    asyncio.run(pool.shutdown())

    # The hung worker's own process is killed with its tree; the one that
    # shut down cleanly keeps its root for ray.kill. Start times are passed
    # through in seconds for the pid-reuse check.
    start_s = _START_MS / 1000.0
    assert seen.tree_kills == [("node-a", [(100, start_s, True), (101, start_s, False)])]
    assert sorted(seen.killed) == ["actor0", "actor1"]
    # Returned only after the GCS was seen reporting both actors DEAD.
    assert gcs.observed_dead == {"actor0", "actor1"}
    # The rank lookup gets the node id and IP of each node, and the actors'
    # (non-zero) start time to reject earlier jobs.
    (drivers, started_after_s, killed_at_s), = seen.driver_calls
    assert drivers == [("node-a", "10.0.0.1", {10_100, 10_101})]
    assert started_after_s == start_s
    assert killed_at_s > start_s
    assert "shutdown RPCs still running for workers [0]" in caplog.text
    assert _errors(caplog) == []
    assert pool._workers == []


def test_tp_rank_actors_of_killed_engines_are_killed_and_confirmed(monkeypatch):
    gcs = _FakeGcs()
    worker = _Actor(0)
    gcs.add(worker.actor_id, pid=100)
    gcs.add("rank0", pid=200, node_id="node-b", ip="10.0.0.2", cls="RayWorkerWrapper")
    seen = _install(monkeypatch, gcs, ranks=[dict(gcs.rows["rank0"])])

    pool = ReplicaPool()
    pool._workers = [worker]
    asyncio.run(pool.shutdown())

    assert ("node-b", [(200, _START_MS / 1000.0, True)]) in seen.tree_kills
    assert gcs.observed_dead == {"actor0", "rank0"}


def test_rank_target_is_retried_after_failed_reads(monkeypatch, caplog):
    gcs = _FakeGcs()
    worker = _Actor(0)
    gcs.add(worker.actor_id, pid=100)
    gcs.add("rank0", pid=200, node_id="node-b", ip="10.0.0.2", cls="RayWorkerWrapper")
    gcs.unreadable_reads["rank0"] = 5
    _install(monkeypatch, gcs, ranks=[dict(gcs.rows["rank0"])])
    caplog.set_level(logging.ERROR, logger=_LOGGER)

    pool = ReplicaPool()
    pool._workers = [worker]
    asyncio.run(pool.shutdown())

    assert "rank0" in gcs.observed_dead
    assert _errors(caplog) == []


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
    errors = _errors(caplog)
    assert len(errors) == 1
    assert "['worker 0', 'worker 1']" in errors[0]
    assert pool._workers == []


def test_hung_gcs_read_is_bounded_by_the_confirm_timeout(monkeypatch, caplog):
    release = threading.Event()

    def read_records(actor_ids):
        release.wait(10.0)
        return [None] * len(actor_ids)

    killed = []
    monkeypatch.setattr(replica_pool_mod, "_read_actor_records", read_records)
    monkeypatch.setattr(replica_pool_mod.ray, "kill", lambda actor: killed.append(actor.index))
    monkeypatch.setenv("ARCTIC_WORKER_KILL_CONFIRM_TIMEOUT_S", "0.2")
    caplog.set_level(logging.ERROR, logger=_LOGGER)

    pool = ReplicaPool()
    pool._workers = [_Actor(0)]

    async def run():
        start = time.monotonic()
        try:
            await asyncio.wait_for(pool.shutdown(), timeout=5.0)
            return time.monotonic() - start
        finally:
            # Let the abandoned read thread finish, so asyncio.run can exit.
            release.set()

    elapsed = asyncio.run(run())

    assert elapsed < 2.0
    assert killed == [0]
    assert any("worker 0 (state unreadable)" in message for message in _errors(caplog))


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
    monkeypatch.setenv("ARCTIC_WORKER_KILL_CONFIRM_TIMEOUT_S", "0.1")

    worker_cls = _WorkerClass(fail_init_index=1)
    pool = ReplicaPool(worker_cls=worker_cls)

    with pytest.raises(RuntimeError, match="worker init failed"):
        asyncio.run(pool.initialize(_config(), model_id="job-1", num_replicas=2))

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

    with pytest.raises(RuntimeError, match="worker init failed"):
        asyncio.run(pool.initialize(_config(), model_id="job-1", num_replicas=1))

    assert pool._workers == []
    assert pool._config is None


# ----------------------------------------------------------------------
# The other paths that drop a worker go through the same hard kill.
# ----------------------------------------------------------------------


class _Scheduler:
    def __init__(self) -> None:
        self.events: list[str] = []

    def mark_worker_unavailable(self, idx):
        self.events.append(f"unavailable {idx}")

    def mark_worker_available(self, idx):
        self.events.append(f"available {idx}")

    async def abort_streams(self, worker_idx):
        pass

    async def drain_worker(self, idx):
        pass

    def remove_last_worker(self):
        self.events.append("removed")

    def update_worker_handle(self, idx, worker):
        self.events.append(f"updated {idx}")

    def add_worker(self, worker, concurrency_limit):
        self.events.append("added")


def _ready_pool(monkeypatch, worker_cls, workers):
    pool = ReplicaPool(worker_cls=worker_cls)
    pool._config = _config()
    pool._workers = list(workers)
    pool._slots = [(None, idx) for idx in range(len(workers))]
    pool._scheduler = _Scheduler()
    hard_kills: list[list[int]] = []
    pool.kill_roots_seen = []

    async def kill_workers_and_confirm(workers, kill_roots=None):
        hard_kills.append([worker.index for worker in workers])
        pool.kill_roots_seen.append(kill_roots)
        pool._scheduler.events.append(f"hard-killed {[worker.index for worker in workers]}")

    monkeypatch.setattr(pool, "_kill_workers_and_confirm", kill_workers_and_confirm)
    return pool, hard_kills


def test_scale_down_hard_kills_all_removed_workers_at_once(monkeypatch):
    workers = [_Actor(0), _Actor(1), _Actor(2)]
    pool, hard_kills = _ready_pool(monkeypatch, _WorkerClass(fail_init_index=-1), workers)

    asyncio.run(pool.scale_down(1))

    # One call for both, after both have left the scheduler, with the roots
    # left to ray.kill.
    assert hard_kills == [[2, 1]]
    assert pool.kill_roots_seen == [None]
    assert pool._scheduler.events == [
        "unavailable 2",
        "removed",
        "unavailable 1",
        "removed",
        "hard-killed [2, 1]",
    ]
    assert pool._workers == [workers[0]]


def test_scale_down_leaves_roots_to_ray_kill(monkeypatch, caplog):
    gcs = _FakeGcs()
    workers = [_Actor(0), _Actor(1), _Actor(2)]
    for worker in workers:
        gcs.add(worker.actor_id, pid=100 + worker.index)
    seen = _install(monkeypatch, gcs)
    caplog.set_level(logging.ERROR, logger=_LOGGER)

    pool = ReplicaPool()
    pool._config = _config()
    pool._workers = list(workers)
    pool._scheduler = _Scheduler()
    asyncio.run(pool.scale_down(1))

    start_s = _START_MS / 1000.0
    assert seen.tree_kills == [("node-a", [(102, start_s, False), (101, start_s, False)])]
    assert sorted(seen.killed) == ["actor1", "actor2"]
    assert gcs.observed_dead == {"actor1", "actor2"}
    assert _errors(caplog) == []


def test_scale_up_init_failure_hard_kills_the_new_worker(monkeypatch):
    worker_cls = _WorkerClass(fail_init_index=0)
    pool, hard_kills = _ready_pool(monkeypatch, worker_cls, [])

    with pytest.raises(RuntimeError, match="worker init failed"):
        asyncio.run(pool.scale_up(1))

    assert hard_kills == [[0]]
    assert pool.kill_roots_seen == [None]
    assert pool._workers == []


def test_single_node_restart_waits_for_old_worker_death_before_replacing(monkeypatch):
    worker_cls = _WorkerClass(fail_init_index=-1)
    old = _Actor(99)
    pool, hard_kills = _ready_pool(monkeypatch, worker_cls, [old])
    made_after_kill = []
    make_worker = pool._make_worker

    def make_worker_checked():
        made_after_kill.append(hard_kills == [[99]])
        return make_worker()

    monkeypatch.setattr(pool, "_make_worker", make_worker_checked)

    asyncio.run(pool._restart_worker(0))

    assert made_after_kill == [True]
    assert pool.kill_roots_seen == [None]
    assert pool._workers == [worker_cls.actors[0]]


def test_single_node_restart_hard_kills_a_replacement_that_fails(monkeypatch):
    worker_cls = _WorkerClass(fail_init_index=0)
    pool, hard_kills = _ready_pool(monkeypatch, worker_cls, [_Actor(99)])

    with pytest.raises(RuntimeError, match="worker init failed"):
        asyncio.run(pool._restart_worker(0))

    assert hard_kills == [[99], [0]]
    assert pool.kill_roots_seen == [None, None]


# ----------------------------------------------------------------------
# TP-rank job matching.
# ----------------------------------------------------------------------


def test_driver_actor_records_matches_engine_core_drivers(monkeypatch):
    from ray._private import state

    def rank(actor_id, state_name="ALIVE", **extra):
        return dict(ActorID=actor_id, ActorClassName="RayWorkerWrapper", State=state_name, **extra)

    jobs = [
        dict(JobID="01000000", DriverIPAddress="10.0.0.1", DriverPid=500, StartTime=10_000, EndTime=0),
        # Same pid, earlier job: a previous driver, not this engine's.
        dict(JobID="02000000", DriverIPAddress="10.0.0.1", DriverPid=500, StartTime=1_000, EndTime=0),
        # Same pid on another node.
        dict(JobID="03000000", DriverIPAddress="10.0.0.2", DriverPid=500, StartTime=10_000, EndTime=0),
        # Same pid, but the job ended before the kill started.
        dict(JobID="04000000", DriverIPAddress="10.0.0.1", DriverPid=501, StartTime=10_000, EndTime=12_000),
        # Matched on node id where the job table has one, so the IP is not used.
        dict(JobID="05000000", DriverIPAddress="10.0.0.9", DriverNodeID="node-a", DriverPid=501, StartTime=10_000),
        dict(JobID="06000000", DriverIPAddress="10.0.0.1", DriverNodeID="node-z", DriverPid=501, StartTime=10_000),
    ]
    actors = {
        "01000000": {
            "r0": rank("r0"),
            "r1": rank("r1", "DEAD"),
            "r2": rank("r2", IsDetached=True),
            "o0": dict(ActorID="o0", ActorClassName="Other", State="ALIVE"),
        },
        "02000000": {"r3": rank("r3")},
        "03000000": {"r4": rank("r4")},
        "04000000": {"r5": rank("r5")},
        "05000000": {"r6": rank("r6")},
        "06000000": {"r7": rank("r7")},
    }
    monkeypatch.setattr(state, "jobs", lambda: jobs)
    monkeypatch.setattr(state, "actors", lambda job_id: actors[job_id.hex()])

    records = replica_pool_mod._driver_actor_records(
        [("node-a", "10.0.0.1", {500, 501})], started_after_s=5.0, killed_at_s=20.0
    )

    assert [r["ActorID"] for r in records] == ["r0", "r6"]
    assert replica_pool_mod._driver_actor_records([], started_after_s=0.0, killed_at_s=0.0) == []


def test_read_actor_records_reads_once_per_job(monkeypatch):
    from ray._private import state

    job = ray.JobID.from_int(7)
    ids = [ray.ActorID.of(job, ray.TaskID.nil(), i).hex() for i in range(3)]
    calls = []

    def actors(job_id):
        calls.append(job_id.hex())
        return {ids[0]: dict(ActorID=ids[0], State="ALIVE"), ids[1]: dict(ActorID=ids[1], State="DEAD")}

    monkeypatch.setattr(replica_pool_mod.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(state, "actors", actors)

    records = replica_pool_mod._read_actor_records([ids[0], ids[1], ids[2], None])

    assert calls == [job.hex()]
    assert [r if r is None else r.get("State") for r in records] == ["ALIVE", "DEAD", None, None]
    assert records[2] == {}


# ----------------------------------------------------------------------
# _kill_process_trees, against real processes.
# ----------------------------------------------------------------------

_TREE = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print(child.pid, flush=True)
time.sleep(60)
"""

# A parent that never reaps its exited child, which stays a zombie.
_ZOMBIE_PARENT = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "pass"])
print(child.pid, flush=True)
time.sleep(60)
"""


def _gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _spawn(source: str) -> tuple[subprocess.Popen, int]:
    root = subprocess.Popen([sys.executable, "-c", source], stdout=subprocess.PIPE, text=True)
    return root, int(root.stdout.readline())


def test_kill_process_trees_kills_descendants():
    root, child_pid = _spawn(_TREE)
    try:
        started = psutil.Process(root.pid).create_time()

        result = replica_pool_mod._kill_process_trees([(root.pid, started, True)], wait_s=10.0)

        # Reported gone even if nothing has reaped the reparented grandchild yet.
        assert result == dict(killed=sorted([root.pid, child_pid]), alive=[], denied=[])
        assert _gone(child_pid)
    finally:
        root.kill()
        root.wait()


def test_kill_process_trees_leaves_root_when_asked():
    root, child_pid = _spawn(_TREE)
    try:
        result = replica_pool_mod._kill_process_trees([(root.pid, 0.0, False)], wait_s=10.0)

        assert result == dict(killed=[child_pid], alive=[], denied=[])
        assert _gone(child_pid)
        assert root.poll() is None
    finally:
        root.kill()
        root.wait()


def test_kill_process_trees_skips_reused_pid():
    root = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        # The actor started well before this process was created, so the pid
        # now belongs to something else.
        started = psutil.Process(root.pid).create_time() - 60.0

        result = replica_pool_mod._kill_process_trees([(root.pid, started, True)], wait_s=1.0)

        assert result == dict(killed=[], alive=[], denied=[])
        assert root.poll() is None
    finally:
        root.kill()
        root.wait()


def test_kill_process_trees_counts_a_zombie_as_gone():
    root, zombie_pid = _spawn(_ZOMBIE_PARENT)
    try:
        deadline = time.monotonic() + 10.0
        while psutil.Process(zombie_pid).status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline
            time.sleep(0.05)

        start = time.monotonic()
        result = replica_pool_mod._kill_process_trees([(root.pid, 0.0, False)], wait_s=5.0)

        # Signalled, then reported gone at once rather than after the wait.
        assert result == dict(killed=[zombie_pid], alive=[], denied=[])
        assert time.monotonic() - start < 2.0
        assert root.poll() is None
    finally:
        root.kill()
        root.wait()


def test_kill_process_trees_continues_past_access_denied(monkeypatch):
    root, child_pid = _spawn(_TREE)
    real_kill = psutil.Process.kill

    def kill(proc):
        if proc.pid == child_pid:
            raise psutil.AccessDenied(proc.pid)
        return real_kill(proc)

    monkeypatch.setattr(psutil.Process, "kill", kill)
    try:
        result = replica_pool_mod._kill_process_trees([(root.pid, 0.0, True)], wait_s=10.0)

        assert result == dict(killed=[root.pid], alive=[], denied=[child_pid])
        assert _gone(root.pid)
    finally:
        monkeypatch.undo()
        for pid in (child_pid, root.pid):
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass
        root.wait()
