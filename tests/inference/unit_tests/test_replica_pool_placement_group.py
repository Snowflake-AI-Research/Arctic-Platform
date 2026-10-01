import asyncio
from types import SimpleNamespace

import pytest

from arctic_platform.inference.server import replica_pool as replica_pool_mod
from arctic_platform.inference.server.replica_pool import ReplicaPool


class _RemoteMethod:
    def __init__(self, func):
        self._func = func

    def remote(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class _FakeActor:
    def __init__(self, index: int, fail_init: bool = False) -> None:
        self.index = index
        self.fail_init = fail_init
        self.initialize_calls = []
        self.shutdown_called = False
        self.initialize = _RemoteMethod(self._initialize)
        self.shutdown = _RemoteMethod(self._shutdown)

    async def _initialize(self, *args, **kwargs):
        self.initialize_calls.append((args, kwargs))
        if self.fail_init:
            raise RuntimeError("worker init failed")
        return {"status": "ok"}

    async def _shutdown(self):
        self.shutdown_called = True
        return {"status": "ok"}


class _ActorFactory:
    def __init__(self, worker_cls):
        self._worker_cls = worker_cls

    def remote(self):
        index = len(self._worker_cls.actors)
        fail_init = (
            self._worker_cls.fail_sequence.pop(0)
            if self._worker_cls.fail_sequence
            else False
        )
        actor = _FakeActor(index=index, fail_init=fail_init)
        self._worker_cls.actors.append(actor)
        return actor


class _FakeWorkerClass:
    def __init__(self, fail_sequence=None) -> None:
        self.actors = []
        self.options_calls = []
        self.fail_sequence = list(fail_sequence or [])

    def options(self, **kwargs):
        self.options_calls.append(kwargs)
        return _ActorFactory(self)


class _FailingWorkerClass:
    def __init__(self) -> None:
        self.options_calls = []

    def options(self, **kwargs):
        self.options_calls.append(kwargs)
        return self

    def remote(self):
        raise RuntimeError("actor creation failed")


class _FakeScheduler:
    def __init__(self) -> None:
        self.unavailable = []
        self.available = []
        self.updated = []
        self.shutdown_called = False

    async def shutdown(self):
        self.shutdown_called = True

    def mark_worker_unavailable(self, index):
        self.unavailable.append(index)

    def mark_worker_available(self, index):
        self.available.append(index)

    def update_worker_handle(self, index, worker):
        self.updated.append((index, worker))


class _FakePlacementGroup:
    def __init__(self, bundles):
        self.bundle_specs = bundles


def _config(
    *,
    tp=2,
    pp=2,
    ray_num_gpus=None,
    extra_env=None,
    engine_kwargs=None,
):
    return SimpleNamespace(
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        ray_num_gpus=ray_num_gpus,
        extra_env=extra_env or {},
        to_engine_kwargs=lambda: {"model": "test-model", **(engine_kwargs or {})},
    )


def _placement_group(world_size=4):
    return _FakePlacementGroup([{"GPU": 1, "CPU": 1} for _ in range(world_size)])


def test_placement_group_initializes_one_zero_gpu_coordinator(monkeypatch):
    worker_cls = _FakeWorkerClass()
    killed = []

    async def run():
        pool = ReplicaPool(worker_cls=worker_cls)
        scheduler = _FakeScheduler()
        pool._make_scheduler = lambda workers: scheduler
        config = _config(extra_env={"NCCL_TEST": 1})
        pg = _placement_group()

        replicas = await pool.initialize(
            config,
            model_id="job-1",
            placement_group=pg,
        )

        assert replicas == 1
        assert pool.num_replicas == 1
        assert pool.uses_placement_group is True
        options = worker_cls.options_calls[0]
        assert options["num_gpus"] == 0
        assert options["num_cpus"] == 0
        assert options["runtime_env"] == {"env_vars": {"NCCL_TEST": "1"}}
        strategy = options["scheduling_strategy"]
        assert strategy.placement_group is pg
        assert strategy.placement_group_bundle_index == 0
        assert strategy.placement_group_capture_child_tasks is True

        engine_kwargs, worker_env, model_id = worker_cls.actors[0].initialize_calls[0][
            0
        ]
        assert engine_kwargs["distributed_executor_backend"] == "ray"
        assert "VLLM_PORT" not in worker_env
        assert model_id == "job-1"

        await pool.shutdown()
        assert pool.uses_placement_group is False
        assert scheduler.shutdown_called is True

    monkeypatch.setattr(
        replica_pool_mod.ray,
        "kill",
        lambda actor: killed.append(actor.index),
    )

    asyncio.run(run())

    assert killed == [0]


def test_placement_group_initialize_failure_clears_reference(monkeypatch):
    worker_cls = _FakeWorkerClass(fail_sequence=[True])
    killed = []
    pg = _placement_group()

    async def run():
        pool = ReplicaPool(worker_cls=worker_cls)

        with pytest.raises(RuntimeError, match="worker init failed"):
            await pool.initialize(
                _config(),
                model_id="job-1",
                placement_group=pg,
            )

        assert pool._config is None
        assert pool._workers == []
        assert pool.uses_placement_group is False

    monkeypatch.setattr(
        replica_pool_mod.ray,
        "kill",
        lambda actor: killed.append(actor.index),
    )

    asyncio.run(run())

    assert killed == [0]
    assert pg.bundle_specs == _placement_group().bundle_specs


def test_placement_group_actor_creation_failure_clears_reference():
    worker_cls = _FailingWorkerClass()
    pool = ReplicaPool(worker_cls=worker_cls)

    with pytest.raises(RuntimeError, match="actor creation failed"):
        asyncio.run(
            pool.initialize(
                _config(),
                placement_group=_placement_group(),
            )
        )

    assert len(worker_cls.options_calls) == 1
    assert pool._config is None
    assert pool._workers == []
    assert pool.uses_placement_group is False


@pytest.mark.parametrize(
    ("config", "pg", "message"),
    [
        (_config(), _placement_group(world_size=3), "exactly 4"),
        (
            _config(),
            _FakePlacementGroup(
                [
                    {"GPU": 1},
                    {"GPU": 1},
                    {"GPU": 0},
                    {"GPU": 1},
                ]
            ),
            "bundle 2",
        ),
        (_config(ray_num_gpus=4), _placement_group(), "ray_num_gpus"),
    ],
)
def test_placement_group_validation_fails_before_worker_creation(
    config,
    pg,
    message,
):
    worker_cls = _FakeWorkerClass()
    pool = ReplicaPool(worker_cls=worker_cls)

    with pytest.raises(ValueError, match=message):
        asyncio.run(pool.initialize(config, placement_group=pg))

    assert worker_cls.actors == []
    assert pool._config is None
    assert pool.uses_placement_group is False


def test_placement_group_rejects_incompatible_executor_before_worker_creation():
    worker_cls = _FakeWorkerClass()
    pool = ReplicaPool(worker_cls=worker_cls)
    config = _config(engine_kwargs={"distributed_executor_backend": "mp"})

    with pytest.raises(ValueError, match="require.*backend='ray'"):
        asyncio.run(pool.initialize(config, placement_group=_placement_group()))

    assert worker_cls.actors == []


def test_single_node_initialization_reserves_tp_times_pp_gpus(monkeypatch):
    worker_cls = _FakeWorkerClass()
    killed = []

    async def run():
        pool = ReplicaPool(worker_cls=worker_cls)
        pool._make_scheduler = lambda workers: _FakeScheduler()

        replicas = await pool.initialize(_config(), num_replicas=1)

        assert replicas == 1
        assert worker_cls.options_calls == [
            {
                "num_gpus": 4.0,
                "max_concurrency": 2048,
            }
        ]
        await pool.shutdown()

    monkeypatch.setattr(
        replica_pool_mod.ray,
        "kill",
        lambda actor: killed.append(actor.index),
    )

    asyncio.run(run())

    assert killed == [0]


def test_placement_group_rejects_replica_scaling():
    pool = ReplicaPool()
    pool._config = _config()
    pool._engine_pg = _placement_group()

    with pytest.raises(ValueError, match="exactly one replica"):
        asyncio.run(pool.scale_up(2))
    with pytest.raises(ValueError, match="exactly one replica"):
        asyncio.run(pool.scale_down(0))


def test_placement_group_restart_retries_after_bundle_release(monkeypatch):
    worker_cls = _FakeWorkerClass(fail_sequence=[True, False])
    old_worker = _FakeActor(index=99)
    killed = []

    async def run():
        pool = ReplicaPool(worker_cls=worker_cls)
        scheduler = _FakeScheduler()
        pool._config = _config()
        pool._model_id = "job-1"
        pool._engine_pg = _placement_group()
        pool._workers = [old_worker]
        pool._scheduler = scheduler

        await pool._restart_worker(0)

        assert len(worker_cls.actors) == 2
        assert pool._workers == [worker_cls.actors[1]]
        assert scheduler.unavailable == [0]
        assert scheduler.available == [0]
        assert scheduler.updated == [(0, worker_cls.actors[1])]

    monkeypatch.setenv("ARCTIC_WORKER_RESTART_TIMEOUT_S", "1")
    monkeypatch.setenv("ARCTIC_WORKER_RESTART_RETRY_S", "0")
    monkeypatch.setattr(
        replica_pool_mod.ray,
        "kill",
        lambda actor: killed.append(actor.index),
    )

    asyncio.run(run())

    assert old_worker.shutdown_called is True
    assert killed == [99, 0]


def test_single_node_restart_keeps_normal_actor_path(monkeypatch):
    worker_cls = _FakeWorkerClass()
    old_worker = _FakeActor(index=99)
    killed = []

    async def run():
        pool = ReplicaPool(worker_cls=worker_cls)
        scheduler = _FakeScheduler()
        pool._config = _config()
        pool._model_id = "job-1"
        pool._workers = [old_worker]
        pool._scheduler = scheduler

        await pool._restart_worker(0)

        assert worker_cls.options_calls == [
            {
                "num_gpus": 4.0,
                "max_concurrency": 2048,
            }
        ]
        assert pool._workers == [worker_cls.actors[0]]
        assert scheduler.updated == [(0, worker_cls.actors[0])]

    monkeypatch.setattr(
        replica_pool_mod.ray,
        "kill",
        lambda actor: killed.append(actor.index),
    )

    asyncio.run(run())

    assert old_worker.shutdown_called is False
    assert killed == [99]
