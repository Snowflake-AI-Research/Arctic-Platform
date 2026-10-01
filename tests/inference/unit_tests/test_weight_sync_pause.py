from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

_ray = types.ModuleType("ray")
_ray.remote = lambda obj=None, **kwargs: obj if obj is not None else (lambda inner: inner)
_ray.init = lambda *args, **kwargs: None
_ray.nodes = lambda: []
_ray.kill = lambda *args, **kwargs: None
_ray.actor = SimpleNamespace(ActorHandle=object)
_ray_util = types.ModuleType("ray.util")
_ray_sched = types.ModuleType("ray.util.scheduling_strategies")
_ray_sched.PlacementGroupSchedulingStrategy = object
_ray.util = _ray_util
sys.modules.setdefault("ray", _ray)
sys.modules.setdefault("ray.util", _ray_util)
sys.modules.setdefault("ray.util.scheduling_strategies", _ray_sched)
vllm_module = types.ModuleType("vllm")
vllm_module.__version__ = "0.23.0"
sys.modules.setdefault("vllm", vllm_module)
vllm_config = types.ModuleType("vllm.config")
vllm_config.VllmConfig = object
sys.modules.setdefault("vllm.config", vllm_config)
vllm_v1 = types.ModuleType("vllm.v1")
vllm_core = types.ModuleType("vllm.v1.core")
vllm_sched = types.ModuleType("vllm.v1.core.sched")
vllm_scheduler = types.ModuleType("vllm.v1.core.sched.scheduler")


def _patched_check_stop(*args, **kwargs):
    return False


_patched_check_stop._arctic_router_replay_patch = True
vllm_scheduler.check_stop = _patched_check_stop
vllm_metrics = types.ModuleType("vllm.v1.metrics")
vllm_loggers = types.ModuleType("vllm.v1.metrics.loggers")
vllm_loggers.StatLoggerBase = object
vllm_stats = types.ModuleType("vllm.v1.metrics.stats")
vllm_stats.IterationStats = object
vllm_stats.SchedulerStats = object
sys.modules.setdefault("vllm.v1", vllm_v1)
sys.modules.setdefault("vllm.v1.core", vllm_core)
sys.modules.setdefault("vllm.v1.core.sched", vllm_sched)
sys.modules.setdefault("vllm.v1.core.sched.scheduler", vllm_scheduler)
sys.modules.setdefault("vllm.v1.metrics", vllm_metrics)
sys.modules.setdefault("vllm.v1.metrics.loggers", vllm_loggers)
sys.modules.setdefault("vllm.v1.metrics.stats", vllm_stats)

from arctic_platform.inference.server.replica_pool import ReplicaPool


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return await self._fn(*args, **kwargs)


class _FakeWorker:
    def __init__(self, events: list[str], *, reset_ok: bool = True):
        self.events = events
        self.reset_ok = reset_ok
        self.pause_generation = _RemoteMethod(self._pause_generation)
        self.resume_generation = _RemoteMethod(self._resume_generation)
        self.reset_prefix_cache = _RemoteMethod(self._reset_prefix_cache)
        self.sync_weights = _RemoteMethod(self._sync_weights)
        self.sync_lora_weights_broadcast = _RemoteMethod(
            self._sync_lora_weights_broadcast
        )

    async def _pause_generation(self, mode: str = "keep", clear_cache: bool = False):
        self.events.append(f"pause_generation:{mode}:{clear_cache}")
        return {"status": "paused", "mode": mode, "clear_cache": clear_cache}

    async def _resume_generation(self):
        self.events.append("resume_generation")
        return {"status": "resumed"}

    async def _reset_prefix_cache(self, timeout_s: float, retry_interval_s: float):
        self.events.append(f"reset_prefix_cache:{timeout_s}")
        return {"status": "ok" if self.reset_ok else "failed", "reset_ok": self.reset_ok}

    async def _sync_weights(self, *args):
        self.events.append("sync_weights")
        return {"status": "done", "params_loaded": 1}

    async def _sync_lora_weights_broadcast(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        lora_int_id: int,
        lora_name: str,
        lora_config: dict,
        bucket_size: int,
        engine_only: bool,
        staging: str,
        evict_first: bool,
    ):
        self.events.append(
            f"sync_lora:{lora_int_id}:{lora_name}:{staging}:evict={evict_first}"
        )
        return {
            "status": "done",
            "weight_format": "lora",
            "lora_int_id": lora_int_id,
            "lora_name": lora_name,
            "lora_sync_staging": staging,
            "lora_evict_first": evict_first,
        }


class _FakeScheduler:
    def __init__(self, events: list[str]):
        self.events = events
        self.drained = False
        self.paused = False

    def pause(self):
        self.paused = True
        self.events.append("scheduler.pause")

    def resume(self):
        self.paused = False
        self.events.append("scheduler.resume")

    def is_worker_available(self, index):
        return True

    async def abort_streams(self):
        self.events.append("scheduler.abort_streams")

    async def drain(self):
        self.events.append("scheduler.drain")
        self.drained = True


def _pool(events: list[str], *, reset_ok: bool = True) -> ReplicaPool:
    pool = ReplicaPool.__new__(ReplicaPool)
    pool._model_id = None
    pool._config = SimpleNamespace(
        tensor_parallel_size=1,
        model="dummy",
        lora_sync_staging="cpu",
    )
    pool._workers = [_FakeWorker(events, reset_ok=reset_ok)]
    pool._scheduler = _FakeScheduler(events)
    pool._lock = asyncio.Lock()
    pool._updating_workers = set()
    pool._synced_lora_name = None
    return pool


def _groups():
    return [{
        "group_id": 0,
        "master_addr": "127.0.0.1",
        "master_port": 12345,
        "world_size": 2,
        "replica_ids": [0],
    }]


def test_sync_weights_rejects_removed_strategies():
    pool = _pool([])
    with pytest.raises(ValueError, match="Unknown strategy"):
        pool._validate_weight_sync_strategy("vllm_keep", "keep")
    with pytest.raises(ValueError, match="Unknown strategy"):
        pool._validate_weight_sync_strategy("freeze", "keep")


def test_pause_keep_sync_soft_fails_prefix_reset():
    async def run():
        events: list[str] = []
        out = await _pool(events, reset_ok=False).sync_weights(
            groups=_groups(),
            strategy="pause",
            pause_mode="keep",
        )
        assert out["strategy"] == "pause"
        assert out["pause_mode"] == "keep"
        assert out["prefix_cache_reset"]["reset_ok"] is False
        assert events == [
            "scheduler.pause",
            "scheduler.abort_streams",
            "pause_generation:keep:False",
            "sync_weights",
            "reset_prefix_cache:0.0",
            "resume_generation",
            "scheduler.resume",
        ]

    asyncio.run(run())


def test_pause_abort_drains_and_hard_fails_prefix_reset():
    async def run():
        events: list[str] = []
        with pytest.raises(RuntimeError, match="Prefix-cache reset failed"):
            await _pool(events, reset_ok=False).sync_weights(
                groups=_groups(),
                strategy="pause",
                pause_mode="abort",
            )
        assert events == [
            "scheduler.pause",
            "scheduler.abort_streams",
            "pause_generation:abort:False",
            "scheduler.drain",
            "sync_weights",
            "reset_prefix_cache:30.0",
        ]

    asyncio.run(run())


def test_lora_pause_keep_reuses_stable_adapter_identity():
    async def run():
        events: list[str] = []
        pool = _pool(events)

        first = await pool.sync_lora_weights_broadcast(
            master_addr="127.0.0.1",
            master_port=12345,
            lora_config={"r": 8},
            strategy="pause",
            pause_mode="keep",
        )
        second = await pool.sync_lora_weights_broadcast(
            master_addr="127.0.0.1",
            master_port=12345,
            lora_config={"r": 8},
            strategy="pause",
            pause_mode="keep",
        )

        assert first["lora_int_id"] == second["lora_int_id"] == 1
        assert first["lora_name"] == second["lora_name"] == "policy"
        assert first["lora_evict_first"] is True
        assert events.count("sync_lora:1:policy:cpu:evict=True") == 2
        assert "scheduler.drain" not in events
        assert events.count("pause_generation:keep:False") == 2
        assert events.count("resume_generation") == 2

    asyncio.run(run())


def test_lora_sync_forwards_gpu_staging():
    async def run():
        events: list[str] = []
        pool = _pool(events)
        pool._config.lora_sync_staging = "gpu"

        result = await pool.sync_lora_weights_broadcast(
            master_addr="127.0.0.1",
            master_port=12345,
            lora_config={"r": 8},
        )

        assert result["lora_sync_staging"] == "gpu"
        assert result["lora_evict_first"] is True
        assert "sync_lora:1:policy:gpu:evict=True" in events

    asyncio.run(run())


def test_lora_hotswap_keeps_resident_adapter_through_receive():
    async def run():
        events: list[str] = []
        result = await _pool(events).sync_lora_weights_broadcast(
            master_addr="127.0.0.1",
            master_port=12345,
            lora_config={"r": 8},
            strategy="hotswap",
        )

        assert result["lora_evict_first"] is False
        assert "sync_lora:1:policy:cpu:evict=False" in events
        assert not any(e.startswith("pause_generation") for e in events)
        assert "resume_generation" not in events
        assert "scheduler.pause" not in events

    asyncio.run(run())
