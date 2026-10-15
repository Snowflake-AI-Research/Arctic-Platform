from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

try:
    from builtins import BaseExceptionGroup
except ImportError:
    from exceptiongroup import BaseExceptionGroup

from arctic_platform.inference.server import api
from arctic_platform.inference.server.config import ModelConfig
from arctic_platform.inference.server.replica_pool import ReplicaPool, ensure_ray
from arctic_platform.inference.server.worker import InferenceWorker

logger = logging.getLogger("arctic_platform.inference.server")


class Driver:
    """Manages multiple models across a GPU cluster.

    Drop-in replacement for :class:`ReplicaPool` as the ``api.backend``.
    Each :meth:`initialize` call loads a model and returns a ``model_id``.
    All subsequent operations route to the correct pool via ``model_id``.
    """

    def __init__(self, worker_cls=InferenceWorker) -> None:
        self._worker_cls = worker_cls
        self._total_gpus: int = 0
        self._ray_initialized = False
        self._pools: dict[str, ReplicaPool] = {}

    # ------------------------------------------------------------------
    # GPU helpers
    # ------------------------------------------------------------------

    def _ensure_ray(self) -> None:
        if self._ray_initialized:
            return
        self._total_gpus = ensure_ray()
        self._ray_initialized = True

    @property
    def _allocated_gpus(self) -> int:
        # GPUs held by a pool = replicas * world_size (TP * PP), since each
        # replica occupies one GPU per rank across all TP ranks and PP stages.
        # Using tp_size alone under-counts PP>1 engines and inflates
        # _available_gpus, which would let the cluster over-subscribe GPUs.
        return sum(
            p.num_replicas * p.world_size
            for p in self._pools.values()
            if not p.uses_placement_group
        )

    @property
    def _available_gpus(self) -> int:
        return self._total_gpus - self._allocated_gpus

    def _compute_even_share(self, engine_gpus: dict[str, int]) -> dict[str, int]:
        """Replicas per model given each engine's GPU footprint.

        ``engine_gpus`` is world_size (TP * PP) per model, so a PP=2 x TP=8
        engine consumes 16 GPUs per replica just like a flat TP=16 one.
        """
        n = len(engine_gpus)
        if n == 0:
            return {}
        gpus_per_model = self._total_gpus // n
        result: dict[str, int] = {}
        for model_id, ws in engine_gpus.items():
            replicas = gpus_per_model // ws
            if replicas == 0:
                raise RuntimeError(
                    f"Not enough GPUs for {model_id!r}: world_size={ws} needs at "
                    f"least {ws} GPUs but only {gpus_per_model} available per "
                    f"model ({self._total_gpus} total / {n} models)"
                )
            result[model_id] = replicas
        return result

    def _rebalance_up(self) -> None:
        engine_gpus = {
            mid: p.world_size
            for mid, p in self._pools.items()
            if not p.uses_placement_group
        }
        if not engine_gpus:
            return
        plan = self._compute_even_share(engine_gpus)
        for mid, p in self._pools.items():
            if p.uses_placement_group:
                continue
            target = plan[mid]
            if target > p.num_replicas:
                logger.info(f"Scaling up {mid!r}: {p.num_replicas} -> {target} replicas (background)")
                # Cancel any earlier in-flight scale before starting a new one,
                # and register the new task on the pool so pool.shutdown() can
                # cancel it cleanly if the model is torn down before scaling
                # finishes.
                if p._scale_task and not p._scale_task.done():
                    p._scale_task.cancel()
                p._scale_task = asyncio.ensure_future(p.scale_up(target))

    def _get_pool(self, model_id: str | None) -> ReplicaPool:
        if model_id is None:
            raise ValueError("model_id is required in multi-model mode")
        try:
            return self._pools[model_id]
        except KeyError:
            raise KeyError(f"Unknown model_id {model_id!r}. Loaded: {list(self._pools)}")

    # ------------------------------------------------------------------
    # Lifecycle  (same interface as ReplicaPool)
    # ------------------------------------------------------------------

    async def initialize(
        self,
        config: ModelConfig,
        model_id: str | None = None,
        num_replicas: int | None = None,
        placement_group: Any | None = None,
    ) -> int:
        self._ensure_ray()

        if model_id is None:
            model_id = uuid.uuid4().hex[:8]
        if model_id in self._pools:
            raise ValueError(f"model_id {model_id!r} already loaded")

        has_placement_pool = any(p.uses_placement_group for p in self._pools.values())
        if placement_group is not None:
            if self._pools:
                raise RuntimeError(
                    "A placement-group-backed model requires exclusive use of "
                    "the Driver; mixed or multiple placement-group pools are unsupported"
                )
            if num_replicas not in (None, 1):
                raise ValueError(
                    "placement-group-backed models require num_replicas=1"
                )
            num_replicas = 1
        elif has_placement_pool:
            raise RuntimeError(
                "Cannot add an ordinary pool to a Driver that already contains "
                "a placement-group-backed model"
            )
        elif num_replicas is None:
            engine_gpus = {mid: p.world_size for mid, p in self._pools.items()}
            engine_gpus[model_id] = (
                config.tensor_parallel_size
                * getattr(config, "pipeline_parallel_size", 1)
            )
            plan = self._compute_even_share(engine_gpus)

            for mid, pool in self._pools.items():
                target = plan[mid]
                if target < pool.num_replicas:
                    logger.info(f"Rebalancing {mid!r}: {pool.num_replicas} -> {target} replicas")
                    await pool.scale_down(target)

            num_replicas = plan[model_id]

        pool = ReplicaPool(worker_cls=self._worker_cls)
        actual_replicas = await pool.initialize(
            config, num_replicas=num_replicas, placement_group=placement_group
        )
        self._pools[model_id] = pool
        return actual_replicas

    async def shutdown(self, model_id: str | None = None) -> None:
        if model_id is not None:
            pool = self._get_pool(model_id)
            try:
                await pool.shutdown()
            finally:
                if self._pools.get(model_id) is pool:
                    del self._pools[model_id]
            if self._pools:
                self._rebalance_up()
        else:
            errors = []
            for pool_id, pool in list(self._pools.items()):
                try:
                    await pool.shutdown()
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    if self._pools.get(pool_id) is pool:
                        del self._pools[pool_id]
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise BaseExceptionGroup("Model pool shutdown failures", errors)

    # ------------------------------------------------------------------
    # Inference  (route by model_id, delegate to pool)
    # ------------------------------------------------------------------

    def stream_generate(self, model_id, request_id, prompt, sampling_params=None, *, limits=None, routing_key=None, strict=False):
        """Return a bounded prepared-generation iterator; close it on early exit."""
        return self._get_pool(model_id).stream_generate(request_id, prompt, sampling_params, limits=limits, routing_key=routing_key, strict=strict)

    async def abort(self, model_id, request_id):
        """Abort one request without unloading the model or affecting other requests."""
        pool = self._pools.get(model_id)
        if pool is None:
            return {"status": "not_found"}
        return await pool.abort(request_id)

    async def generate(
        self,
        prompts: str | list[int] | list[str | list[int]],
        sampling_params: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        model_id: str | None = None,
        routing_key: str | list[str | None] | None = None,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        """Route a /generate to the pool for *model_id*.

        ``routing_key`` and ``strict`` are forwarded to
        :meth:`ReplicaPool.generate` so multi-turn rollouts can pin
        same-keyed prompts to one replica for KV cache reuse. See the
        scheduler module for the full affinity contract.
        """
        return await self._get_pool(model_id).generate(
            prompts, sampling_params,
            routing_key=routing_key, strict=strict,
        )

    def submit_generate_futures(
        self,
        prompts: str | list[int] | list[str | list[int]],
        sampling_params: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        model_id: str | None = None,
        routing_key: str | list[str | None] | None = None,
        strict: bool = False,
    ) -> list[asyncio.Future]:
        return self._get_pool(model_id).submit_generate_futures(
            prompts, sampling_params,
            routing_key=routing_key, strict=strict,
        )

    # ------------------------------------------------------------------
    # Weight sync  (route by model_id, delegate to pool)
    # ------------------------------------------------------------------

    def get_weights_info(self, model_id: str | None = None) -> list[dict]:
        return self._get_pool(model_id).get_weights_info()

    async def sync_weights(
        self,
        groups: list[dict[str, Any]] | None = None,
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "pause",
        pause_mode: str = "keep",
        clear_cache: bool = False,
        engine_only: bool = False,
        direct_mode: bool = False,
        reverse: bool = False,
        model_id: str | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
        world_size: int | None = None,
    ) -> dict[str, Any]:
        return await self._get_pool(model_id).sync_weights(
            groups, bucket_size, strategy=strategy,
            pause_mode=pause_mode, clear_cache=clear_cache,
            engine_only=engine_only, direct_mode=direct_mode,
            reverse=reverse,
            master_addr=master_addr, master_port=master_port,
            world_size=world_size,
        )

    async def sync_weights_broadcast(
        self,
        master_addr: str,
        master_port: int,
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "pause",
        pause_mode: str = "keep",
        clear_cache: bool = False,
        engine_only: bool = False,
        model_id: str | None = None,
        weight_format: str = "vllm",
    ) -> dict[str, Any]:
        return await self._get_pool(model_id).sync_weights_broadcast(
            master_addr=master_addr, master_port=master_port,
            bucket_size=bucket_size, strategy=strategy, engine_only=engine_only,
            pause_mode=pause_mode, clear_cache=clear_cache,
            weight_format=weight_format,
        )

    async def sync_lora_weights_broadcast(
        self,
        master_addr: str,
        master_port: int,
        lora_config: dict[str, Any],
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "pause",
        pause_mode: str = "keep",
        clear_cache: bool = False,
        engine_only: bool = False,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._get_pool(model_id).sync_lora_weights_broadcast(
            master_addr=master_addr, master_port=master_port,
            lora_config=lora_config, bucket_size=bucket_size, strategy=strategy,
            pause_mode=pause_mode, clear_cache=clear_cache, engine_only=engine_only,
        )

    async def close_weight_sync(self, model_id: str | None = None) -> dict[str, Any]:
        return await self._get_pool(model_id).close_weight_sync()

    async def reset_prefix_cache(
        self,
        model_id: str | None = None,
        *,
        drain: bool = True,
        timeout_s: float = 60.0,
        retry_interval_s: float = 0.1,
    ) -> dict[str, Any]:
        return await self._get_pool(model_id).reset_prefix_cache(
            drain=drain,
            timeout_s=timeout_s,
            retry_interval_s=retry_interval_s,
        )

    async def init_router_replay(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        model_id: str | None = None,
        is_server: bool = False,
    ) -> dict[str, Any]:
        return await self._get_pool(model_id).init_router_replay(
            master_addr=master_addr,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            is_server=is_server,
        )

    async def send_router_replay(self, model_id: str | None = None) -> dict[str, Any]:
        return await self._get_pool(model_id).send_router_replay()

    async def discard_router_replay(
        self,
        sample_ids: list[str],
        model_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._get_pool(model_id).discard_router_replay(sample_ids)

    async def close_router_replay(self, model_id: str | None = None) -> dict[str, Any]:
        return await self._get_pool(model_id).close_router_replay()

    # ------------------------------------------------------------------
    # Sleep / Wake
    # ------------------------------------------------------------------

    async def sleep(self, model_id: str, level: int = 1) -> dict[str, Any]:
        """Free GPU memory for *model_id* (drain requests first)."""
        pool = self._get_pool(model_id)
        return await pool.sleep(level=level)

    async def wake_up(
        self, model_id: str, tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Restore GPU memory for *model_id* and resume serving."""
        pool = self._get_pool(model_id)
        return await pool.wake_up(tags=tags)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self) -> dict[str, Any]:
        models: dict[str, Any] = {}
        for mid, pool in self._pools.items():
            models[mid] = await pool.get_status()
        return {
            "total_gpus": self._total_gpus,
            "allocated_gpus": self._allocated_gpus,
            "available_gpus": self._available_gpus,
            "models": models,
        }

    async def get_chat_support(self, model_id: str | None = None) -> dict[str, bool]:
        return await self._get_pool(model_id).get_chat_support()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def drain_metrics(
        self, model_id: str | None = None,
    ) -> dict[str, Any]:
        """Drain metrics for one model (when ``model_id`` is set) or all."""
        if model_id is not None:
            payload = await self._get_pool(model_id).drain_metrics()
            payload["model_id"] = model_id
            return payload
        models: dict[str, Any] = {}
        for mid, pool in self._pools.items():
            models[mid] = await pool.drain_metrics()
        return {"models": models}


# Swap backend and reuse the app — no endpoint duplication.
api.backend = Driver()
app = api.app
