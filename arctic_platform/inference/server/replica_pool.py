from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from arctic_platform.inference.server.config import ModelConfig
from arctic_platform.inference.server.scheduler import (
    RoutingFn,
    Scheduler,
    prefix_affinity_routing,
)
from arctic_platform.inference.server.worker import InferenceWorker

logger = logging.getLogger("arctic_platform.inference.server")

WEIGHT_SYNC_STRATEGIES = {"pause", "drain", "skip", "hotswap"}
WEIGHT_SYNC_PAUSE_MODES = {"keep", "abort"}
_DEFAULT_WORKER_SHUTDOWN_CONCURRENCY = 64
_SYNCED_LORA_ADAPTER_ID = 1


def _worker_result_failed(result: Any) -> bool:
    if isinstance(result, Exception):
        return True
    return isinstance(result, dict) and result.get("status") in {"error", "failed"}


def _failed_worker_indices(results: list[Any]) -> list[int]:
    return [i for i, result in enumerate(results) if _worker_result_failed(result)]


def _worker_error_messages(results: list[Any], indices: list[int]) -> list[str]:
    messages: list[str] = []
    for idx in indices:
        result = results[idx]
        if isinstance(result, Exception):
            messages.append(str(result))
        elif isinstance(result, dict):
            messages.append(str(result.get("message") or result.get("reason") or result))
        else:
            messages.append(str(result))
    return messages


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%r is below %d; using %d", name, raw, minimum, minimum)
        return minimum
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a float; using %.3f", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%r is below %.3f; using %.3f", name, raw, minimum, minimum)
        return minimum
    return value


async def _await_maybe(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def ensure_ray() -> int:
    """Initialize Ray and return the total number of GPUs in the cluster."""
    ray.init(ignore_reinit_error=True, log_to_driver=True)
    nodes = [n for n in ray.nodes() if n["Alive"]]
    if not nodes:
        raise RuntimeError("No alive Ray nodes")
    total = sum(
        int(n["Resources"].get("GPU", torch.cuda.device_count()))
        for n in nodes
    )
    if total == 0:
        raise RuntimeError("No GPUs available in the Ray cluster")
    logger.info(f"{total} GPUs available across {len(nodes)} node(s)")
    return total


class ReplicaPool:
    """Manages a set of worker replicas for a single model.

    Owns the workers and an internal :class:`Scheduler` that handles
    request routing and concurrency control.

    Args:
        worker_cls: Ray actor class for inference workers.
    """

    def __init__(
        self,
        worker_cls=InferenceWorker,
        routing_fn: RoutingFn | None = None,
        enable_prefix_hash: bool | None = None,
    ) -> None:
        self._worker_cls = worker_cls
        _opt_out = os.environ.get(
            "ARCTIC_PREFIX_ROUTING_DISABLED", ""
        ).lower() in ("1", "true", "yes")
        if _opt_out:
            self._routing_fn = None
            self._enable_prefix_hash = False
        elif enable_prefix_hash is None and routing_fn is None:
            self._routing_fn = prefix_affinity_routing
            self._enable_prefix_hash = True
        else:
            self._routing_fn = routing_fn
            self._enable_prefix_hash = bool(enable_prefix_hash)
        self._config: ModelConfig | None = None
        self._model_id: str | None = None
        self._workers: list[ray.actor.ActorHandle] = []
        self._scheduler: Scheduler | None = None
        self._lock = asyncio.Lock()
        self._stream_admission_blocked = False
        self._stop_monitoring = False
        self._health_task: asyncio.Task | None = None
        # Background scale_up task scheduled by Driver._rebalance_up. Tracked
        # so shutdown() can cancel it before worker init finishes — otherwise
        # the task races with shutdown and tries to add_worker on a None
        # scheduler.
        self._scale_task: asyncio.Task | None = None
        self._updating_workers: set[int] = set()
        self._cached_weights_info: list[dict] | None = None
        self._cached_spec_weights_info: list[dict] | None = None
        self._sleeping = False
        self._synced_lora_name: str | None = None
        # Cross-node placement group for a node-spanning engine, built and
        # owned by the caller (dss-platform) and threaded in via initialize().
        # When set, this pool runs the multi-node path: a single 0-GPU
        # coordinator actor is scheduled into this PG (bundle 0) and vLLM's Ray
        # executor inherits it via capture_child_tasks. ``None`` means the
        # single-node path (the pool reserves ``world_size`` GPUs on one actor).
        # The pool does NOT create or remove this PG; lifetime is the caller's.
        self._engine_pg: Any | None = None

    @property
    def config(self) -> ModelConfig:
        if self._config is None:
            raise RuntimeError("ReplicaPool not initialized")
        return self._config

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def tp_size(self) -> int:
        return self.config.tensor_parallel_size

    @property
    def pp_size(self) -> int:
        return getattr(self.config, "pipeline_parallel_size", 1)

    @property
    def world_size(self) -> int:
        """GPUs one engine occupies = TP ranks * PP stages."""
        return self.tp_size * self.pp_size

    @property
    def num_replicas(self) -> int:
        return len(self._workers)

    @property
    def uses_placement_group(self) -> bool:
        return self._engine_pg is not None

    @property
    def _ray_num_gpus(self) -> float:
        if self._config is not None and self._config.ray_num_gpus is not None:
            return self._config.ray_num_gpus
        return float(self.world_size)

    # ------------------------------------------------------------------
    # Multi-node (Approach A): let vLLM own the TP GPUs via a cross-node
    # placement group. The PG is built and owned by the caller (dss-platform)
    # and threaded in via initialize(); this pool only consumes it.
    # ------------------------------------------------------------------

    def _is_multi_node(self) -> bool:
        """True when this engine runs the node-spanning path.

        The caller (dss-platform) decides node-spanning by building a cross-node
        placement group and passing it to :meth:`initialize`. The pool simply
        keys off that PG's presence: when set, it schedules a 0-GPU coordinator
        into the PG; when ``None`` it takes the single-node path.
        """
        return self._engine_pg is not None

    def _make_worker(self) -> ray.actor.ActorHandle:
        """Create one inference-worker actor.

        Single-node: the actor reserves ``tp_size`` GPUs and runs an mp-backed
        vLLM engine (unchanged). Multi-node: the actor is a 0-GPU coordinator
        pinned into a cross-node PG; vLLM inherits that PG via
        ``get_current_placement_group()`` and schedules one rank per bundle.
        """
        if self._is_multi_node():
            options: dict[str, Any] = dict(
                num_gpus=0,
                num_cpus=0,
                max_concurrency=2048,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self._engine_pg,
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=True,
                ),
            )
            # vLLM's Ray executor spawns each rank's worker with the
            # *coordinator's* runtime_env (arg_utils.py reads
            # ``ray.get_runtime_context().runtime_env`` and threads it to every
            # worker). vLLM's own env-copy allowlist excludes NCCL_/DSS_/TORCH_
            # vars, so cross-node NCCL tuning (e.g. NCCL_SOCKET_IFNAME) set only
            # via ``extra_env`` -- which is applied to os.environ *inside* the
            # coordinator, after the actor exists -- would never reach the
            # per-rank workers on other nodes. Seeding the coordinator's Ray
            # runtime_env from extra_env at actor-creation time is what makes it
            # propagate cluster-wide. (Single-node mp workers inherit the
            # coordinator's os.environ directly, so they need no runtime_env.)
            worker_env = {
                k: str(v) for k, v in (self._config.extra_env or {}).items()
            }
            if worker_env:
                options["runtime_env"] = {"env_vars": worker_env}
            return self._worker_cls.options(**options).remote()
        return self._worker_cls.options(
            num_gpus=self._ray_num_gpus,
            max_concurrency=2048,
        ).remote()

    @staticmethod
    def _validate_placement_group(
        config: ModelConfig,
        placement_group: Any,
    ) -> None:
        if getattr(config, "ray_num_gpus", None) is not None:
            raise ValueError(
                "ray_num_gpus cannot be combined with a caller-provided placement group"
            )

        expected = (
            int(config.tensor_parallel_size)
            * int(getattr(config, "pipeline_parallel_size", 1))
        )
        bundle_specs = getattr(placement_group, "bundle_specs", None)
        if callable(bundle_specs):
            bundle_specs = bundle_specs()
        if bundle_specs is None:
            raise ValueError(
                "caller-provided placement group must expose bundle_specs"
            )
        if len(bundle_specs) != expected:
            raise ValueError(
                "caller-provided placement group has "
                f"{len(bundle_specs)} bundles; TP x PP requires exactly {expected}"
            )
        for index, bundle in enumerate(bundle_specs):
            if float(bundle.get("GPU", 0)) < 1:
                raise ValueError(
                    "caller-provided placement group bundle "
                    f"{index} must reserve at least one GPU"
                )

    @staticmethod
    def _engine_kwargs_for(
        config: ModelConfig,
        *,
        multi_node: bool,
    ) -> dict[str, Any]:
        """Build engine kwargs and enforce the multi-node Ray executor."""
        kwargs = config.to_engine_kwargs()
        if multi_node:
            backend = kwargs.get("distributed_executor_backend")
            if backend not in (None, "ray"):
                raise ValueError(
                    "placement-group-backed engines require "
                    "distributed_executor_backend='ray', "
                    f"got {backend!r}"
                )
            kwargs["distributed_executor_backend"] = "ray"
        return kwargs

    def _engine_kwargs(self) -> dict[str, Any]:
        """Engine kwargs with the multi-node Ray executor forced on.

        vLLM only spreads TP across nodes with its Ray distributed executor;
        the default mp executor is single-node.
        """
        return self._engine_kwargs_for(
            self.config,
            multi_node=self._is_multi_node(),
        )

    def _check_model_id(self, model_id: str | None) -> None:
        if model_id is not None and self._model_id is not None and model_id != self._model_id:
            raise ValueError(
                f"model_id mismatch: got {model_id!r}, "
                f"expected {self._model_id!r}"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _worker_concurrency_limit(self) -> int:
        return _env_int("ARCTIC_WORKER_CONCURRENCY_LIMIT", 128)

    @asynccontextmanager
    async def _stream_lifecycle(self):
        async with self._lock:
            self._stream_admission_blocked = True
            try:
                yield
            finally:
                self._stream_admission_blocked = False

    def _make_scheduler(self, workers: list[ray.actor.ActorHandle]) -> Scheduler:
        kwargs: dict[str, Any] = {
            "workers": workers,
            "initial_concurrency": self._worker_concurrency_limit(),
        }
        if self._routing_fn is not None:
            kwargs["routing_fn"] = self._routing_fn
        if self._enable_prefix_hash:
            kwargs["enable_prefix_hash"] = True
        return Scheduler(**kwargs)

    def _reset_lifecycle_state(self) -> None:
        self._workers.clear()
        # The engine PG is owned by the caller (dss-platform); we only drop our
        # reference to it here and never remove it.
        self._engine_pg = None
        self._config = None
        self._model_id = None
        self._cached_weights_info = None
        self._cached_spec_weights_info = None
        self._sleeping = False
        self._synced_lora_name = None

    async def _cleanup_failed_initialize(self) -> None:
        await self._shutdown_workers(list(self._workers))
        self._reset_lifecycle_state()

    async def _shutdown_workers(self, workers: list[ray.actor.ActorHandle]) -> None:
        n = len(workers)
        if n == 0:
            return

        timeout_s = _env_float("ARCTIC_WORKER_SHUTDOWN_TIMEOUT_S", 30.0)
        concurrency = min(
            n,
            _env_int(
                "ARCTIC_WORKER_SHUTDOWN_CONCURRENCY",
                min(n, _DEFAULT_WORKER_SHUTDOWN_CONCURRENCY),
            ),
        )
        logger.info(
            "Shutting down %d inference workers with concurrency=%d timeout=%.3fs",
            n,
            concurrency,
            timeout_s,
        )
        semaphore = asyncio.Semaphore(concurrency)
        started: set[int] = set()

        async def _shutdown_one(idx: int, worker: ray.actor.ActorHandle) -> Any:
            async with semaphore:
                started.add(idx)
                try:
                    return await _await_maybe(worker.shutdown.remote())
                except Exception as exc:
                    logger.warning("Worker %d shutdown failed: %r", idx, exc)
                    return exc

        tasks = [
            asyncio.create_task(_shutdown_one(idx, worker))
            for idx, worker in enumerate(workers)
        ]
        pending: set[asyncio.Task] = set(tasks)
        try:
            _done, pending = await asyncio.wait(tasks, timeout=timeout_s)
            if pending:
                started_pending_indices = [
                    idx for idx, task in enumerate(tasks)
                    if task in pending and idx in started
                ]
                not_started_indices = [
                    idx for idx, task in enumerate(tasks)
                    if task in pending and idx not in started
                ]
                logger.warning(
                    "Timed out after %.3fs shutting down inference workers; "
                    "shutdown RPCs still running for workers %s; "
                    "shutdown RPCs not yet started for workers %s",
                    timeout_s,
                    started_pending_indices,
                    not_started_indices,
                )

            results: list[Any] = [None] * n
            for idx, task in enumerate(tasks):
                if task in pending:
                    continue
                if task.cancelled():
                    results[idx] = RuntimeError("worker shutdown cancelled")
                    continue
                try:
                    results[idx] = task.result()
                except Exception as exc:
                    results[idx] = exc
            failures = _failed_worker_indices(results)
            if failures:
                logger.warning(
                    "Worker shutdown completed with failures on workers %s: %s",
                    failures,
                    _worker_error_messages(results, failures),
                )
        finally:
            for task in pending:
                task.cancel()
            for idx, worker in enumerate(workers):
                try:
                    ray.kill(worker)
                except Exception:
                    logger.debug("ray.kill failed for worker %d", idx, exc_info=True)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _initialize_workers(
        self,
        engine_kwargs: dict[str, Any],
        extra_env: dict[str, str] | None,
    ) -> None:
        n = len(self._workers)
        default_concurrency = n
        concurrency = min(n, _env_int("ARCTIC_WORKER_INIT_CONCURRENCY", default_concurrency))
        stagger_s = _env_float("ARCTIC_WORKER_INIT_STAGGER_S", 0.0)
        if concurrency < n or stagger_s > 0:
            logger.info(
                "Initializing %d workers with startup concurrency=%d stagger=%.3fs",
                n,
                concurrency,
                stagger_s,
            )

        port_base = _env_int("ARCTIC_VLLM_PORT_BASE", 8000, minimum=1)
        port_stride = _env_int("ARCTIC_VLLM_PORT_STRIDE", 100, minimum=1)
        # The per-replica VLLM_PORT pin only exists to keep multiple single-node
        # replicas on one host from colliding. A node-spanning engine has a
        # single replica whose vLLM Ray executor derives its torch.distributed
        # rendezvous port from VLLM_PORT; pinning it collides with the
        # co-located EngineCore (EADDRINUSE). Let vLLM pick free ports instead.
        pin_vllm_port = not self._is_multi_node()

        for start in range(0, n, concurrency):
            batch = self._workers[start : start + concurrency]
            refs = []
            for offset, worker in enumerate(batch):
                worker_idx = start + offset
                logger.info("Initializing worker %d/%d", worker_idx + 1, n)
                worker_env = dict(extra_env or {})
                if pin_vllm_port:
                    worker_env["VLLM_PORT"] = str(port_base + worker_idx * port_stride)
                refs.append(worker.initialize.remote(engine_kwargs, worker_env, self._model_id))
                if stagger_s > 0 and offset + 1 < len(batch):
                    await asyncio.sleep(stagger_s)
            await asyncio.gather(*refs)

    async def initialize(
        self,
        config: ModelConfig,
        model_id: str | None = None,
        num_replicas: int | None = None,
        placement_group: Any | None = None,
    ) -> int:
        """Set configuration, create worker actors, start scheduler.

        Args:
            config: Model configuration (defines TP size, model name, etc.).
            model_id: Ignored in single-model mode.
            num_replicas: Number of replicas. If ``None``, uses all
                available GPUs (``total_gpus // world_size`` where
                world_size = tensor_parallel_size * pipeline_parallel_size).
                Ignored when ``placement_group`` is given (a node-spanning
                engine is a single replica bound to that one PG).
            placement_group: Optional pre-built cross-node Ray placement group
                from the caller (dss-platform). When given, this pool runs the
                node-spanning path: one 0-GPU coordinator actor is scheduled
                into the PG (bundle 0) and vLLM's Ray executor inherits it via
                capture_child_tasks. The pool does not create or remove the PG.
                ``None`` keeps the single-node path (the pool reserves
                ``world_size`` GPUs on one actor).

        Returns the number of workers created.
        """
        if self._config is not None:
            raise RuntimeError("Already initialized. Call shutdown() first.")

        if placement_group is not None:
            if num_replicas not in (None, 1):
                raise ValueError(
                    "placement-group-backed pools require num_replicas=1"
                )
            self._validate_placement_group(config, placement_group)
        engine_kwargs = self._engine_kwargs_for(
            config,
            multi_node=placement_group is not None,
        )

        self._config = config
        self._model_id = model_id
        self._engine_pg = placement_group

        if self._engine_pg is not None:
            # A node-spanning engine occupies its whole PG as a single replica;
            # the caller sized the PG to exactly this engine's world_size.
            num_replicas = 1
        elif num_replicas is None:
            total_gpus = ensure_ray()
            num_replicas = total_gpus // self.world_size
            if num_replicas == 0:
                raise RuntimeError(
                    f"Not enough GPUs: TP={self.tp_size} x PP={self.pp_size} "
                    f"needs at least {self.world_size} GPUs but only "
                    f"{total_gpus} available"
                )

        n = num_replicas
        multi_node = self._is_multi_node()
        logger.info(
            f"Creating {n} workers (TP={self.tp_size}, PP={self.pp_size}, "
            f"world_size={self.world_size}, "
            f"{'multi-node: 0-GPU coordinator + cross-node PG' if multi_node else 'single-node'})"
        )
        logger.info(
            "clear_cache_on_weight_sync=%s (applies to strategy='pause', "
            "pause_mode='keep' weight syncs)",
            self._config.clear_cache_on_weight_sync,
        )

        extra_env = self._config.extra_env or None
        try:
            self._workers = []
            for _ in range(n):
                self._workers.append(self._make_worker())
            await self._initialize_workers(engine_kwargs, extra_env)
        except asyncio.CancelledError:
            logger.info(
                "ReplicaPool initialization cancelled; cleaning up %d workers",
                len(self._workers),
            )
            await self._cleanup_failed_initialize()
            raise
        except BaseException:
            logger.exception(
                "ReplicaPool worker initialization failed; cleaning up %d workers",
                len(self._workers),
            )
            await self._cleanup_failed_initialize()
            raise

        self._scheduler = self._make_scheduler(self._workers)

        self._stop_monitoring = False
        self._health_task = asyncio.create_task(self._monitor_health())

        logger.info(f"ReplicaPool ready: {n} workers")
        return n

    async def shutdown(self, model_id: str | None = None) -> None:
        self._check_model_id(model_id)
        self._stop_monitoring = True

        # Cancel any background scale_up before tearing down the scheduler;
        # otherwise the in-flight scale_up will finish loading a worker and
        # then try to add_worker on a None scheduler.
        if self._scale_task and not self._scale_task.done():
            self._scale_task.cancel()
            try:
                await self._scale_task
            except (asyncio.CancelledError, Exception):
                pass
        self._scale_task = None

        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        try:
            if self._scheduler:
                await self._scheduler.shutdown()
        finally:
            self._scheduler = None
            try:
                await self._shutdown_workers(list(self._workers))
            finally:
                self._reset_lifecycle_state()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def stream_generate(self, request_id, prompt, sampling_params=None, model_id=None, *, limits=None, routing_key=None, strict=False):
        self._check_model_id(model_id)
        if self._scheduler is None or self._stop_monitoring or self._sleeping or self._stream_admission_blocked:
            raise RuntimeError("ReplicaPool is not available for streaming")
        return self._scheduler.stream_generate(request_id, prompt, sampling_params, limits=limits, routing_key=routing_key, strict=strict)

    async def abort(self, request_id, model_id=None):
        self._check_model_id(model_id)
        if self._scheduler is None:
            return {"status": "not_found"}
        return await self._scheduler.abort(request_id)

    async def generate(
        self,
        prompts: str | list[int] | list[str | list[int]],
        sampling_params: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        model_id: str | None = None,
        routing_key: str | list[str | None] | None = None,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        """Generate completions for one or more prompts.

        Args:
            prompts: A single prompt (``str`` or token-id ``list[int]``) or a
                batch of prompts (``list[str | list[int]]``).
            sampling_params: vLLM sampling parameters (dict or SamplingParams).
            model_id: Ignored in single-model mode.
            routing_key: Optional per-batch affinity key. A single string
                applies to every prompt in the batch; a list must match the
                batch size element-wise. Hashed by the scheduler and used in
                place of the prompt hash to pin same-keyed requests to the
                same worker (e.g. all turns of one multi-turn rollout).
            strict: When True, enforce hard pinning to the keyed worker even
                under load, instead of ringing to the next worker on overload.
        """
        self._check_model_id(model_id)
        futures = self.submit_generate_futures(
            prompts, sampling_params,
            routing_key=routing_key, strict=strict,
        )
        return list(await asyncio.gather(*futures))

    def submit_generate_futures(
        self,
        prompts: str | list[int] | list[str | list[int]],
        sampling_params: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        routing_key: str | list[str | None] | None = None,
        strict: bool = False,
    ) -> list[asyncio.Future]:
        """Submit generation requests and return one future per prompt."""
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")
        if self._sleeping:
            raise RuntimeError("Model is sleeping; call /wake_up first")
        params = sampling_params or {}

        def _single_key() -> str | None:
            if routing_key is None or isinstance(routing_key, str):
                return routing_key
            if len(routing_key) != 1:
                raise ValueError(
                    f"Got {len(routing_key)} routing keys for a single prompt"
                )
            return routing_key[0]

        def _single_params() -> dict[str, Any]:
            if not isinstance(params, list):
                return params
            if len(params) != 1:
                raise ValueError(
                    f"Got {len(params)} sampling params for a single prompt"
                )
            return params[0] or {}

        if isinstance(prompts, str):
            return [self._scheduler.submit(
                prompts, _single_params(), routing_key=_single_key(), strict=strict,
            )]
        if prompts and isinstance(prompts[0], int):
            return [self._scheduler.submit(
                prompts, _single_params(), routing_key=_single_key(), strict=strict,
            )]
        if isinstance(routing_key, list):
            if len(routing_key) != len(prompts):
                raise ValueError(
                    f"routing_key list length ({len(routing_key)}) must match "
                    f"prompts length ({len(prompts)})"
                )
            keys: list[str | None] = list(routing_key)
        else:
            keys = [routing_key] * len(prompts)
        if isinstance(params, list):
            if len(params) != len(prompts):
                raise ValueError(
                    f"sampling_params list length ({len(params)}) must match "
                    f"prompts length ({len(prompts)})"
                )
            params_list = [p or {} for p in params]
        else:
            params_list = [params] * len(prompts)
        return [
            self._scheduler.submit(prompt, param, routing_key=key, strict=strict)
            for prompt, param, key in zip(prompts, params_list, keys)
        ]

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    async def scale_down(self, target_count: int) -> None:
        """Remove workers from the end until *target_count* remain."""
        if target_count < 0:
            raise ValueError(f"target_count must be >= 0, got {target_count}")
        if self._is_multi_node() and target_count != 1:
            raise ValueError(
                "placement-group-backed pools must contain exactly one replica"
            )
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        async with self._stream_lifecycle():
            while len(self._workers) > target_count:
                idx = len(self._workers) - 1
                self._scheduler.mark_worker_unavailable(idx)
                try:
                    await self._scheduler.abort_streams(worker_idx=idx)
                    await self._scheduler.drain_worker(idx)
                finally:
                    worker = self._workers.pop()
                    try:
                        ray.kill(worker)
                    except Exception:
                        logger.warning("Failed to kill removed worker %d", idx, exc_info=True)
                    finally:
                        self._scheduler.remove_last_worker()
                    logger.info(f"Scaled down: removed worker {idx}")

    async def scale_up(self, target_count: int) -> None:
        """Add workers until *target_count* are running.

        Safe to cancel mid-flight: any half-built actor is killed and the
        pool's worker list is left consistent.
        """
        if self._is_multi_node() and target_count != 1:
            raise ValueError(
                "placement-group-backed pools must contain exactly one replica"
            )
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        async with self._lock:
            engine_kwargs = self._engine_kwargs()
            extra_env = self._config.extra_env or None

            while len(self._workers) < target_count:
                worker = self._make_worker()
                try:
                    await worker.initialize.remote(engine_kwargs, extra_env, self._model_id)
                except asyncio.CancelledError:
                    try:
                        ray.kill(worker)
                    except Exception:
                        pass
                    raise
                except Exception:
                    try:
                        ray.kill(worker)
                    except Exception:
                        pass
                    raise

                # The pool may have been torn down while we awaited the
                # (slow) worker init. Don't touch the now-None scheduler;
                # discard the freshly-built worker.
                if self._scheduler is None:
                    try:
                        ray.kill(worker)
                    except Exception:
                        pass
                    logger.info(
                        "scale_up aborted: pool was shut down during worker init"
                    )
                    return

                self._workers.append(worker)
                self._scheduler.add_worker(
                    worker,
                    concurrency_limit=self._worker_concurrency_limit(),
                )
                logger.info(f"Scaled up: added worker {len(self._workers) - 1}")

    # ------------------------------------------------------------------
    # Sleep / Wake
    # ------------------------------------------------------------------

    @property
    def sleeping(self) -> bool:
        return self._sleeping

    async def sleep(self, model_id: str | None = None, level: int = 1) -> dict[str, Any]:
        """Drain requests, then free GPU memory on every worker."""
        self._check_model_id(model_id)
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")
        if self._sleeping:
            return {"status": "already_sleeping", "level": level}

        async with self._stream_lifecycle():
            scheduler_was_paused = self._scheduler.paused
            self._scheduler.pause()
            try:
                await self._scheduler.abort_streams()
                await self._scheduler.drain()
            except BaseException:
                if not scheduler_was_paused:
                    self._scheduler.resume()
                raise

            results = await asyncio.gather(
                *[w.sleep.remote(level=level) for w in self._workers],
                return_exceptions=True,
            )
            self._sleeping = True

            per_worker = [
                {"status": "error", "message": str(r)} if isinstance(r, Exception) else r
                for r in results
            ]
            logger.info("ReplicaPool sleeping (%d workers, level=%d)", len(self._workers), level)
            return {"status": "sleeping", "level": level, "workers": per_worker}

    async def wake_up(self, model_id: str | None = None, tags: list[str] | None = None) -> dict[str, Any]:
        """Restore GPU memory on every worker, then resume scheduling."""
        self._check_model_id(model_id)
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")
        if not self._sleeping:
            return {"status": "already_ready"}

        async with self._stream_lifecycle():
            results = await asyncio.gather(
                *[w.wake_up.remote(tags=tags) for w in self._workers],
                return_exceptions=True,
            )
            self._sleeping = False
            self._scheduler.resume()

            per_worker = [
                {"status": "error", "message": str(r)} if isinstance(r, Exception) else r
                for r in results
            ]
            logger.info("ReplicaPool awake (%d workers)", len(self._workers))
            return {"status": "ready", "workers": per_worker}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self) -> dict[str, Any]:
        if self._config is None:
            return {"status": "not_initialized"}
        states = await asyncio.gather(*[w.get_state.remote() for w in self._workers])
        return {
            "model": self._config.model,
            "num_replicas": len(self._workers),
            "sleeping": self._sleeping,
            "replica_states": list(states),
        }

    async def get_chat_support(self, model_id: str | None = None) -> dict[str, bool]:
        """Report ``{"chat_prompt": bool, "thinking_optional": bool}`` for the loaded model.

        ``chat_prompt``: streams take a ``ChatPrompt``. ``thinking_optional``:
        ``reasoning_effort="none"`` turns thinking off.
        """
        self._check_model_id(model_id)
        if not self._workers:
            raise RuntimeError("ReplicaPool not initialized")
        # Every replica loads the same model with the same engine kwargs.
        return await self._workers[0].get_chat_support.remote()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def drain_metrics(self, model_id: str | None = None) -> dict[str, Any]:
        """Drain per-replica snapshots and per-request records.

        See :meth:`Scheduler.drain_metrics` for the payload shape; the
        ``model`` field is added at this level so consumers tagging metrics
        by ``model_id`` don't have to look it up separately.
        """
        self._check_model_id(model_id)
        if self._scheduler is None:
            return {
                "model": self._config.model if self._config else None,
                "drained_at": time.time(),
                "requests": [],
                "replicas": [],
            }
        payload = await self._scheduler.drain_metrics()
        payload["model"] = self._config.model if self._config else None
        payload["drained_at"] = time.time()
        return payload

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def get_weights_info(self, model_id: str | None = None) -> list[dict]:
        self._check_model_id(model_id)
        if self._config is None:
            raise RuntimeError("ReplicaPool not initialized")
        if self._cached_weights_info is None:
            from arctic_platform.inference.server.weight_sync import build_weights_info
            infos = build_weights_info(self._config.model)
            self._cached_weights_info = [wi.to_dict() for wi in infos]
        return self._cached_weights_info

    def _validate_weight_sync_strategy(self, strategy: str, pause_mode: str) -> None:
        if strategy not in WEIGHT_SYNC_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. Use: pause, drain, skip, hotswap"
            )
        if pause_mode not in WEIGHT_SYNC_PAUSE_MODES:
            raise ValueError(f"Unknown pause_mode: {pause_mode!r}. Use: keep, abort")

    async def _pause_generation_for_weight_sync(
        self,
        pause_mode: str,
        clear_cache: bool,
    ) -> bool:
        """Pause generation on every worker; return the clear_cache value sent."""
        clear_cache = clear_cache or (
            self.config.clear_cache_on_weight_sync and pause_mode == "keep"
        )
        pause_results = await asyncio.gather(
            *[
                worker.pause_generation.remote(
                    mode=pause_mode,
                    clear_cache=clear_cache,
                )
                for worker in self._workers
            ],
            return_exceptions=True,
        )
        failed_workers = _failed_worker_indices(pause_results)
        if failed_workers:
            await asyncio.gather(
                *[worker.resume_generation.remote() for worker in self._workers],
                return_exceptions=True,
            )
            raise RuntimeError(
                f"Failed to pause inference workers {failed_workers}: "
                f"{_worker_error_messages(pause_results, failed_workers)}"
            )
        return clear_cache

    async def _resume_generation_after_weight_sync(self) -> None:
        resume_results = await asyncio.gather(
            *[worker.resume_generation.remote() for worker in self._workers],
            return_exceptions=True,
        )
        failed_workers = _failed_worker_indices(resume_results)
        if failed_workers:
            raise RuntimeError(
                f"Failed to resume inference workers {failed_workers}: "
                f"{_worker_error_messages(resume_results, failed_workers)}"
            )

    async def _reset_prefix_cache_after_weight_sync(
        self,
        *,
        require_success: bool,
    ) -> dict[str, Any]:
        timeout_s = (
            _env_float("ARCTIC_INFERENCE_WEIGHT_SYNC_PREFIX_RESET_TIMEOUT_S", 30.0)
            if require_success
            else 0.0
        )
        retry_interval_s = _env_float(
            "ARCTIC_INFERENCE_WEIGHT_SYNC_PREFIX_RESET_RETRY_INTERVAL_S",
            0.1,
            minimum=0.01,
        )
        results = await asyncio.gather(
            *[
                worker.reset_prefix_cache.remote(timeout_s, retry_interval_s)
                for worker in self._workers
            ],
            return_exceptions=True,
        )
        per_worker = [
            {"status": "error", "message": str(result)}
            if isinstance(result, Exception)
            else result
            for result in results
        ]
        reset_ok = all(
            isinstance(result, dict) and bool(result.get("reset_ok"))
            for result in per_worker
        )
        payload = {"reset_ok": reset_ok, "workers": per_worker}
        if require_success and not reset_ok:
            raise RuntimeError(
                "Prefix-cache reset failed after weight sync; refusing to resume "
                "generation with stale prefix blocks."
            )
        if not reset_ok:
            logger.warning(
                "Prefix-cache reset did not fully complete after pause-mode weight sync; "
                "continuing because keep mode can retain running request KV blocks. "
                "reset_result=%s",
                payload,
            )
        return payload

    async def reset_prefix_cache(
        self,
        model_id: str | None = None,
        *,
        drain: bool = True,
        timeout_s: float = 60.0,
        retry_interval_s: float = 0.1,
    ) -> dict[str, Any]:
        self._check_model_id(model_id)
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")
        async with self._stream_lifecycle():
            if drain:
                self._scheduler.pause()
                await self._scheduler.drain()
            try:
                results = await asyncio.gather(
                    *[
                        worker.reset_prefix_cache.remote(timeout_s, retry_interval_s)
                        for worker in self._workers
                    ],
                    return_exceptions=True,
                )
            finally:
                if drain:
                    self._scheduler.resume()
            per_worker = [
                {"status": "error", "message": str(result)}
                if isinstance(result, Exception)
                else result
                for result in results
            ]
            return {
                "status": "ok" if all(
                    isinstance(result, dict) and bool(result.get("reset_ok"))
                    for result in per_worker
                ) else "failed",
                "reset_ok": all(
                    isinstance(result, dict) and bool(result.get("reset_ok"))
                    for result in per_worker
                ),
                "workers": per_worker,
            }

    async def init_router_replay(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        is_server: bool = False,
    ) -> dict[str, Any]:
        if self._workers is None:
            raise RuntimeError("ReplicaPool not initialized")
        results = await asyncio.gather(
            *[
                worker.init_router_replay.remote(
                    master_addr=master_addr,
                    master_port=master_port,
                    rank=rank_offset + idx,
                    world_size=world_size,
                    is_server=is_server and idx == 0,
                )
                for idx, worker in enumerate(self._workers)
            ],
            return_exceptions=True,
        )
        return {
            "n_replicas": len(self._workers),
            "workers": [
                {"status": "error", "message": str(result)}
                if isinstance(result, Exception)
                else result
                for result in results
            ],
        }

    async def send_router_replay(self) -> dict[str, Any]:
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        async with self._lock:
            scheduler = self._scheduler
            if scheduler is None:
                raise RuntimeError("ReplicaPool not initialized")

            scheduler_was_paused = scheduler.paused
            scheduler.pause()
            replay_send: asyncio.Future | None = None
            try:
                await scheduler.drain()
                replay_send = asyncio.gather(
                    *[
                        worker.send_router_replay.remote()
                        for worker in self._workers
                    ],
                    return_exceptions=True,
                )
                try:
                    results = await asyncio.shield(replay_send)
                except asyncio.CancelledError:
                    # Ray actor work continues after the local await is
                    # cancelled. Keep the lifecycle lock and scheduler pause
                    # until every started collective has reached a known state.
                    while not replay_send.done():
                        try:
                            await asyncio.shield(replay_send)
                        except asyncio.CancelledError:
                            continue
                    raise
            finally:
                if (
                    not scheduler_was_paused
                    and (replay_send is None or replay_send.done())
                ):
                    scheduler.resume()

        return {
            "n_replicas": len(self._workers),
            "workers": [
                {"status": "error", "message": str(result)}
                if isinstance(result, Exception)
                else result
                for result in results
            ],
        }

    async def discard_router_replay(self, sample_ids: list[str]) -> dict[str, Any]:
        if self._workers is None:
            raise RuntimeError("ReplicaPool not initialized")
        results = await asyncio.gather(
            *[
                worker.discard_router_replay.remote(sample_ids)
                for worker in self._workers
            ],
            return_exceptions=True,
        )
        removed = 0
        workers = []
        for result in results:
            if isinstance(result, Exception):
                workers.append({"status": "error", "message": str(result)})
            else:
                workers.append(result)
                removed += int(result.get("removed", 0))
        return {"requested": len(sample_ids), "removed": removed, "workers": workers}

    async def close_router_replay(self) -> dict[str, Any]:
        if self._workers is None:
            return {"status": "no_workers"}
        results = await asyncio.gather(
            *[worker.close_router_replay.remote() for worker in self._workers],
            return_exceptions=True,
        )
        return {
            "n_replicas": len(self._workers),
            "workers": [
                {"status": "error", "message": str(result)}
                if isinstance(result, Exception)
                else result
                for result in results
            ],
        }

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
        """Receive weights from sender(s) and load into all replicas.

        Accepts either ``groups`` or legacy flat fields (``master_addr``,
        ``master_port``, ``world_size``).  The *strategy* controls how
        in-flight requests are handled:

          - **pause**: pause scheduler admission, pause vLLM generation,
            sync, reset prefix cache, resume.
          - **drain**: pause scheduler, wait for in-flight to finish, sync, resume.
          - **skip**: mark workers unavailable, cancel in-flight, sync, re-enable.
          - **hotswap**: sync while serving continues. Unsafe during active generation.

        With **pause** and ``pause_mode="keep"``, the pause also clears the
        prefix cache when ``clear_cache`` is true or the run's
        ``ModelConfig.clear_cache_on_weight_sync`` is true (the default). A
        request can force clearing on but not off; a run opts out by setting
        ``clear_cache_on_weight_sync: false`` in its ``vllm_config``. Other
        pause modes use ``clear_cache`` as given, and the other strategies
        issue no pause. The response's ``clear_cache`` is the value sent to
        the workers' pause, or ``False`` when no pause was issued.
        """
        self._validate_weight_sync_strategy(strategy, pause_mode)
        self._check_model_id(model_id)
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        if groups is None:
            if master_addr is None or master_port is None:
                raise ValueError(
                    "Provide either 'groups' or legacy flat fields "
                    "(master_addr, master_port, world_size)"
                )
            n = self.num_replicas
            tp = self.tp_size
            groups = [{
                "group_id": 0,
                "master_addr": master_addr,
                "master_port": master_port,
                "world_size": world_size or (1 + n * tp),
                "replica_ids": list(range(n)),
            }]

        effective_strategy = "hotswap" if engine_only else strategy

        async with self._stream_lifecycle():
            t0 = time.time()
            n = len(self._workers)
            scheduler_was_paused = self._scheduler.paused
            available_workers = [
                index for index in range(n)
                if self._scheduler.is_worker_available(index)
            ]
            scheduler_paused = False
            skipped_workers = False
            generation_paused = False
            pause_clear_cache = False
            weight_update_started = False
            update_succeeded = False
            prefix_cache_reset: dict[str, Any] | None = None
            results: list[Any] = []

            try:
                if effective_strategy in {"pause", "drain"}:
                    self._scheduler.pause()
                    scheduler_paused = True
                elif effective_strategy == "skip":
                    skipped_workers = True
                    for i in range(n):
                        self._scheduler.mark_worker_unavailable(i)
                await self._scheduler.abort_streams()
                if effective_strategy == "pause":
                    pause_clear_cache = await self._pause_generation_for_weight_sync(
                        pause_mode, clear_cache
                    )
                    generation_paused = True
                    if pause_mode == "abort":
                        await self._scheduler.drain()
                elif effective_strategy == "drain":
                    await self._scheduler.drain()
                elif effective_strategy == "skip":
                    for i in range(n):
                        self._scheduler.cancel_worker_inflight(i)

                tp = self.tp_size
                replica_to_group: dict[int, dict[str, Any]] = {}
                for g in groups:
                    base_port = g["master_port"]
                    for rid in g["replica_ids"]:
                        replica_to_group[rid] = {
                            "master_addr": g["master_addr"],
                            "master_port": base_port + rid * tp,
                            "world_size": 2,
                            "rank_offset": 1,
                        }

                self._updating_workers = set(range(n))

                tasks = []
                for i, worker in enumerate(self._workers):
                    gcfg = replica_to_group.get(i)
                    if gcfg is None:
                        raise RuntimeError(
                            f"Replica {i} not assigned to any group. "
                            f"Groups cover replicas: {[g['replica_ids'] for g in groups]}"
                        )
                    weight_update_started = True
                    tasks.append(
                        worker.sync_weights.remote(
                            gcfg["master_addr"], gcfg["master_port"],
                            gcfg["rank_offset"], gcfg["world_size"],
                            bucket_size, engine_only, direct_mode, reverse,
                        )
                    )

                results = await asyncio.gather(*tasks, return_exceptions=True)
                failed_workers = _failed_worker_indices(results)
                if failed_workers:
                    raise RuntimeError(
                        f"Weight sync failed on workers {failed_workers}: "
                        f"{_worker_error_messages(results, failed_workers)}"
                    )
                if effective_strategy == "pause":
                    prefix_cache_reset = await self._reset_prefix_cache_after_weight_sync(
                        require_success=pause_mode == "abort",
                    )
                update_succeeded = True
            finally:
                self._updating_workers.clear()
                can_restore_scheduling = update_succeeded or not weight_update_started
                if effective_strategy == "pause":
                    if generation_paused and update_succeeded:
                        await self._resume_generation_after_weight_sync()
                    if scheduler_paused and (update_succeeded or not generation_paused):
                        if not scheduler_was_paused:
                            self._scheduler.resume()
                    elif scheduler_paused:
                        logger.error(
                            "Weight sync failed after vLLM generation was paused; "
                            "leaving scheduler and workers paused for operator recovery."
                        )
                elif scheduler_paused and can_restore_scheduling and not scheduler_was_paused:
                    self._scheduler.resume()
                if skipped_workers and can_restore_scheduling:
                    for index in available_workers:
                        self._scheduler.mark_worker_available(index)


            per_worker = [
                {"status": "error", "message": str(r)} if isinstance(r, Exception) else r
                for r in results
            ]
            elapsed = time.time() - t0
            logger.info(f"Weight sync: {n} workers, {len(groups)} group(s) in {elapsed:.2f}s")
            response = {
                "elapsed": elapsed,
                "num_groups": len(groups),
                "strategy": strategy,
                "pause_mode": pause_mode,
                "clear_cache": pause_clear_cache,
                "strategy_elapsed": elapsed,
                "workers": per_worker,
            }
            if prefix_cache_reset is not None:
                response["prefix_cache_reset"] = prefix_cache_reset
            return response

    async def _run_broadcast_weight_sync(
        self,
        *,
        build_tasks,
        strategy: str,
        pause_mode: str,
        clear_cache: bool,
        engine_only: bool,
        model_id: str | None,
        fail_label: str,
        log_extra: str = "",
        extra_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Shared pause/resume lifecycle for broadcast weight syncs.

        Only the broadcast payload differs between full-weight and LoRA syncs:
        ``build_tasks(n, tp, world_size)`` returns the per-worker ``.remote(...)``
        handles. ``extra_response`` is merged into the returned dict.
        """
        self._validate_weight_sync_strategy(strategy, pause_mode)
        self._check_model_id(model_id)
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        effective_strategy = "hotswap" if engine_only else strategy

        async with self._stream_lifecycle():
            t0 = time.time()
            n = len(self._workers)
            tp = self.tp_size
            world_size = 1 + n * tp
            scheduler_was_paused = self._scheduler.paused
            available_workers = [
                index for index in range(n)
                if self._scheduler.is_worker_available(index)
            ]
            scheduler_paused = False
            skipped_workers = False
            generation_paused = False
            pause_clear_cache = False
            weight_update_started = False
            update_succeeded = False
            prefix_cache_reset: dict[str, Any] | None = None
            results: list[Any] = []

            try:
                if effective_strategy in {"pause", "drain"}:
                    self._scheduler.pause()
                    scheduler_paused = True
                elif effective_strategy == "skip":
                    skipped_workers = True
                    for i in range(n):
                        self._scheduler.mark_worker_unavailable(i)
                await self._scheduler.abort_streams()
                if effective_strategy == "pause":
                    pause_clear_cache = await self._pause_generation_for_weight_sync(
                        pause_mode, clear_cache
                    )
                    generation_paused = True
                    if pause_mode == "abort":
                        await self._scheduler.drain()
                elif effective_strategy == "drain":
                    await self._scheduler.drain()
                elif effective_strategy == "skip":
                    for i in range(n):
                        self._scheduler.cancel_worker_inflight(i)

                self._updating_workers = set(range(n))
                weight_update_started = True
                results = await asyncio.gather(
                    *build_tasks(n, tp, world_size), return_exceptions=True
                )
                failed_workers = _failed_worker_indices(results)
                if failed_workers:
                    raise RuntimeError(
                        f"{fail_label} failed on workers {failed_workers}: "
                        f"{_worker_error_messages(results, failed_workers)}"
                    )
                if effective_strategy == "pause":
                    prefix_cache_reset = await self._reset_prefix_cache_after_weight_sync(
                        require_success=pause_mode == "abort",
                    )
                update_succeeded = True
            finally:
                self._updating_workers.clear()
                can_restore_scheduling = update_succeeded or not weight_update_started
                if effective_strategy == "pause":
                    if generation_paused and update_succeeded:
                        await self._resume_generation_after_weight_sync()
                    if scheduler_paused and (update_succeeded or not generation_paused):
                        if not scheduler_was_paused:
                            self._scheduler.resume()
                    elif scheduler_paused:
                        logger.error(
                            f"{fail_label} failed after vLLM generation was paused; "
                            "leaving scheduler and workers paused for operator recovery."
                        )
                elif scheduler_paused and can_restore_scheduling and not scheduler_was_paused:
                    self._scheduler.resume()
                if skipped_workers and can_restore_scheduling:
                    for index in available_workers:
                        self._scheduler.mark_worker_available(index)

            per_worker = [
                {"status": "error", "message": str(r)} if isinstance(r, Exception) else r
                for r in results
            ]
            elapsed = time.time() - t0
            logger.info(
                "%s: %d replicas (world_size=%d)%s in %.2fs",
                fail_label, n, world_size, log_extra, elapsed,
            )
            response = {
                "elapsed": elapsed,
                "world_size": world_size,
                "strategy": strategy,
                "pause_mode": pause_mode,
                "clear_cache": pause_clear_cache,
                "workers": per_worker,
            }
            if extra_response:
                response.update(extra_response)
            if prefix_cache_reset is not None:
                response["prefix_cache_reset"] = prefix_cache_reset
            return response

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
        """Join one NCCL broadcast group (rank 0 = trainer) and load weights into all replicas."""
        def build_tasks(n, tp, world_size):
            return [
                worker.sync_weights_broadcast.remote(
                    master_addr, master_port,
                    rank_offset=1 + i * tp,
                    world_size=world_size,
                    bucket_size=bucket_size,
                    engine_only=engine_only,
                    weight_format=weight_format,
                )
                for i, worker in enumerate(self._workers)
            ]

        return await self._run_broadcast_weight_sync(
            build_tasks=build_tasks,
            strategy=strategy,
            pause_mode=pause_mode,
            clear_cache=clear_cache,
            engine_only=engine_only,
            model_id=model_id,
            fail_label="Broadcast weight sync",
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
        lora_name_prefix: str = "policy",
    ) -> dict[str, Any]:
        """Join the NCCL broadcast group (rank 0 = trainer) and install a LoRA
        adapter into every replica's LoRA manager.

        Adapter-only payload over :meth:`sync_weights_broadcast`'s lifecycle.
        Every sync replaces the same adapter id and name so paused requests
        continue to reference the resident adapter when generation resumes.
        """
        if self._synced_lora_name is None:
            lora_name = lora_name_prefix
            if not engine_only:
                self._synced_lora_name = lora_name
        else:
            lora_name = self._synced_lora_name
            if not engine_only and lora_name_prefix != lora_name:
                raise ValueError(
                    "The synced LoRA adapter name must remain stable across "
                    f"updates: expected {lora_name!r}, got {lora_name_prefix!r}"
                )
        lora_int_id = _SYNCED_LORA_ADAPTER_ID
        evict_first = strategy != "hotswap"

        def build_tasks(n, tp, world_size):
            return [
                worker.sync_lora_weights_broadcast.remote(
                    master_addr, master_port,
                    rank_offset=1 + i * tp,
                    world_size=world_size,
                    lora_int_id=lora_int_id,
                    lora_name=lora_name,
                    lora_config=lora_config,
                    bucket_size=bucket_size,
                    engine_only=engine_only,
                    staging=self.config.lora_sync_staging,
                    evict_first=evict_first,
                )
                for i, worker in enumerate(self._workers)
            ]

        return await self._run_broadcast_weight_sync(
            build_tasks=build_tasks,
            strategy=strategy,
            pause_mode=pause_mode,
            clear_cache=clear_cache,
            engine_only=engine_only,
            model_id=model_id,
            fail_label="LoRA weight sync",
            log_extra=f" lora_int_id={lora_int_id}",
            extra_response={
                "weight_format": "lora",
                "lora_int_id": lora_int_id,
                "lora_name": lora_name,
                "lora_sync_staging": self.config.lora_sync_staging,
                "lora_evict_first": evict_first,
            },
        )

    def get_spec_weights_info(self, model_id: str | None = None) -> list[dict]:
        """Return weight metadata for the spec (drafter) model."""
        self._check_model_id(model_id)
        if self._config is None:
            raise RuntimeError("ReplicaPool not initialized")
        spec_model = getattr(self._config, "speculative_model", None)
        if not spec_model:
            raise RuntimeError("No speculative_model configured")
        if self._cached_spec_weights_info is None:
            from arctic_platform.inference.server.weight_sync import build_weights_info
            infos = build_weights_info(spec_model)
            self._cached_spec_weights_info = [wi.to_dict() for wi in infos]
        return self._cached_spec_weights_info

    async def sync_spec_weights(
        self,
        groups: list[dict[str, Any]] | None = None,
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "hotswap",
        engine_only: bool = False,
        reverse: bool = False,
        model_id: str | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
        world_size: int | None = None,
    ) -> dict[str, Any]:
        """Receive spec (drafter) weights from sender(s) and load into all replicas.

        Same semantics as :meth:`sync_weights` but targets the drafter model.
        """
        self._check_model_id(model_id)
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        if groups is None:
            if master_addr is None or master_port is None:
                raise ValueError(
                    "Provide either 'groups' or legacy flat fields "
                    "(master_addr, master_port, world_size)"
                )
            n = self.num_replicas
            tp = self.tp_size
            groups = [{
                "group_id": 0,
                "master_addr": master_addr,
                "master_port": master_port,
                "world_size": world_size or (1 + n * tp),
                "replica_ids": list(range(n)),
            }]

        async with self._stream_lifecycle():
            t0 = time.time()
            n = len(self._workers)

            scheduler_was_paused = self._scheduler.paused
            available_workers = [
                index for index in range(n)
                if self._scheduler.is_worker_available(index)
            ]
            if strategy == "drain":
                self._scheduler.pause()
            elif strategy == "skip":
                for i in range(n):
                    self._scheduler.mark_worker_unavailable(i)
            elif strategy != "hotswap":
                raise ValueError(
                    f"Unknown strategy: {strategy!r}. "
                    "Use: drain, skip, hotswap"
                )

            try:
                await self._scheduler.abort_streams()
                if strategy == "drain":
                    await self._scheduler.drain()
                elif strategy == "skip":
                    for i in range(n):
                        self._scheduler.cancel_worker_inflight(i)
            except BaseException:
                if strategy == "drain" and not scheduler_was_paused:
                    self._scheduler.resume()
                elif strategy == "skip":
                    for index in available_workers:
                        self._scheduler.mark_worker_available(index)
                raise

            tp = self.tp_size
            replica_to_group: dict[int, dict[str, Any]] = {}
            for g in groups:
                base_port = g["master_port"]
                for rid in g["replica_ids"]:
                    replica_to_group[rid] = {
                        "master_addr": g["master_addr"],
                        "master_port": base_port + rid * tp,
                        "world_size": 2,
                        "rank_offset": 1,
                    }

            tasks = []
            for i, worker in enumerate(self._workers):
                gcfg = replica_to_group.get(i)
                if gcfg is None:
                    raise RuntimeError(
                        f"Replica {i} not assigned to any group. "
                        f"Groups cover replicas: "
                        f"{[g['replica_ids'] for g in groups]}"
                    )
                tasks.append(
                    worker.sync_spec_weights.remote(
                        gcfg["master_addr"], gcfg["master_port"],
                        gcfg["rank_offset"], gcfg["world_size"],
                        bucket_size, engine_only, reverse,
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            if strategy == "drain" and not scheduler_was_paused:
                self._scheduler.resume()
            elif strategy == "skip":
                for index in available_workers:
                    self._scheduler.mark_worker_available(index)

            per_worker = [
                {"status": "error", "message": str(r)}
                if isinstance(r, Exception) else r
                for r in results
            ]
            elapsed = time.time() - t0
            logger.info(
                "Spec weight sync: %d workers, %d group(s) in %.2fs",
                n, len(groups), elapsed,
            )
            return {
                "elapsed": elapsed,
                "num_groups": len(groups),
                "strategy": strategy,
                "strategy_elapsed": elapsed,
                "workers": per_worker,
            }

    async def close_weight_sync(self, model_id: str | None = None) -> dict[str, Any]:
        self._check_model_id(model_id)
        async with self._stream_lifecycle():
            results = await asyncio.gather(*[w.close_weight_sync.remote() for w in self._workers])
            return {"status": "ok", "workers": list(results)}

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    async def _monitor_health(self) -> None:
        while not self._stop_monitoring:
            await asyncio.sleep(10)
            for i, w in enumerate(self._workers):
                if i in self._updating_workers:
                    continue
                if self._scheduler is not None and not self._scheduler.is_worker_available(i):
                    continue
                try:
                    healthy = await asyncio.wait_for(w.is_healthy.remote(), timeout=30)
                except Exception as exc:
                    logger.warning(
                        "Worker %d health check failed; %s: %s",
                        i,
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )
                    continue
                if not healthy:
                    logger.warning("Worker %d unhealthy", i)
                    continue

    async def _restart_worker(self, idx: int) -> None:
        old = self._workers[idx]
        if self._scheduler is not None:
            self._scheduler.mark_worker_unavailable(idx)

        engine_kwargs = self._engine_kwargs()
        extra_env = self._config.extra_env or None

        if not self._is_multi_node():
            try:
                ray.kill(old)
            except Exception:
                pass
            new_worker = self._make_worker()
            try:
                await new_worker.initialize.remote(
                    engine_kwargs,
                    extra_env,
                    self._model_id,
                )
            except BaseException:
                try:
                    ray.kill(new_worker)
                except Exception:
                    pass
                raise
        else:
            # vLLM's Ray child actors may release PG bundles asynchronously.
            # Ask the coordinator to shut down first, then retry creation until
            # the placement-group restart deadline expires.
            await self._shutdown_workers([old])
            new_worker = await self._restart_placement_worker(
                engine_kwargs,
                extra_env,
            )

        self._workers[idx] = new_worker

        if self._scheduler is not None:
            self._scheduler.update_worker_handle(idx, new_worker)
            self._scheduler.mark_worker_available(idx)

        logger.info(f"Worker {idx} restarted successfully")

    async def _restart_placement_worker(
        self,
        engine_kwargs: dict[str, Any],
        extra_env: dict[str, str] | None,
    ) -> ray.actor.ActorHandle:
        total_timeout = _env_float(
            "ARCTIC_WORKER_RESTART_TIMEOUT_S", 600.0, minimum=0.001
        )
        retry_s = _env_float(
            "ARCTIC_WORKER_RESTART_RETRY_S", 1.0, minimum=0.0
        )
        deadline = time.monotonic() + total_timeout
        attempt = 0

        while True:
            attempt += 1
            new_worker = None
            try:
                new_worker = self._make_worker()
                remaining = max(deadline - time.monotonic(), 0.001)
                await asyncio.wait_for(
                    _await_maybe(
                        new_worker.initialize.remote(
                            engine_kwargs,
                            extra_env,
                            self._model_id,
                        )
                    ),
                    timeout=remaining,
                )
                return new_worker
            except asyncio.CancelledError:
                if new_worker is not None:
                    try:
                        ray.kill(new_worker)
                    except Exception:
                        pass
                raise
            except Exception:
                if new_worker is not None:
                    try:
                        ray.kill(new_worker)
                    except Exception:
                        pass
                if not self._is_multi_node() or time.monotonic() >= deadline:
                    raise
                delay = min(retry_s, max(deadline - time.monotonic(), 0.0))
                logger.warning(
                    "Placement-group worker restart attempt %d failed; "
                    "retrying in %.3fs",
                    attempt,
                    delay,
                    exc_info=True,
                )
                await asyncio.sleep(delay)
