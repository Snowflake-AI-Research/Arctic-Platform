from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

import ray
from arctic_platform.inference.server.streaming import ClientStream, MAX_WORKER_STREAMS, StreamLimits, validate_request

from arctic_platform.inference.server.metrics import (
    ConcurrencyHistory,
    RequestRecord,
    _BoundedDeque,
)

logger = logging.getLogger("arctic_platform.inference.server")

_LOG_AFFINITY = os.environ.get("ARCTIC_LOG_AFFINITY", "") == "1"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


@dataclass
class _Request:
    id: int
    prompt: str | list[int]
    sampling_params: dict[str, Any]
    future: asyncio.Future
    worker_idx: int | None = None
    created_at: float = field(default_factory=time.time)
    prefix_hash: int | None = None
    routing_key: str | None = None
    strict: bool = False


@dataclass
class WorkerState:
    handle: ray.actor.ActorHandle
    concurrency_limit: int
    active_requests: int = 0
    available: bool = True
    latest_cache_utilization: float = 0.0
    running_reqs: int = 0
    waiting_reqs: int = 0
    streaming_requests: int = 0
    quarantined: bool = False

    @property
    def schedulable(self) -> bool:
        return self.available and not self.quarantined

    def quarantine(self) -> None:
        self.quarantined = True
        self.available = False


@dataclass
class _AffinityAssignment:
    worker_idx: int
    last_used: float


@dataclass
class _RouteDiagnostics:
    reason: str
    existing_assignment: bool
    active_requests: list[int]
    concurrency_limits: list[int]
    assignments: int
    routing_key_preview: str | None


RoutingFn = Callable[[_Request, list[WorkerState]], int]
ConcurrencyFn = Callable[[list[WorkerState]], list[int]]


def _compute_prefix_hash(prompt: str | list[int]) -> int:
    """Hash the full prompt for affinity routing."""
    if isinstance(prompt, str):
        return hash(prompt)
    return hash(tuple(prompt))


def least_loaded_routing(request: _Request, workers: list[WorkerState]) -> int:
    """Route to the worker with the lowest load ratio."""
    candidates = [i for i, ws in enumerate(workers) if ws.schedulable and ws.concurrency_limit > 0]
    if not candidates:
        candidates = list(range(len(workers)))
    return min(
        candidates,
        key=lambda i: workers[i].active_requests / max(workers[i].concurrency_limit, 1),
    )


_PREFIX_LOAD_THRESHOLD = 0.85


def prefix_affinity_routing(request: _Request, workers: list[WorkerState]) -> int:
    """Route requests with the same prefix hash to the same worker.

    Falls back through a hash ring on overload and ultimately to
    ``least_loaded_routing`` when all candidates exceed the load threshold.
    """
    candidates = [
        i for i, ws in enumerate(workers)
        if ws.schedulable and ws.concurrency_limit > 0
    ]
    if not candidates:
        return least_loaded_routing(request, workers)

    if request.prefix_hash is None:
        return least_loaded_routing(request, workers)

    preferred = candidates[request.prefix_hash % len(candidates)]
    ws = workers[preferred]
    if ws.active_requests < ws.concurrency_limit * _PREFIX_LOAD_THRESHOLD:
        return preferred

    for offset in range(1, len(candidates)):
        alt = candidates[(request.prefix_hash + offset) % len(candidates)]
        ws_alt = workers[alt]
        if ws_alt.active_requests < ws_alt.concurrency_limit * _PREFIX_LOAD_THRESHOLD:
            return alt

    return least_loaded_routing(request, workers)


def strict_affinity_routing(request: _Request, workers: list[WorkerState]) -> int:
    """Always route to the worker keyed by ``request.prefix_hash``, no ringing.

    Unlike :func:`prefix_affinity_routing`, this never spills to alternate
    workers when the preferred one is overloaded; the scheduler's outer loop
    backs off and retries until capacity opens up on the pinned worker. Used
    for multi-turn rollouts where cache reuse is more valuable than throughput
    smoothing.
    """
    candidates = [
        i for i, ws in enumerate(workers)
        if ws.schedulable and ws.concurrency_limit > 0
    ]
    if not candidates:
        return least_loaded_routing(request, workers)
    if request.prefix_hash is None:
        return least_loaded_routing(request, workers)
    return candidates[request.prefix_hash % len(candidates)]


def _preview_routing_key(routing_key: str | None, *, max_len: int = 160) -> str | None:
    if routing_key is None:
        return None
    if len(routing_key) <= max_len:
        return routing_key
    return f"{routing_key[:max_len]}..."


def utilization_based_concurrency(workers: list[WorkerState]) -> list[int]:
    """Adjust per-worker concurrency limits based on KV cache utilization."""
    TARGET_UTIL = 0.90
    HIGH_UTIL = 0.95
    MAX_QUEUE = 2
    MIN_LIMIT, MAX_LIMIT = 8, 2048
    GROWTH_STEP = 2
    AGGRESSIVE_FACTOR = 10
    BACKOFF_STEP = 1 if random.random() < 0.3 else 0
    PROBE_PROB = 0.1

    new_limits: list[int] = []
    for ws in workers:
        current = max(ws.concurrency_limit, MIN_LIMIT)
        util = ws.latest_cache_utilization

        if util == 0.0 and ws.running_reqs == 0 and ws.waiting_reqs == 0:
            if ws.active_requests > 64:
                new = max(MIN_LIMIT, int(ws.active_requests * 0.95))
            else:
                new = min(max(MIN_LIMIT, int(ws.active_requests * 1.5)), MAX_LIMIT)
        elif util > HIGH_UTIL:
            new = current - BACKOFF_STEP
        elif ws.waiting_reqs > MAX_QUEUE:
            new = current - BACKOFF_STEP
        elif util < TARGET_UTIL:
            if ws.active_requests >= (current - 2):
                gap = max(0.0, TARGET_UTIL - util)
                new = current + GROWTH_STEP + int(gap * AGGRESSIVE_FACTOR)
            else:
                new = current
        else:
            if ws.active_requests >= current and ws.waiting_reqs == 0 and random.random() < PROBE_PROB:
                new = current + 1
            else:
                new = current

        new_limits.append(max(MIN_LIMIT, min(new, MAX_LIMIT)))
    return new_limits


class Scheduler:
    """Routes generation requests across workers."""

    def __init__(
        self,
        workers: list[ray.actor.ActorHandle],
        initial_concurrency: int = 64,
        routing_fn: RoutingFn = least_loaded_routing,
        concurrency_fn: ConcurrencyFn = utilization_based_concurrency,
        poll_interval: float = 0.5,
        adjust_interval: float = 0.5,
        enable_prefix_hash: bool = False,
        max_request_records: int = 100_000,
        dynamic_concurrency: bool | None = None,
        enable_group_affinity: bool = True,
    ) -> None:
        if not workers:
            raise ValueError("Scheduler requires at least one worker")

        self._workers = [
            WorkerState(handle=w, concurrency_limit=max(1, initial_concurrency))
            for w in workers
        ]
        self._routing_fn = routing_fn
        self._concurrency_fn = concurrency_fn
        self._enable_prefix_hash = enable_prefix_hash
        self._poll_interval = poll_interval
        self._adjust_interval = adjust_interval
        self._dynamic_concurrency = (
            _env_bool("ARCTIC_DYNAMIC_CONCURRENCY", False)
            if dynamic_concurrency is None else bool(dynamic_concurrency)
        )
        self._next_id = 0
        self._streams = {}
        self._stream_retired = OrderedDict()
        # Retired streams whose engine cleanup never confirmed; pruned with _stream_retired.
        self._stream_retired_unconfirmed = set()
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._stopped = False
        self._poll_task: asyncio.Task | None = None
        self._adjust_task: asyncio.Task | None = None
        self._inflight_tasks: dict[int, set] = {i: set() for i in range(len(workers))}
        self._group_affinity_enabled = enable_group_affinity
        self._group_affinity_ttl_s = max(1.0, _env_float("ARCTIC_GROUP_AFFINITY_TTL_S", 3600.0))
        self._group_affinity_max_keys = max(1, _env_int("ARCTIC_GROUP_AFFINITY_MAX_KEYS", 500_000))
        self._group_affinity_assignments: dict[str, _AffinityAssignment] = {}
        self._group_affinity_last_cleanup = time.time()
        self._last_route_diagnostics: _RouteDiagnostics | None = None
        self._next_group_affinity_worker_idx = 0

        # Per-request metric ring (drained by `drain_metrics`).
        self._request_records: _BoundedDeque = _BoundedDeque(max_request_records)
        # Per-replica `concurrency_limit` history; used to back-fill
        # `max_concurrency` on per-step worker snapshots when draining.
        self._concurrency_history: list[ConcurrencyHistory] = [
            ConcurrencyHistory() for _ in workers
        ]
        # Seed the history so the first drain has something to look up.
        for h, ws in zip(self._concurrency_history, self._workers):
            h.record(ws.concurrency_limit)
        # Best-effort: tell each worker its index so snapshots are
        # tagged with the right replica_id.
        for idx, w in enumerate(workers):
            try:
                w.set_replica_id.remote(idx)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        prompt: str | list[int],
        sampling_params: dict[str, Any],
        routing_key: str | None = None,
        strict: bool = False,
    ) -> asyncio.Future:
        """Submit a single prompt for generation.

        Args:
            routing_key: Optional opaque string used as the affinity key for
                routing. By default, the first request for a key is assigned
                to the least-loaded worker and later requests for the same
                key are pinned to that worker. Intended for multi-turn
                rollouts where turn N+1 must hit the same replica as turn N
                to reuse its KV cache. The key also replaces the prompt hash
                as the request's ``prefix_hash`` for non-default routing.
            strict: When True, use :func:`strict_affinity_routing` instead of
                the pool's configured routing function, pinning the request
                to the keyed worker even under load. Ignored if no
                ``routing_key`` (and no prefix hash) is available.
        """
        if self._stopped:
            raise RuntimeError("Scheduler is stopped")

        self._ensure_background_tasks()

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if routing_key is not None:
            prefix_hash: int | None = hash(routing_key)
        elif self._enable_prefix_hash:
            prefix_hash = _compute_prefix_hash(prompt)
        else:
            prefix_hash = None
        req = _Request(
            id=self._next_id,
            prompt=prompt,
            sampling_params=sampling_params,
            future=future,
            prefix_hash=prefix_hash,
            routing_key=routing_key,
            strict=strict,
        )
        self._next_id += 1
        asyncio.create_task(self._process_request(req))
        return future

    def stream_generate(self, request_id, prompt, sampling_params=None, *, limits=None, routing_key=None, strict=False):
        if self._stopped:
            raise RuntimeError("Scheduler is stopped")
        if not isinstance(request_id, str) or not 0 < len(request_id) <= 128:
            raise ValueError("request_id must contain 1..128 characters")
        self._prune_retired_streams()
        if request_id in self._streams or request_id in self._stream_retired:
            raise ValueError("Duplicate or recently retired request_id")
        prompt, params = validate_request(prompt, sampling_params)
        limits = limits or StreamLimits()
        if not isinstance(limits, StreamLimits):
            raise TypeError("limits must be StreamLimits")
        request = _Request(
            id=self._next_id, prompt=prompt, sampling_params=params,
            future=asyncio.get_running_loop().create_future(),
            prefix_hash=hash(routing_key) if routing_key is not None else (_compute_prefix_hash(prompt) if self._enable_prefix_hash else None),
            routing_key=routing_key, strict=strict,
        )
        self._next_id += 1
        self._ensure_background_tasks()
        stream = ClientStream(self, request_id, request, params, limits)
        self._streams[request_id] = stream
        return stream

    def _select_stream_worker(self, request):
        if self._group_affinity_enabled and request.routing_key is not None:
            return self._group_affinity_routing(request)
        routing = strict_affinity_routing if request.strict and request.prefix_hash is not None else self._routing_fn
        index = routing(request, self._workers)
        if not request.strict and self._workers[index].streaming_requests >= MAX_WORKER_STREAMS:
            candidates = [idx for idx, worker in enumerate(self._workers) if worker.schedulable and worker.streaming_requests < MAX_WORKER_STREAMS and worker.active_requests < worker.concurrency_limit]
            if candidates:
                return min(candidates, key=lambda idx: self._workers[idx].active_requests)
        return index

    def _prune_retired_streams(self):
        cutoff = time.monotonic() - 3600
        while self._stream_retired:
            oldest = next(iter(self._stream_retired.values()))
            if oldest > cutoff and len(self._stream_retired) <= 100000:
                break
            request_id, _ = self._stream_retired.popitem(last=False)
            self._stream_retired_unconfirmed.discard(request_id)

    def _retire_stream(self, request_id, cleanup_unconfirmed=False):
        self._stream_retired[request_id] = time.monotonic()
        if cleanup_unconfirmed:
            self._stream_retired_unconfirmed.add(request_id)
        self._stream_retired.move_to_end(request_id)
        self._prune_retired_streams()

    async def abort(self, request_id):
        self._prune_retired_streams()
        stream = self._streams.get(request_id)
        if stream is not None:
            return await stream.abort()
        if request_id in self._stream_retired_unconfirmed:
            return {"status": "cleanup_unconfirmed"}
        return {"status": "already_terminal" if request_id in self._stream_retired else "not_found"}

    async def abort_streams(self, worker_idx=None):
        worker = self._workers[worker_idx] if worker_idx is not None else None
        streams = [stream for stream in list(self._streams.values()) if worker_idx is None or stream.worker is worker]
        results = await asyncio.gather(*(stream.abort("lifecycle_change") for stream in streams), return_exceptions=True)
        if any(isinstance(result, BaseException) or result["status"] == "cleanup_unconfirmed" for result in results):
            raise RuntimeError("Streaming cleanup unconfirmed")

    async def submit_batch(
        self,
        prompts: list[str | list[int]],
        sampling_params: dict[str, Any] | list[dict[str, Any] | None],
        routing_key: str | list[str | None] | None = None,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        if isinstance(routing_key, list):
            if len(routing_key) != len(prompts):
                raise ValueError(
                    f"routing_key list length ({len(routing_key)}) must match "
                    f"prompts length ({len(prompts)})"
                )
            keys: list[str | None] = list(routing_key)
        else:
            keys = [routing_key] * len(prompts)
        if isinstance(sampling_params, list):
            if len(sampling_params) != len(prompts):
                raise ValueError(
                    f"sampling_params list length ({len(sampling_params)}) must match "
                    f"prompts length ({len(prompts)})"
                )
            params_list = [params or {} for params in sampling_params]
        else:
            params_list = [sampling_params] * len(prompts)
        futures = [
            self.submit(prompt, params, routing_key=key, strict=strict)
            for prompt, params, key in zip(prompts, params_list, keys)
        ]
        return list(await asyncio.gather(*futures))

    async def drain(self) -> None:
        """Block until all in-flight requests have completed."""
        while any(ws.active_requests > 0 for ws in self._workers):
            await asyncio.sleep(0.1)

    async def drain_worker(self, idx: int, timeout: float = 15) -> None:
        """Block until worker *idx* has zero active requests, or timeout."""
        self._check_idx(idx)
        deadline = time.time() + timeout
        ws = self._workers[idx]
        while ws.active_requests > 0 and time.time() < deadline:
            await asyncio.sleep(0.1)
        if ws.active_requests > 0:
            logger.warning(f"Drain timeout: worker {idx} still has {ws.active_requests} active requests")

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True
        self._pause_event.clear()

    def resume(self) -> None:
        self._paused = False
        self._pause_event.set()

    @property
    def total_concurrency_limit(self) -> int:
        return sum(ws.concurrency_limit for ws in self._workers if ws.schedulable)

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def is_worker_available(self, idx: int) -> bool:
        self._check_idx(idx)
        return self._workers[idx].schedulable

    def mark_worker_unavailable(self, idx: int) -> None:
        self._check_idx(idx)
        self._workers[idx].available = False

    def mark_worker_available(self, idx: int) -> None:
        self._check_idx(idx)
        self._workers[idx].available = not self._workers[idx].quarantined

    def update_worker_handle(self, idx: int, new_handle: ray.actor.ActorHandle) -> None:
        self._check_idx(idx)
        self._workers[idx].handle = new_handle
        self._workers[idx].quarantined = False
        try:
            new_handle.set_replica_id.remote(idx)
        except Exception:
            pass

    def cancel_worker_inflight(self, idx: int) -> int:
        """Cancel all in-flight tasks for worker *idx*. Returns count cancelled."""
        self._check_idx(idx)
        tasks = self._inflight_tasks.get(idx, set())
        cancelled = 0
        for task in list(tasks):
            if not task.done():
                task.cancel()
                cancelled += 1
        return cancelled

    def add_worker(self, handle: ray.actor.ActorHandle, concurrency_limit: int = 64) -> None:
        """Add a new worker to the scheduler."""
        idx = len(self._workers)
        new_limit = max(1, concurrency_limit)
        self._workers.append(WorkerState(handle=handle, concurrency_limit=new_limit))
        self._inflight_tasks[idx] = set()
        history = ConcurrencyHistory()
        history.record(new_limit)
        self._concurrency_history.append(history)
        try:
            handle.set_replica_id.remote(idx)
        except Exception:
            pass

    def remove_last_worker(self) -> None:
        """Remove the last worker. Raises if no workers remain."""
        if not self._workers:
            raise RuntimeError("No workers to remove")
        idx = len(self._workers) - 1
        for task in list(self._inflight_tasks.get(idx, set())):
            if not task.done():
                task.cancel()
        self._inflight_tasks.pop(idx, None)
        self._workers.pop()
        if self._concurrency_history:
            self._concurrency_history.pop()
        for key, assignment in list(self._group_affinity_assignments.items()):
            if assignment.worker_idx >= len(self._workers):
                self._group_affinity_assignments.pop(key, None)

    async def shutdown(self) -> None:
        self._stopped = True
        self._pause_event.set()
        try:
            await self.abort_streams()
        finally:
            tasks = [task for task in (self._poll_task, self._adjust_task) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_idx(self, idx: int) -> None:
        if not (0 <= idx < len(self._workers)):
            raise IndexError(f"Worker index {idx} out of range (have {len(self._workers)} workers)")

    def _ensure_background_tasks(self) -> None:
        if not self._dynamic_concurrency:
            return
        loop = asyncio.get_running_loop()
        if self._poll_task is None:
            self._poll_task = loop.create_task(self._utilization_poll_loop())
        if self._adjust_task is None:
            self._adjust_task = loop.create_task(self._concurrency_adjust_loop())

    def _worker_can_receive_new_requests(self, idx: int) -> bool:
        return (
            0 <= idx < len(self._workers)
            and self._workers[idx].schedulable
            and self._workers[idx].concurrency_limit > 0
        )

    def _least_loaded_worker_idx(self) -> int:
        candidates = [
            i for i, ws in enumerate(self._workers)
            if ws.schedulable and ws.concurrency_limit > 0
        ]
        if not candidates:
            candidates = list(range(len(self._workers)))
        assignment_counts = Counter(
            assignment.worker_idx
            for assignment in self._group_affinity_assignments.values()
            if 0 <= assignment.worker_idx < len(self._workers)
        )
        start = self._next_group_affinity_worker_idx % max(len(self._workers), 1)
        idx = min(
            candidates,
            key=lambda i: (
                self._workers[i].active_requests / max(self._workers[i].concurrency_limit, 1),
                self._workers[i].active_requests,
                assignment_counts[i],
                (i - start) % max(len(self._workers), 1),
            ),
        )
        self._next_group_affinity_worker_idx = (idx + 1) % max(len(self._workers), 1)
        return idx

    def _cleanup_group_affinity_assignments(self, now: float) -> None:
        if now - self._group_affinity_last_cleanup < 30.0 and (
            len(self._group_affinity_assignments) <= self._group_affinity_max_keys
        ):
            return

        self._group_affinity_last_cleanup = now
        cutoff = now - self._group_affinity_ttl_s
        for key, assignment in list(self._group_affinity_assignments.items()):
            if assignment.last_used < cutoff or not self._worker_can_receive_new_requests(assignment.worker_idx):
                self._group_affinity_assignments.pop(key, None)

        overflow = len(self._group_affinity_assignments) - self._group_affinity_max_keys
        if overflow > 0:
            oldest = sorted(
                self._group_affinity_assignments.items(),
                key=lambda item: item[1].last_used,
            )
            for key, _assignment in oldest[:overflow]:
                self._group_affinity_assignments.pop(key, None)

    def _route_diagnostics(
        self,
        request: _Request,
        *,
        reason: str,
        existing_assignment: bool = False,
    ) -> _RouteDiagnostics:
        return _RouteDiagnostics(
            reason=reason,
            existing_assignment=existing_assignment,
            active_requests=[ws.active_requests for ws in self._workers],
            concurrency_limits=[ws.concurrency_limit for ws in self._workers],
            assignments=len(self._group_affinity_assignments),
            routing_key_preview=_preview_routing_key(request.routing_key),
        )

    def _set_route_diagnostics(self, diagnostics: _RouteDiagnostics) -> None:
        self._last_route_diagnostics = diagnostics

    def _group_affinity_routing(self, request: _Request) -> int:
        if request.routing_key is None:
            idx = least_loaded_routing(request, self._workers)
            self._set_route_diagnostics(self._route_diagnostics(request, reason="no_routing_key"))
            return idx

        now = time.time()
        self._cleanup_group_affinity_assignments(now)
        assignment = self._group_affinity_assignments.get(request.routing_key)
        if assignment is not None and self._worker_can_receive_new_requests(assignment.worker_idx):
            assignment.last_used = now
            self._set_route_diagnostics(
                self._route_diagnostics(
                    request,
                    reason="existing_assignment",
                    existing_assignment=True,
                )
            )
            return assignment.worker_idx

        idx = self._least_loaded_worker_idx()
        self._group_affinity_assignments[request.routing_key] = _AffinityAssignment(
            worker_idx=idx,
            last_used=now,
        )
        reason = "new_assignment" if assignment is None else "reassigned_unavailable_worker"
        self._set_route_diagnostics(self._route_diagnostics(request, reason=reason))
        return idx

    async def _process_request(self, req: _Request) -> None:
        ws: WorkerState | None = None
        current_task = asyncio.current_task()
        tracked_worker_idx: int | None = None
        submitted_time: float = 0.0
        result: Any = None

        if self._group_affinity_enabled and req.routing_key is not None:
            routing_fn: RoutingFn | None = None
            routing_name = "group_affinity_routing"
        else:
            routing_fn = (
                strict_affinity_routing
                if req.strict and req.prefix_hash is not None
                else self._routing_fn
            )
            routing_name = routing_fn.__name__

        try:
            while not self._stopped:
                if self._paused:
                    await self._pause_event.wait()
                    continue
                route_diagnostics: _RouteDiagnostics | None = None
                if routing_fn is None:
                    idx = self._group_affinity_routing(req)
                    route_diagnostics = self._last_route_diagnostics
                else:
                    idx = routing_fn(req, self._workers)
                ws = self._workers[idx]
                if ws.schedulable and ws.active_requests < ws.concurrency_limit:
                    ws.active_requests += 1
                    req.worker_idx = idx
                    submitted_time = time.time()
                    if _LOG_AFFINITY:
                        # Opt-in audit trail: every dispatched request emits
                        # one line so we can grep '(prefix_hash, worker)' and
                        # confirm same-keyed requests land on the same worker.
                        # Emit at WARNING since the env var is the gate and
                        # the default Python logger threshold drops INFO.
                        logger.warning(
                            "AFFINITY req=%d worker=%d prefix_hash=%s "
                            "strict=%s routing_fn=%s route_reason=%s "
                            "existing_assignment=%s active_before=%s "
                            "limits=%s assignments=%s "
                            "routing_key=%s",
                            req.id, idx, req.prefix_hash,
                            req.strict, routing_name,
                            route_diagnostics.reason if route_diagnostics is not None else None,
                            route_diagnostics.existing_assignment if route_diagnostics is not None else None,
                            route_diagnostics.active_requests if route_diagnostics is not None else None,
                            route_diagnostics.concurrency_limits if route_diagnostics is not None else None,
                            route_diagnostics.assignments if route_diagnostics is not None else None,
                            route_diagnostics.routing_key_preview if route_diagnostics is not None else None,
                        )
                    break
                await asyncio.sleep(0.005)
            else:
                if not req.future.done():
                    req.future.set_exception(RuntimeError("Scheduler stopped"))
                return

            tracked_worker_idx = req.worker_idx
            if current_task is not None and tracked_worker_idx is not None:
                self._inflight_tasks[tracked_worker_idx].add(current_task)

            try:
                result = await ws.handle.generate.remote(req.prompt, req.sampling_params)
                if not req.future.done():
                    req.future.set_result(result)
            except asyncio.CancelledError:
                if not req.future.done():
                    req.future.set_exception(RuntimeError("Request cancelled for weight update"))
            except Exception as exc:
                logger.warning(
                    "Worker %d generate failed for request %d: %s: %s",
                    tracked_worker_idx,
                    req.id,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                raise

        except asyncio.CancelledError:
            if not req.future.done():
                req.future.set_exception(RuntimeError("Request cancelled for weight update"))
        except Exception as e:
            if not req.future.done():
                req.future.set_exception(e)
        finally:
            if current_task is not None and tracked_worker_idx is not None:
                self._inflight_tasks[tracked_worker_idx].discard(current_task)
            if ws is not None:
                ws.active_requests = max(0, ws.active_requests - 1)
            # Record the per-request metric — only if we successfully
            # submitted to a worker. Skipping cancelled / failed requests
            # keeps the buffer clean for downstream analysis.
            if (
                tracked_worker_idx is not None
                and submitted_time > 0.0
                and isinstance(result, dict)
            ):
                self._request_records.push(RequestRecord(
                    request_id=req.id,
                    replica_id=tracked_worker_idx,
                    arrival_time=req.created_at,
                    submitted_time=submitted_time,
                    completion_time=time.time(),
                    prompt_len=int(result.get("prompt_len", 0) or 0),
                    generation_len=int(result.get("generation_len", 0) or 0),
                    prefix_cache_len=int(result.get("prefix_cache_len", 0) or 0),
                ))

    async def _utilization_poll_loop(self) -> None:
        while not self._stopped:
            await asyncio.sleep(self._poll_interval)
            refs = [ws.handle.get_stats.remote() for ws in self._workers]
            results = await asyncio.gather(*refs, return_exceptions=True)
            for ws, result in zip(self._workers, results):
                if isinstance(result, Exception):
                    ws.latest_cache_utilization = 0.0
                    ws.running_reqs = 0
                    ws.waiting_reqs = 0
                else:
                    ws.latest_cache_utilization = result.get("gpu_cache_usage", 0.0)
                    ws.running_reqs = int(result.get("num_requests_running", 0))
                    ws.waiting_reqs = int(result.get("num_requests_waiting", 0))

    async def _concurrency_adjust_loop(self) -> None:
        while not self._stopped:
            await asyncio.sleep(self._adjust_interval)
            try:
                new_limits = self._concurrency_fn(self._workers)
                for idx, (ws, limit) in enumerate(zip(self._workers, new_limits)):
                    new_limit = max(1, int(limit))
                    ws.concurrency_limit = new_limit
                    if idx < len(self._concurrency_history):
                        self._concurrency_history[idx].record(new_limit)
            except Exception as e:
                logger.error(f"Concurrency adjustment failed: {e}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def drain_metrics(self) -> dict[str, Any]:
        """Drain per-replica snapshots and per-request records.

        Returns a dict with two top-level keys:

          * ``requests``: list of :class:`RequestRecord` dicts (one per
            generation call completed since the last drain).
          * ``replicas``: snapshots, lifetime engine totals, and replay-cache
            counters. Each snapshot has ``max_concurrency`` back-filled from
            the scheduler's per-replica concurrency-limit history.

        Calling this is idempotent and does not block scheduling.
        """
        # Pull snapshots from each worker. Drains the worker-side ring as a
        # side effect.
        replicas_payload: list[dict[str, Any]] = []
        # Resolve handles to current ones so worker-restart doesn't strand
        # us holding a dead actor.
        worker_handles = [(idx, ws.handle) for idx, ws in enumerate(self._workers)]
        results = await asyncio.gather(
            *[h.drain_metrics.remote() for _, h in worker_handles],
            return_exceptions=True,
        )
        for (idx, _), res in zip(worker_handles, results):
            if isinstance(res, Exception):
                replicas_payload.append({
                    "replica_id": idx,
                    "snapshots": [],
                    "error": str(res),
                })
                continue
            snaps = res.get("snapshots", []) if isinstance(res, dict) else []
            history = (
                self._concurrency_history[idx]
                if idx < len(self._concurrency_history)
                else None
            )
            for s in snaps:
                # Worker doesn't know its own concurrency_limit; back-fill
                # from the scheduler's per-replica history.
                s["replica_id"] = idx
                if history is not None:
                    s["max_concurrency"] = history.at(s.get("timestamp", 0.0))
            replicas_payload.append({
                "replica_id": idx,
                "snapshots": snaps,
                "engine_totals": res["engine_totals"],
                "action_mask_replay_cache": (
                    res.get("action_mask_replay_cache", {})
                    if isinstance(res, dict)
                    else {}
                ),
            })

        requests_payload = [r.to_dict() for r in self._request_records.drain()]
        return {
            "requests": requests_payload,
            "replicas": replicas_payload,
        }
