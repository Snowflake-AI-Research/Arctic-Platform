from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import random
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Hashable

import ray
from arctic_platform.inference.server.chat import ChatPrompt
from arctic_platform.inference.server.streaming import ClientStream, MAX_WORKER_STREAMS, StreamLimits, validate_request
from arctic_platform.inference.server.token_prompts import is_token_tensor

from arctic_platform.inference.server.metrics import RequestRecord, _BoundedDeque

logger = logging.getLogger("arctic_platform.inference.server")

_LOG_AFFINITY = os.environ.get("ARCTIC_LOG_AFFINITY", "") == "1"
_MAX_KEYS = 500_000
# Requests a worker may hold, as a multiple of capacity, before the rest wait here for whichever worker frees first.
_BACKLOG = 2


@dataclass
class _Request:
    id: int
    prompt: Any  # str, list[int], 1-D integer tensor, or ChatPrompt
    sampling_params: dict[str, Any]
    future: asyncio.Future
    # Affinity key: the routing key, else the prompt's hash under prefix routing.
    key: Hashable | None = None
    # Never spills from its key's worker: set by a routing key or strict.
    sticky: bool = False
    worker_idx: int | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class WorkerState:
    handle: ray.actor.ActorHandle
    active_requests: int = 0
    available: bool = True
    streaming_requests: int = 0
    quarantined: bool = False
    # Futures of the generate calls running on this worker.
    inflight: set = field(default_factory=set)

    @property
    def schedulable(self) -> bool:
        return self.available and not self.quarantined

    def quarantine(self) -> None:
        self.quarantined = True
        self.available = False


def _compute_prefix_hash(prompt: Any) -> int:
    """Hash the full prompt for affinity routing.

    A tensor prompt hashes its int32 bytes, so the same tokens route the same
    way whatever integer dtype they arrive in. It hashes differently from the
    same tokens sent as a list.
    """
    if isinstance(prompt, ChatPrompt):
        # Already validated as plain JSON.
        return hash(json.dumps(asdict(prompt), sort_keys=True))
    if isinstance(prompt, str):
        return hash(prompt)
    if is_token_tensor(prompt):
        import torch

        return hash(prompt.to(torch.int32).contiguous().numpy().tobytes())
    return hash(tuple(prompt))


class Scheduler:
    """Routes each request to a worker, letting each engine queue a backlog of its own.

    A request with no key goes to the less loaded of two random workers. A
    keyed one goes to the worker its key was first given, chosen that way then;
    a prefix-hash key spills while its worker holds ``capacity`` requests and
    another has room, and a sticky one never spills. A worker takes up to
    ``_BACKLOG * capacity`` generate requests; past that on every worker, or
    while paused, requests wait here in order and go to the first worker that
    frees up.
    """

    def __init__(
        self,
        workers: list[ray.actor.ActorHandle],
        capacity: int,
        prefix_routing: bool = False,
        max_request_records: int = 100_000,
    ) -> None:
        if not workers:
            raise ValueError("Scheduler requires at least one worker")

        self._workers = [WorkerState(handle=w) for w in workers]
        self._capacity = capacity
        self._backlog = _BACKLOG * capacity
        self._prefix_routing = prefix_routing
        self._keys: OrderedDict[Hashable, int] = OrderedDict()
        # Requests no worker could take yet, dispatched in order by `_drain`.
        self._waiting: deque[_Request] = deque()
        self._next_id = 0
        self._streams = {}
        self._stream_retired = OrderedDict()
        # Retired streams whose engine cleanup never confirmed; pruned with _stream_retired.
        self._stream_retired_unconfirmed = set()
        self._paused = False
        self._stopped = False
        self._wake = asyncio.Event()
        # Per-request metric ring (drained by `drain_metrics`).
        self._request_records: _BoundedDeque = _BoundedDeque(max_request_records)
        # Best-effort: tell each worker its index so snapshots are
        # tagged with the right replica_id.
        for idx, w in enumerate(workers):
            try:
                w.set_replica_id.remote(idx)
            except Exception:
                pass

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_wake"] = None
        state["_waiting"] = deque()
        state["_workers"] = [replace(ws, inflight=set()) for ws in self._workers]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._wake = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        prompt: Any,
        sampling_params: dict[str, Any],
        routing_key: str | None = None,
        strict: bool = False,
    ) -> asyncio.Future:
        """Submit a single prompt for generation.

        Args:
            prompt: Text, ``list[int]``, or a 1-D integer CPU tensor of token
                ids; a tensor is sent to the worker as is.
            routing_key: Optional opaque string used as the affinity key. The
                first request for a key goes to the least-loaded worker and
                later ones stay on that worker, so turn N+1 of a multi-turn
                rollout reuses turn N's KV cache and router-replay capture.
            strict: Keep a prompt-hash keyed request on its worker under load
                too; a routing key already does.
        """
        if self._stopped:
            raise RuntimeError("Scheduler is stopped")
        future = asyncio.get_running_loop().create_future()
        self._waiting.append(self._request(prompt, sampling_params, future, routing_key, strict))
        self._drain()
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
        future = asyncio.get_running_loop().create_future()
        request = self._request(prompt, params, future, routing_key, strict)
        stream = ClientStream(self, request_id, request, params, limits)
        self._streams[request_id] = stream
        return stream

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

    def resume(self) -> None:
        self._paused = False
        self._notify()

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def is_worker_available(self, idx: int) -> bool:
        self._check_idx(idx)
        return self._workers[idx].schedulable

    async def wait_for_schedulable_worker(self, timeout: float) -> ray.actor.ActorHandle:
        """A worker that takes requests, waiting up to ``timeout`` seconds for one."""
        deadline = time.monotonic() + timeout
        while not self._stopped:
            for ws in self._workers:
                if ws.schedulable:
                    return ws.handle
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"No schedulable worker within {timeout:g}s")
            try:
                await asyncio.wait_for(self._wake.wait(), remaining)
            except asyncio.TimeoutError:
                pass
        raise RuntimeError("Scheduler stopped")

    def mark_worker_unavailable(self, idx: int) -> None:
        self._check_idx(idx)
        self._workers[idx].available = False

    def mark_worker_available(self, idx: int) -> None:
        self._check_idx(idx)
        self._workers[idx].available = not self._workers[idx].quarantined
        self._notify()

    def update_worker_handle(self, idx: int, new_handle: ray.actor.ActorHandle) -> None:
        self._check_idx(idx)
        self._workers[idx].handle = new_handle
        self._workers[idx].quarantined = False
        try:
            new_handle.set_replica_id.remote(idx)
        except Exception:
            pass
        self._notify()

    def cancel_worker_inflight(self, idx: int) -> int:
        """Cancel all in-flight requests for worker *idx*. Returns count cancelled."""
        self._check_idx(idx)
        return sum(future.cancel() for future in list(self._workers[idx].inflight))

    def add_worker(self, handle: ray.actor.ActorHandle) -> None:
        """Add a new worker to the scheduler."""
        idx = len(self._workers)
        self._workers.append(WorkerState(handle=handle))
        try:
            handle.set_replica_id.remote(idx)
        except Exception:
            pass
        self._notify()

    def remove_last_worker(self) -> None:
        """Remove the last worker. Raises if no workers remain."""
        if not self._workers:
            raise RuntimeError("No workers to remove")
        self.cancel_worker_inflight(len(self._workers) - 1)
        self._workers.pop()

    async def shutdown(self) -> None:
        self._stopped = True
        for req in self._waiting:
            if not req.future.done():
                req.future.set_exception(RuntimeError("Scheduler stopped"))
        self._waiting.clear()
        self._notify()
        await self.abort_streams()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_idx(self, idx: int) -> None:
        if not (0 <= idx < len(self._workers)):
            raise IndexError(f"Worker index {idx} out of range (have {len(self._workers)} workers)")

    def _notify(self) -> None:
        """Retry waiting requests, and wake every stream waiter to try again."""
        self._drain()
        self._wake.set()
        self._wake = asyncio.Event()

    def _drain(self) -> None:
        """Dispatch waiting requests in order, until one can't go anywhere yet."""
        while self._waiting:
            req = self._waiting[0]
            if not req.future.done() and not self._dispatch(req):
                return
            self._waiting.popleft()

    def _request(self, prompt, sampling_params, future, routing_key, strict) -> _Request:
        if routing_key is not None:
            key: Hashable | None = routing_key
        elif self._prefix_routing:
            key = _compute_prefix_hash(prompt)
        else:
            key = None
        self._next_id += 1
        return _Request(
            id=self._next_id - 1, prompt=prompt, sampling_params=sampling_params, future=future,
            key=key, sticky=routing_key is not None or strict,
        )

    def _take(self, req: _Request, stream: bool = False) -> WorkerState | None:
        """Count ``req`` against its worker, or None while none can take it."""
        if self._paused or self._stopped:
            return None
        idx = self._route(req, stream)
        if idx is None:
            return None
        ws = self._workers[idx]
        ws.active_requests += 1
        if stream:
            ws.streaming_requests += 1
        req.worker_idx = idx
        if _LOG_AFFINITY:
            # Emit at WARNING since the env var is the gate and the default
            # Python logger threshold drops INFO.
            logger.warning(
                "AFFINITY req=%d worker=%d key=%s sticky=%s active=%s",
                req.id, idx, req.key, req.sticky, [w.active_requests for w in self._workers],
            )
        return ws

    def _route(self, req: _Request, stream: bool) -> int | None:
        idx = self._keys.get(req.key) if req.key is not None else None
        if idx is None or idx >= len(self._workers) or not self._workers[idx].schedulable:
            idx = self._least_loaded(stream)
            if idx is not None and req.key is not None:
                self._keys[req.key] = idx
                if len(self._keys) > _MAX_KEYS:
                    self._keys.popitem(last=False)
            return idx
        self._keys.move_to_end(req.key)
        ws = self._workers[idx]
        room = self._has_room(ws, stream)
        if room and (req.sticky or ws.active_requests < self._capacity):
            return idx
        if req.sticky:
            return None
        alt = self._least_loaded(stream)
        # Spilling only helps when another worker has room; otherwise it just splits the key's cache.
        if alt is not None and self._workers[alt].active_requests < self._capacity:
            return alt
        return idx if room else alt

    def _has_room(self, ws: WorkerState, stream: bool) -> bool:
        if stream:
            return ws.streaming_requests < MAX_WORKER_STREAMS
        return ws.active_requests < self._backlog

    def _least_loaded(self, stream: bool) -> int | None:
        """The less loaded of two random workers that can take a request, else the least loaded of all that can."""
        def can_take(ws: WorkerState) -> bool:
            return ws.schedulable and self._has_room(ws, stream)

        n = len(self._workers)
        i = random.randrange(n)
        j = (i + 1 + random.randrange(n - 1)) % n if n > 1 else i
        a, b = self._workers[i], self._workers[j]
        if can_take(a) and can_take(b):
            return i if a.active_requests <= b.active_requests else j
        candidates = [k for k, ws in enumerate(self._workers) if can_take(ws)]
        return min(candidates, key=lambda k: self._workers[k].active_requests, default=None)

    def _dispatch(self, req: _Request) -> bool:
        """Send ``req`` to a worker, or False while none can take it."""
        ws = self._take(req)
        if ws is None:
            return False
        future = ws.handle.generate.remote(req.prompt, req.sampling_params).as_future()
        ws.inflight.add(future)
        future.add_done_callback(functools.partial(self._done, req, ws, time.time()))
        return True

    def _done(self, req: _Request, ws: WorkerState, submitted_time: float, future: asyncio.Future) -> None:
        ws.inflight.discard(future)
        ws.active_requests -= 1
        self._drain()
        if future.cancelled():
            error: BaseException | None = RuntimeError("Request cancelled for weight update")
        elif (error := future.exception()) is not None:
            logger.warning(
                "Worker %d generate failed for request %d: %s: %s",
                req.worker_idx, req.id, type(error).__name__, error, exc_info=error,
            )
        if error is not None:
            if not req.future.done():
                req.future.set_exception(error)
            return
        result = future.result()
        if not req.future.done():
            req.future.set_result(result)
        if isinstance(result, dict):
            self._request_records.push(RequestRecord(
                request_id=req.id,
                replica_id=req.worker_idx,
                arrival_time=req.created_at,
                submitted_time=submitted_time,
                completion_time=time.time(),
                prompt_len=int(result.get("prompt_len", 0) or 0),
                generation_len=int(result.get("generation_len", 0) or 0),
                prefix_cache_len=int(result.get("prefix_cache_len", 0) or 0),
            ))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def drain_metrics(self) -> dict[str, Any]:
        """Drain per-replica snapshots and per-request records.

        Returns a dict with two top-level keys:

          * ``requests``: list of :class:`RequestRecord` dicts (one per
            generation call completed since the last drain).
          * ``replicas``: snapshots, lifetime engine totals, speculative-decoding
            counters, and replay-cache counters.

        Calling this is idempotent and does not block scheduling.
        """
        # Resolve handles to current ones so worker-restart doesn't strand
        # us holding a dead actor.
        handles = [ws.handle for ws in self._workers]
        results = await asyncio.gather(*[h.drain_metrics.remote() for h in handles], return_exceptions=True)
        replicas_payload: list[dict[str, Any]] = []
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                replicas_payload.append({"replica_id": idx, "snapshots": [], "error": str(res)})
                continue
            res = res if isinstance(res, dict) else {}
            snaps = res.get("snapshots", [])
            for s in snaps:
                s["replica_id"] = idx
            replicas_payload.append({
                "replica_id": idx,
                "snapshots": snaps,
                "engine_totals": res.get("engine_totals", {}),
                "spec_decoding": res.get("spec_decoding", {}),
                "action_mask_replay_cache": res.get("action_mask_replay_cache", {}),
            })

        requests_payload = [r.to_dict() for r in self._request_records.drain()]
        return {
            "requests": requests_payload,
            "replicas": replicas_payload,
        }
