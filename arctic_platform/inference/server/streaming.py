"""Bounded prepared-generation streams and their version-one wire contract."""

from __future__ import annotations

import asyncio
import math
import json
import time
from functools import wraps
from collections import deque
from dataclasses import asdict, dataclass
from typing import AsyncIterator
from uuid import uuid4

MAX_WORKER_STREAMS = 128
CONTEXT_LIMIT_SOURCES = frozenset({"prompt", "completion_budget"})


class StreamError(RuntimeError):
    def __init__(self, code: str, *, context_limit_source=None):
        if code == "context_length_exceeded":
            if (
                not isinstance(context_limit_source, str)
                or context_limit_source not in CONTEXT_LIMIT_SOURCES
            ):
                raise ValueError(
                    "context_length_exceeded requires a valid context_limit_source"
                )
        elif context_limit_source is not None:
            raise ValueError(
                "context_limit_source is valid only for context_length_exceeded"
            )
        super().__init__(code)
        self.code = code
        self.context_limit_source = context_limit_source


def classify_engine_error(exc):
    try:
        from vllm.exceptions import VLLMValidationError
    except ImportError:
        return "engine_error", None
    if not isinstance(exc, VLLMValidationError):
        return "engine_error", None
    if getattr(exc, "parameter", None) == "input_tokens":
        return "context_length_exceeded", "prompt"
    message = str(exc)
    if (
        message.startswith("This model's maximum context length is ")
        and "your prompt contains" in message
        and " input tokens" in message
        and "Please reduce the length of the input prompt or the number of "
        "requested output tokens." in message
    ):
        return "context_length_exceeded", "prompt"
    if (
        message.startswith("The decoder prompt (length ")
        and "longer than the maximum model length of " in message
    ):
        return "context_length_exceeded", "prompt"
    return "engine_error", None


async def bounded_cleanup(awaitable, timeout):
    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if not done:
            raise asyncio.TimeoutError("Cleanup deadline exceeded")
        return task.result()
    finally:
        try:
            if not task.done():
                task.cancel()
                await asyncio.wait({task}, timeout=timeout)
        finally:
            task.add_done_callback(
                lambda completed: completed.exception()
                if not completed.cancelled()
                else None
            )


def stream_lifecycle_change(method):
    @wraps(method)
    async def wrapped(self, *args, **kwargs):
        self._stream_lifecycle_depth = getattr(self, "_stream_lifecycle_depth", 0) + 1
        try:
            await self.abort_all_streams()
            return await method(self, *args, **kwargs)
        finally:
            self._stream_lifecycle_depth -= 1

    return wrapped


@dataclass(frozen=True)
class StreamLimits:
    timeout_s: float = 300
    stall_timeout_s: float = 30
    cleanup_timeout_s: float = 10
    max_buffer_events: int = 128
    max_buffer_bytes: int = 1024 * 1024
    max_event_bytes: int = 256 * 1024

    def __post_init__(self):
        for name in ("timeout_s", "stall_timeout_s", "cleanup_timeout_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= 3600
            ):
                raise ValueError(f"{name} must be finite and in (0, 3600]")
        for name in ("max_buffer_events", "max_buffer_bytes", "max_event_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_buffer_events > 1024 or self.max_buffer_bytes > 16 * 1024 * 1024:
            raise ValueError("Stream buffer exceeds server ceiling")
        if self.max_event_bytes > self.max_buffer_bytes:
            raise ValueError("max_event_bytes must not exceed max_buffer_bytes")


def validate_request(prompt, sampling_params):
    if isinstance(prompt, str):
        if not prompt or len(prompt.encode("utf-8")) > 1024 * 1024:
            raise ValueError("Prompt must contain 1..1048576 UTF-8 bytes")
    elif isinstance(prompt, list):
        if not 0 < len(prompt) <= 131072 or any(
            type(token) is not int or not 0 <= token < 2**31 for token in prompt
        ):
            raise ValueError("Prompt must contain 1..131072 nonnegative token IDs")
        prompt = list(prompt)
    else:
        raise ValueError("Expected one prepared text prompt or token-ID list")
    params = dict(sampling_params or {})
    unsupported = params.keys() - {
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "max_tokens",
        "stop",
        "n",
        "seed",
    }
    if unsupported:
        raise ValueError(f"Unsupported streaming parameters: {sorted(unsupported)}")
    params.setdefault("max_tokens", 4096)
    params.setdefault("n", 1)
    for name, ceiling in (("max_tokens", 131072), ("n", 8)):
        if type(params[name]) is not int or not 1 <= params[name] <= ceiling:
            raise ValueError(f"{name} must be an integer in [1, {ceiling}]")
    for name, default, lower, upper in (
        ("temperature", 1.0, 0, 2),
        ("top_p", 1.0, 0, 1),
        ("frequency_penalty", 0.0, -2, 2),
        ("presence_penalty", 0.0, -2, 2),
    ):
        value = params.setdefault(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not lower <= value <= upper
        ):
            raise ValueError(f"Invalid {name}")
    if params["top_p"] == 0:
        raise ValueError("top_p must be greater than zero")
    if "seed" in params and type(params["seed"]) is not int:
        raise ValueError("seed must be an integer")
    stop = params.get("stop")
    if stop is not None:
        stops = [stop] if isinstance(stop, str) else stop
        if (
            not isinstance(stops, list)
            or not 1 <= len(stops) <= 4
            or any(
                not isinstance(item, str)
                or not item
                or len(item.encode("utf-8")) > 4096
                for item in stops
            )
        ):
            raise ValueError(
                "stop must be a string or 1..4 nonempty strings of at most 4096 bytes"
            )
        params["stop"] = list(stops)
    return prompt, params


def event_size(event):
    return len(
        json.dumps(event, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    )


class EventBuffer:
    def __init__(self, limits):
        self.limits = limits
        self.events = deque()
        self.bytes = 0
        self.peak_bytes = 0
        self.peak_events = 0
        self.ready = asyncio.Event()
        self.error = None
        self.done = False
        # Newest undelivered delta per choice, which later text may join. With
        # merging, a stream that falls behind needs at most one slot per choice
        # for text, but its end needs 2n + 2 (finish events never merge), so
        # max_buffer_events must be at least that.
        self._open_deltas = {}

    def put(self, event):
        if event.get("type") == "delta" and self._merge(event):
            return
        size = event_size(event)
        if size > self.limits.max_event_bytes:
            raise StreamError("event_too_large")
        if (
            len(self.events) >= self.limits.max_buffer_events
            or self.bytes + size > self.limits.max_buffer_bytes
        ):
            raise StreamError("buffer_overflow")
        entry = [event, size]
        self.events.append(entry)
        if event.get("type") == "delta":
            self._open_deltas[event["choice_index"]] = entry
        elif "choice_index" in event:
            self._open_deltas.pop(event["choice_index"], None)
        else:
            self._open_deltas.clear()
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.peak_events = max(self.peak_events, len(self.events))
        self.ready.set()

    def _merge(self, event):
        """Append a delta's text to its choice's undelivered delta, if any.

        Tokens that arrive while the reader is behind then share one event
        instead of each taking a buffer slot, so a slow reader sees fewer,
        larger deltas rather than an overflow. A reader that keeps up takes
        each delta before the next arrives, so nothing merges.
        """
        entry = self._open_deltas.get(event["choice_index"])
        if entry is None:
            return False
        merged = {**entry[0], "text": entry[0]["text"] + event["text"]}
        size = event_size(merged)
        if size > self.limits.max_event_bytes:
            return False
        if self.bytes - entry[1] + size > self.limits.max_buffer_bytes:
            raise StreamError("buffer_overflow")
        self.bytes += size - entry[1]
        entry[0], entry[1] = merged, size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        return True

    def _pop(self):
        entry = self.events.popleft()
        event, size = entry
        self.bytes -= size
        if self._open_deltas.get(event.get("choice_index")) is entry:
            del self._open_deltas[event["choice_index"]]
        return event

    def fail(self, code, *, context_limit_source=None):
        if self.error is None:
            self.error = StreamError(
                code, context_limit_source=context_limit_source
            )
        self.events.clear()
        self._open_deltas.clear()
        self.bytes = 0
        self.ready.set()

    def drain(self):
        """Remove and return every event already buffered, without waiting."""
        drained = []
        while self.events and self.error is None:
            drained.append(self._pop())
        return drained

    async def get(self):
        while True:
            if self.error:
                raise self.error
            if self.events:
                return self._pop()
            if self.done:
                raise StopAsyncIteration
            self.ready.clear()
            await self.ready.wait()


class EngineStream:
    def __init__(self, owner, attempt_id, prompt, params, expires_at, limits):
        self.owner = owner
        self.attempt_id = attempt_id
        self.prompt = prompt
        self.params = params
        self.expires_at = expires_at
        self.limits = limits
        self.buffer = EventBuffer(limits)
        self.ack = asyncio.Event()
        self.pending_sequence = None
        self.last_progress = time.monotonic()
        self.reader_started = False
        self.cleanup_confirmed = False
        self.engine_cleanup_task = None
        self.stop_task = None
        self.pump = asyncio.create_task(self.produce())
        self.watchdog = asyncio.create_task(self.watch())

    async def produce(self):
        counts = [0] * self.params["n"]
        finished = set()
        prompt_tokens = None
        outputs = None
        try:
            params = self.owner._stream_sampling_params(self.params)
            prepared = (
                {"prompt_token_ids": self.prompt}
                if isinstance(self.prompt, list)
                else self.prompt
            )
            kwargs = {"request_id": self.attempt_id}
            adapter = self.owner._active_lora_request()
            if adapter is not None:
                kwargs["lora_request"] = adapter
            outputs = self.owner.llm.generate(prepared, params, **kwargs)
            async for output in outputs:
                if output.prompt_token_ids is not None:
                    prompt_tokens = len(output.prompt_token_ids)
                    max_model_len = self.owner.llm.model_config.max_model_len
                    if prompt_tokens + self.params["max_tokens"] > max_model_len:
                        raise StreamError(
                            "context_length_exceeded",
                            context_limit_source="completion_budget",
                        )
                for choice in output.outputs:
                    index = choice.index
                    if not 0 <= index < len(counts) or index in finished:
                        raise StreamError("invalid_engine_output")
                    counts[index] += len(choice.token_ids)
                    if counts[index] > self.params["max_tokens"]:
                        raise StreamError("invalid_engine_output")
                    if choice.text or choice.token_ids:
                        self.buffer.put(
                            {
                                "type": "delta",
                                "choice_index": index,
                                "text": choice.text,
                                "token_ids": list(choice.token_ids),
                            }
                        )
                    if choice.finish_reason is not None:
                        if choice.finish_reason not in {"stop", "length"}:
                            raise StreamError("engine_aborted")
                        finished.add(index)
                        self.buffer.put(
                            {
                                "type": "choice_finished",
                                "choice_index": index,
                                "finish_reason": choice.finish_reason,
                            }
                        )
            if len(finished) != len(counts) or prompt_tokens is None:
                raise StreamError("incomplete_engine_output")
            closing = outputs
            outputs = None
            try:
                await bounded_cleanup(closing.aclose(), self.limits.cleanup_timeout_s)
            except BaseException:
                self.owner._stream_cleanup_failed = True
                raise StreamError("cleanup_unconfirmed") from None
            if self.buffer.error:
                raise self.buffer.error
            self.buffer.put(
                {
                    "type": "usage",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": sum(counts),
                    "total_tokens": prompt_tokens + sum(counts),
                }
            )
            self.buffer.put({"type": "completed"})
            self.buffer.done = True
            self.buffer.ready.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if isinstance(exc, StreamError):
                code = exc.code
                context_limit_source = exc.context_limit_source
            else:
                code, context_limit_source = classify_engine_error(exc)
            self.buffer.fail(
                code, context_limit_source=context_limit_source
            )
            self.ack.set()
        finally:
            if outputs is not None:
                try:
                    await bounded_cleanup(
                        outputs.aclose(), self.limits.cleanup_timeout_s
                    )
                except BaseException:
                    self.owner._stream_cleanup_failed = True
                    self.buffer.fail("cleanup_unconfirmed")
            if self.buffer.error:
                await self.abort_engine()

    async def abort_engine(self):
        if self.engine_cleanup_task is None:
            self.engine_cleanup_task = asyncio.create_task(self._abort_engine())
        await asyncio.shield(self.engine_cleanup_task)

    async def _abort_engine(self):
        try:
            await bounded_cleanup(
                self.owner.llm.abort(self.attempt_id), self.limits.cleanup_timeout_s
            )
            self.cleanup_confirmed = True
        except BaseException:
            self.cleanup_confirmed = False
            self.owner._stream_cleanup_failed = True

    async def stop(self, code):
        if self.stop_task is None:
            self.buffer.fail(code)
            self.ack.set()
            self.stop_task = asyncio.create_task(self._stop())
        return await asyncio.shield(self.stop_task)

    async def _stop(self):
        if not self.pump.done():
            self.pump.cancel()
        try:
            await asyncio.wait_for(
                asyncio.shield(self.pump), self.limits.cleanup_timeout_s
            )
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            self.owner._stream_cleanup_failed = True
            await self.abort_engine()
            return "cleanup_unconfirmed"
        await self.abort_engine()
        return (
            "aborted"
            if self.cleanup_confirmed and not getattr(self.owner, "_stream_cleanup_failed", False)
            else "cleanup_unconfirmed"
        )

    async def watch(self):
        try:
            while True:
                remaining = self.expires_at - time.monotonic()
                if not self.reader_started or self.pending_sequence is not None:
                    remaining = min(
                        remaining,
                        self.limits.stall_timeout_s
                        - (time.monotonic() - self.last_progress),
                    )
                if remaining <= 0:
                    await self.stop(
                        "deadline_exceeded"
                        if time.monotonic() >= self.expires_at
                        else "consumer_stalled"
                    )
                    self.owner._engine_streams.pop(self.attempt_id, None)
                    return
                await asyncio.sleep(min(remaining, 0.25))
        except asyncio.CancelledError:
            pass


class StreamingWorkerMixin:
    def streaming_status(self):
        sessions = list(getattr(self, "_engine_streams", {}).values())
        return {
            "active_sessions": len(sessions),
            "buffered_bytes": sum(session.buffer.bytes for session in sessions),
            "buffered_events": sum(len(session.buffer.events) for session in sessions),
            "cleanup_failed": getattr(self, "_stream_cleanup_failed", False),
            "engine_unfinished_requests": self.llm.get_num_unfinished_requests(),
        }

    def _stream_sampling_params(self, params):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        return SamplingParams(
            **params,
            output_kind=RequestOutputKind.DELTA,
            include_stop_str_in_output=False,
        )

    def start_stream(self, attempt_id, prompt, sampling_params, remaining_s, limits):
        if (
            getattr(self.state, "value", self.state) != "ready"
            or getattr(self, "_stream_cleanup_failed", False)
            or getattr(self, "_stream_lifecycle_depth", 0)
            or getattr(self, "_stream_engine_paused", False)
        ):
            raise RuntimeError("Worker is not ready for streaming")
        limits = StreamLimits(**limits)
        if (
            isinstance(remaining_s, bool)
            or not isinstance(remaining_s, (int, float))
            or not math.isfinite(remaining_s)
            or not 0 < remaining_s <= limits.timeout_s
        ):
            raise ValueError("Expired or invalid stream deadline")
        if not isinstance(attempt_id, str) or not 0 < len(attempt_id) <= 128:
            raise ValueError("Invalid attempt ID")
        prompt, params = validate_request(prompt, sampling_params)
        if not hasattr(self, "_engine_streams"):
            self._engine_streams = {}
        if attempt_id in self._engine_streams:
            raise ValueError("Duplicate attempt ID")
        if len(self._engine_streams) >= MAX_WORKER_STREAMS:
            return {"status": "rejected", "code": "stream_capacity_exceeded"}
        self._engine_streams[attempt_id] = EngineStream(
            self, attempt_id, prompt, params, time.monotonic() + remaining_s, limits
        )
        return {"status": "registered"}

    async def stream_events(self, attempt_id):
        """Yield buffered events in batches, one acknowledgement per batch.

        Each batch holds the next event plus everything already buffered behind
        it. With no backlog a batch is one event, as before. When the engine
        outruns the consumer, the backlog is handed over in one round trip
        instead of one per event, so the buffer drains instead of overflowing.
        """
        session = self._engine_streams[attempt_id]
        if session.reader_started:
            raise ValueError("Stream already has a reader")
        session.reader_started = True
        sequence = 0
        completed = False
        try:
            while True:
                try:
                    events = [await session.buffer.get()]
                except StreamError as exc:
                    event = {
                        "type": "terminal_error",
                        "code": exc.code,
                        "sequence": sequence,
                        "version": 1,
                    }
                    if exc.context_limit_source is not None:
                        event["context_limit_source"] = exc.context_limit_source
                    yield [event]
                    return
                except StopAsyncIteration:
                    return
                events.extend(session.buffer.drain())
                batch = []
                for event in events:
                    batch.append({**event, "sequence": sequence, "version": 1})
                    sequence += 1
                    if event["type"] == "completed":
                        completed = True
                        break
                session.ack.clear()
                session.pending_sequence = batch[-1]["sequence"]
                session.last_progress = time.monotonic()
                if completed:
                    session.watchdog.cancel()
                    self._engine_streams.pop(attempt_id, None)
                yield batch
                if completed:
                    return
                await session.ack.wait()
        finally:
            if not completed:
                await session.stop(session.buffer.error or "cancelled")
            session.watchdog.cancel()
            self._engine_streams.pop(attempt_id, None)

    def acknowledge_stream(self, attempt_id, sequence):
        session = self._engine_streams.get(attempt_id)
        if session is None:
            return False
        if session.pending_sequence != sequence:
            raise ValueError("Unexpected stream acknowledgement")
        session.last_progress = time.monotonic()
        session.pending_sequence = None
        session.ack.set()
        return True

    async def abort_stream(self, attempt_id):
        session = getattr(self, "_engine_streams", {}).get(attempt_id)
        if session is None:
            return {
                "status": "cleanup_unconfirmed"
                if getattr(self, "_stream_cleanup_failed", False)
                else "not_found"
            }
        status = await self._stop_and_release_stream(session, "cancelled")
        return {"status": status}

    async def _stop_and_release_stream(self, session, code):
        try:
            return await session.stop(code)
        except BaseException:
            self._stream_cleanup_failed = True
            raise
        finally:
            session.watchdog.cancel()
            if self._engine_streams.get(session.attempt_id) is session:
                self._engine_streams.pop(session.attempt_id, None)

    async def abort_all_streams(self):
        sessions = list(getattr(self, "_engine_streams", {}).values())
        results = await asyncio.gather(
            *(
                self._stop_and_release_stream(session, "worker_lifecycle_change")
                for session in sessions
            ),
            return_exceptions=True,
        )
        if getattr(self, "_stream_cleanup_failed", False) or any(
            isinstance(result, BaseException) or result == "cleanup_unconfirmed"
            for result in results
        ):
            raise RuntimeError("Streaming engine cleanup unconfirmed")


class ClientStream(AsyncIterator):
    def __init__(self, scheduler, request_id, request, params, limits):
        self.scheduler = scheduler
        self.request_id = request_id
        self.request = request
        self.params = params
        self.limits = limits
        self.attempt_id = uuid4().hex
        self.expires_at = time.monotonic() + limits.timeout_s
        self.last_read = time.monotonic()
        self.worker = None
        self.registration = None
        self.registration_rejected = False
        self.submitted_at = None
        self.remote_stream = None
        self.previous_sequence = None
        self.reading = False
        self.closed = False
        self.error = None
        self.cleanup = None
        self.error_delivered = False
        self.usage = None
        self.first_delta_time = None
        self.finished_choices = set()
        self.next_sequence = 0
        # Events received in the current batch, not yet handed to the reader.
        self.pending_events = deque()
        self.watchdog = asyncio.create_task(self.watch())

    def __aiter__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def watch(self):
        try:
            while not self.closed:
                if time.monotonic() >= self.expires_at:
                    await self.abort("deadline_exceeded")
                    return
                if (
                    not self.reading
                    and time.monotonic() - self.last_read >= self.limits.stall_timeout_s
                ):
                    await self.abort("consumer_stalled")
                    return
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    async def __anext__(self):
        if self.reading:
            raise RuntimeError("Concurrent stream reads are not supported")
        if self.error_delivered:
            raise StopAsyncIteration
        if self.error:
            raise StreamError(self.error)
        if self.closed:
            raise StopAsyncIteration
        self.reading = True
        self.last_read = time.monotonic()
        try:
            return await asyncio.wait_for(
                self.read_event(), max(0.001, self.expires_at - time.monotonic())
            )
        except asyncio.TimeoutError:
            await self.abort("deadline_exceeded")
            raise StreamError("deadline_exceeded") from None
        except BaseException as exc:
            if not isinstance(exc, StopAsyncIteration):
                await self.abort(
                    exc.code if isinstance(exc, StreamError) else "stream_interrupted"
                )
            raise
        finally:
            self.reading = False
            self.last_read = time.monotonic()

    async def read_buffered(self, limit):
        """Return up to ``limit`` more events already received, with no round trip.

        For callers that relay events onward in batches: after a normal read,
        this hands over the rest of the batch that read fetched. Returns an
        empty list when nothing is buffered. A bad event aborts the stream as a
        normal read would; events accepted before it are still returned, and
        the next read raises the error.
        """
        if self.reading:
            raise RuntimeError("Concurrent stream reads are not supported")
        events = []
        self.reading = True
        try:
            while (
                self.pending_events
                and len(events) < limit
                and not (self.closed or self.error or self.error_delivered)
            ):
                events.append(await self._accept_event(self.pending_events.popleft()))
        except StreamError as exc:
            await self.abort(exc.code)
            if not events:
                raise
        except BaseException:
            await self.abort("stream_interrupted")
            raise
        finally:
            self.reading = False
            self.last_read = time.monotonic()
        return events

    async def read_event(self):
        if not self.pending_events:
            await self._fetch_batch()
        event = self.pending_events.popleft()
        return await self._accept_event(event)

    async def _fetch_batch(self):
        if self.worker is None:
            while self.worker is None:
                if self.error:
                    raise StreamError(self.error)
                if self.scheduler._stopped:
                    raise StreamError("scheduler_stopped")
                if not self.scheduler._paused and self.scheduler._workers:
                    index = self.scheduler._select_stream_worker(self.request)
                    candidate = self.scheduler._workers[index]
                    if (
                        candidate.schedulable
                        and candidate.active_requests < candidate.concurrency_limit
                        and candidate.streaming_requests < MAX_WORKER_STREAMS
                    ):
                        self.worker = candidate
                        self.submitted_at = time.time()
                        self.request.worker_idx = index
                        candidate.active_requests += 1
                        candidate.streaming_requests += 1
                        self.registration = asyncio.ensure_future(
                            candidate.handle.start_stream.remote(
                                self.attempt_id,
                                self.request.prompt,
                                self.params,
                                self.expires_at - time.monotonic(),
                                asdict(self.limits),
                            )
                        )
                        break
                await asyncio.sleep(0.005)
            result = await asyncio.shield(self.registration)
            if result.get("status") == "rejected":
                self.registration_rejected = True
                raise StreamError(result["code"])
            if self.error:
                raise StreamError(self.error)
            self.remote_stream = self.worker.handle.stream_events.remote(
                self.attempt_id
            )
        if self.previous_sequence is not None:
            await self.worker.handle.acknowledge_stream.remote(
                self.attempt_id, self.previous_sequence
            )
        try:
            reference = await self.remote_stream.__anext__()
            batch = await reference
        except StopAsyncIteration:
            raise StreamError("incomplete_stream") from None
        if not isinstance(batch, list) or not batch:
            raise StreamError("invalid_event_sequence")
        self.pending_events.extend(batch)

    async def _accept_event(self, event):
        if self.error:
            raise StreamError(self.error)
        if (
            not isinstance(event, dict)
            or event.get("version") != 1
            or event.get("sequence") != self.next_sequence
        ):
            raise StreamError("invalid_event_sequence")
        self.next_sequence += 1
        self.previous_sequence = event["sequence"]
        kind = event.get("type")
        if kind == "terminal_error":
            context_limit_source_present = "context_limit_source" in event
            context_limit_source = event.get("context_limit_source")
            if event.get("code") == "context_length_exceeded":
                if (
                    not context_limit_source_present
                    or not isinstance(context_limit_source, str)
                    or context_limit_source not in CONTEXT_LIMIT_SOURCES
                ):
                    raise StreamError("invalid_terminal_error")
            elif context_limit_source_present:
                raise StreamError("invalid_terminal_error")
            await self.abort(event["code"])
            self.error_delivered = True
        elif kind in {"delta", "choice_finished"}:
            index = event.get("choice_index")
            if (
                type(index) is not int
                or not 0 <= index < self.params["n"]
                or index in self.finished_choices
                or self.usage is not None
            ):
                raise StreamError("invalid_choice_event")
            if kind == "delta" and self.first_delta_time is None:
                self.first_delta_time = time.time()
            if kind == "choice_finished":
                if event.get("finish_reason") not in {"stop", "length"}:
                    raise StreamError("invalid_finish_reason")
                self.finished_choices.add(index)
        elif kind == "usage":
            if self.usage is not None or len(self.finished_choices) != self.params["n"]:
                raise StreamError("invalid_usage_event")
            if (
                any(
                    type(event.get(key)) is not int or event[key] < 0
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                )
                or event["total_tokens"]
                != event["prompt_tokens"] + event["completion_tokens"]
            ):
                raise StreamError("invalid_usage_event")
            self.usage = event
        elif kind == "completed":
            if self.usage is None:
                raise StreamError("incomplete_stream")
            await self.finish()
        else:
            raise StreamError("unknown_event")
        return {**event, "request_id": self.request_id}

    async def abort(self, code="cancelled"):
        if self.closed:
            return {"status": "already_terminal"}
        if self.cleanup is None:
            self.error = code
            self.cleanup = asyncio.create_task(self._abort())
        return await asyncio.shield(self.cleanup)

    async def _abort(self):
        status = "aborted"
        if self.worker is not None:
            try:
                if self.registration is not None:
                    import ray

                    try:
                        result = await asyncio.wait_for(
                            asyncio.shield(self.registration),
                            self.limits.cleanup_timeout_s,
                        )
                    except ray.exceptions.RayTaskError:
                        result = {"status": "rejected"}
                    self.registration_rejected = result.get("status") == "rejected"
                if self.registration_rejected:
                    status = "not_started"
                else:
                    result = await asyncio.wait_for(
                        self.worker.handle.abort_stream.remote(self.attempt_id),
                        self.limits.cleanup_timeout_s,
                    )
                    status = result["status"]
                    if status == "not_found":
                        status = "already_terminal"
            except Exception:
                status = "cleanup_unconfirmed"
        if self.remote_stream is not None:
            import ray

            try:
                ray.cancel(self.remote_stream)
            except Exception:
                status = "cleanup_unconfirmed"
        if status == "cleanup_unconfirmed" and self.worker is not None:
            self.worker.quarantine()
        await self.finish()
        return {"status": status}

    async def finish(self):
        if self.closed:
            return
        self.closed = True
        if self.worker is not None:
            self.worker.active_requests = max(0, self.worker.active_requests - 1)
            self.worker.streaming_requests = max(0, self.worker.streaming_requests - 1)
        if self.usage is not None and self.error is None:
            from arctic_platform.inference.server.metrics import RequestRecord

            self.scheduler._request_records.push(
                RequestRecord(
                    request_id=self.request.id,
                    replica_id=self.request.worker_idx,
                    arrival_time=self.request.created_at,
                    submitted_time=self.submitted_at,
                    completion_time=time.time(),
                    prompt_len=self.usage["prompt_tokens"],
                    generation_len=self.usage["completion_tokens"],
                    prefix_cache_len=0,
                    streaming=True,
                    first_delta_time=self.first_delta_time,
                )
            )
        self.scheduler._streams.pop(self.request_id, None)
        self.scheduler._retire_stream(self.request_id)
        if asyncio.current_task() is not self.watchdog:
            self.watchdog.cancel()

    async def aclose(self):
        await self.abort()
