# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cortex (cortex-training) transport over SnowAPI.

Each op is one `_Route`: its URL, how its body becomes HTTP sends, whether it
polls, whether it may be replayed, and how its reply is normalized. The
transport only executes those sends, on its own event loop, so sync and async
callers share one implementation. Cortex-specific request and response shapes
are translated here until the servers share one API. Log-probs is unsupported.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any
from typing import Iterator

from tenacity import AsyncRetrying
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential_jitter

from arctic_platform import wire
from arctic_platform.client.config import ArcticClientConfig
from arctic_platform.client.config import CortexConfig
from arctic_platform.client.config import JobId
from arctic_platform.client.requests import cancel_job_request
from arctic_platform.client.requests import get_job_request
from arctic_platform.client.transport import JOB_TYPES
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transport import Request
from arctic_platform.client.transport import Transport

logger = logging.getLogger(__name__)

_MAX_OCTET_BYTES = 60 * 1024 * 1024  # matches the SnowAPI per-request cap
# HTTP statuses worth retrying: the request was well-formed, so the same call may
# succeed once transient load/infra clears (429 rate-limit, 5xx, plus 404/409 seen
# while GS/ZMD restart). Other 4xx are client errors that won't fix themselves.
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504, 404, 409}
_OCTET_HEADERS = {"Content-Type": "application/octet-stream"}
_CHUNKED_RESULTS = {"response_options": {"format": "dssst1", "delivery": "chunked"}}
# `forward` has no octet endpoint on SnowAPI, so it travels as an /operation RPC
# whose payload is the base64'd DSSST1 frame. The control plane's REST surface
# marshals that RPC as JSON, so the frame inflates 4/3 on the way in and the
# budget below is a JSON-body budget, not a frame budget. These mirror
# dss_client's constants so both clients split at the same boundary.
_OPERATION_MAX_JSON_BYTES = 16 * 1024 * 1024
# Room for the envelope's keys, the operation type and the sub-job token.
_OPERATION_ENVELOPE_OVERHEAD_BYTES = 4096
_CHUNK_GROUP_RESTART_REQUIRED = "chunk_group_restart_required"
_CHUNK_GROUP_ERROR_CODES = {_CHUNK_GROUP_RESTART_REQUIRED, "chunk_group_conflict", "chunk_group_missing_chunks"}
_JOB_TERMINAL = ("failed", "done", "cancelled", "canceled")
_REQUEST_DONE = ("completed", "done", "succeeded")
_REQUEST_FAILED = ("failed", "cancelled", "canceled")
# JobHandles role -> Cortex sub-job job_type name.
_SUB_JOB_KEY = {"training": "training", "sampling": "sampling", "log_prob": "log_probability"}

Send = dict[str, Any]  # keyword arguments for one HTTP request


def _parent_id(job_id: JobId) -> str:
    return str(job_id).split(":", 1)[0]


# ── encoders: a filtered body -> the HTTP sends that deliver it ─────────────
def _no_body(request: Request, body: dict) -> list[Send]:
    return [{}]


def _json(request: Request, body: dict) -> list[Send]:
    return [{"json": body}]


def _octet(wire_op: str, *, force_chunk: bool = False) -> Callable[[Request, dict], list[Send]]:
    """DSSST1 octet bodies, one send per chunk; ``wire_op`` labels the chunk envelope."""

    def encode(request: Request, body: dict) -> list[Send]:
        frame = wire.dumps(body, metadata=_CHUNKED_RESULTS)
        chunks = wire.encode_byte_chunks(
            frame, kind="request", operation=wire_op, max_bytes=_MAX_OCTET_BYTES, force_chunk=force_chunk
        )
        return [{"data": chunk, "headers": _OCTET_HEADERS} for chunk in chunks]

    return encode


def _forward(request: Request, body: dict) -> list[Send]:
    """One /operation envelope per slice of the forward frame.

    Unlike the octet routes, chunking here happens *inside* the envelope: each
    POST is a complete, valid `/operation` call whose payload is one slice of
    the frame, and the zone stages them until the last one arrives. Only that
    final POST answers with a request_id -- the earlier ones report the chunk
    was cached -- so they must be posted in order and the last reply kept.
    """
    frame = wire.dumps(body, metadata=_CHUNKED_RESULTS)
    # The budget is on the JSON body, and base64 costs 4 bytes per 3, so the
    # frame slice has to be the smaller 3/4 share of what is left.
    max_payload = max(1, (_OPERATION_MAX_JSON_BYTES - _OPERATION_ENVELOPE_OVERHEAD_BYTES) * 3 // 4)
    return [
        {
            "json": {
                "operation_type": "forward",
                "sub_job_id": str(request.job_id),
                "payload": {
                    "content_type": "application/octet-stream",
                    "payload_b64": base64.b64encode(frame[at : at + max_payload]).decode("ascii"),
                },
            }
        }
        for at in range(0, len(frame), max_payload)
    ]


# ── normalizers: a Cortex reply -> the canonical result ─────────────────────
def _identity(value: Any) -> Any:
    return value


def _fwd_bwd_result(result: dict) -> dict:
    if "post_process_outputs" not in result:
        return result
    if "batch" in result:
        raise ValueError("Cortex fwd-bwd returned both 'batch' and 'post_process_outputs'")

    import torch

    batch = dict(result["post_process_outputs"] or {})
    for key in ("logprobs", "entropy"):
        if key in batch and batch[key] is not None:
            batch[key] = torch.as_tensor(batch[key])
    canonical = dict(result)
    canonical.pop("post_process_outputs")
    canonical["batch"] = batch
    return canonical


def _unwrap_forward_payload(result: dict) -> dict:
    """Unwrap forward's DSSST1 frame from the JSON result envelope.

    The zone base64s `wire.dumps({job_id, logprobs})` into `payload_b64`, but
    without the `wire_format` marker `_decode_result` keys off, so the generic
    path passes the envelope through and the caller gets a base64 string where
    the tensors should be. Nothing between here and the zone unwraps it: the
    only code that does is the standalone dev gateway, which the Cortex path
    does not go through.
    """
    if not isinstance(result, dict) or not isinstance(result.get("payload_b64"), str):
        return result
    return wire.loads(base64.b64decode(result["payload_b64"]))


def _to_python(obj: Any) -> Any:
    import torch

    if torch.is_tensor(obj):
        return obj.cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


@dataclass(frozen=True, kw_only=True)
class _Route:
    method: str
    # Under the prefix. `{job}` is the parent of request.job_id; other fields come from the body.
    path: str
    encode: Callable[[Request, dict], list[Send]] = _json
    poll: bool = False
    # False: retry only sends that provably never reached the server.
    replay_safe: bool = True
    # Re-post the whole chunk group once on `chunk_group_restart_required`. Only endpoints
    # that stage chunks on the job's actor (forward-backward) send that code; it is not a
    # general retry flag.
    restartable: bool = False
    # `operation_type`s that may answer in the POST reply instead of with a request_id.
    inline_types: frozenset[str] = frozenset()
    normalize: Callable[[Any], Any] = _identity

    def prepare(self, prefix: str, request: Request) -> tuple[str, list[Send]]:
        body = {k: v for k, v in request.body.items() if v is not None}
        job = None if request.job_id is None else _parent_id(request.job_id)
        return prefix + self.path.format_map(body | {"job": job}), self.encode(request, body)

    def submitted(self, request: Request, reply: dict) -> str | dict:
        """The request_id to poll, or the reply itself for an inline type.

        A missing request_id anywhere else raises: it would otherwise read as a
        finished op that was never waited on.
        """
        if reply.get("request_id") is not None:
            return str(reply["request_id"])
        operation_type = request.body.get("operation_type")
        if operation_type in self.inline_types:
            return reply
        raise RuntimeError(f"cortex {operation_type or request.op} response carried no request_id: {reply}")

    def finish(self, chunks: list[bytes], result: dict) -> Any:
        return self.normalize(wire.decode_result_chunks(chunks) if chunks else _decode_result(result))


class _NoopRoute(_Route):
    """Sends nothing and answers `{}`, whatever the body's shape."""

    def prepare(self, prefix: str, request: Request) -> tuple[str, list[Send]]:
        return "", []


# Neutrino never colocates training and sampling on the same GPUs, so it exposes no
# sleep/wake surface. These send nothing rather than erroring, which keeps shared
# client flows like sync_weights() (wake → operation → wake → reset) portable.
_NOOP = _NoopRoute(method="NOOP", path="")

_ROUTES = {
    "sleep-inference": _NOOP,
    "wake-inference": _NOOP,
    "sleep-training": _NOOP,
    "wake-training": _NOOP,
    # A replayed create would spawn a second job.
    "create-job": _Route(method="POST", path="", replay_safe=False),
    "get-job": _Route(method="GET", path="/{job}", encode=_no_body),
    "cancel-job": _Route(method="POST", path="/{job}:cancel", encode=_no_body),
    # forward-backward always carries a chunk envelope, even when the frame fits: its
    # chunk_group_id is the server's idempotency key. A mid-stream chunk-group desync
    # (GS restart) is recoverable only by re-posting the whole group.
    "forward-backward": _Route(
        method="POST",
        path="/{job}/forward-backward",
        encode=_octet("fwd-bwd", force_chunk=True),
        poll=True,
        restartable=True,
        normalize=_fwd_bwd_result,
    ),
    # SnowAPI requires DSSST1 octet bodies here even though generate carries no tensors.
    "generate": _Route(
        method="POST", path="/{job}/generate", encode=_octet("generate"), poll=True, normalize=_to_python
    ),
    "forward": _Route(
        method="POST", path="/{job}/operation", encode=_forward, poll=True, normalize=_unwrap_forward_payload
    ),
    "step": _Route(method="POST", path="/{job}/step", poll=True),
    "save": _Route(method="POST", path="/{job}/save", poll=True),
    # /operation is polymorphic: weight-sync is polled, reset-prefix-cache answers inline.
    "operation": _Route(
        method="POST", path="/{job}/operation", poll=True, inline_types=frozenset({"reset-prefix-cache"})
    ),
}


def _route(op: str) -> _Route:
    if op not in _ROUTES:
        raise NotImplementedError(f"cortex has no {op}")
    return _ROUTES[op]


def _is_transient(exc: BaseException) -> bool:
    """Retry connection/timeout errors and transient HTTP statuses (safe for reads)."""
    import aiohttp

    if isinstance(exc, (aiohttp.ClientConnectionError, asyncio.TimeoutError)):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in _TRANSIENT_STATUSES
    return False


def _is_connect_error(exc: BaseException) -> bool:
    """Only failures proving the request never reached the server (safe for mutating POSTs)."""
    import aiohttp

    return isinstance(exc, (aiohttp.ClientConnectorError, aiohttp.ConnectionTimeoutError))


class _ChunkGroupError(Exception):
    """A SnowAPI chunk-group error (missing/conflict/restart).

    No retry predicate matches it: re-posting one chunk cannot fix the group,
    only re-posting the whole group can.
    """

    def __init__(self, detail: dict) -> None:
        super().__init__(f"cortex chunk group error: {detail}")
        self.detail = detail

    @property
    def restart_required(self) -> bool:
        return self.detail["code"] == _CHUNK_GROUP_RESTART_REQUIRED


def _iter_error_dicts(value: Any, *, depth: int = 0) -> Iterator[dict]:
    """Yield error dicts from direct or GS-wrapped (JSON-in-string) error bodies."""
    if depth > 8:
        return
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_error_dicts(child, depth=depth + 1)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_error_dicts(child, depth=depth + 1)
    elif isinstance(value, str):
        decoder = json.JSONDecoder()
        for index, char in enumerate(value):
            if char in "[{":
                try:
                    decoded, _ = decoder.raw_decode(value[index:])
                except ValueError:
                    continue
                yield from _iter_error_dicts(decoded, depth=depth + 1)


def _chunk_group_detail(body: Any) -> dict | None:
    """The chunk-group error dict in an error body, else None."""
    for candidate in _iter_error_dicts(body):
        if candidate.get("code") in _CHUNK_GROUP_ERROR_CODES:
            return candidate
        if (
            candidate.get("chunk_group_id")
            and str(candidate.get("message") or "") == "request chunk group is missing chunks"
        ):
            return {**candidate, "code": "chunk_group_missing_chunks"}
    return None


async def _raise_for_status(resp: Any) -> None:
    """Raise for an error response, keeping the server's error body in the message.

    GS explains a 4xx in the response body; the stock error shows only the status
    line, which turns an actionable rejection into a guessing game. Stays a
    `ClientResponseError` (carrying `status`) so `_is_transient` keeps classifying
    it. A chunk-group error raises `_ChunkGroupError` instead.
    """
    import aiohttp

    body = (await resp.text()).strip()
    detail = _chunk_group_detail(body)
    if detail is not None:
        raise _ChunkGroupError(detail)
    message = f"{resp.reason}: {body[:2000]}" if body else (resp.reason or "")
    raise aiohttp.ClientResponseError(
        resp.request_info,
        resp.history,
        status=resp.status,
        message=message,
        headers=resp.headers,
    )


def job_create_payload(sub_job_configs: list[dict], config: CortexConfig) -> dict:
    """Job-create body. Unset fields and an empty comment are omitted."""
    payload: dict[str, Any] = {"sub_job_configs": sub_job_configs}
    if config.job_comment is not None and config.job_comment != "":
        payload["comment"] = config.job_comment
    for key in ("idle_timeout_seconds", "pending_timeout_seconds"):
        value = getattr(config, key)
        if value is not None:
            payload[key] = value
    return payload


class CortexTransport(Transport):
    def __init__(self, config: CortexConfig) -> None:
        self.config = config
        self.jobs = JobHandles()
        self.job_id: str | None = None
        self._roles: list[str] = []
        self.max_retries = config.max_retries
        self.poll_interval = 0.5
        self._start_loop()

    # Every HTTP exchange runs on this transport's own loop thread, so sync and async
    # callers share one implementation and one aiohttp session. Encoding (`prepare`)
    # and decoding (`finish`) stay on the caller's thread so a large frame never
    # stalls the loop for other callers.
    def _start_loop(self) -> None:
        self._session = None  # aiohttp.ClientSession, built on the loop at first send
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, name="cortex-transport", daemon=True).start()

    def _on_loop(self, coro) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # A loop and its thread are not picklable; the receiving side starts its own.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_loop"], state["_session"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._start_loop()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def initialize(self, client_config: ArcticClientConfig) -> JobHandles:
        self.start(client_config)
        return self.wait_until_ready()

    def start(self, client_config: ArcticClientConfig) -> JobHandles:
        self._roles = [role for role in JOB_TYPES if client_config.gpus_for(role) > 0]
        reconnect = JobHandles.from_config(client_config)
        if reconnect.any_set:  # reattach; a sub-job token embeds its parent job id
            self.job_id = _parent_id(reconnect.require_any())
        else:
            body = job_create_payload(client_config.to_cortex(), self.config)
            self.job_id = self.call(Request("create-job", body=body))["job_id"]
            # Logged before the wait: a job can stay PLACING for a long time.
            logger.info("created cortex job %s", self.job_id)
        return self.jobs

    def wait_until_ready(self) -> JobHandles:
        self._wait_running()
        sub_jobs = self._capture_sub_jobs()
        # JobHandles holds each role's sub-job token, so the client's op bodies
        # (e.g. weight-sync source/target) already carry Cortex-correct ids with no
        # transport-side rewrite -- exactly as on-prem does with plain job ids.
        for role in self._roles:
            self.jobs.set(role, sub_jobs[_SUB_JOB_KEY[role]])
        return self.jobs

    def shutdown(self) -> None:
        import aiohttp

        if self.job_id is not None:
            # Best-effort teardown, tolerating failure (the job may be gone).
            with contextlib.suppress(aiohttp.ClientError, asyncio.TimeoutError):
                self.call(cancel_job_request(self.job_id))
            self.job_id = None
        self._on_loop(self._close_session()).result()

    def list_jobs(self) -> list[dict]:
        """Every Cortex job on this account, with its status."""
        payload = self._on_loop(self._send("GET", self._prefix)).result()
        if isinstance(payload, list):
            return payload
        return payload.get("jobs") or payload.get("data") or []

    def cancel_job(self, job_id: str) -> None:
        """Release one job's GPUs by id."""
        self._on_loop(self._send("POST", f"{self._prefix}/{job_id}:cancel")).result()

    # ── deliver one op ───────────────────────────────────────────────────────
    def call(self, request: Request) -> Any:
        route, future = self._submit(request)
        return route.finish(*future.result())

    async def acall(self, request: Request) -> Any:
        route, future = self._submit(request)
        return route.finish(*await asyncio.wrap_future(future))

    def _submit(self, request: Request) -> tuple[_Route, Future]:
        route = _route(request.op)
        url, sends = route.prepare(self._prefix, request)
        return route, self._on_loop(self._deliver(route, request, url, sends))

    async def _deliver(self, route: _Route, request: Request, url: str, sends: list[Send]) -> tuple[list[bytes], dict]:
        reply = await self._send_group(route, url, sends)
        if not route.poll:
            return [], reply
        return await self._poll(route.submitted(request, reply))

    async def _send_group(self, route: _Route, url: str, sends: list[Send]) -> dict:
        try:
            return await self._send_each(route, url, sends)
        except _ChunkGroupError as exc:
            if not (route.restartable and exc.restart_required):
                raise
        return await self._send_each(route, url, sends)

    async def _send_each(self, route: _Route, url: str, sends: list[Send]) -> dict:
        """Post every send in order; the last reply is the op's answer."""
        retry_on = _is_transient if route.replay_safe else _is_connect_error
        reply: dict = {}
        for kwargs in sends:
            reply = await self._send(route.method, url, retry_on=retry_on, **kwargs)
        return reply

    def _request_url(self, request_id: str, cursor: str | None) -> tuple[str, dict | None]:
        url = f"{self._prefix}/{self.job_id}/requests/{request_id}"
        return url, ({"cursor": cursor} if cursor else None)

    async def _poll(self, submitted: str | dict) -> tuple[list[bytes], dict]:
        """The finished request's result chunks and inline result, left for the caller to decode."""
        # Inline types (`_Route.inline_types`) return their body from POST.
        if isinstance(submitted, dict):
            return [], submitted
        request_id = submitted
        deadline = time.monotonic() + self.config.poll_timeout
        delay = self.poll_interval
        chunks: list[bytes] = []
        cursor: str | None = None
        while time.monotonic() < deadline:
            url, params = self._request_url(request_id, cursor)
            action, value = _poll_progress(await self._send("GET", url, params=params), chunks, request_id)
            if action == "done":
                return chunks, value
            if action == "drain":
                cursor = value  # more result chunks queued; re-poll without backing off
                continue
            await asyncio.sleep(delay)
            delay = _next_delay(delay)
        raise TimeoutError(f"cortex request {request_id} did not complete within {self.config.poll_timeout}s")

    def _wait_running(self) -> None:
        deadline = time.monotonic() + self.config.job_ready_timeout
        delay = self.poll_interval
        started = time.monotonic()
        announced: str | None = None
        last_note = 0.0
        while time.monotonic() < deadline:
            job = self._job()
            state = _short(job.get("status")) or "pending"
            elapsed = time.monotonic() - started
            if state != announced:
                announced = state
                last_note = elapsed
                if state == "running":
                    logger.info("Cortex job is running (%s).", _clock(elapsed))
                else:
                    logger.info(
                        "Cortex job is %s. Workers are starting and loading the model.",
                        state,
                    )
            elif state != "running" and elapsed - last_note >= 15:
                last_note = elapsed
                logger.info("Still %s (%s).", state, _clock(elapsed))
            if state == "running":
                return
            if state in _JOB_TERMINAL:
                # Surface the server-side reason and any per-sub-job status so
                # callers can distinguish rate limits, allowlist rejections,
                # capacity exhaustion, and genuine sub-job crashes.
                reason = job.get("reason") or "(no reason)"
                sub_states = ", ".join(
                    f"{_short(sj.get('job_type', ''), 'job_type_')}={_short(sj.get('status'))}"
                    for sj in (job.get("sub_jobs") or [])
                )
                detail = f" reason={reason!r}"
                if sub_states:
                    detail += f" sub_jobs=[{sub_states}]"
                raise RuntimeError(f"cortex job {self.job_id} reached terminal state '{state}';{detail}")
            time.sleep(delay)
            delay = _next_delay(delay)
        raise TimeoutError(f"cortex job {self.job_id} did not become running within {self.config.job_ready_timeout}s")

    def _capture_sub_jobs(self) -> dict[str, str]:
        sub_jobs: dict[str, str] = {}
        for sub in self._job().get("sub_jobs", []) or []:
            sub_jobs[_short(sub.get("job_type"), "job_type_")] = str(sub["sub_job_id"])
        for role in _SUB_JOB_KEY.values():
            sub_jobs.setdefault(role, f"{self.job_id}:{role}:0")
        return sub_jobs

    # ── HTTP + auth ──────────────────────────────────────────────────────────
    @property
    def _prefix(self) -> str:
        cx = self.config
        base = (cx.base_url or f"https://{cx.host}").rstrip("/")
        return f"{base}/api/v2/databases/{cx.database}/schemas/{cx.schema_}/{cx.endpoint}"

    def _auth_headers(self) -> dict[str, str]:
        cx = self.config
        if cx.base_url is not None:  # local/dev host: no PAT auth
            return {}
        return {  # config validated the PAT is present for host/PAT auth
            "Authorization": f"Bearer {cx.pat.get_secret_value()}",
            "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
        }

    def _job(self) -> dict:
        return self.call(get_job_request(self.job_id))

    async def _send(self, method: str, url: str, *, retry_on=_is_transient, **kwargs: Any) -> dict:
        # Every SnowAPI call goes through here so transient 429/5xx/connection blips
        # are retried with exponential-jitter backoff (the neutrino client's policy).
        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
                headers=self._auth_headers(),
            )
        session = self._session

        async def attempt() -> dict:
            async with session.request(method, url, **kwargs) as resp:
                if resp.status >= 400:
                    await _raise_for_status(resp)
                body = await resp.read()
                # `:cancel` answers 200 with an empty body. Decoding that as JSON
                # would report an already-completed release as a failure.
                return json.loads(body) if body.strip() else {}

        retryer = AsyncRetrying(
            retry=retry_if_exception(retry_on),
            stop=stop_after_attempt(1 + self.max_retries),
            wait=wait_exponential_jitter(initial=0.5, max=10.0),
            reraise=True,
        )
        return await retryer(attempt)

    async def _close_session(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def _next_delay(delay: float) -> float:
    """Poll backoff: 1.25x growth capped at 6s."""
    return min(delay * 1.25, 6.0)


def _poll_progress(status: dict, chunks: list[bytes], request_id: str) -> tuple[str, Any]:
    """Fold one poll response into an action, mutating `chunks` with any result chunks.

    Returns ``("drain", next_cursor)`` (more chunks queued — re-poll now),
    ``("done", inline_result)`` (finished; `chunks` holds any chunked result), or
    ``("wait", None)`` (still running — back off). Raises ``RuntimeError`` if the
    request ended failed.
    """
    chunks.extend(c for c in map(_result_chunk, status.get("events") or []) if c is not None)
    if status.get("next_cursor"):
        return "drain", status["next_cursor"]
    state = _short(status.get("status"))
    if state in _REQUEST_DONE:
        return "done", status.get("result") or {}
    if state in _REQUEST_FAILED:
        raise RuntimeError(f"cortex request {request_id} ended '{state}': {status.get('error', '')}")
    return "wait", None


def _short(status: Any, prefix: str = "request_state_") -> str:
    """Full enum names (``REQUEST_STATE_DONE``/``JOB_STATE_RUNNING``) -> short form."""
    text = str(status or "").lower()
    return text.removeprefix(prefix).removeprefix("job_state_")


def _result_chunk(event: Any) -> bytes | None:
    if not isinstance(event, dict) or event.get("type") != "result_chunk":
        return None
    payload = base64.b64decode(event["payload_b64"])
    expected = event.get("payload_sha256")
    if expected and hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError("cortex result_chunk payload_sha256 mismatch")
    return payload


def _decode_result(result: dict) -> dict:
    """Decode a small inline result: a base64 DSSST1 frame, else pass-through JSON."""
    if isinstance(result, dict) and result.get("wire_format") == wire.WIRE_FORMAT_VERSION:
        return wire.loads(base64.b64decode(result["payload_b64"]))
    return result
