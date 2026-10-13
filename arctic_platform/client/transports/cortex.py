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
transport only executes those sends, so sync and async differ only at the HTTP
call. Cortex-specific request and response shapes are translated here until the
servers share one API. Log-probs is unsupported.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import Iterator

import requests
from tenacity import AsyncRetrying
from tenacity import Retrying
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential_jitter
from urllib3.exceptions import NewConnectionError

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
# Neutrino never colocates training and sampling on the same GPUs, so it exposes no
# sleep/wake surface. They resolve to no-ops here rather than errors, which keeps
# shared client flows like sync_weights() (wake → operation → wake → reset) portable.
_NOOP_OPS = {"sleep-inference", "wake-inference", "sleep-training", "wake-training"}
# /operation is polymorphic: async ops (weight-sync, ...) return {request_id: ...} and we
# poll to completion; inline ops answer with the finished body directly. Whitelisting the
# inline set lets us tell the two apart without silently dropping either kind of response.
_INLINE_OPERATION_TYPES = {
    "bootstrap-router-replay",
    "cancel-request",
    "reset-prefix-cache",
    "router-replay-discard",
    "tail-logs",
}
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


def _query(request: Request, body: dict) -> list[Send]:
    return [{"params": body}]


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


def _capacity_result(value: dict) -> dict:
    """All six capacity fields; proto3 JSON omits false and zero values."""
    return {
        "has_reservation": bool(value.get("has_reservation", False)),
        "max_total_gpus": int(value.get("max_total_gpus", 0)),
        "reserved_gpus": int(value.get("reserved_gpus", 0)),
        "in_use_gpus": int(value.get("in_use_gpus", 0)),
        "pending_gpus": int(value.get("pending_gpus", 0)),
        "available_gpus": int(value.get("available_gpus", 0)),
    }


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


@dataclass(frozen=True)
class _Route:
    method: str
    # Under the prefix. `{job}` is the parent of request.job_id; other fields come from the body.
    path: str
    encode: Callable[[Request, dict], list[Send]] = _json
    poll: bool = False
    # False: retry only sends that provably never reached the server.
    replay_safe: bool = True
    # Re-post the whole chunk group once when the server reports it lost track of it.
    restartable: bool = False
    normalize: Callable[[Any], Any] = _identity

    def prepare(self, prefix: str, request: Request) -> tuple[str, list[Send]]:
        body = {k: v for k, v in request.body.items() if v is not None}
        job = None if request.job_id is None else _parent_id(request.job_id)
        return prefix + self.path.format_map(body | {"job": job}), self.encode(request, body)


_ROUTES = {
    # A replayed create would spawn a second job.
    "create-job": _Route("POST", "", replay_safe=False),
    "list-jobs": _Route("GET", "", _query, normalize=lambda reply: reply.get("jobs", [])),
    "get-job": _Route("GET", "/{job}", _no_body),
    "cancel-job": _Route("POST", "/{job}:cancel", _no_body),
    "capacity": _Route("GET", "/capacity", _query, normalize=_capacity_result),
    "list-checkpoints": _Route(
        "GET", "/{job}/checkpoints", _no_body, normalize=lambda reply: reply.get("checkpoints", [])
    ),
    "export-checkpoint": _Route("POST", "/{job}/checkpoints/{checkpoint_id}:export", _no_body),
    # A replayed delete answers 404 for a checkpoint the first send already removed.
    "delete-checkpoint": _Route("DELETE", "/{job}/checkpoints/{checkpoint_id}", _no_body, replay_safe=False),
    # forward-backward always carries a chunk envelope, even when the frame fits: its
    # chunk_group_id is the server's idempotency key. A mid-stream chunk-group desync
    # (GS restart) is recoverable only by re-posting the whole group.
    "forward-backward": _Route(
        "POST",
        "/{job}/forward-backward",
        _octet("fwd-bwd", force_chunk=True),
        poll=True,
        restartable=True,
        normalize=_fwd_bwd_result,
    ),
    # SnowAPI requires DSSST1 octet bodies here even though generate carries no tensors.
    "generate": _Route("POST", "/{job}/generate", _octet("generate"), poll=True, normalize=_to_python),
    "forward": _Route("POST", "/{job}/operation", _forward, poll=True, normalize=_unwrap_forward_payload),
    "step": _Route("POST", "/{job}/step", poll=True),
    "save": _Route("POST", "/{job}/save", poll=True),
    "operation": _Route("POST", "/{job}/operation", poll=True),
}


def _route(op: str) -> _Route:
    if op not in _ROUTES:
        raise NotImplementedError(f"cortex has no {op}")
    return _ROUTES[op]


def _is_transient(exc: BaseException) -> bool:
    """Retry connection/timeout errors and transient HTTP statuses (safe for reads)."""
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        resp = exc.response
        return resp is not None and resp.status_code in _TRANSIENT_STATUSES
    return False


def _is_transient_async(exc: BaseException) -> bool:
    """`_is_transient` for the aiohttp path (different exception hierarchy)."""
    import aiohttp

    if isinstance(exc, (aiohttp.ClientConnectionError, asyncio.TimeoutError)):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in _TRANSIENT_STATUSES
    return False


def _is_connect_error(exc: BaseException) -> bool:
    """Only failures proving the request never reached the server (safe for mutating POSTs)."""
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        cause = exc.args[0] if exc.args else None
        reason = getattr(cause, "reason", None)
        return isinstance(cause, NewConnectionError) or isinstance(reason, NewConnectionError)
    return False


def _is_connect_error_async(exc: BaseException) -> bool:
    """Async failures proving a mutating request was never delivered."""
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


def _submitted(op: str, body: dict, response: dict) -> str | dict:
    """A poll handle for an async op, or the finished result for an inline one.

    SnowAPI's contract on ``/operation`` is "poll if the response carries a
    ``request_id``, else consume it inline". Only operations in
    ``_INLINE_OPERATION_TYPES`` are allowed to answer inline; anything else must
    return a ``request_id`` and be polled to completion.
    """
    request_id = response.get("request_id")
    if request_id is not None:
        return str(request_id)
    operation_type = body.get("operation_type")
    if operation_type in _INLINE_OPERATION_TYPES:
        return response
    raise RuntimeError(f"cortex {operation_type or op} response carried no request_id: {response}")


def _raise_for_status(resp: requests.Response) -> None:
    """``raise_for_status()`` that keeps the server's error body in the message.

    GS explains a 4xx in the response body; the stock HTTPError shows only the
    status line, which turns an actionable rejection into a guessing game. The
    ``response`` is preserved so retry inspection still works. A chunk-group
    error raises `_ChunkGroupError` instead.
    """
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        body = (resp.text or "").strip()
        detail = _chunk_group_detail(body)
        if detail is not None:
            raise _ChunkGroupError(detail) from exc
        if not body:
            raise
        raise requests.exceptions.HTTPError(f"{exc}: {body[:2000]}", response=resp) from None


async def _araise_for_status(resp: Any) -> None:
    """`_raise_for_status` for the aiohttp path.

    Stays a `ClientResponseError` (carrying `status`) so `_is_transient_async` keeps
    classifying it, but folds the body into the message.
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


class CortexTransport(Transport):
    def __init__(self, config: CortexConfig) -> None:
        self.config = config
        self.jobs = JobHandles()
        self.job_id: str | None = None
        self.max_retries = config.max_retries
        self.poll_interval = 0.5
        self.session = self._build_session()
        self._asession = None  # aiohttp.ClientSession, lazy on first acall
        self._asession_loop = None  # the event loop that session is bound to

    # ── lifecycle ──────────────────────────────────────────────────────────
    def initialize(self, client_config: ArcticClientConfig) -> JobHandles:
        reconnect = JobHandles.from_config(client_config)
        if reconnect.any_set:  # reattach; a sub-job token embeds its parent job id
            self.job_id = _parent_id(reconnect.require_any())
        else:
            create = Request("create-job", body={"sub_job_configs": client_config.to_cortex()})
            self.job_id = self.call(create)["job_id"]
        self._wait_running()
        sub_jobs = self._capture_sub_jobs()
        # JobHandles holds each role's sub-job token, so the client's op bodies
        # (e.g. weight-sync source/target) already carry Cortex-correct ids with no
        # transport-side rewrite -- exactly as on-prem does with plain job ids.
        for role in JOB_TYPES:
            if client_config.gpus_for(role) > 0:
                self.jobs.set(role, sub_jobs[_SUB_JOB_KEY[role]])
        return self.jobs

    def shutdown(self) -> None:
        if self.job_id is None:
            return
        # Best-effort teardown, tolerating failure (the job may be gone).
        with contextlib.suppress(requests.exceptions.RequestException):
            self.call(cancel_job_request(self.job_id))

    # ── deliver one op ───────────────────────────────────────────────────────
    def call(self, request: Request) -> Any:
        if request.op in _NOOP_OPS:
            return {}
        route = _route(request.op)
        url, sends = route.prepare(self._prefix, request)
        reply = self._send_group(route, url, sends)
        if route.poll:
            reply = self._poll(_submitted(request.op, request.body, reply))
        return route.normalize(reply)

    async def acall(self, request: Request) -> Any:
        if request.op in _NOOP_OPS:
            return {}
        route = _route(request.op)
        url, sends = route.prepare(self._prefix, request)
        reply = await self._asend_group(route, url, sends)
        if route.poll:
            reply = await self._apoll(_submitted(request.op, request.body, reply))
        return route.normalize(reply)

    def _send_group(self, route: _Route, url: str, sends: list[Send]) -> dict:
        try:
            return self._send_each(route, url, sends)
        except _ChunkGroupError as exc:
            if not (route.restartable and exc.restart_required):
                raise
        return self._send_each(route, url, sends)

    def _send_each(self, route: _Route, url: str, sends: list[Send]) -> dict:
        """Post every send in order; the last reply is the op's answer."""
        retry_on = _is_transient if route.replay_safe else _is_connect_error
        reply: dict = {}
        for kwargs in sends:
            reply = self._send(route.method, url, retry_on=retry_on, **kwargs)
        return reply

    async def _asend_group(self, route: _Route, url: str, sends: list[Send]) -> dict:
        try:
            return await self._asend_each(route, url, sends)
        except _ChunkGroupError as exc:
            if not (route.restartable and exc.restart_required):
                raise
        return await self._asend_each(route, url, sends)

    async def _asend_each(self, route: _Route, url: str, sends: list[Send]) -> dict:
        retry_on = _is_transient_async if route.replay_safe else _is_connect_error_async
        reply: dict = {}
        for kwargs in sends:
            reply = await self._asend(route.method, url, retry_on=retry_on, **kwargs)
        return reply

    def _request_url(self, request_id: str, cursor: str | None) -> tuple[str, dict | None]:
        url = f"{self._prefix}/{self.job_id}/requests/{request_id}"
        return url, ({"cursor": cursor} if cursor else None)

    def _poll(self, submitted: str | dict) -> dict:
        # Inline ops (_INLINE_OPERATION_TYPES) return their body from POST.
        if isinstance(submitted, dict):
            return submitted
        request_id = submitted
        deadline = time.monotonic() + self.config.poll_timeout
        delay = self.poll_interval
        chunks: list[bytes] = []
        cursor: str | None = None
        while time.monotonic() < deadline:
            url, params = self._request_url(request_id, cursor)
            action, value = _poll_progress(self._send("GET", url, params=params), chunks, request_id)
            if action == "done":
                return value
            if action == "drain":
                cursor = value  # more result chunks queued; re-poll without backing off
                continue
            time.sleep(delay)
            delay = _next_delay(delay)
        raise TimeoutError(f"cortex request {request_id} did not complete within {self.config.poll_timeout}s")

    async def _apoll(self, submitted: str | dict) -> dict:
        if isinstance(submitted, dict):
            return submitted  # inline op — see _poll
        request_id = submitted
        deadline = time.monotonic() + self.config.poll_timeout
        delay = self.poll_interval
        chunks: list[bytes] = []
        cursor: str | None = None
        while time.monotonic() < deadline:
            url, params = self._request_url(request_id, cursor)
            action, value = _poll_progress(await self._asend("GET", url, params=params), chunks, request_id)
            if action == "done":
                return value
            if action == "drain":
                cursor = value
                continue
            await asyncio.sleep(delay)
            delay = _next_delay(delay)
        raise TimeoutError(f"cortex request {request_id} did not complete within {self.config.poll_timeout}s")

    def _wait_running(self) -> None:
        deadline = time.monotonic() + self.config.job_ready_timeout
        delay = self.poll_interval
        while time.monotonic() < deadline:
            job = self._job()
            state = _short(job.get("status"))
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

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(self._auth_headers())
        return session

    def _job(self) -> dict:
        return self.call(get_job_request(self.job_id))

    def _send(self, method: str, url: str, *, retry_on=None, **kwargs: Any) -> dict:
        # Every SnowAPI call goes through here so transient 429/5xx/connection blips
        # are retried with exponential-jitter backoff (the neutrino client's policy).
        def attempt() -> dict:
            resp = self.session.request(method, url, timeout=self.config.request_timeout, **kwargs)
            _raise_for_status(resp)
            # `:cancel` answers 200 with an empty body. Decoding that as JSON
            # would report an already-completed release as a failure.
            return resp.json() if resp.content.strip() else {}

        retryer = Retrying(
            retry=retry_if_exception(retry_on or _is_transient),
            stop=stop_after_attempt(1 + self.max_retries),
            wait=wait_exponential_jitter(initial=0.5, max=10.0),
            reraise=True,
        )
        return retryer(attempt)

    async def _ensure_asession(self):
        # A ClientSession is bound to the loop it's built on; reuse it only on that
        # same loop. On a new loop (e.g. a fresh asyncio.run) rebuild -- the stale
        # one can't be awaited closed from here.
        loop = asyncio.get_running_loop()
        if self._asession is None or self._asession.closed or self._asession_loop is not loop:
            import aiohttp

            self._asession = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
                headers=self._auth_headers(),
            )
            self._asession_loop = loop
        return self._asession

    async def _asend(self, method: str, url: str, *, retry_on=None, **kwargs: Any) -> dict:
        session = await self._ensure_asession()

        async def attempt() -> dict:
            async with session.request(method, url, **kwargs) as resp:
                if resp.status >= 400:
                    await _araise_for_status(resp)
                body = await resp.read()
                return json.loads(body) if body.strip() else {}

        retryer = AsyncRetrying(
            retry=retry_if_exception(retry_on or _is_transient_async),
            stop=stop_after_attempt(1 + self.max_retries),
            wait=wait_exponential_jitter(initial=0.5, max=10.0),
            reraise=True,
        )
        return await retryer(attempt)

    async def aclose(self) -> None:
        # Only the loop that owns the session can close it; on any other loop just
        # drop the reference.
        if self._asession is not None:
            if not self._asession.closed and self._asession_loop is asyncio.get_running_loop():
                await self._asession.close()
            self._asession = None
            self._asession_loop = None


def _next_delay(delay: float) -> float:
    """Poll backoff: 1.25x growth capped at 6s."""
    return min(delay * 1.25, 6.0)


def _poll_progress(status: dict, chunks: list[bytes], request_id: str) -> tuple[str, Any]:
    """Fold one poll response into an action, mutating `chunks` with any result chunks.

    Returns ``("drain", next_cursor)`` (more chunks queued — re-poll now),
    ``("done", result)`` (finished, decoded result), or ``("wait", None)`` (still
    running — back off). Raises ``RuntimeError`` if the request ended failed.
    """
    chunks.extend(c for c in map(_result_chunk, status.get("events") or []) if c is not None)
    if status.get("next_cursor"):
        return "drain", status["next_cursor"]
    state = _short(status.get("status"))
    if state in _REQUEST_DONE:
        return "done", (wire.decode_result_chunks(chunks) if chunks else _decode_result(status.get("result") or {}))
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
