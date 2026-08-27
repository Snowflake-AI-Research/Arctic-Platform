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

SnowAPI is async: every op submits and returns a ``request_id`` that is polled to
completion. So an op is just submit + poll -> final result dict, the same
contract the on-prem transports expose. `call` runs it over ``requests``; `acall`
runs the identical flow over ``aiohttp`` for the async client. The only Cortex
specifics live in `_submit`, because SnowAPI is not uniform: forward-backward and
generate carry DSSST1 octet bodies (byte-chunked), while step/save/operation post
their JSON body as-is (the client assembles the full `/operation` envelope, incl.
sub-job routing). Unsupported ops (`forward`, `log-probs`) raise NotImplementedError.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import time
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
from arctic_platform.client.cortex_batch import is_cortex_shaped
from arctic_platform.client.cortex_batch import lower_fwd_bwd_batch
from arctic_platform.client.transport import JOB_TYPES
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transport import Request
from arctic_platform.client.transport import Transport

_MAX_OCTET_BYTES = 60 * 1024 * 1024  # matches the SnowAPI per-request cap
# HTTP statuses worth retrying: the request was well-formed, so the same call may
# succeed once transient load/infra clears (429 rate-limit, 5xx, plus 404/409 seen
# while GS/ZMD restart). Other 4xx are client errors that won't fix themselves.
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504, 404, 409}
# SnowAPI requires DSSST1 octet bodies for these ops. This is a wire requirement of
# the endpoint, not payload binary-ness (generate carries no tensors), so it lives in
# the transport rather than on Request.binary.
_OCTET_OPS = {"forward-backward", "generate"}
_OCTET_HEADERS = {"Content-Type": "application/octet-stream"}
# The chunk envelope's operation label is the wire's own name, not our canonical op name.
_WIRE_OPERATION = {"forward-backward": "fwd-bwd", "generate": "generate"}
# forward-backward always carries a chunk envelope, even when the frame would fit in one
# request: its chunk_group_id is the server's idempotency key, so a bare frame is off
# contract. generate posts the bare frame when it fits.
_FORCE_CHUNK_OPS = {"forward-backward"}
# Ops whose chunked frame can be re-posted from scratch on a chunk-group error.
# forward-backward carries the large gradient frame; a mid-stream chunk-group
# desync (GS restart) is recoverable only by re-posting the whole group.
_GROUP_RESTART_OPS = {"forward-backward"}
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


class _ChunkGroupError(Exception):
    """A SnowAPI chunk-group error (missing/conflict/restart) — never retried per-chunk."""

    def __init__(self, detail: dict) -> None:
        super().__init__(detail.get("code", "chunk_group_error"))
        self.detail = detail


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


def _is_chunk_post_transient(exc: BaseException) -> bool:
    """`_is_transient`, but never retry a single chunk on a chunk-group error — the
    whole group must be re-posted instead (see `_submit_octet`)."""
    if (
        isinstance(exc, requests.exceptions.HTTPError)
        and _chunk_group_detail(_response_json(exc.response)) is not None
    ):
        return False
    return _is_transient(exc)


# Cortex reports as top-level fields what on-prem reports inside `metrics`
# (avg_loss on forward-backward; last_lr / global_steps on step). Mirror them in
# so callers written against the on-prem shape work unchanged. Purely additive:
# the top-level keys stay put, so callers reading those are unaffected.
_METRIC_MIRRORS = {
    "forward-backward": (("avg_loss", "loss"),),
    "step": (("last_lr", "last_lr"), ("global_steps", "global_steps")),
}


def _mirror_metrics(op: str, result: Any) -> Any:
    mirrors = _METRIC_MIRRORS.get(op)
    if not mirrors or not isinstance(result, dict):
        return result
    metrics = dict(result.get("metrics") or {})
    for source, target in mirrors:
        if source in result and target not in metrics:
            metrics[target] = result[source]
    result["metrics"] = metrics
    return result


def _zero_logprobs(body: dict) -> dict:
    """Stand in for the `/forward` op Cortex does not have.

    Correct *only* for single-epoch on-policy GRPO without KL: server-side `grpo`
    defaults π_old to `logprobs.detach()`, so the caller's copy is never read as a
    ratio denominator and `approx_kl` / `clip_ratio` come back 0. Any recipe that
    genuinely consumes these -- a KL penalty, or `ppo_epochs > 1`, where π_old
    must be the pre-update policy -- has to be refused before it reaches here;
    see the verl adapter's Cortex preflight.
    """
    import torch

    tensors = body.get("batch") if isinstance(body.get("batch"), dict) else body
    ids = tensors.get("input_ids") if isinstance(tensors, dict) else None
    rows, cols = (int(ids.shape[0]), int(ids.shape[-1])) if torch.is_tensor(ids) else (1, 1)
    zeros = torch.zeros((rows, max(cols, 1)), dtype=torch.float32)
    # verl reads log_probs/entropy, SkyRL reads logprobs/entropies.
    return {"batch": {"logprobs": zeros, "log_probs": zeros, "entropy": zeros, "entropies": zeros}}


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
    ``response`` is preserved so retry / chunk-group inspection still works.
    """
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        body = (resp.text or "").strip()
        if not body:
            raise
        raise requests.exceptions.HTTPError(f"{exc}: {body[:2000]}", response=resp) from None


def _response_json(resp: requests.Response | None) -> Any:
    if resp is None:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


async def _araise_for_status(resp: Any) -> None:
    """`_raise_for_status` for the aiohttp path.

    Stays a `ClientResponseError` (carrying `status`) so `_is_transient_async` keeps
    classifying it, but folds the body into the message.
    """
    import aiohttp

    body = (await resp.text()).strip()
    message = f"{resp.reason}: {body[:2000]}" if body else (resp.reason or "")
    raise aiohttp.ClientResponseError(
        resp.request_info,
        resp.history,
        status=resp.status,
        message=message,
        headers=resp.headers,
    )


async def _aread_json(resp: Any) -> Any:
    try:
        return await resp.json(content_type=None)
    except Exception:
        return None


class CortexTransport(Transport):
    def __init__(self, config: ArcticClientConfig) -> None:
        self.config = config
        self.jobs = JobHandles()
        self.job_id: str | None = None
        self.request_timeout = config.request_timeout
        self.max_retries = config.backend.max_retries
        self.poll_interval = 0.5
        self.poll_timeout = config.job_ready_timeout
        self.session = self._build_session()
        self._asession = None  # aiohttp.ClientSession, lazy on first acall
        self._asession_loop = None  # the event loop that session is bound to

    # ── lifecycle ──────────────────────────────────────────────────────────
    def initialize(self) -> JobHandles:
        cfg = self.config
        reconnect = JobHandles.from_config(cfg)
        if reconnect.any_set:  # reattach; a sub-job token embeds its parent job id
            token = next(t for t in (reconnect.training, reconnect.sampling, reconnect.log_prob) if t is not None)
            self.job_id = str(token).split(":", 1)[0]
        else:
            # A mutating create: only retry when the request provably never landed,
            # so we can't spawn duplicate jobs (matches the neutrino client).
            created = self._send(
                "POST", self._prefix, retry_on=_is_connect_error, json={"sub_job_configs": self._sub_job_configs()}
            )
            self.job_id = created["job_id"]
        self._wait_running()
        sub_jobs = self._capture_sub_jobs()
        # JobHandles holds each role's sub-job token, so the client's op bodies
        # (e.g. weight-sync source/target) already carry Cortex-correct ids with no
        # transport-side rewrite -- exactly as on-prem does with plain job ids.
        for role in JOB_TYPES:
            if cfg.gpus_for(role) > 0:
                self.jobs.set(role, sub_jobs[_SUB_JOB_KEY[role]])
        return self.jobs

    def shutdown(self) -> None:
        if self.job_id is None:
            return
        # GS uses colon-action syntax: /{job_id}:cancel. Best-effort teardown routed
        # through _send (retry + auth), tolerating failure (the job may be gone).
        with contextlib.suppress(requests.exceptions.RequestException):
            self._send("POST", f"{self._prefix}/{self.job_id}:cancel")

    # ── deliver one op: submit + poll to completion ──────────────────────────
    def call(self, request: Request) -> dict:
        if request.op in _NOOP_OPS:
            return {}
        if request.op == "forward":
            return _zero_logprobs(request.body)
        result = self._poll(self._submit(request))
        # generate returns token ids as DSSST1 tensors; on-prem returns plain
        # lists, so match that contract.
        if request.op == "generate":
            return _to_python(result)
        return _mirror_metrics(request.op, result)

    async def acall(self, request: Request) -> dict:
        if request.op in _NOOP_OPS:
            return {}
        if request.op == "forward":
            return _zero_logprobs(request.body)
        result = await self._apoll(await self._asubmit(request))
        if request.op == "generate":
            return _to_python(result)
        return _mirror_metrics(request.op, result)

    def _op_target(self, request: Request) -> tuple[str, dict]:
        """The url + JSON body for one op (None-valued keys dropped).

        forward-backward is additionally lowered from the verl-GRPO
        ``{batch, meta}`` contract onto Cortex's RPC frame, so callers holding a
        verl-shaped batch don't each need their own reshape. Callers that already
        built the RPC frame (the standalone recipes) pass through untouched.
        """
        body = {k: v for k, v in request.body.items() if v is not None}
        if request.op == "forward-backward" and not is_cortex_shaped(body):
            body = lower_fwd_bwd_batch(body)
        return (f"{self._prefix}/{self.job_id}/{request.op}", body)

    def _submit(self, request: Request) -> str | dict:
        # Same shape as on-prem's call: build the url, then pick the wire. Octet ops
        # (forward-backward, generate) go DSSST1; the rest post JSON as-is. The client
        # assembles the full /operation envelope (incl. sub-job routing hints), so
        # `operation` is just another JSON post. `forward`/`log-probs` don't exist here.
        op = request.op
        url, body = self._op_target(request)
        if op in _OCTET_OPS:
            return self._submit_octet(url, op, body)
        if op in ("step", "save", "operation"):
            return _submitted(op, body, self._send("POST", url, json=body))
        raise NotImplementedError(f"cortex has no {op}")

    def _octet_chunks(self, body: dict, op: str) -> list[bytes]:
        frame = wire.dumps(body, metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}})
        return list(
            wire.encode_byte_chunks(
                frame,
                kind="request",
                operation=_WIRE_OPERATION[op],
                max_bytes=_MAX_OCTET_BYTES,
                force_chunk=op in _FORCE_CHUNK_OPS,
            )
        )

    def _submit_octet(self, url: str, op: str, body: dict) -> str:
        # Post the frame chunk-by-chunk. Transient blips retry per-chunk; a
        # chunk-group desync (only forward-backward) re-posts the whole group once.
        chunks = self._octet_chunks(body, op)
        allow_restart = op in _GROUP_RESTART_OPS
        retry_on = _is_chunk_post_transient if allow_restart else _is_transient
        restarts = idx = 0
        final: dict = {}
        while idx < len(chunks):
            try:
                final = self._send("POST", url, retry_on=retry_on, data=chunks[idx], headers=_OCTET_HEADERS)
            except requests.exceptions.HTTPError as exc:
                detail = _chunk_group_detail(_response_json(exc.response))
                if not (allow_restart and restarts < 1 and detail and detail["code"] == _CHUNK_GROUP_RESTART_REQUIRED):
                    raise
                restarts, idx, final = restarts + 1, 0, {}
                continue
            idx += 1
        return final["request_id"]

    async def _asubmit(self, request: Request) -> str | dict:
        op = request.op
        url, body = self._op_target(request)
        if op in _OCTET_OPS:
            return await self._asubmit_octet(url, op, body)
        if op in ("step", "save", "operation"):
            return _submitted(op, body, await self._asend("POST", url, json=body))
        raise NotImplementedError(f"cortex has no {op}")

    async def _asubmit_octet(self, url: str, op: str, body: dict) -> str:
        chunks = self._octet_chunks(body, op)
        allow_restart = op in _GROUP_RESTART_OPS
        restarts = idx = 0
        final: dict = {}
        while idx < len(chunks):
            try:
                final = await self._apost_octet_chunk(url, chunks[idx], allow_restart=allow_restart)
            except _ChunkGroupError as exc:
                if not (restarts < 1 and exc.detail["code"] == _CHUNK_GROUP_RESTART_REQUIRED):
                    raise
                restarts, idx, final = restarts + 1, 0, {}
                continue
            idx += 1
        return final["request_id"]

    async def _apost_octet_chunk(self, url: str, chunk: bytes, *, allow_restart: bool) -> dict:
        # aiohttp's ClientResponseError drops the body, so read it here to spot a
        # chunk-group error (surfaced as _ChunkGroupError, never retried per-chunk).
        session = await self._ensure_asession()

        async def attempt() -> dict:
            async with session.post(url, data=chunk, headers=_OCTET_HEADERS) as resp:
                if resp.status >= 400:
                    detail = _chunk_group_detail(await _aread_json(resp)) if allow_restart else None
                    if detail is not None:
                        raise _ChunkGroupError(detail)
                    resp.raise_for_status()
                return await resp.json(content_type=None)

        retryer = AsyncRetrying(
            retry=retry_if_exception(_is_transient_async),
            stop=stop_after_attempt(1 + self.max_retries),
            wait=wait_exponential_jitter(initial=0.5, max=10.0),
            reraise=True,
        )
        return await retryer(attempt)

    def _request_url(self, request_id: str, cursor: str | None) -> tuple[str, dict | None]:
        url = f"{self._prefix}/{self.job_id}/requests/{request_id}"
        return url, ({"cursor": cursor} if cursor else None)

    def _poll(self, submitted: str | dict) -> dict:
        # Inline ops (_INLINE_OPERATION_TYPES) return their body from POST.
        if isinstance(submitted, dict):
            return submitted
        request_id = submitted
        deadline = time.monotonic() + self.poll_timeout
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
        raise TimeoutError(f"cortex request {request_id} did not complete within {self.poll_timeout}s")

    async def _apoll(self, submitted: str | dict) -> dict:
        if isinstance(submitted, dict):
            return submitted  # inline op — see _poll
        request_id = submitted
        deadline = time.monotonic() + self.poll_timeout
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
        raise TimeoutError(f"cortex request {request_id} did not complete within {self.poll_timeout}s")

    def _wait_running(self) -> None:
        deadline = time.monotonic() + self.poll_timeout
        delay = self.poll_interval
        while time.monotonic() < deadline:
            state = _short(self._job().get("status"))
            if state == "running":
                return
            if state in _JOB_TERMINAL:
                raise RuntimeError(f"cortex job {self.job_id} reached terminal state '{state}'")
            time.sleep(delay)
            delay = _next_delay(delay)
        raise TimeoutError(f"cortex job {self.job_id} did not become running within {self.poll_timeout}s")

    def _capture_sub_jobs(self) -> dict[str, str]:
        job = self._job()
        job = job.get("job", job)
        sub_jobs: dict[str, str] = {}
        for sub in job.get("sub_jobs", []) or []:
            sub_jobs[_short(sub.get("job_type"), "job_type_")] = str(sub["sub_job_id"])
        for role in _SUB_JOB_KEY.values():
            sub_jobs.setdefault(role, f"{self.job_id}:{role}:0")
        return sub_jobs

    # ── create-job body (SubJobConfig wire shape) ────────────────────────────
    def _sub_job_configs(self) -> list[dict]:
        return self.config.to_cortex()

    # ── HTTP + auth ──────────────────────────────────────────────────────────
    @property
    def _prefix(self) -> str:
        cfg = self.config
        cx = cfg.backend
        base = (cx.base_url or f"https://{cx.host}").rstrip("/")
        return f"{base}/api/v2/databases/{cx.database}/schemas/{cx.schema_}/{cx.endpoint}"

    def _auth_headers(self) -> dict[str, str]:
        cx = self.config.backend
        if cx.base_url is not None:  # local/dev host: no PAT auth
            return {}
        return {  # config validated resolve_pat() is present for host/PAT auth
            "Authorization": f"Bearer {cx.resolve_pat()}",
            "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
        }

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(self._auth_headers())
        return session

    def _job(self) -> dict:
        return self._send("GET", f"{self._prefix}/{self.job_id}")

    def _send(self, method: str, url: str, *, retry_on=None, **kwargs: Any) -> dict:
        # Every SnowAPI call goes through here so transient 429/5xx/connection blips
        # are retried with exponential-jitter backoff (the neutrino client's policy).
        def attempt() -> dict:
            resp = self.session.request(method, url, timeout=self.request_timeout, **kwargs)
            _raise_for_status(resp)
            return resp.json()

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
                timeout=aiohttp.ClientTimeout(total=self.request_timeout),
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
                return await resp.json(content_type=None)

        retryer = AsyncRetrying(
            retry=retry_if_exception(retry_on or _is_transient_async),
            stop=stop_after_attempt(1 + self.max_retries),
            wait=wait_exponential_jitter(initial=0.5, max=10.0),
            reraise=True,
        )
        return await retryer(attempt)

    async def aclose(self) -> None:
        # Only the loop that owns the session can close it; on any other loop just
        # drop the reference (matches the on-prem HTTP transport).
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


def _to_python(obj: Any) -> Any:
    import torch

    if torch.is_tensor(obj):
        return obj.cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj
