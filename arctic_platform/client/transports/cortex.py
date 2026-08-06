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
import hashlib
import os
import time
from typing import Any

import requests
from tenacity import AsyncRetrying
from tenacity import Retrying
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential_jitter
from urllib3.exceptions import NewConnectionError

from arctic_platform import wire
from arctic_platform.client.config import ArcticRLClientConfig
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


class CortexTransport(Transport):
    def __init__(self, config: ArcticRLClientConfig) -> None:
        self.config = config
        self.jobs = JobHandles()
        self.job_id: str | None = None
        self.request_timeout = config.request_timeout
        self.max_retries = config.backend_config.max_retries
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
            created = self._send("POST", self._prefix, retry_on=_is_connect_error, json={"sub_job_configs": self._sub_job_configs()})
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
        if self.job_id is not None:
            # GS uses colon-action syntax: /{job_id}:cancel
            self.session.post(f"{self._prefix}/{self.job_id}:cancel", timeout=self.request_timeout)

    # ── deliver one op: submit + poll to completion ──────────────────────────
    def call(self, request: Request) -> dict:
        result = self._poll(self._submit(request))
        # generate returns token ids as DSSST1 tensors; on-prem returns plain
        # lists, so match that contract.
        return _to_python(result) if request.op == "generate" else result

    async def acall(self, request: Request) -> dict:
        result = await self._apoll(await self._asubmit(request))
        return _to_python(result) if request.op == "generate" else result

    def _submit(self, request: Request) -> str:
        # Same shape as on-prem's call: build the url, then pick the wire. Octet ops
        # (forward-backward, generate) go DSSST1; the rest post JSON as-is. The client
        # assembles the full /operation envelope (incl. sub-job routing hints), so
        # `operation` is just another JSON post. `forward`/`log-probs` don't exist here.
        op = request.op
        body = {k: v for k, v in request.body.items() if v is not None}
        url = f"{self._prefix}/{self.job_id}/{op}"
        if op in _OCTET_OPS:
            return self._submit_octet(url, op, body)
        if op in ("step", "save", "operation"):
            return self._send("POST", url, json=body)["request_id"]
        raise NotImplementedError(f"cortex has no {op}")

    def _submit_octet(self, url: str, op: str, body: dict) -> str:
        frame = wire.dumps(body, metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}})
        final: dict = {}
        for chunk in wire.encode_byte_chunks(frame, kind="request", operation=op, max_bytes=_MAX_OCTET_BYTES):
            final = self._send("POST", url, data=chunk, headers={"Content-Type": "application/octet-stream"})
        return final["request_id"]

    async def _asubmit(self, request: Request) -> str:
        op = request.op
        body = {k: v for k, v in request.body.items() if v is not None}
        url = f"{self._prefix}/{self.job_id}/{op}"
        if op in _OCTET_OPS:
            return await self._asubmit_octet(url, op, body)
        if op in ("step", "save", "operation"):
            return (await self._asend("POST", url, json=body))["request_id"]
        raise NotImplementedError(f"cortex has no {op}")

    async def _asubmit_octet(self, url: str, op: str, body: dict) -> str:
        frame = wire.dumps(body, metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}})
        final: dict = {}
        for chunk in wire.encode_byte_chunks(frame, kind="request", operation=op, max_bytes=_MAX_OCTET_BYTES):
            final = await self._asend("POST", url, data=chunk, headers={"Content-Type": "application/octet-stream"})
        return final["request_id"]

    def _poll(self, request_id: str) -> dict:
        deadline = time.monotonic() + self.poll_timeout
        delay = self.poll_interval
        chunks: list[bytes] = []
        cursor: str | None = None
        while time.monotonic() < deadline:
            status = self._send("GET", f"{self._prefix}/{self.job_id}/requests/{request_id}", params={"cursor": cursor} if cursor else None)
            state = _short(status.get("status"))
            chunks.extend(c for c in map(_result_chunk, status.get("events") or []) if c is not None)
            if status.get("next_cursor"):
                cursor = status["next_cursor"]
                continue  # drain remaining result chunks before backing off
            if state in _REQUEST_DONE:
                if chunks:
                    return wire.decode_result_chunks(chunks)
                return _decode_result(status.get("result") or {})
            if state in _REQUEST_FAILED:
                raise RuntimeError(f"cortex request {request_id} ended '{state}': {status.get('error', '')}")
            time.sleep(delay)
            delay = min(delay * 1.25, 6.0)
        raise TimeoutError(f"cortex request {request_id} did not complete within {self.poll_timeout}s")

    async def _apoll(self, request_id: str) -> dict:
        deadline = time.monotonic() + self.poll_timeout
        delay = self.poll_interval
        chunks: list[bytes] = []
        cursor: str | None = None
        while time.monotonic() < deadline:
            status = await self._asend(
                "GET", f"{self._prefix}/{self.job_id}/requests/{request_id}", params={"cursor": cursor} if cursor else None
            )
            state = _short(status.get("status"))
            chunks.extend(c for c in map(_result_chunk, status.get("events") or []) if c is not None)
            if status.get("next_cursor"):
                cursor = status["next_cursor"]
                continue  # drain remaining result chunks before backing off
            if state in _REQUEST_DONE:
                if chunks:
                    return wire.decode_result_chunks(chunks)
                return _decode_result(status.get("result") or {})
            if state in _REQUEST_FAILED:
                raise RuntimeError(f"cortex request {request_id} ended '{state}': {status.get('error', '')}")
            await asyncio.sleep(delay)
            delay = min(delay * 1.25, 6.0)
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
            delay = min(delay * 1.25, 6.0)
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
        cx = cfg.backend_config
        base = (cx.base_url or f"https://{cx.host}").rstrip("/")
        return f"{base}/api/v2/databases/{cx.database}/schemas/{cx.schema_}/{cx.endpoint}"

    def _auth_headers(self) -> dict[str, str]:
        cx = self.config.backend_config
        if cx.base_url is not None:  # local/dev host: no PAT auth
            return {}
        return {
            "Authorization": f"Bearer {os.environ[cx.pat_env_var]}",
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
            resp.raise_for_status()
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
                resp.raise_for_status()
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
