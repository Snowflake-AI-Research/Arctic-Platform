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
"""HTTP transport (on-prem). Sync via ``requests``; async via ``aiohttp``. Tensor bodies use DSSST1."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from arctic_platform import wire
from arctic_platform.client.config import ArcticRLClientConfig
from arctic_platform.client.config import JobId
from arctic_platform.client.transport import JOB_TYPES
from arctic_platform.client.transport import Request
from arctic_platform.client.transports.onprem import OnPremTransport

# Ops the server wants as DSSST1 octet even without tensors in the body: a wire
# requirement of the endpoint (matching Cortex/SnowAPI), not payload binary-ness.
_OCTET_OPS = frozenset({"generate"})


class HttpTransport(OnPremTransport):
    """HTTP over the shared DSSST1 wire. Serves onprem (local/remote)."""

    def __init__(self, config: ArcticRLClientConfig) -> None:
        super().__init__(config)
        import requests

        self.base_url = f"http://{config.backend_config.host}:{config.backend_config.port}"
        self.timeout = config.request_timeout
        self.session = requests.Session()
        self._asession = None  # aiohttp.ClientSession, lazy on first acall
        self._asession_loop = None  # the event loop that session is bound to
        self.proc = None
        if config.backend_config.launch_local_server:
            self._launch_server()

    def _start(self, payload: dict) -> JobId:
        resp = self.session.post(f"{self.base_url}/initialize", json=payload, timeout=self.timeout)
        if not resp.ok:
            # Surface the server-side validation/error body (e.g. 422 detail);
            # raise_for_status alone hides it, which makes remote debugging hard.
            raise RuntimeError(f"/initialize failed ({resp.status_code}): {resp.text}")
        return resp.json()["job_id"]

    def _http_args(self, request: Request) -> tuple[str, dict[str, Any]]:
        """URL + post kwargs shared by ``requests`` and ``aiohttp``."""
        url = f"{self.base_url}/{request.op}"
        params = {} if request.job_id is None else {"job_id": request.job_id}
        if request.binary or request.op in _OCTET_OPS:
            return url, {
                "params": params,
                "data": wire.dumps(request.body),
                "headers": {"Content-Type": "application/octet-stream"},
            }
        return url, {"params": params, "json": request.body}

    def call(self, request: Request) -> dict:
        from arctic_platform.common.utils import sft_profile

        with sft_profile.timed("serialize"):
            url, kwargs = self._http_args(request)
        with sft_profile.timed("rpc"):
            resp = self.session.post(url, timeout=self.timeout, **kwargs)
            resp.raise_for_status()
            if "application/octet-stream" in resp.headers.get("Content-Type", ""):
                result = wire.loads(resp.content)
            else:
                result = resp.json()
        if sft_profile.enabled() and request.op in ("forward-backward", "forward", "step"):
            # Client-side buckets only; server attaches its own under metrics._profile_ms.
            sft_profile.maybe_print(f"client {request.op}")
            sft_profile.take_last()
        return result

    async def _ensure_asession(self):
        # A ClientSession is bound to the loop it's built on; reuse it only on
        # that same loop. On a new loop (e.g. a fresh asyncio.run) we abandon the
        # stale one -- it can't be awaited closed from here -- and rebuild.
        loop = asyncio.get_running_loop()
        if self._asession is None or self._asession.closed or self._asession_loop is not loop:
            import aiohttp

            self._asession = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=aiohttp.TCPConnector(limit=0),
            )
            self._asession_loop = loop
        return self._asession

    async def acall(self, request: Request) -> dict:
        session = await self._ensure_asession()
        url, kwargs = self._http_args(request)
        async with session.post(url, **kwargs) as resp:
            resp.raise_for_status()
            if "application/octet-stream" in resp.headers.get("Content-Type", ""):
                return wire.loads(await resp.read())
            return await resp.json()

    async def aclose(self) -> None:
        # Only the loop that owns the session can close it; on any other loop
        # (a stale session from a prior loop) just drop the reference.
        if self._asession is not None:
            if not self._asession.closed and self._asession_loop is asyncio.get_running_loop():
                await self._asession.close()
            self._asession = None
            self._asession_loop = None

    def _destroy(self, job_id: JobId, job_type: str) -> None:
        self.session.post(
            f"{self.base_url}/destroy", params={"job_id": job_id}, json={"job_type": job_type}, timeout=self.timeout
        )

    def _wait_running(self) -> None:
        for job_type in JOB_TYPES:
            job_id = getattr(self.jobs, job_type)
            if job_id is not None:
                self._poll(
                    lambda jid=job_id: self._is_running(jid), self.config.job_ready_timeout, f"{job_type} {job_id}"
                )

    def shutdown(self) -> None:
        super().shutdown()
        self._terminate_server()

    def _terminate_server(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
        self.proc = None

    def _is_running(self, job_id: JobId) -> bool:
        resp = self.session.get(f"{self.base_url}/job/{job_id}", timeout=self.timeout)
        return resp.ok and resp.json().get("status") == "RUNNING"

    def _launch_server(self) -> None:
        import os
        import subprocess
        import sys

        cfg = self.config
        bc = cfg.backend_config
        cmd = [
            sys.executable,
            "-m",
            "arctic_platform.common.http_server",
            "--host",
            "0.0.0.0",
            "--port",
            str(bc.port),
            "--training-gpus",
            str(cfg.training_gpus),
            "--sampling-gpus",
            str(cfg.sampling_gpus),
            "--log-prob-gpus",
            str(cfg.log_prob_gpus),
        ]
        if bc.colocate:
            cmd.append("--colocate")
        env = os.environ.copy()
        if bc.server_cuda_visible_devices is not None:
            # Client may run with CUDA_VISIBLE_DEVICES= (empty); give the
            # server subprocess an explicit GPU list so Ray workers see devices.
            env["CUDA_VISIBLE_DEVICES"] = bc.server_cuda_visible_devices
        self.proc = subprocess.Popen(cmd, env=env)
        try:
            self._poll(
                lambda: self.session.get(f"{self.base_url}/health", timeout=self.timeout).ok,
                bc.startup_timeout,
                "server",
            )
        except Exception:
            # Health timeout runs in __init__ (before client init guards); kill the orphan.
            self._terminate_server()
            raise

    @staticmethod
    def _poll(pred, timeout: float, what: str) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if pred():
                    return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(f"Timed out waiting for {what} after {timeout}s")
