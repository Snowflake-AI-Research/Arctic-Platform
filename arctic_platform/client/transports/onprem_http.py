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
"""Blocking HTTP transport (on-prem). Tensor bodies use the shared DSSST1 codec."""

from __future__ import annotations

import time

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
    """Blocking HTTP over the shared DSSST1 wire. Serves onprem (local/remote)."""

    def __init__(self, config: ArcticRLClientConfig) -> None:
        super().__init__(config)
        import requests

        self.base_url = f"http://{config.backend_config.host}:{config.backend_config.port}"
        self.timeout = config.request_timeout
        self.session = requests.Session()
        self.proc = None
        if config.backend_config.launch_local_server:
            self._launch_server()

    def _start(self, payload: dict) -> JobId:
        resp = self.session.post(f"{self.base_url}/initialize", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["job_id"]

    def call(self, request: Request) -> dict:
        # Tensor-bearing ops (and generate) send a DSSST1 octet body; rest send JSON.
        octet = request.binary or request.op in _OCTET_OPS
        payload = (
            {"data": wire.dumps(request.body), "headers": {"Content-Type": "application/octet-stream"}}
            if octet
            else {"json": request.body}
        )
        params = {} if request.job_id is None else {"job_id": request.job_id}
        resp = self.session.post(f"{self.base_url}/{request.op}", params=params, timeout=self.timeout, **payload)
        resp.raise_for_status()
        # Tensor-bearing responses come back as DSSST1 octet; everything else is JSON.
        if "application/octet-stream" in resp.headers.get("Content-Type", ""):
            return wire.loads(resp.content)
        return resp.json()

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
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()

    def _is_running(self, job_id: JobId) -> bool:
        resp = self.session.get(f"{self.base_url}/job/{job_id}", timeout=self.timeout)
        return resp.ok and resp.json().get("status") == "RUNNING"

    def _launch_server(self) -> None:
        import subprocess
        import sys

        cfg = self.config
        cmd = [
            sys.executable,
            "-m",
            "arctic_platform.rl.http_server",
            "--host",
            "0.0.0.0",
            "--port",
            str(cfg.backend_config.port),
            "--training-gpus",
            str(cfg.training_gpus),
            "--sampling-gpus",
            str(cfg.sampling_gpus),
            "--log-prob-gpus",
            str(cfg.log_prob_gpus),
        ]
        if cfg.backend_config.colocate:
            cmd.append("--colocate")
        self.proc = subprocess.Popen(cmd)
        self._poll(
            lambda: self.session.get(f"{self.base_url}/health", timeout=self.timeout).ok,
            cfg.backend_config.startup_timeout,
            "server",
        )

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
