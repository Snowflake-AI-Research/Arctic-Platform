# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Blocking HTTP transport (onprem/dss) + tensor codec."""

from __future__ import annotations

import time
from typing import Any

from arctic_platform.client.config import ArcticRLClientConfig
from arctic_platform.client.config import JobId
from arctic_platform.client.transport import JOB_TYPES
from arctic_platform.client.transports.onprem import OnPremTransport


def _dumps(obj: Any) -> bytes:
    import io

    import torch

    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _loads(data: bytes) -> Any:
    import io

    import torch

    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


class HttpTransport(OnPremTransport):
    """Blocking HTTP + tensor codec. Serves onprem (local/remote) and dss."""

    def __init__(self, config: ArcticRLClientConfig) -> None:
        super().__init__(config)
        import requests

        self.base_url = f"http://{config.host}:{config.port}"
        self.session = requests.Session()
        self.proc = None
        if config.launch_local_server:
            self._launch_server()

    def _start(self, payload: dict) -> JobId:
        resp = self.session.post(f"{self.base_url}/initialize", json=payload)
        resp.raise_for_status()
        return resp.json()["job_id"]

    def _rpc(self, op: str, job_id: JobId, body: dict) -> dict:
        resp = self.session.post(
            f"{self.base_url}/{op}",
            params={"job_id": job_id},
            data=_dumps(body),
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        # Tensor-bearing ops (fwd-bwd/fwd-no-grad) reply with torch-pickled bytes;
        # everything else (step, sync-weights, ...) replies with JSON.
        if "application/octet-stream" in resp.headers.get("Content-Type", ""):
            return _loads(resp.content)
        return resp.json()

    def _destroy(self, job_id: JobId, job_type: str) -> None:
        try:
            self.session.post(f"{self.base_url}/destroy", params={"job_id": job_id}, json={"job_type": job_type})
        except Exception:
            pass

    def _wait_running(self) -> None:
        for job_type in JOB_TYPES:
            job_id = getattr(self.jobs, job_type)
            if job_id is not None:
                self._poll(lambda jid=job_id: self._is_running(jid), self.config.job_ready_timeout, f"{job_type} {job_id}")

    def shutdown(self) -> None:
        super().shutdown()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()

    def _is_running(self, job_id: JobId) -> bool:
        resp = self.session.get(f"{self.base_url}/job/{job_id}", timeout=5)
        return resp.ok and resp.json().get("status") == "RUNNING"

    def _launch_server(self) -> None:
        import subprocess
        import sys

        cfg = self.config
        cmd = [
            sys.executable, "-m", "arctic_platform.rl.http_server",
            "--host", "0.0.0.0", "--port", str(cfg.port),
            "--training-gpus", str(cfg.training_gpus),
            "--sampling-gpus", str(cfg.sampling_gpus),
            "--log-prob-gpus", str(cfg.log_prob_gpus),
        ]
        if cfg.colocate:
            cmd.append("--colocate")
        self.proc = subprocess.Popen(cmd)
        self._poll(lambda: self.session.get(f"{self.base_url}/health", timeout=3).ok, cfg.startup_timeout, "server")

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
