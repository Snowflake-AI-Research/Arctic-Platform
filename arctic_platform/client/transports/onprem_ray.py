# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""In-process Ray transport — no HTTP, no serialization."""

from __future__ import annotations

from arctic_platform.client.config import ArcticRLClientConfig
from arctic_platform.client.config import JobId
from arctic_platform.client.transports.onprem import OnPremTransport


class RayTransport(OnPremTransport):
    """In-process Ray actor calls — no HTTP, no serialization.

    Prototype: only the ops the on-prem example exercises are wired
    (initialize / fwd-bwd / step / destroy). The real server splits into a
    ``state`` actor (job creation) and an ``ArcticRLRayServer`` wrapper (typed
    async ops) that snapshots workers at construction, so the wrapper is built
    lazily after jobs are initialized.

    ============================ TODO: ALIGN TRANSPORTS ============================
    The whole point of this unified client is that the transport is a DUMB
    forwarder: the client emits one canonical Request(op, target, body) and the
    transport hands (op, job_id, body) straight to the matching server method
    with NO per-op arg/data rewiring. HttpTransport achieves this because the
    HTTP server exposes a uniform POST /{op}?job_id=<id> surface. This Ray path
    does NOT: ArcticRLRayServer is a bag of typed async methods with per-op
    signatures (fwd_bwd(job_id, batch), step(job_id), sync_weights(request), ...),
    so we are forced into the op-by-op if/else in `_rpc` below.

    GOAL: make every backend (ray-onprem, http-onprem, cortex) accept the exact
    same args/kwargs per op so no transport has to rewire anything. Align this
    to CORTEX's contract EXACTLY (see transports/cortex.py) — treat it as the
    source of truth for each op's arg/kwarg shape. The right fix lives server-side:
    give ArcticRLRayServer a single uniform entrypoint (e.g.
    `async def call(op, job_id, body)`) mirroring the HTTP /{op} surface, so this
    `_rpc` collapses back to a one-liner and the per-op arg binding lives once on
    the server instead of being duplicated into each transport.
    ==============================================================================
    """

    def __init__(self, config: ArcticRLClientConfig) -> None:
        super().__init__(config)
        from arctic_platform.rl.ray_server import create_arctic_rl_ray_server_state

        self._state = create_arctic_rl_ray_server_state(
            training_gpus=config.training_gpus,
            sampling_gpus=config.sampling_gpus,
            log_prob_gpus=config.log_prob_gpus,
            log_prob_engine="deepspeed",
            colocate=config.colocate,
        )
        self._server = None  # ArcticRLRayServer, built once jobs exist

    def _start(self, payload: dict) -> JobId:
        import ray

        return ray.get(self._state.initialize.remote(payload))["job_id"]

    def _wait_running(self) -> None:
        # Build the wrapper after initialize(): it snapshots training_workers.
        from arctic_platform.rl.ray_server import ArcticRLRayServer

        self._server = ArcticRLRayServer(self._state)

    def _rpc(self, op: str, job_id: JobId, body: dict) -> dict:
        import asyncio

        if op == "fwd-bwd":
            coro = self._server.fwd_bwd(job_id, body)
        elif op == "step":
            coro = self._server.step(job_id)
        else:
            raise NotImplementedError(f"RayTransport (prototype) does not wire op {op!r}")
        return asyncio.run(coro)

    def _destroy(self, job_id: JobId, job_type: str) -> None:
        import asyncio

        asyncio.run(self._server.destroy(job_id, job_type))
