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
"""In-process Ray transport — no HTTP, no serialization."""

from __future__ import annotations

import asyncio

from arctic_platform.client.config import ArcticRLClientConfig
from arctic_platform.client.config import JobId
from arctic_platform.client.transport import Request
from arctic_platform.client.transport import method_name
from arctic_platform.client.transports.onprem import OnPremTransport


class RayTransport(OnPremTransport):
    """In-process Ray actor calls — no HTTP, no serialization.

    The server splits into a ``state`` actor (job creation) and an
    ``ArcticRLRayServer`` wrapper (typed async ops) that snapshots workers at
    construction, so the wrapper is built lazily after jobs are initialized.

    `call` resolves ``op -> method`` and forwards the request unchanged; async
    callers use `acall`, which awaits the actor coroutine on the caller's loop.

    Reconnect: when handed an existing ``server_state`` actor (plus reconnect
    job ids on the config) the transport reattaches to the already-running
    server instead of creating one. Because the wrapper snapshots live workers
    at construction, in reconnect mode it is built immediately (the jobs already
    exist on the shared state actor).
    """

    def __init__(self, config: ArcticRLClientConfig, server_state: object | None = None) -> None:
        super().__init__(config)
        self._reconnect = server_state is not None
        if self._reconnect:
            self._state = server_state
        else:
            from arctic_platform.rl.ray_server import create_arctic_rl_ray_server_state

            self._state = create_arctic_rl_ray_server_state(
                training_gpus=config.training_gpus,
                sampling_gpus=config.sampling_gpus,
                log_prob_gpus=config.log_prob_gpus,
                log_prob_engine=config.sampling.log_prob_engine,
                colocate=config.backend_config.colocate,
            )
        self._server = None  # ArcticRLRayServer, built once jobs exist
        # One long-lived loop for every sync op instead of asyncio.run() per call.
        self._loop = asyncio.new_event_loop()

    # The whole client (this transport included) is cloudpickled when verl hands
    # a reconnected backend to the ArcticLLMServer Ray actor. An event loop is not
    # picklable, so drop it on pickle and rebuild a fresh one on the receiving
    # side. The Ray handles (_state, _server) pickle fine and are what actually
    # addresses the running server.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_loop", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._loop = asyncio.new_event_loop()

    def initialize(self):  # type: ignore[override]
        jobs = super().initialize()
        # Reconnect returns early from the base initialize() (job ids preset, so
        # _wait_running() is skipped) — build the server wrapper here since the
        # workers already exist on the shared state actor.
        if self._server is None:
            self._build_server()
        return jobs

    def _build_server(self) -> None:
        from arctic_platform.rl.ray_server import ArcticRLRayServer

        self._server = ArcticRLRayServer(self._state)
        self._check_op_coverage(self._server)

    def get_server_state(self) -> object:
        """The Ray state actor, for reattaching from another process."""
        return self._state

    def _start(self, payload: dict) -> JobId:
        import ray

        return ray.get(self._state.initialize.remote(payload))["job_id"]

    def _wait_running(self) -> None:
        # Fresh path: build the wrapper after initialize() so it snapshots the
        # freshly created training_workers.
        self._build_server()

    def _resolve(self, request: Request):
        method = getattr(self._server, method_name(request.op))
        # Pass job_id only when set (ops omitting it call method(body)), then the body.
        args = (request.body,) if request.job_id is None else (request.job_id, request.body)
        return method, args

    def _run_on_loop(self, coro):
        """Run ``coro`` on the private loop.

        Sync callers (``SyncArcticRLClient``, destroy from a non-async thread) use
        ``run_until_complete`` directly. Async callers (``ArcticRLClient.shutdown``
        invoked under verl's ``asyncio.run``) already have a running loop in this
        thread, so nesting ``run_until_complete`` raises; hop to a worker thread.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._loop.run_until_complete(coro)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(self._loop.run_until_complete, coro).result()

    def call(self, request: Request) -> dict:
        method, args = self._resolve(request)
        return self._run_on_loop(method(*args))

    async def acall(self, request: Request) -> dict:
        method, args = self._resolve(request)
        return await method(*args)

    def _destroy(self, job_id: JobId, job_type: str) -> None:
        self._run_on_loop(self._server.destroy(job_id, job_type))

    def shutdown(self) -> None:
        # In reconnect mode the jobs/server are owned by the driver process;
        # a reattached transport must not tear them down.
        if not self._reconnect:
            super().shutdown()
        self._loop.close()
