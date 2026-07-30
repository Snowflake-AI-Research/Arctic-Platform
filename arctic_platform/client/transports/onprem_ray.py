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

    `_rpc` resolves ``op -> method`` and forwards the request unchanged.
    """

    def __init__(self, config: ArcticRLClientConfig, *, server_state: object | None = None) -> None:
        super().__init__(config)
        from arctic_platform.rl.ray_server import create_arctic_rl_ray_server_state

        # Reconnect: forwarder workers (verl) and Ray-remote entrypoints (SkyRL)
        # rebuild the client in a different process from the driver. They pass
        # the driver's state actor back in via `server_state=` so we reattach
        # instead of spinning up fresh Ray actors that would race the existing
        # jobs. Non-reconnect path: build a fresh state actor as before.
        if server_state is not None:
            self._state = server_state
        else:
            self._state = create_arctic_rl_ray_server_state(
                training_gpus=config.training_gpus,
                sampling_gpus=config.sampling_gpus,
                log_prob_gpus=config.log_prob_gpus,
                log_prob_engine="deepspeed",
                colocate=config.colocate,
            )
        self._server = None  # ArcticRLRayServer, built once jobs exist
        # One long-lived loop for every op instead of asyncio.run() per call.
        self._loop = asyncio.new_event_loop()

    def _start(self, payload: dict) -> JobId:
        import ray

        return ray.get(self._state.initialize.remote(payload))["job_id"]

    def _wait_running(self) -> None:
        # Build the wrapper after initialize(): it snapshots training_workers.
        from arctic_platform.rl.ray_server import ArcticRLRayServer

        self._server = ArcticRLRayServer(self._state)
        self._check_op_coverage(self._server)

    def _rpc(self, request: Request) -> dict:
        # Pass job_id only when set (some ops, e.g. sync-weights, omit it), then the body.
        method = getattr(self._server, method_name(request.op))
        args = (request.body,) if request.job_id is None else (request.job_id, request.body)
        return self._loop.run_until_complete(method(*args))

    def _destroy(self, job_id: JobId, job_type: str) -> None:
        self._loop.run_until_complete(self._server.destroy(job_id, job_type))

    def shutdown(self) -> None:
        super().shutdown()
        self._loop.close()

    def get_server_state(self) -> object:
        """Expose the state actor for reconnect flows.

        The SkyRL entrypoint runs the driver-side client in the parent
        process, then dispatches training to a Ray remote task with a
        `reconnect_config` and this server_state so the worker can reattach
        to the same actors. Non-Ray transports return None from the client
        wrapper `ArcticRLClient.get_server_state`.
        """
        return self._state
