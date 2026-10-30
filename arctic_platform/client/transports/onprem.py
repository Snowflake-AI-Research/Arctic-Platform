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
"""On-prem transport over in-process Ray actor calls."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading

from arctic_platform.client.config import ArcticClientConfig
from arctic_platform.client.config import JobId
from arctic_platform.client.config import OnPremConfig
from arctic_platform.client.transport import CLUSTER_OPS
from arctic_platform.client.transport import JOB_CREATE_ORDER
from arctic_platform.client.transport import JOB_TYPES
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transport import Request
from arctic_platform.client.transport import Transport
from arctic_platform.client.transport import method_name
from arctic_platform.client.transport import unresolved_ops

logger = logging.getLogger(__name__)

# The server answers generate inline, so it has no request to poll or cancel.
_UNSUPPORTED_OPS = frozenset({"generate-stream", "request-status", "cancel-request"})
# Cancel in-flight ops, then wait for the loop thread to exit, before close().
_LOOP_SHUTDOWN_TIMEOUT_S = 30


class OnPremTransport(Transport):
    """The server splits into a ``state`` actor (job creation) and an
    ``ArcticRLRayServer`` wrapper (typed async ops) that snapshots workers at
    construction, so the wrapper is built after jobs are initialized.

    `call` and `acall` both submit the actor coroutine to the private loop.
    `call` blocks on that future. `acall` awaits it, so a server
    ``asyncio.Lock`` is only used from that one loop.

    Reconnect: when handed an existing ``server_state`` actor (plus reconnect
    job ids on the config) the transport reattaches to the already-running
    server instead of creating one, and never tears it down.
    """

    def __init__(self, config: OnPremConfig, server_state: object | None = None) -> None:
        self.config = config
        self.jobs = JobHandles()
        self._reconnect = server_state is not None
        self._state = server_state
        self._server = None  # ArcticRLRayServer, built once jobs exist
        # One long-lived loop, driven by its OWN thread, for every op. TRL's AsyncGRPOTrainer calls in from two
        # threads concurrently -- the background rollout worker (generate / engine_old_log_probs) and the main
        # trainer thread (forward_backward, weight_sync) -- so a per-caller `run_until_complete` on a shared loop
        # races and raises "This event loop is already running". Running the loop in a dedicated thread and
        # submitting via `run_coroutine_threadsafe` makes dispatch safe for any number of caller threads.
        #
        # Training RPCs on this loop await under ``ArcticRLRayServer._training_op_lock`` (#104).
        # A blocking gather on the loop thread stalls generate for the whole step.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = self._start_loop_thread(self._loop)

    @staticmethod
    def _start_loop_thread(loop: asyncio.AbstractEventLoop) -> threading.Thread:
        thread = threading.Thread(target=loop.run_forever, name="ray-transport-loop", daemon=True)
        thread.start()
        return thread

    # The whole client (this transport included) is cloudpickled when verl hands
    # a reconnected backend to the ArcticLLMServer Ray actor. An event loop is not
    # picklable, so drop it on pickle and rebuild a fresh one on the receiving
    # side. The Ray handles (_state, _server) pickle fine and are what actually
    # addresses the running server.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_loop", None)
        state.pop("_loop_thread", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._loop = asyncio.new_event_loop()
        self._loop_thread = self._start_loop_thread(self._loop)

    def initialize(self, client_config: ArcticClientConfig) -> JobHandles:
        if not self._reconnect:
            from arctic_platform.rl.ray_server import create_arctic_rl_ray_server_state

            self._state = create_arctic_rl_ray_server_state(
                training_gpus=client_config.training_gpus,
                sampling_gpus=client_config.sampling_gpus,
                log_prob_gpus=client_config.log_prob_gpus,
                log_prob_engine=client_config.sampling.log_prob_engine,
                colocate=self.config.colocate,
            )
        reconnect = JobHandles.from_config(client_config)
        if reconnect.any_set:
            self.jobs = reconnect
            self._build_server()
            return self.jobs
        try:
            for job_type in JOB_CREATE_ORDER:
                if client_config.gpus_for(job_type) > 0:
                    self.jobs.set(job_type, self._start(client_config.to_onprem(job_type)))
            self._build_server()
        except Exception:
            # Tear down partial jobs so GPUs are not orphaned.
            try:
                self.shutdown()
            except Exception:
                logger.exception("cleanup after failed initialize() also failed")
            raise
        return self.jobs

    def _build_server(self) -> None:
        from arctic_platform.rl.ray_server import ArcticRLRayServer

        self._server = ArcticRLRayServer(self._state)
        missing = [op for op in unresolved_ops(self._server) if op not in _UNSUPPORTED_OPS]
        if missing:
            logger.warning("%s cannot resolve ops (they will fail if called): %s", type(self).__name__, missing)

    def get_server_state(self) -> object:
        """The Ray state actor, for reattaching from another process."""
        return self._state

    def _start(self, payload: dict) -> JobId:
        import ray

        return ray.get(self._state.initialize.remote(payload))["job_id"]

    def _resolve(self, request: Request):
        if request.op in CLUSTER_OPS:
            raise NotImplementedError(f"on-prem does not support cluster operation {request.op!r}")
        if request.op in _UNSUPPORTED_OPS:
            raise NotImplementedError(f"on-prem does not support job operation {request.op!r}")
        method = getattr(self._server, method_name(request.op))
        # Pass job_id only when set (ops omitting it call method(body)), then the body.
        args = (request.body,) if request.job_id is None else (request.job_id, request.body)
        return method, args

    def _submit(self, coro) -> concurrent.futures.Future:
        """Schedule ``coro`` on the private loop.

        Blocking on the returned future from the loop thread deadlocks: the
        loop cannot run ``coro`` until this call returns. ``acall`` awaits the
        future from the caller's loop instead of blocking this thread.
        """
        if threading.current_thread() is self._loop_thread:
            coro.close()
            raise RuntimeError(
                "OnPremTransport._submit was called from the transport loop thread; "
                "waiting on that future would deadlock. Await the server method on this loop instead."
            )
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _run_on_loop(self, coro):
        """Block until ``coro`` finishes on the private loop."""
        return self._submit(coro).result()

    def _cancel_pending(self) -> None:
        """Cancel tasks still on the private loop, then wait until they finish."""
        loop = self._loop

        async def _cancel() -> None:
            current = asyncio.current_task()
            pending = [task for task in asyncio.all_tasks(loop) if task is not current]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            self._submit(_cancel()).result(timeout=_LOOP_SHUTDOWN_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            logger.warning("in-flight Ray transport ops were still running at shutdown", exc_info=True)

    def call(self, request: Request) -> dict:
        method, args = self._resolve(request)
        return self._submit(method(*args)).result()

    async def acall(self, request: Request) -> dict:
        method, args = self._resolve(request)
        return await asyncio.wrap_future(self._submit(method(*args)))

    def shutdown(self) -> None:
        # In reconnect mode the jobs/server are owned by the driver process;
        # a reattached transport must not tear them down. Handles are cleared
        # after destroy so a second call (transport + client guard) is a no-op.
        if threading.current_thread() is self._loop_thread:
            raise RuntimeError("OnPremTransport.shutdown cannot run on the transport loop thread")
        if not self._reconnect:
            for job_type in JOB_TYPES:
                job_id = getattr(self.jobs, job_type)
                if job_id is not None:
                    self._run_on_loop(self._server.destroy(job_id, job_type))
            self.jobs = JobHandles()
        # Cancel in-flight ops before stopping, and close the loop only after its
        # thread has actually exited. close() on a still-running loop raises.
        thread = self._loop_thread
        if thread.is_alive():
            self._cancel_pending()
            self._loop.call_soon_threadsafe(self._loop.stop)
            thread.join(timeout=_LOOP_SHUTDOWN_TIMEOUT_S)
            if thread.is_alive():
                raise RuntimeError(
                    "Ray transport loop thread did not exit within "
                    f"{_LOOP_SHUTDOWN_TIMEOUT_S:.0f}s; refusing to close a running event loop"
                )
        if not self._loop.is_closed():
            self._loop.close()
