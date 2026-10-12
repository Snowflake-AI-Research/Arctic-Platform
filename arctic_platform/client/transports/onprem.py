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
import logging
import threading

from arctic_platform.client.config import ArcticClientConfig
from arctic_platform.client.config import JobId
from arctic_platform.client.transport import JOB_CREATE_ORDER
from arctic_platform.client.transport import JOB_TYPES
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transport import Request
from arctic_platform.client.transport import Transport
from arctic_platform.client.transport import method_name
from arctic_platform.client.transport import unresolved_ops

logger = logging.getLogger(__name__)


class OnPremTransport(Transport):
    """The server splits into a ``state`` actor (job creation) and an
    ``ArcticRLRayServer`` wrapper (typed async ops) that snapshots workers at
    construction, so the wrapper is built after jobs are initialized.

    `call` resolves ``op -> method`` and forwards the request unchanged; async
    callers use `acall`, which awaits the actor coroutine on the caller's loop.

    Reconnect: when handed an existing ``server_state`` actor (plus reconnect
    job ids on the config) the transport reattaches to the already-running
    server instead of creating one, and never tears it down.
    """

    def __init__(self, config: ArcticClientConfig, server_state: object | None = None) -> None:
        self.config = config
        self.jobs = JobHandles()
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
                colocate=config.backend.colocate,
            )
        self._server = None  # ArcticRLRayServer, built once jobs exist
        # One long-lived loop, driven by its OWN thread, for every op. TRL's AsyncGRPOTrainer calls in from two
        # threads concurrently -- the background rollout worker (generate / engine_old_log_probs) and the main
        # trainer thread (forward_backward, weight_sync) -- so a per-caller `run_until_complete` on a shared loop
        # races and raises "This event loop is already running". Running the loop in a dedicated thread and
        # submitting via `run_coroutine_threadsafe` makes dispatch safe for any number of caller threads.
        #
        # That overlap is only safe together with Arctic-Platform #104. #104 stops blocking this loop on
        # training RPCs and holds one lock across every training-worker op (forward/step/save/load and also
        # weight_sync, sleep/wake, and empty-cache). Without #104, generation stalls for the whole training
        # step. With #104, a training op that is not under that lock can overlap a DP collective.
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

    def initialize(self) -> JobHandles:
        reconnect = JobHandles.from_config(self.config)
        if reconnect.any_set:
            self.jobs = reconnect
            self._build_server()
            return self.jobs
        cfg = self.config
        try:
            for job_type in JOB_CREATE_ORDER:
                if cfg.gpus_for(job_type) > 0:
                    self.jobs.set(job_type, self._start(cfg.to_onprem(job_type)))
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
        missing = unresolved_ops(self._server)
        if missing:
            logger.warning("%s cannot resolve ops (they will fail if called): %s", type(self).__name__, missing)

    def get_server_state(self) -> object:
        """The Ray state actor, for reattaching from another process."""
        return self._state

    def _start(self, payload: dict) -> JobId:
        import ray

        return ray.get(self._state.initialize.remote(payload))["job_id"]

    def _resolve(self, request: Request):
        method = getattr(self._server, method_name(request.op))
        # Pass job_id only when set (ops omitting it call method(body)), then the body.
        args = (request.body,) if request.job_id is None else (request.job_id, request.body)
        return method, args

    def _run_on_loop(self, coro):
        """Run ``coro`` on the private loop and block for its result.

        The loop runs forever in its own thread, so this works uniformly for
        every caller: sync callers (``ArcticRLClient`` / ``SyncArcticRLClient``,
        destroy from a non-async thread), concurrent sync callers (TRL's trainer
        thread and background rollout worker), and async callers
        (``AsyncArcticRLClient.shutdown`` under verl's ``asyncio.run``, whose
        loop is a different loop). ``run_coroutine_threadsafe`` schedules on the
        loop thread and ``.result()`` blocks the caller -- no thread drives the
        loop itself, so there is no "event loop is already running" race.

        Calling this from the loop thread deadlocks: ``.result()`` waits for a
        callback that cannot run until this call returns.
        """
        if threading.current_thread() is self._loop_thread:
            coro.close()
            raise RuntimeError(
                "OnPremTransport._run_on_loop was called from the transport loop thread; "
                "Future.result() would deadlock. Await the server method on this loop instead."
            )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

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
            asyncio.run_coroutine_threadsafe(_cancel(), loop).result(timeout=30)
        except Exception:
            logger.warning("in-flight Ray transport ops were still running at shutdown", exc_info=True)

    def call(self, request: Request) -> dict:
        method, args = self._resolve(request)
        return self._run_on_loop(method(*args))

    async def acall(self, request: Request) -> dict:
        method, args = self._resolve(request)
        return await method(*args)

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
            thread.join(timeout=30)
            if thread.is_alive():
                raise RuntimeError(
                    "Ray transport loop thread did not exit within 30s; refusing to close a running event loop"
                )
        if not self._loop.is_closed():
            self._loop.close()
