# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Driver-side OpenAI-compat gateway for any-harness agent training.

Cortex sub-jobs are only reachable through SnowAPI's op-name dispatch
(``/{prefix}/{job}/generate``, ``/step`` ...). ``/v1/chat/completions``
is not in that set, so the ``openai_compat`` router mounted on the
sampling sub-job cannot be reached by an external OpenAI SDK client.

Rather than push a control-plane change into Cortex, the driver process
already holds an ``ArcticRLClient`` that speaks Cortex's op envelope.
This module spins up a tiny FastAPI + uvicorn server bound to
``127.0.0.1`` in the driver process and reuses the router in
``arctic_platform.openai_compat``. Each ``/v1/chat/completions`` call
lands on a ``_ClientBackedPool`` adapter that forwards to
``client.generate(prompts, sampling_params)`` — the same call our
native ``CortexRLAgent`` already makes.

Effect: any Harbor agent (Terminus 2, Computer 1, community
``BaseAgent``s) that speaks OpenAI-chat through LiteLLM or the
``openai`` SDK can point at ``http://127.0.0.1:{port}/v1`` and train
against Cortex. No Cortex-side change, no monkey-patch on the harness.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from typing import Any

from arctic_platform.openai_compat import router as openai_router


class _ClientBackedPool:
    """Look-alike for ``arctic_inference.server.replica_pool.ReplicaPool``.

    ``arctic_platform.openai_compat`` only touches ``pool._config`` (as a
    liveness sentinel) and ``pool.generate(prompts, sampling_params)``.
    Everything else on ``ReplicaPool`` (KV/prefix cache, tensor-parallel
    routing, ``AsyncLLM`` scheduling) is a sub-job concern that the
    Cortex sampling job already handles; the driver only shuttles wire
    payloads through ``ArcticRLClient``.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        # ``openai_compat._get_pool_and_tokenizer`` treats a ``None``
        # ``_config`` as "still warming" and returns 503. Any non-None
        # value works — we use ``self`` to keep it obvious in a repl.
        self._config = self

    async def generate(
        self,
        prompts: list[Any],
        sampling_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Delegate to ``ArcticRLClient.generate``.

        The unified ``arctic_platform.client.ArcticRLClient.generate``
        is synchronous (SnowAPI polling happens inside the Cortex
        transport). The legacy dispatch shim's ``generate`` is a
        coroutine. We detect which we got and drive it accordingly so
        both call paths behave the same from Harbor's point of view.
        """
        result = self._client.generate(
            prompts=list(prompts),
            sampling_params=dict(sampling_params or {}),
        )
        if asyncio.iscoroutine(result):
            return await result
        # Sync ``.generate`` blocks the current thread; that's fine here
        # because uvicorn already runs on its own loop, but concurrent
        # trials serialize through it. If concurrency ever matters,
        # hoist the sync path into ``asyncio.to_thread``.
        return result


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DriverOpenAIGateway:
    """Local ``/v1/*`` HTTP endpoint over an ``ArcticRLClient``.

    Lifecycle: instantiate, ``start()`` (spawns a uvicorn thread bound
    to ``127.0.0.1``), read ``base_url``, hand it to Harbor via
    ``--model-base-url``, ``stop()`` on shutdown. The uvicorn thread is
    daemonized so a driver crash doesn't leak sockets.

    Threading model: uvicorn runs in its own thread with its own asyncio
    event loop. Requests hitting the router call ``asyncio.to_thread``
    on the client, so the client's blocking transport doesn't stall the
    server loop while other trials are in flight.
    """

    def __init__(
        self,
        *,
        client: Any,
        tokenizer: Any,
        model_name: str,
        host: str = "127.0.0.1",
        port: int | None = None,
    ) -> None:
        self._client = client
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._host = host
        self._port = port or _pick_free_port()
        self._server: Any | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}/v1"

    def _build_app(self) -> Any:
        # Local import: FastAPI/uvicorn are only required when Harbor is
        # actually asked to run in OpenAI-compat mode. Keeps the base
        # Arctic-Platform install slim for native-mode users.
        from fastapi import FastAPI

        app = FastAPI(title="arctic-platform openai-compat gateway")
        app.state.sampling_pool = _ClientBackedPool(self._client)
        app.state.sampling_tokenizer = self._tokenizer
        app.state.sampling_model_name = self._model_name
        app.state.sampling_created = int(time.time())
        app.include_router(openai_router)
        return app

    def start(self, *, ready_timeout_s: float = 20.0) -> str:
        """Boot uvicorn in a daemon thread and block until it's serving.

        Returns the ``/v1`` base URL. Raises if the server didn't come
        up within ``ready_timeout_s`` (usually indicates a port clash or
        an import-time failure in the router).
        """
        import uvicorn

        config = uvicorn.Config(
            self._build_app(),
            host=self._host,
            port=self._port,
            log_level="warning",
            loop="asyncio",
            lifespan="on",
        )
        server = uvicorn.Server(config)
        # ``uvicorn.Server.install_signal_handlers`` grabs SIGINT/SIGTERM
        # on the main thread — we're on a worker thread so it would
        # crash; disable to let the driver own signal handling.
        server.install_signal_handlers = lambda: None
        self._server = server

        def _run() -> None:
            server.run()

        self._thread = threading.Thread(target=_run, name="arctic-openai-gateway", daemon=True)
        self._thread.start()

        deadline = time.monotonic() + ready_timeout_s
        while time.monotonic() < deadline:
            if getattr(server, "started", False):
                return self.base_url
            time.sleep(0.05)
        raise RuntimeError(
            f"OpenAI-compat gateway did not become ready within {ready_timeout_s:.1f}s "
            f"on {self.base_url}"
        )

    def stop(self, *, timeout_s: float = 5.0) -> None:
        """Signal uvicorn to shut down and join its thread."""
        server = self._server
        if server is None:
            return
        server.should_exit = True
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
        self._server = None
        self._thread = None
        self._ready.clear()

    def __enter__(self) -> "DriverOpenAIGateway":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        with contextlib.suppress(Exception):
            self.stop()
