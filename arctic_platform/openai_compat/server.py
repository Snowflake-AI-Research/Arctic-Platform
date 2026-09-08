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
"""Serving shells around the router.

Two shapes, one app:

- :func:`build_app` / :class:`OpenAIGateway` for a driver that already holds a
  client and wants a ``/v1`` URL for the length of a run.
- ``python -m arctic_platform.openai_compat`` for the standalone case: attach
  to a sampling job that already exists and serve until interrupted.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import socket
import threading
import time
from pathlib import Path
from typing import Any

from arctic_platform._dependency_groups import require_any_dep_group
from arctic_platform.openai_compat.backend import DEFAULT_MAX_CONCURRENCY
from arctic_platform.openai_compat.backend import backend_for
from arctic_platform.openai_compat.errors import OpenAIError
from arctic_platform.openai_compat.errors import openai_error_handler
from arctic_platform.openai_compat.errors import unhandled_error_handler
from arctic_platform.openai_compat.router import GatewayState
from arctic_platform.openai_compat.router import router

logger = logging.getLogger(__name__)

_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


def build_app(
    *,
    backend: Any,
    tokenizer: Any,
    model_name: str,
    max_model_len: int,
    api_key: str | None = None,
) -> Any:
    """A FastAPI app serving ``/v1`` over ``backend``."""
    require_any_dep_group("openai")
    from fastapi import FastAPI

    app = FastAPI(title="arctic-platform OpenAI-compatible endpoint")
    app.state.openai_compat = GatewayState(
        backend=backend,
        tokenizer=tokenizer,
        model_name=model_name,
        max_model_len=max_model_len,
        api_key=api_key,
    )
    # Both handlers exist so that *every* failure leaves as OpenAI's envelope.
    # FastAPI's defaults render `{"detail": ...}`, which the openai SDK reports
    # as a bare status code with no message.
    app.add_exception_handler(OpenAIError, openai_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
    app.include_router(router)
    return app


def app_for_client(
    client: Any,
    *,
    tokenizer: Any,
    model_name: str,
    max_model_len: int,
    api_key: str | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> Any:
    return build_app(
        backend=backend_for(client, max_concurrency=max_concurrency),
        tokenizer=tokenizer,
        model_name=model_name,
        max_model_len=max_model_len,
        api_key=api_key,
    )


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def check_bind(host: str, api_key: str | None) -> None:
    """Refuse to expose an unauthenticated endpoint off-box.

    Binding beyond loopback is what lets a container or another host reach the
    gateway, and it is also what makes an open endpoint reachable. Requiring a
    key at that point is cheap; discovering later that anyone on the network
    could spend the job's GPUs is not.
    """
    if host in _LOOPBACK or api_key is not None:
        return
    raise ValueError(
        f"Refusing to bind {host} without an API key: the endpoint would accept unauthenticated requests from"
        " anywhere that can route to this host. Pass an api_key, or bind 127.0.0.1."
    )


class OpenAIGateway:
    """Run :func:`build_app` on a background thread for the life of a driver.

    Owns the server, never the job: ``stop()`` shuts down uvicorn and leaves
    the sampling job running. Tearing the job down here would cancel an
    endpoint the caller may still be using -- and on Cortex, ``client.shutdown()``
    cancels the whole parent job, GPUs included.
    """

    def __init__(
        self,
        *,
        client: Any = None,
        app: Any = None,
        tokenizer: Any = None,
        model_name: str = "",
        max_model_len: int = 0,
        host: str = "127.0.0.1",
        port: int | None = None,
        api_key: str | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        if app is None:
            if client is None:
                raise ValueError("OpenAIGateway needs either a built app or a client to build one from.")
            app = app_for_client(
                client,
                tokenizer=tokenizer,
                model_name=model_name,
                max_model_len=max_model_len,
                api_key=api_key,
                max_concurrency=max_concurrency,
            )
        # The app owns auth. Reading the key back off it (rather than trusting
        # the argument) means a pre-built app passed in with no key configured
        # still fails the bind check instead of being served wide open.
        check_bind(host, getattr(getattr(app.state, "openai_compat", None), "api_key", None))
        self._app = app
        self._host = host
        self._port = port or _pick_free_port()
        self._server: Any = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}/v1"

    def start(self, *, ready_timeout_s: float = 30.0) -> str:
        import uvicorn

        config = uvicorn.Config(self._app, host=self._host, port=self._port, log_level="warning", lifespan="on")
        server = uvicorn.Server(config)
        # uvicorn installs SIGINT/SIGTERM handlers, which only works on the main
        # thread; the driver keeps signal handling.
        server.install_signal_handlers = lambda: None
        self._server = server
        self._thread = threading.Thread(target=server.run, name="arctic-openai-compat", daemon=True)
        self._thread.start()

        deadline = time.monotonic() + ready_timeout_s
        while time.monotonic() < deadline:
            if getattr(server, "started", False):
                return self.base_url
            if not self._thread.is_alive():
                raise RuntimeError(f"OpenAI-compatible endpoint died during startup on {self.base_url}")
            time.sleep(0.02)
        raise RuntimeError(f"OpenAI-compatible endpoint was not ready within {ready_timeout_s:.1f}s")

    def stop(self, *, timeout_s: float = 10.0) -> None:
        server, thread = self._server, self._thread
        self._server = self._thread = None
        if server is None:
            return
        server.should_exit = True
        if thread is not None:
            thread.join(timeout=timeout_s)

    def __enter__(self) -> OpenAIGateway:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        with contextlib.suppress(Exception):
            self.stop()


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def main(argv: list[str] | None = None) -> None:
    """Attach to an existing sampling job and serve ``/v1`` until interrupted."""
    parser = argparse.ArgumentParser(
        prog="python -m arctic_platform.openai_compat",
        description="Serve an OpenAI-compatible endpoint over an Arctic sampling job.",
    )
    parser.add_argument("--config", required=True, type=Path, help="ArcticClientConfig JSON/YAML.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--api-key",
        default=None,
        help="Require this bearer token. Mandatory when --host is not loopback.",
    )
    parser.add_argument("--served-model-name", default=None, help="Name advertised at /v1/models.")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    check_bind(args.host, args.api_key)

    from transformers import AutoTokenizer

    from arctic_platform.client.config import ArcticClientConfig

    config = ArcticClientConfig.model_validate(_load_config(args.config))
    if config.sampling_job_id is None:
        raise SystemExit("--config must set sampling_job_id: this serves an endpoint, it does not create one.")

    from arctic_platform.client.base import ArcticClient

    client = ArcticClient(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    app = app_for_client(
        client,
        tokenizer=tokenizer,
        model_name=args.served_model_name or config.model_name,
        max_model_len=config.max_seq_len,
        api_key=args.api_key,
        max_concurrency=args.max_concurrency,
    )

    import uvicorn

    logger.info("Serving %s at http://%s:%s/v1", config.model_name, args.host, args.port)
    logger.info("The sampling job stays up when this process exits; tear it down with the Cortex CLI.")
    # Deliberately no client.shutdown() on exit: on Cortex that cancels the
    # parent job, which would take the endpoint (and its GPUs) down with the
    # gateway.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
