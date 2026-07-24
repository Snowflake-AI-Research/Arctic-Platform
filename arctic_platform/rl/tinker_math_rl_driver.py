# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Thin entry point that runs ``tinker_cookbook.recipes.math_rl.train`` with a
non-default httpx timeout.

The Tinker SDK ships a 60 s HTTP timeout (see
``tinker._constants.DEFAULT_TIMEOUT``), which is fine for the hosted service
but too short for a self-hosted backend doing multi-GPU ZeRO-3 forward +
backward passes on a 2048-datum batch. We patch the constant *before*
``tinker`` is imported anywhere else and then hand off to the unmodified
recipe. Set ``TINKER_HTTP_TIMEOUT`` (seconds) to tune; default 1800.
"""

from __future__ import annotations

import asyncio
import os

import httpx

_TIMEOUT = float(os.environ.get("TINKER_HTTP_TIMEOUT", "1800"))
_TIMEOUT_OBJ = httpx.Timeout(timeout=_TIMEOUT, connect=10.0)

# ``tinker._base_client`` does ``from ._constants import DEFAULT_TIMEOUT`` at
# import time, so patching only ``tinker._constants`` won't rebind the name
# already captured on ``_base_client``. Poke both -- and also monkey-patch
# httpx's underlying constructors so any future rebindings still land on our
# generous timeout.
import tinker._constants as _tinker_constants  # noqa: E402
import tinker._base_client as _tinker_base_client  # noqa: E402

_tinker_constants.DEFAULT_TIMEOUT = _TIMEOUT_OBJ
_tinker_base_client.DEFAULT_TIMEOUT = _TIMEOUT_OBJ

_orig_httpx_async_init = httpx.AsyncClient.__init__


def _patched_httpx_async_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    kwargs["timeout"] = _TIMEOUT_OBJ
    return _orig_httpx_async_init(self, *args, **kwargs)


httpx.AsyncClient.__init__ = _patched_httpx_async_init  # type: ignore[method-assign]

# Import *after* the patch so the constant binds correctly wherever it's read.
import chz  # noqa: E402
from tinker_cookbook.recipes.math_rl.train import CLIConfig, cli_main  # noqa: E402


def main() -> None:
    config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(config))


if __name__ == "__main__":
    main()
