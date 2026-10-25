"""A real vLLM ``AsyncLLM`` over a restored semi-p engine (``SEMIP_FRONTEND=asyncllm``).

The restored child turns its in-process ``EngineCore`` into a ZMQ-served one
(``serve_core``), and an ``AsyncLLM`` here attaches to it the way vLLM attaches
to an engine it did not launch, through ``client_addresses``. Requests,
streaming, aborts, pause and resume, weight-sync RPCs and stats are then
vLLM's own; ``semip_frontend._SemiPAsyncLLMMixin`` keeps sleep, wake, teardown
and a watch on the child.

Imports vLLM at module scope, unlike ``semip_frontend``, so it is loaded only
when the flag is on.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from vllm.utils.network_utils import get_open_zmq_ipc_path
from vllm.v1.engine.async_llm import AsyncLLM

from arctic_platform.inference.server.metrics import RingStatLogger
from arctic_platform.inference.server.semip_frontend import (
    _SemiPAsyncLLMMixin,
    _teardown,
    compare_core_config,
    core_summary,
    serve_core,
)

logger = logging.getLogger("arctic_platform.inference.server.semip")


class SemiPAsyncLLM(_SemiPAsyncLLMMixin, AsyncLLM):
    """``AsyncLLM`` driving the engine core inside a restored semi-p child."""


async def attach(handle: Any, vllm_config: Any) -> SemiPAsyncLLM:
    """Serve ``handle``'s restored engine with an ``AsyncLLM`` built from ``vllm_config``.

    ``vllm_config`` is the front end's own config, built from the same engine
    kwargs the image was dumped with; the child's core must agree with it on
    every identity field, or the attach fails before any request is sent.
    Built on the event loop, as on the plain path. On failure the restored
    engine is torn down, since nothing else owns it yet.
    """
    t0 = time.perf_counter()
    addresses = {"input_address": get_open_zmq_ipc_path(),
                 "output_address": get_open_zmq_ipc_path()}
    try:
        summary = await asyncio.to_thread(
            serve_core, handle.inst, addresses["input_address"],
            addresses["output_address"])
        t_serve = time.perf_counter() - t0
        mismatches = compare_core_config(summary, core_summary(vllm_config))
        if mismatches:
            message = (
                "semi_p: the restored engine core does not match this front "
                "end's config: " + "; ".join(mismatches) + ". Both were built "
                "from this job's engine kwargs, so the child and the front end "
                "resolve them differently; unset SEMIP_FRONTEND to serve "
                "through the default front end meanwhile.")
            logger.error("%s", message)
            raise RuntimeError(message)
        t1 = time.perf_counter()
        llm = SemiPAsyncLLM.from_vllm_config(
            vllm_config,
            stat_loggers=[RingStatLogger],
            client_addresses=addresses,
        )
        t_llm = time.perf_counter() - t1
    except BaseException:
        _teardown(handle.inst, handle.agents)
        raise
    llm._init_semi_p(handle.inst, handle.agents)
    llm._start_death_watch(llm._on_child_death)
    logger.info("semi_p: front end attached in %.2fs (serve_core %.2fs, "
                "AsyncLLM %.2fs)", time.perf_counter() - t0, t_serve, t_llm)
    return llm
