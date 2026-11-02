"""A real vLLM ``AsyncLLM`` over a restored semi-p engine.

The restored child turns its in-process ``EngineCore`` into a ZMQ-served one
(``serve_core``), and an ``AsyncLLM`` here attaches to it the way vLLM attaches
to an engine it did not launch, through ``client_addresses``. Requests,
streaming, aborts, pause and resume, weight-sync RPCs and stats are then
vLLM's own; ``semip_frontend._SemiPAsyncLLMMixin`` keeps sleep, wake, teardown
and a watch on the child.

Imports vLLM at module scope, unlike ``semip_frontend``, so the GPU-free tests
load it against fake vLLM modules.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

import vllm.v1.engine.async_llm as _async_llm
from vllm.renderers import renderer_from_config
from vllm.utils.network_utils import get_open_zmq_ipc_path
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.input_processor import InputProcessor

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


async def restore_and_attach(engine_kwargs: dict[str, Any], agents: Any,
                             vllm_config: Any) -> SemiPAsyncLLM:
    """Restore the engine ``engine_kwargs`` describe and attach to it.

    ``vllm_config`` is the worker's, built from the same kwargs. Its scheduler
    limits are pinned into the kwargs the child builds its engine from,
    because the child uses ``LLM()``, whose usage-context defaults differ.
    Stats are turned on for the same kind of reason: ``LLM()`` turns them off,
    and the core builds its stats objects with the engine. The reasoning
    parser is pinned too: the worker adds chat's reasoner to ``vllm_config``
    after building it, so the kwargs alone would build a core without one.
    """
    from arctic_platform.inference.server.semip_engine import restore_engine

    kwargs = dict(engine_kwargs)
    for key in ("max_num_seqs", "max_num_batched_tokens"):
        if kwargs.get(key) is None:
            kwargs[key] = getattr(vllm_config.scheduler_config, key)
    kwargs.setdefault("disable_log_stats", False)
    reasoner = getattr(vllm_config.structured_outputs_config, "reasoning_parser", None)
    if reasoner and not kwargs.get("reasoning_parser"):
        kwargs["reasoning_parser"] = reasoner
    building = asyncio.ensure_future(asyncio.to_thread(_build_inputs, vllm_config))
    try:
        handle = await asyncio.to_thread(restore_engine, kwargs, agents)
    finally:
        renderer, input_processor, render_s = await building
    return await attach(handle, vllm_config, renderer, render_s, input_processor)


def _build_inputs(vllm_config: Any) -> tuple[Any, Any, float | None]:
    """The front end's renderer (tokenizer and multimodal processor) and input
    processor (which sizes a multimodal model's encoder cache), built while
    the engine restores, and how long that took. A part that fails to build
    is None, and the ``AsyncLLM`` then builds its own."""
    t0 = time.perf_counter()
    try:
        renderer = renderer_from_config(vllm_config)
    except Exception:  # noqa: BLE001 - AsyncLLM retries it and raises then
        logger.warning("semi_p: building the renderer during the restore "
                       "failed; the AsyncLLM will build it", exc_info=True)
        return None, None, None
    try:
        input_processor = InputProcessor(vllm_config, renderer)
    except Exception:  # noqa: BLE001 - as above
        logger.warning("semi_p: building the input processor during the "
                       "restore failed; the AsyncLLM will build it", exc_info=True)
        input_processor = None
    return renderer, input_processor, time.perf_counter() - t0


@contextlib.contextmanager
def _prebuilt_inputs(renderer: Any, input_processor: Any):
    """Has ``AsyncLLM.__init__`` take ``renderer`` and ``input_processor``
    rather than build them.

    It calls its module's ``renderer_from_config`` and ``InputProcessor`` by
    name; were that to change, it would build its own, which is only slower.
    """
    swaps = [(name, value) for name, value in (
        ("renderer_from_config", renderer), ("InputProcessor", input_processor))
        if value is not None]
    originals = {name: getattr(_async_llm, name) for name, _ in swaps}
    for name, value in swaps:
        setattr(_async_llm, name, lambda *args, _value=value, **kwargs: _value)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(_async_llm, name, original)


async def attach(handle: Any, vllm_config: Any, renderer: Any = None,
                 render_s: float | None = None,
                 input_processor: Any = None) -> SemiPAsyncLLM:
    """Serve ``handle``'s restored engine with an ``AsyncLLM`` built from ``vllm_config``.

    ``vllm_config`` is the front end's own config, built from the same engine
    kwargs the image was dumped with; the child's core must agree with it on
    every identity field, or the attach fails before any request is sent.
    ``renderer`` and ``input_processor``, when given, are the ones
    ``restore_and_attach`` built from ``vllm_config`` during the restore, and
    ``render_s`` how long that took; ``input_processor`` is used only with
    the ``renderer`` it was built on.
    Built on the event loop, as on the plain path. On failure the restored
    engine is torn down, since nothing else owns it yet.
    """
    if renderer is None:
        input_processor = None
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
                "resolve them differently; disable semi_p for this job "
                "meanwhile.")
            logger.error("%s", message)
            raise RuntimeError(message)
        t1 = time.perf_counter()
        with _prebuilt_inputs(renderer, input_processor):
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
                "AsyncLLM %.2fs, %s)", time.perf_counter() - t0, t_serve, t_llm,
                ("renderer and input processor" if input_processor is not None
                 else "renderer") + f" built during the restore in {render_s:.2f}s"
                if renderer is not None else "renderer built by the AsyncLLM")
    return llm
