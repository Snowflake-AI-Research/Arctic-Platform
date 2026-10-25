"""The front end over a restored semi-p engine: the part that needs no vLLM.

A restore (``semip_engine.restore_engine``) ends in a ``RestoredEngine``: the
``Instance`` whose vLLM child holds the engine, the agents for the other pods
of a pod-spanning engine, and where the tokenizer lives. ``SemiPAsyncLLM``
(``semip_asyncllm.py``) serves it: a real vLLM ``AsyncLLM`` attached over ZMQ
to the child's engine core.

What that front end adds to ``AsyncLLM`` lives here, importable without vLLM so
the GPU-free tests can reach it: semi-p's sleep and wake, teardown, a watch on
the child, and the check that the front end agrees with the core it attaches to.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

logger = logging.getLogger("arctic_platform.inference.server.semip")

# The weight-sync RPCs that write new base or drafter weights into the engine.
# After one, the host copy semi-p wakes from is stale. A LoRA sync is not one:
# the child stages LoRA state at every sleep.
_WEIGHT_UPDATE_RPCS = frozenset(
    {"sync_weights", "sync_weights_broadcast", "sync_spec_weights"}
)

_DEATH_WATCH_INTERVAL_S = 1.0


def _teardown(inst: Any, agents: list[Any]) -> None:
    """Bring a restored engine down: the leader's tree, then the other pods'."""
    if inst is not None:
        try:
            inst.teardown()
            # teardown() is a non-blocking _send, and the actual kill of the
            # vLLM child happens in the Instance's worker process when it
            # picks the command up (_kill_process_tree in worker.py's
            # teardown handler). Without this wait, /destroy lets the actor
            # exit first, the worker dies before draining the command, and
            # the child survives -- reparented to init, invisible to a
            # `ps | grep vllm` because its comm is plain "python", and
            # holding the entire GPU. That is exactly what a restored tree
            # does: criu_restore sets child_proc=None, so worker_loop's
            # force-kill branch is skipped and this handler is the only
            # thing that reaps it. Bounded by the worker's own 30s/5s joins.
            inst.wait()
        except Exception:  # pragma: no cover - best effort
            logger.warning("semi_p: teardown failed", exc_info=True)
    # Fan out to the other node-partitions. After the leader is down they hold a
    # process tree that can never be driven again -- its executor was the
    # leader's -- so leaving them alive would pin a pod's GPUs until the
    # job ended.
    agents = list(agents or ())
    if not agents:
        return
    import ray
    for agent in agents:
        try:
            ray.get(agent.teardown.remote(), timeout=120)
        except Exception:  # pragma: no cover - best effort
            logger.warning("semi_p: agent teardown failed", exc_info=True)


# ---------------------------------------------------------------------------
# What the front end must agree on with the core it drives
# ---------------------------------------------------------------------------

# The fields compared before attaching. The rest of the summary is reported
# but not compared: vLLM's ready handshake syncs max_model_len and the KV
# block counts from the core, and prefix caching is the core's alone.
_IDENTITY_FIELDS = (
    "model", "tokenizer", "dtype", "tensor_parallel_size",
    "worker_extension_cls", "max_num_seqs", "max_num_batched_tokens",
    "reasoning_parser", "structured_outputs_backend", "speculative_method",
    "num_speculative_tokens", "lora",
)

_NOT_REPORTED = "<not reported>"


def core_summary(vllm_config: Any) -> dict[str, Any]:
    """``vllm_child._semip_core_summary`` for the front end's own config.

    The child cannot be imported here, so the reading is repeated;
    ``tests/test_semip_frontend.py`` holds the two to the same output.
    """
    mc = vllm_config.model_config
    pc = vllm_config.parallel_config
    sc = vllm_config.scheduler_config
    cc = vllm_config.cache_config
    so = getattr(vllm_config, "structured_outputs_config", None)
    spec = getattr(vllm_config, "speculative_config", None)
    return {
        "model": str(mc.model),
        "tokenizer": str(mc.tokenizer),
        "dtype": str(mc.dtype).removeprefix("torch."),
        "tensor_parallel_size": int(pc.tensor_parallel_size),
        "worker_extension_cls": str(getattr(pc, "worker_extension_cls", "") or ""),
        "max_num_seqs": int(sc.max_num_seqs),
        "max_num_batched_tokens": int(sc.max_num_batched_tokens),
        "reasoning_parser": str(getattr(so, "reasoning_parser", "") or ""),
        "structured_outputs_backend": str(getattr(so, "backend", "") or ""),
        "speculative_method": str(getattr(spec, "method", "") or ""),
        "num_speculative_tokens": int(
            getattr(spec, "num_speculative_tokens", 0) or 0),
        "lora": getattr(vllm_config, "lora_config", None) is not None,
        "max_model_len": int(mc.max_model_len),
        "block_size": int(cc.block_size),
        "num_gpu_blocks": int(cc.num_gpu_blocks or 0),
        "enable_prefix_caching": bool(cc.enable_prefix_caching),
    }


def compare_core_config(summary: dict[str, Any],
                        vllm_fields: dict[str, Any]) -> list[str]:
    """Every identity field on which the core and the front end disagree.

    ``summary`` is what the child's ``serve_core`` reported, ``vllm_fields``
    the front end's ``core_summary``. Each entry names the field and both
    values; an empty list means they agree.
    """
    mismatches = []
    for name in _IDENTITY_FIELDS:
        core = summary.get(name, _NOT_REPORTED)
        front = vllm_fields.get(name, _NOT_REPORTED)
        if core != front:
            mismatches.append(f"{name}: core={core!r} front end={front!r}")
    return mismatches


def serve_core(inst: Any, input_address: str,
               output_address: str) -> dict[str, Any]:
    """Have the restored child serve its engine core on two ZMQ addresses.

    Blocking. Returns the child's summary of its config (``serve_core``'s
    info). ``Instance`` has no method for this command, so this is the one
    place that uses its private ``_send``; ``wait`` raises if the child
    refused or failed it.
    """
    inst._send("serve_core", input_address=input_address,
               output_address=output_address)
    inst.wait()
    return dict(inst.last_info.get("serve_core") or {})


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` is a running process; a zombie counts as dead."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


# ---------------------------------------------------------------------------
# The front end
# ---------------------------------------------------------------------------

class _SemiPAsyncLLMMixin:
    """What ``SemiPAsyncLLM`` changes about ``AsyncLLM``, ahead of it in the MRO.

    Kept apart from the vLLM class so it can be tested against a fake base.
    Everything not here -- generate, abort, pause and resume, prefix-cache
    reset, the weight-sync RPCs -- is vLLM's, over ZMQ to the child's core.

    Sleep and wake are semi-p's: sleep drops the GPU weights and KV cache, and
    wake puts the weights back from the host copy taken at restore. A weight
    sync leaves that copy stale, so the next sleep first stages the GPU
    weights into it.
    """

    def _init_semi_p(self, inst: Any, agents: list[Any] | None = None) -> None:
        self._inst = inst
        # The other node-partitions of a pod-spanning engine. They hold GPUs and a
        # restored process tree, so they have to come down with this engine;
        # nothing else owns them.
        self._agents = list(agents or ())
        self._host_copy_stale = False
        self._death_watch_stop: threading.Event | None = None

    async def collective_rpc(self, method: str, timeout: float | None = None,
                             args: tuple = (), kwargs: dict | None = None):
        if method == "sleep":
            await self.sleep()
            return [None]
        if method == "wake_up":
            await self.wake_up()
            return [None]
        try:
            return await super().collective_rpc(
                method, timeout=timeout, args=args, kwargs=kwargs)
        finally:
            # Even on failure: a sync that raised part way may still have
            # written some of the weights.
            if method in _WEIGHT_UPDATE_RPCS:
                self._host_copy_stale = True

    # level and tags are ignored: semi-p always sleeps at level 2 and wakes
    # everything.
    async def sleep(self, level: int = 1, mode: str = "abort") -> None:
        await asyncio.to_thread(self._sleep_blocking)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        await asyncio.to_thread(self._wake_blocking)

    def _sleep_blocking(self):
        if self._host_copy_stale:
            logger.info("semi_p: staging the synced weights into the host "
                        "copy before sleep")
            self._inst.stage()
            self._inst.wait()
            self._host_copy_stale = False
        self._inst.sleep()
        self._inst.wait()

    def _wake_blocking(self):
        # restore_weights() is not optional here: sleep() is level 2, which
        # frees the weight memory, and wake_up_weights() only re-allocates the
        # parameter tensors without repopulating them. Without it the model
        # generates from uninitialized weights and never errors. No
        # plan_restore_weights() here: the chunk plan and the double buffer
        # formed at restore time stay on the workers across sleep.
        self._inst.wake_up_weights()
        self._inst.restore_weights()
        self._inst.wake_up_kv_cache()
        self._inst.wait()

    def _start_death_watch(self, on_death,
                           interval_s: float = _DEATH_WATCH_INTERVAL_S) -> None:
        """Call ``on_death`` once if the engine child exits before ``close``."""
        pid = getattr(self._inst, "pid", None)
        if not pid:
            logger.warning("semi_p: the Instance reports no child pid, so the "
                           "front end will not notice if the child dies")
            return
        stop = threading.Event()
        self._death_watch_stop = stop

        def watch():
            while not stop.wait(interval_s):
                if _pid_alive(pid):
                    continue
                logger.warning(
                    "semi_p: the engine child (pid %d) exited unexpectedly; "
                    "failing the front end's requests", pid)
                try:
                    on_death()
                except Exception:
                    logger.warning("semi_p: shutting the front end down after "
                                   "the child died also failed", exc_info=True)
                return

        threading.Thread(target=watch, daemon=True,
                         name="semip-death-watch").start()

    def _stop_death_watch(self) -> None:
        stop = getattr(self, "_death_watch_stop", None)
        if stop is not None:
            stop.set()

    def _on_child_death(self) -> None:
        # What vLLM's own engine monitor does; it never runs for an engine
        # vLLM did not launch. Pending and later requests get EngineDeadError.
        self.engine_core.resources.engine_dead = True
        self.engine_core.shutdown()

    def close(self) -> None:
        self._stop_death_watch()
        try:
            self.shutdown()
        finally:
            inst, self._inst = getattr(self, "_inst", None), None
            agents, self._agents = list(getattr(self, "_agents", ())), []
            _teardown(inst, agents)
