"""Front ends over a restored semi-p engine: the part that needs no vLLM.

A restore (``semip_engine.restore_engine``) ends in a ``RestoredEngine``: the
``Instance`` whose vLLM child holds the engine, the agents for the other pods
of a pod-spanning engine, and where the tokenizer lives. Two front ends serve
it, chosen in ``InferenceWorker.initialize`` by ``SEMIP_FRONTEND``:

* ``_SemiPEngine``, the default, presents the subset of the ``AsyncLLM``
  surface the worker touches and forwards each ``generate`` over the
  ``Instance`` pipe.
* ``SemiPAsyncLLM`` (``semip_asyncllm.py``, ``SEMIP_FRONTEND=asyncllm``) is a
  real vLLM ``AsyncLLM`` attached over ZMQ to the child's engine core.

What both share lives here, importable without vLLM so the GPU-free tests can
reach it: semi-p's sleep and wake, teardown, and the check that a front end
agrees with the core it attaches to.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger("arctic_platform.inference.server.semip")

# collective_rpc method names that are weight-sync (RL) features. Deferred on
# _SemiPEngine.
_WEIGHT_SYNC_RPCS = frozenset(
    {"sync_weights", "sync_weights_broadcast", "sync_spec_weights", "close_weight_sync"}
)

# The weight-sync RPCs that write new weights into the engine. After one, the
# host copy semi-p wakes from is stale.
_WEIGHT_UPDATE_RPCS = frozenset(
    {"sync_weights", "sync_weights_broadcast", "sync_spec_weights",
     "sync_lora_weights_broadcast"}
)

# SamplingParams fields the Instance's vLLM child must not inherit: it reads
# cumulative outputs once a request finishes, whatever the caller streams.
_SAMPLING_PARAM_SKIP = frozenset({"output_kind", "stream_interval", "skip_clone"})

_DEATH_WATCH_INTERVAL_S = 1.0


def _sampling_params_to_dict(params: Any) -> dict[str, Any]:
    """A vLLM ``SamplingParams`` as kwargs the child rebuilds it from.

    Every field a caller can set (those ``from_optional`` takes) is copied,
    ``None`` included: an explicit ``max_tokens=None`` means unbounded, where
    an omitted one means vLLM's default of 16.
    """
    if isinstance(params, dict):
        return dict(params)
    import inspect
    names = inspect.signature(type(params).from_optional).parameters
    return {name: getattr(params, name) for name in names
            if name not in _SAMPLING_PARAM_SKIP and hasattr(params, name)}


def _dense_from_numpy(prompt_logprobs: Any) -> Any:
    """Dense prompt logprobs cross the child's pipe as numpy; the worker wants
    the tensors the dense patch produced. Anything else passes through."""
    if not isinstance(prompt_logprobs, dict):
        return prompt_logprobs
    import torch
    return {key: torch.from_numpy(value) if hasattr(value, "dtype") else value
            for key, value in prompt_logprobs.items()}


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

def _teardown_leader(inst: Any) -> None:
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


def _teardown_agents(agents: list[Any]) -> None:
    # Fan out to the other node-partitions. After the leader is down they hold a
    # process tree that can never be driven again -- its executor was the
    # leader's -- so leaving them alive would pin a pod's GPUs until the
    # job ended.
    if not agents:
        return
    import ray
    for agent in agents:
        try:
            ray.get(agent.teardown.remote(), timeout=120)
        except Exception:  # pragma: no cover - best effort
            logger.warning("semi_p: agent teardown failed", exc_info=True)


def _teardown(inst: Any, agents: list[Any]) -> None:
    """Bring a restored engine down: the leader's tree, then the other pods'."""
    if inst is not None:
        _teardown_leader(inst)
    _teardown_agents(list(agents or ()))


# ---------------------------------------------------------------------------
# What a front end must agree on with the core it drives
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
# Lifecycle shared by both front ends
# ---------------------------------------------------------------------------

class _SemiPLifecycle:
    """Semi-p's side of a front end's lifecycle, written once for both.

    Sleep and wake are always semi-p's: sleep drops the GPU weights and KV
    cache, and wake puts the weights back from the host copy taken at
    restore. A weight sync leaves that copy stale, so the next sleep first
    stages the GPU weights into it.
    """

    def _init_semi_p(self, inst: Any, agents: list[Any] | None = None) -> None:
        self._inst = inst
        # The other node-partitions of a pod-spanning engine. They hold GPUs and a
        # restored process tree, so they have to come down with this engine;
        # nothing else owns them.
        self._agents = list(agents or ())
        self._host_copy_stale = False
        self._death_watch_stop: threading.Event | None = None

    def _mark_weights_updated(self) -> None:
        self._host_copy_stale = True

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

    def _close_semi_p(self) -> None:
        self._stop_death_watch()
        inst, self._inst = getattr(self, "_inst", None), None
        agents, self._agents = list(getattr(self, "_agents", ())), []
        _teardown(inst, agents)


class _SemiPAsyncLLMMixin(_SemiPLifecycle):
    """What ``SemiPAsyncLLM`` changes about ``AsyncLLM``, ahead of it in the MRO.

    Kept apart from the vLLM class so it can be tested against a fake base.
    Everything not here -- generate, abort, pause and resume, prefix-cache
    reset, the weight-sync RPCs -- is vLLM's, over ZMQ to the child's core.
    """

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
                self._mark_weights_updated()

    # level and tags are ignored, as on _SemiPEngine: semi-p always sleeps at
    # level 2 and wakes everything.
    async def sleep(self, level: int = 1, mode: str = "abort") -> None:
        await asyncio.to_thread(self._sleep_blocking)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        await asyncio.to_thread(self._wake_blocking)

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
            self._close_semi_p()


# ---------------------------------------------------------------------------
# The default front end: requests over the Instance pipe
# ---------------------------------------------------------------------------

class _SemiPEngine(_SemiPLifecycle):
    """Stands in for ``self.llm`` (a vLLM ``AsyncLLM``) over a semi-p Instance.

    Generates run concurrently, as they do on ``AsyncLLM``: each is its own
    ``generate`` command, and the child batches every request it holds into
    one engine step. Everything else (sleep, wake, pause, resume) runs alone,
    after the in-flight generates finish and before any new one starts.
    """

    def __init__(self, inst: Any, *, tokenizer_path: str | None,
                 model: str | None, agents: list[Any] | None = None):
        self._init_semi_p(inst, agents)
        self._tokenizer_path = tokenizer_path
        self._model = model
        self._tokenizer: Any = None
        # Held by control operations for their whole run, and by a generate
        # only while it submits.
        self._lock = asyncio.Lock()
        # req_id -> (loop, future) of each generate the child still holds.
        # Written on the event loop, popped on the demuxer thread.
        self._waiters: dict[str, tuple[asyncio.AbstractEventLoop,
                                       asyncio.Future]] = {}
        self._waiters_lock = threading.Lock()
        self._inflight = 0
        self._drained = asyncio.Event()
        self._drained.set()
        inst.add_cmd_listener("generate", self._on_generate_done)

    @classmethod
    def from_restored(cls, handle: Any) -> "_SemiPEngine":
        return cls(handle.inst, tokenizer_path=handle.tokenizer_path,
                   model=handle.model, agents=handle.agents)

    # -- generate -----------------------------------------------------------

    async def generate(self, prompt_input: Any, params: Any,
                       request_id: str | None = None,
                       reasoning_ended: bool | None = None):
        """Async generator yielding a single vLLM-``RequestOutput``-shaped final."""
        sp = _sampling_params_to_dict(params)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        async with self._lock:
            with self._waiters_lock:
                self._inst.generate([prompt_input], sp,
                                    reasoning_ended=reasoning_ended)
                self._waiters[self._inst.last_req_id] = (loop, future)
            self._inflight += 1
            self._drained.clear()
        yield self._to_request_output(await future, sp)

    def _on_generate_done(self, cmd: str, elapsed: float,
                          error: object | None, info: Any) -> None:
        """Demuxer listener: hand each finished generate to its waiter."""
        rid = info.get("req_id") if isinstance(info, dict) else None
        with self._waiters_lock:
            if rid is None:
                # The worker reports a dead child without a req_id, and
                # nothing the child held will ever finish.
                done = list(self._waiters.items())
                self._waiters.clear()
            else:
                waiter = self._waiters.pop(rid, None)
                done = [(rid, waiter)] if waiter is not None else []
        for done_rid, (loop, future) in done:
            result = self._inst.generate_results.pop(done_rid, None) or {}
            loop.call_soon_threadsafe(self._settle, future, result, error)

    def _settle(self, future: asyncio.Future, result: dict[str, Any],
                error: object | None) -> None:
        # Counted here, not where the caller awaits: a caller that gives up
        # leaves its request running in the child all the same.
        self._inflight -= 1
        if not self._inflight:
            self._drained.set()
        if future.done():
            return
        if error is not None:
            future.set_exception(
                RuntimeError(f"semi_p: generate failed: {error}"))
        else:
            future.set_result(result)

    @contextlib.asynccontextmanager
    async def _exclusive(self):
        """Run a control operation with no generate in flight."""
        async with self._lock:
            await self._drained.wait()
            # Every generate failure already reached its caller through
            # _settle, but the demuxer also latches the first one, and the
            # control operation's own wait() would raise it as its own.
            try:
                self._inst.wait()
            except RuntimeError as exc:
                logger.debug("semi_p: dropping a reported generate error: %s",
                             exc)
            yield

    @staticmethod
    def _first(seq, i=0):
        if isinstance(seq, (list, tuple)) and len(seq) > i:
            return seq[i]
        return None

    def _to_request_output(self, res: dict[str, Any],
                           sp: dict[str, Any] | None = None) -> SimpleNamespace:
        # outputs: [[text per sample] per prompt]; take prompt 0, sample 0.
        prompt_texts = self._first(res.get("outputs"))
        text = self._first(prompt_texts) or ""

        # completion_token_ids: [[ids per sample] per prompt].
        token_ids = self._first(self._first(res.get("completion_token_ids")))
        prompt_ids = self._first(res.get("prompt_token_ids"))
        if token_ids is None or prompt_ids is None:
            # Callers train on these ids, so a stand-in would be silently
            # wrong rather than merely missing.
            raise RuntimeError(
                "semi_p: the restored engine reported no token ids for this "
                f"request (result keys: {sorted(res)})")

        logprobs = self._first(self._first(res.get("completion_logprobs")))
        prompt_logprobs = _dense_from_numpy(
            self._first(res.get("prompt_logprobs")))
        sp = sp or {}
        for name, value in (("logprobs", logprobs),
                            ("prompt_logprobs", prompt_logprobs)):
            if sp.get(name) is not None and value is None:
                raise RuntimeError(
                    f"semi_p: {name}={sp[name]} was requested but the "
                    f"restored engine returned none")

        finish_reasons = res.get("finish_reasons") or []
        finish_reason = finish_reasons[0] if finish_reasons else "stop"

        choice = SimpleNamespace(
            index=0,
            text=text,
            token_ids=token_ids,
            finish_reason=finish_reason,
            logprobs=logprobs,
            cumulative_logprob=None,
        )
        return SimpleNamespace(
            outputs=[choice],
            prompt_token_ids=prompt_ids,
            prompt_logprobs=prompt_logprobs,
            num_cached_tokens=res.get("num_cached_tokens"),
        )

    # -- tokenizer ----------------------------------------------------------

    def get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_path, trust_remote_code=True
            )
        return self._tokenizer

    # -- sleep / wake / pause / resume / prefix-cache -----------------------

    async def collective_rpc(self, method: str, args: tuple = (),
                             kwargs: dict | None = None):
        if method == "sleep":
            async with self._exclusive():
                await asyncio.to_thread(self._sleep_blocking)
            return [None]
        if method == "wake_up":
            async with self._exclusive():
                await asyncio.to_thread(self._wake_blocking)
            return [None]
        if method == "_arl_cuda_sync":
            # CUDA barrier only needed for weight-sync; no-op for sampling.
            return []
        if method in _WEIGHT_SYNC_RPCS:
            raise NotImplementedError(
                f"semi_p: weight-sync not implemented yet ({method})"
            )
        raise NotImplementedError(f"semi_p: collective_rpc({method}) not supported")

    async def pause_generation(self, mode: str = "keep", clear_cache: bool = False):
        async with self._exclusive():
            await asyncio.to_thread(self._pause_blocking)

    def _pause_blocking(self):
        self._inst.pause()
        self._inst.wait()

    async def resume_generation(self):
        # Not _exclusive: a generate submitted while paused is parked in the
        # child until this resume, so waiting for it to drain would deadlock.
        async with self._lock:
            await asyncio.to_thread(self._resume_blocking)

    def _resume_blocking(self):
        self._inst.resume()
        self._inst.wait()

    async def reset_prefix_cache(self) -> bool:
        # No-op: the published Instance has no reset_prefix_cache primitive.
        # The only live caller is replica_pool's post-weight-sync reset, and
        # weight sync already raises NotImplementedError above, so returning
        # False (reset did not happen) is honest and unreachable in practice.
        # Restoring the primitive across instance/worker/vllm_child is the
        # alternative if weight sync ever lands.
        logger.debug("semi_p: reset_prefix_cache is a no-op")
        return False

    # -- teardown -----------------------------------------------------------

    def close(self) -> None:
        inst = getattr(self, "_inst", None)
        if inst is not None:
            _teardown_leader(inst)
            self._inst = None
            with self._waiters_lock:
                waiters = list(self._waiters.values())
                self._waiters.clear()
            for loop, future in waiters:
                loop.call_soon_threadsafe(
                    self._settle, future, {}, "engine closed")
        agents, self._agents = list(getattr(self, "_agents", ())), []
        _teardown_agents(agents)

    def __del__(self):  # pragma: no cover - GC path
        try:
            self.close()
        except Exception:
            pass
