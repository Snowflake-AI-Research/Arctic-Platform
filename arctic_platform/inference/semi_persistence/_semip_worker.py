"""Custom vLLM worker for TP semi-persistence.

Standalone ``worker_cls`` (subclasses vanilla vLLM ``Worker``; no
arctic_inference dependency).  Wired via ``vllm_config["worker_cls"] =
"_semip_worker.SemipGPUWorker"`` for TP>1 instances.

Two hooks:
  * ``init_device`` remaps the vLLM ``local_rank`` to a physical GPU via
    ``SEMIP_GPU_MAP`` (set by vllm_child before spawn), so a TP group can be
    placed on an arbitrary set of physical GPUs while keeping all GPUs
    visible (required for the cuda-checkpoint physical-GPU addressing).
  * ``compile_or_warm_up_model`` captures the cold-start CUDA graphs with
    keep_graph=True, so the preserved graph is reuse-friendly and
    ``ca_graph_rebind`` can rewrite its baked addresses after CRIU restore.
    The patches are withdrawn at the end of that window (try/finally) so
    runtime is unaffected.

The CustomAllreduce copy path (registered=False) is forced from
``init_device`` rather than from the warmup hook, because 0.30 captures once
more, earlier, inside ``determine_available_memory``. Both hooks install it;
the withdrawal stays in the warmup ``finally``.

``SEMIP_SUPPRESS_PROFILE_CA_REGISTER=1`` additionally skips CA graph-buffer
registration for that earlier capture alone, as the fallback if the copy path
turns out not to cover every pointer it records. Off by default.

Both hooks return whatever the base class returns, so this stays agnostic to
the ``compile_or_warm_up_model`` return type (a float on older vLLM, a
``CompilationTimes`` on newer).
"""
import os
import sys

from vllm.v1.worker.gpu_worker import Worker


class SemipGPUWorker(Worker):
    def init_device(self):
        # fd 1/2 are the pod-log pipe the child handed down, which Python
        # block-buffers by default; several replicas share it, so flush per line.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(line_buffering=True)
            except (AttributeError, OSError):
                pass
        gpu_map = os.environ.get("SEMIP_GPU_MAP")
        if gpu_map:
            self.local_rank = int(gpu_map.split(",")[self.local_rank])
        result = super().init_device()
        # The FlashInfer workspace probe has to be live from here, not from
        # `compile_or_warm_up_model`. GLM-5.3 runs *two* PIECEWISE capture
        # passes, and job 07fd79b4 measured the workspace being built 57%
        # through the FIRST one -- before this worker's warmup hook is entered
        # at all. The proof is three lines in that log read together: the probe
        # reported `patched=[all three] missing=[] suppress=True`, no
        # `first ... call via` line ever printed, and `Initialized FlashInfer
        # Allreduce` still appeared once. A patch that is installed and silent
        # while the thing it refuses still happens can only mean the allocation
        # is outside the window. Installing here covers both captures; the
        # withdrawal stays in `compile_or_warm_up_model`'s finally, which still
        # runs long before the dump, so nothing rides into the CRIU image.
        self._semip_install_fi_ar_probe("init_device")
        # Same reasoning, second victim. 0.30 runs a CUDA-graph capture inside
        # `determine_available_memory` (profile_cudagraph_memory ->
        # capture_model(profile_only=True)), which is over before
        # `compile_or_warm_up_model` is entered. `CustomAllreduce.capture()`
        # exports an IPC handle for every pointer that capture recorded, and
        # with `enable_sleep_mode` those pointers can be cumem-backed, which
        # the legacy `cudaIpcGetMemHandle` cannot export. Scoped to the warmup
        # hook, the copy-path patch could not protect the earlier capture: job
        # 37414847 died there on all eight ranks, from inside CUDACHECK, with
        # nothing for Python to catch. Install here so both captures record
        # only the pre-registered staging buffer.
        self._semip_install_force_copy("init_device")
        self._semip_arm_profile_register_guard()
        return result

    def _semip_install_fi_ar_probe(self, where):
        """Install the workspace probe and say what stuck. Idempotent: the
        second caller gets the live state back rather than re-wrapping."""
        try:
            import ca_graph_rebind
        except Exception as exc:  # noqa: BLE001 - probe is never load-bearing
            print(f"[fi-ar-probe] install({where}): ca_graph_rebind "
                  f"unavailable: {type(exc).__name__}: {exc}", flush=True)
            return
        info = ca_graph_rebind.install_fi_ar_workspace_probe()
        print(f"[fi-ar-probe] install({where}): patched={info['patched']} "
              f"missing={info['missing']} suppress={info['suppress']} "
              f"err={info['err']}", flush=True)

    def _semip_install_force_copy(self, where):
        """Install the CustomAllreduce copy-path patch and say what stuck.

        Idempotent, and the second call is not redundant: it re-states the
        live state next to the capture, and `car_calls` is cumulative across
        installs (the early return in `install_force_copy_patch` precedes the
        counter reset). So `install(warmup): active=True car_calls=0` says the
        profiling capture ran unprotected, which is the failure this hook
        exists to rule out -- and it says it before the capture that would
        otherwise prove it by dying."""
        try:
            import ca_graph_rebind
        except Exception as exc:  # noqa: BLE001
            print(f"[force-copy] install({where}): ca_graph_rebind "
                  f"unavailable: {type(exc).__name__}: {exc}", flush=True)
            return
        ca_graph_rebind.install_force_copy_patch()
        state = ca_graph_rebind.force_copy_state()
        print(f"[force-copy] install({where}): active={state['active']} "
              f"ar_calls={state['ar_calls']} car_calls={state['car_calls']}",
              flush=True)

    def _semip_arm_profile_register_guard(self):
        """Optionally skip CA graph-buffer registration for the profiling
        capture. Off unless `SEMIP_SUPPRESS_PROFILE_CA_REGISTER=1`.

        The pre-armed fallback for the copy-path fix above. If the copy path
        covers every pointer the profiling capture records, this is not
        needed. If Qwen3.8-Flash-Next still dies at `cuh:164` with
        `car_calls > 0`, then something reached `graph_unreg_buffers_` through
        a path `install_force_copy_patch` does not wrap -- it patches
        `all_reduce` and `custom_all_reduce`, while 0.30's CA also has
        `custom_all_gather` and `custom_reduce_scatter` -- and skipping
        registration outright is the answer for a capture whose graphs are
        discarded anyway. Shipping it dark makes that a resubmit with one
        `extra_env` key rather than another image build.

        Deliberately scoped to the profiling window and released at the top
        of `compile_or_warm_up_model`, not in its finally: the warmup capture
        is the one whose graphs survive into the image, and its registration
        stays untouched."""
        if os.environ.get("SEMIP_SUPPRESS_PROFILE_CA_REGISTER", "0") != "1":
            return
        try:
            import ca_graph_rebind
        except Exception as exc:  # noqa: BLE001
            print(f"[profile-register] arm: ca_graph_rebind unavailable: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            return
        ca_graph_rebind.install_suppress_register_patch()
        state = ca_graph_rebind.suppress_register_state()
        print(f"[profile-register] arm: active={state['active']}", flush=True)

    def _semip_release_profile_register_guard(self):
        """Hand registration back before the warmup capture, and say how many
        it swallowed. `skipped=0` with the guard armed means the profiling
        capture never reached a registration, which is a different finding
        from the guard having caught one."""
        try:
            import ca_graph_rebind
        except Exception:  # noqa: BLE001
            return
        state = ca_graph_rebind.suppress_register_state()
        if not state["active"]:
            return
        ca_graph_rebind.restore_suppress_register_patch()
        print(f"[profile-register] release: skipped={state['skipped']}",
              flush=True)

    def compile_or_warm_up_model(self):
        try:
            import ca_graph_rebind
        except Exception:
            return super().compile_or_warm_up_model()
        self._semip_release_profile_register_guard()
        self._semip_install_force_copy("warmup")
        ca_graph_rebind.install_keepgraph_patch()
        # Normally a no-op that reports the state `init_device` established --
        # but it is not redundant. This is the only install if `init_device`
        # ran before this class was in place, and it is where the state gets
        # re-stated next to the capture it used to be scoped to.
        self._semip_install_fi_ar_probe("warmup")
        try:
            result = super().compile_or_warm_up_model()
        finally:
            ca_graph_rebind.restore_force_copy_patch()
            ca_graph_rebind.restore_keepgraph_patch()
            ca_graph_rebind.restore_fi_ar_workspace_probe()
        # keep_graph=True is what lets ca_graph_rebind read the captured
        # topology, but torch's capture_end() instantiates only when
        # keep_graph is false, so the patch above also leaves every graph
        # without a cudaGraphExec_t.  Stock vLLM has all of them instantiated
        # at this point; without this call the image is dumped holding execs
        # only for the shapes the dump-time job happens to run, and the
        # post-restore warmup becomes the first code to build the rest.
        # Undoing that side effect here keeps the two configurations equal.
        # Unconditional since 2026-09-24 (was SEMIP_WARM_IMAGE): this is the
        # instantiate half of image warming, and warming is part of init.
        ca_graph_rebind.instantiate_captured_graphs(self)
        # Unconditional: the census is read-only and costs a bool check per
        # graph, and gating it is how a dump gets wasted.
        ca_graph_rebind.graph_exec_census(self)
        return result
