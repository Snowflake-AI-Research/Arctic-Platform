"""Give a plain (non-semi_p) engine the collective setup a semi_p engine runs with.

At ``tensor_parallel_size >= 2`` a semi_p engine does not run vLLM's stock
collective path. ``vllm_child`` pins NCCL NVLS, symmetric-memory and FlashInfer
all-reduce off, turns off the fused all-reduce + RMSNorm pass and the MoE
shared-experts side stream, and enables sleep mode; ``Instance`` runs every TP
worker as ``_semip_worker.SemipGPUWorker``, which refuses the FlashInfer
all-reduce workspace and forces CustomAllreduce's copy path while the CUDA
graphs are captured. Each of those changes a reduction order, so a stock plain
engine differs from a semi_p one in the last bits of its logits -- enough to
flip near-tie greedy tokens, and nothing to do with checkpoint or restore.

A plain job whose ``extra_env`` sets ``SEMIP_MATCH_COLLECTIVES=1`` gets the
same setup from ``apply``, so a semi_p job can be compared against it token for
token and any difference left is semi_p's own. ``tests/test_matched_baseline.py``
holds the values here to what ``vllm_child`` and ``Instance`` actually set.

Single-pod engines only: a pod-spanning semi_p engine also pins its NCCL
network settings at the cold start (``multinode``), which a plain engine's
remote workers would not inherit from here.
"""
from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from typing import Any

MATCH_ENV = "SEMIP_MATCH_COLLECTIVES"

# vllm_child's TP>=2 block: assigned outright ...
COLLECTIVE_ENV = {
    "NCCL_NVLS_ENABLE": "0",
    "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
    "VLLM_ALLREDUCE_USE_FLASHINFER": "0",
}
# ... and defaulted, so extra_env can still override it.
COLLECTIVE_ENV_DEFAULTS = {"SEMIP_SUPPRESS_FI_AR_WORKSPACE": "1"}
PASS_CONFIG = {"fuse_allreduce_rms": False}
WORKER_CLS = "_semip_worker.SemipGPUWorker"
SEMIP_DIR = os.path.dirname(os.path.abspath(__file__))


def requested(environ: MutableMapping[str, str] = os.environ) -> bool:
    return environ.get(MATCH_ENV, "") == "1"


def apply(engine_kwargs: dict[str, Any], environ: MutableMapping[str, str] = os.environ) -> list[str]:
    """Match ``engine_kwargs`` and ``environ`` to a semi_p engine of the same
    tensor-parallel size, in place. A no-op below TP 2, where semi_p changes
    nothing. Returns what it set, for the log."""
    tp = int(engine_kwargs.get("tensor_parallel_size") or 1)
    if tp < 2:
        return []
    changed = []
    for key, value in COLLECTIVE_ENV.items():
        environ[key] = value
        changed.append(f"{key}={value}")
    for key, value in COLLECTIVE_ENV_DEFAULTS.items():
        environ.setdefault(key, value)
        changed.append(f"{key}={environ[key]}")

    compilation = engine_kwargs.get("compilation_config")
    if compilation is None or isinstance(compilation, dict):
        compilation = dict(compilation or {})
        compilation["pass_config"] = {**(compilation.get("pass_config") or {}), **PASS_CONFIG}
        engine_kwargs["compilation_config"] = compilation
    else:
        for key, value in PASS_CONFIG.items():
            setattr(compilation.pass_config, key, value)
    changed.append("pass_config.fuse_allreduce_rms=False")

    engine_kwargs["enable_sleep_mode"] = True
    engine_kwargs.setdefault("worker_cls", WORKER_CLS)
    changed += ["enable_sleep_mode=True", f"worker_cls={engine_kwargs['worker_cls']}"]

    # The worker class and ca_graph_rebind are imported by bare module name in
    # the processes vLLM spawns: spawn copies sys.path, and PYTHONPATH covers
    # anything started another way.
    if SEMIP_DIR not in sys.path:
        sys.path.insert(0, SEMIP_DIR)
    paths = [p for p in environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if SEMIP_DIR not in paths:
        environ["PYTHONPATH"] = os.pathsep.join([SEMIP_DIR, *paths])
    return changed
