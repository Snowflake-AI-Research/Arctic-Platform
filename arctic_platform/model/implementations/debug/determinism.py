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
"""Determinism controls, in one place, for two use cases that must not be confused.

**Replay determinism** is a production requirement and costs almost nothing. Activation checkpointing runs a block
twice -- once forward, once during the backward recompute -- and the two passes must agree. For an MoE that is not
a nicety: the router's ``topk`` over the gate scores is decided by the last bits of a matmul, so a recompute that
re-derives it can select a different expert set, which changes ``num_tokens_per_expert``, which changes tensor
shapes inside the checkpoint, which raises ``CheckpointError: recomputed values have different metadata``. Router
replay (``arctic_platform.model.implementations.gpu.router_replay_recompute``) captures the decision on the forward and replays it.

Of the two replay knobs, this module owns :data:`CHECKPOINT_PRESERVE_RNG_STATE`, which every checkpoint call site
reads so they cannot drift apart. Whether to install router replay stays with the recipe's
``ac_config.router_replay_recompute``, which defaults to on and should stay on -- a parity run whose expert
assignment may differ between a forward and its recompute is not measuring the change under test.

**Full determinism** is a test-only sledgehammer, and it is expensive: deterministic kernels, cuDNN autotuning
off, deterministic reductions in cuBLAS, NCCL and flash attention, and a seeded run. It exists so a repeat of a
run reproduces its bits, and it should never be enabled in a production run. It is requested with a single flag,
``debug.full_determinism``, mirroring Arctic Platform's flag of the same name
(``arctic_platform/common/utils/debug.py``).

It leaves the arithmetic exactly as it was: TF32 and reduced-precision accumulation are precision decisions, not
determinism ones, and they belong to ``debug.fp32_precision`` in ``precision.py``. A bf16 run already reproduces
bit-exactly, so raising precision buys reproducibility nothing -- what it buys is invariance to how a sum was
decomposed, which is a different property and only matters to a parity comparison.

Full determinism implies replay determinism, never the reverse: a run can be perfectly replayable and still not
reproduce another run's bits.

The two halves of the full-determinism setup live in different places, because some settings are read before any
of our code runs:

* :func:`determinism_worker_env` -- variables the libraries consume while initializing. ``CUBLAS_WORKSPACE_CONFIG``
  is read when cuBLAS creates its handle, ``NCCL_DETERMINISTIC`` when NCCL initializes, and ``PYTHONHASHSEED`` only
  affects hash randomization if the interpreter sees it at startup. The head merges these into the worker
  environment at spawn, because setting them inside the worker would be too late.
* :func:`configure_full_determinism` -- torch state, applied first thing in the worker.

One of those variables is not part of the flag. :data:`CUBLAS_WORKSPACE` goes into every worker's environment
whichever way the flag is set, because what it buys is not reproducibility between runs but a projection whose
result for a token does not depend on how many other tokens share its call. A trainer that answers differently
for a token depending on the height of the batch it arrived in is wrong in production, not merely awkward to
test. ``docs/determinism.md`` carries the measurement, the cost, and the call shapes it still does not cover.

Seeding is deliberately not applied here: the worker resolves one seed via :func:`resolve_seed` and applies it per
rank, so doing it here as well would only give two sources of truth.
"""

from __future__ import annotations

import functools
import inspect
import os
from typing import Any
from typing import Mapping
from typing import Optional

from .config import training_debug_config

DEFAULT_SEED = 42

# The variable a full-determinism run exports for flash attention. Hugging Face's attention integration
# reads it; a model that calls a flash-attention entry point directly has to read it itself.
FLASH_ATTENTION_DETERMINISTIC_ENV = "FLASH_ATTENTION_DETERMINISTIC"

# Whether activation checkpointing restores RNG state before recomputing a block. Saving and restoring the CPU and
# CUDA generators around every checkpointed block is not free, and it only buys anything when the block consumes
# randomness -- dropout, stochastic depth. These recipes train with dropout disabled, so the recompute is already
# a pure function of its inputs and router replay covers the one decision that is not. Flip this here, in one
# place, if a recipe ever introduces randomness inside a checkpointed block.
CHECKPOINT_PRESERVE_RNG_STATE = False

# The cuBLAS workspace every worker gets, determinism flag or not. Both values that permit deterministic
# reductions make a repeat of one call reproducible; only this, the smaller one, makes a product independent of
# how many rows share the call, because a split-K kernel needs workspace for its partial products and cuBLASLt
# will select one when the row count is low and the output narrow. That independence is a property of a single
# run and not only of a comparison between two: a log-probability that moves when the same token arrives in a
# call of a different height moves the importance ratio the policy update multiplies by. One home for the value,
# read by both branches of :func:`determinism_worker_env`, so the two cannot drift.
CUBLAS_WORKSPACE = ":16:8"

_LEGACY_FLAG = "deterministic_algorithms"


def full_determinism_enabled(training_config: Mapping[str, Any]) -> bool:
    debug_config = training_debug_config(training_config)
    if _LEGACY_FLAG in debug_config:
        # Silently ignoring it would leave a parity test believing it runs deterministically while it does not,
        # which is the one failure mode this flag exists to prevent.
        raise ValueError(
            f"debug.{_LEGACY_FLAG} is now debug.full_determinism, which also seeds the run and sets the "
            "deterministic worker environment. For fp32 accumulation and TF32 off, set debug.fp32_precision"
        )
    value = debug_config.get("full_determinism", False)
    if not isinstance(value, bool):
        raise TypeError(f"full_determinism in the training debug config must be a bool, got {type(value).__name__}")
    return value


def resolve_seed(seed: Any, training_config: Mapping[str, Any]) -> Optional[int]:
    """The seed the worker should apply, defaulting only when full determinism was requested.

    An explicit ``seed`` is honored either way. Without one, a determinism request still has to seed something or
    the run is not reproducible -- the flag would pin the kernels while leaving the initial weights and every
    dropout mask to chance -- so it falls back to :data:`DEFAULT_SEED`. Leaving the seed unset otherwise preserves
    the default of not touching global RNG state.
    """
    if seed is not None:
        return int(seed)
    if full_determinism_enabled(training_config):
        return DEFAULT_SEED
    return None


def determinism_worker_env(training_config: Mapping[str, Any], seed: Optional[int]) -> dict[str, str]:
    """Environment for a training worker: the cuBLAS workspace always, the determinism variables only on request.

    cuBLASLt selects a split-K kernel for a bfloat16 product when the row count is low enough to leave the device
    underoccupied and the contraction long enough for the split to pay for itself, and split-K groups that
    contraction differently from the unsplit kernel a taller call selects. The larger workspace, ``:4096:8``,
    leaves room for the split's partial products, so which kernel runs -- and therefore the last bit of the
    result -- follows how many rows share the call. :data:`CUBLAS_WORKSPACE` denies that room, and the unsplit
    kernel runs at every row count.

    Two things make it wrong to leave to the flag. Selection needs a narrow output as well as a low row count,
    so it is the attention output and MLP down projections that move while qkv, gate/up and the language-model
    head do not, and the deviation then amplifies through depth: at hidden 4096 over 28 layers it reaches
    7.0e-02 to 1.7e-01 on the importance ratio the policy update multiplies by, against a same-shape repeat
    floor of 0.000e+00. And sequence parallelism is not required to reach it -- any change in how many rows a
    call carries does, which a token-budget packer varies by construction.

    It is not the slower value either: the split-K kernel the larger workspace makes eligible is not faster at
    these shapes, and both the isolated GEMMs and a production-shaped optimizer step measure the same either
    way. Pinning a workspace at all does cost, and every run pays it rather than tests alone;
    ``docs/determinism.md`` carries the number, the shapes this does *not* cover -- calls of one, two and three
    rows still move -- and why a trainer-side variable cannot reach the vLLM sampler.
    ``tests/sp/test_sp_projection_row_count_invariance.py`` holds the property under the flag and
    ``tests/sp/test_production_logprob_row_count_invariance.py`` holds it for a worker with no debug section.
    """
    if not full_determinism_enabled(training_config):
        return {"CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE}
    return {
        "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE,
        "NCCL_DETERMINISTIC": "1",
        FLASH_ATTENTION_DETERMINISTIC_ENV: "1",
        "PYTHONHASHSEED": str(seed),
    }


def configure_full_determinism(training_config: Mapping[str, Any]) -> None:
    """Make a run reproduce itself: same inputs, same kernels, same result.

    This is determinism only, deliberately. It does not change which arithmetic runs, so it does not make results
    invariant to *how* a computation is decomposed -- see the note in ``docs/determinism.md``. Precision is a
    separate axis with its own settings, such as ``fp32_lm_head``.
    """
    if not full_determinism_enabled(training_config):
        return

    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flash_attention_determinism_requested() -> bool:
    """Whether the worker environment asks flash attention for a deterministic backward.

    :func:`determinism_worker_env` exports :data:`FLASH_ATTENTION_DETERMINISTIC_ENV` for a
    ``debug.full_determinism`` run, and Hugging Face's attention integration consumes it. A model that calls a
    flash-attention entry point itself has no such consumer, so it has to read the variable too or the request
    stops at the process boundary while the configuration still claims the run is reproducible.
    """
    return os.environ.get(FLASH_ATTENTION_DETERMINISTIC_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _varlen_backward_error(varlen_func, head_dim: int, *, deterministic: bool) -> Optional[str]:
    """Run one minimal varlen backward and return what it raised, or ``None`` if it completed.

    The inputs are zeros rather than a draw from the global generator: this runs while the model is being built,
    and consuming randomness there would move every seeded quantity in the run that follows.
    """
    import torch

    tokens = 8
    shape = (tokens, 1, head_dim)
    device = torch.device("cuda")
    q, k, v = (torch.zeros(shape, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    cu_seqlens = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    try:
        out = varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=tokens,
            max_seqlen_k=tokens,
            causal=True,
            deterministic=deterministic,
        )
        if isinstance(out, tuple):
            out = out[0]
        out.sum().backward()
    except Exception as exc:  # the kernel's own refusal is the answer being collected
        return f"{type(exc).__name__}: {exc}"
    return None


@functools.lru_cache(maxsize=None)
def flash_attention_deterministic_backward_refusal(varlen_func, head_dim: int) -> Optional[str]:
    """The installed kernel's reason for refusing a deterministic backward at ``head_dim``, or ``None``.

    The question goes to the kernel rather than to a version string because the answer belongs to the build: one
    entry point accepts the keyword at some head dimensions and rejects it at others, so a version test would
    carry a threshold only the build knows and would be wrong the first time a build moved it.

    A failure counts as a determinism refusal only when the same call succeeds with the flag off. That separates
    "this build will not run a deterministic backward at this head dimension" from "this build does not support
    this head dimension at all", which is a different failure and is left to surface where it always did.
    """
    try:
        parameters = inspect.signature(varlen_func).parameters
    except (TypeError, ValueError):
        parameters = None
    if parameters is not None and "deterministic" not in parameters:
        return f"{varlen_func.__module__}.{varlen_func.__qualname__} accepts no 'deterministic' argument"
    if _varlen_backward_error(varlen_func, head_dim, deterministic=False) is not None:
        return None
    return _varlen_backward_error(varlen_func, head_dim, deterministic=True)


def resolve_flash_attention_determinism(varlen_func, head_dim: int, implementation: str) -> bool:
    """Whether to ask ``varlen_func`` for a deterministic backward, refusing when the request cannot be honoured.

    Returning ``False`` whenever nothing asked keeps the product path exactly as it is and leaves this the only
    place that decides. Raising when the request cannot be met is the point of the check: a run that keeps both
    ``debug.full_determinism`` and a kernel that will not honour it reports a reproducibility it does not have,
    and being able to trust that report is the flag's only purpose.

    This answers for the softmax-attention layers alone. The linear-attention layers and the expert combine have
    their own reduction orders, which no flag here pins.
    """
    if not flash_attention_determinism_requested():
        return False
    refusal = flash_attention_deterministic_backward_refusal(varlen_func, head_dim)
    if refusal is None:
        return True
    raise RuntimeError(
        "debug.full_determinism asks for a deterministic attention backward and the installed "
        f"{implementation} kernel refuses one at head_dim={head_dim}: {refusal.rstrip('. ')}. Either run this "
        "configuration with attn_implementation='sdpa', whose backward is deterministic at every head dimension, "
        "or drop debug.full_determinism and treat the run as not replayable."
    )
