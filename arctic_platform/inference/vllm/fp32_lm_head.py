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

"""FP32 LM head: run the lm_head matmul in fp32 without changing weights.

For RL workloads we need precise logits / log-probs at the LM head:
- the lm_head matmul must be computed in fp32 (not bf16/fp16), and
- the softmax must be in fp32.

vLLM's V1 sampler already runs softmax in fp32 (it casts logits to
fp32 before softmax — see ``vllm/v1/sample/sampler.py`` lines 90 and
207, and ``vllm/v1/sample/ops/topk_topp_sampler.py`` which uses
``softmax(..., dtype=torch.float32)``). The remaining bf16 op is the
LM-head matmul itself.

This module fixes that by patching ``LogitsProcessor._apply_head``, the
single chokepoint every logits path runs the projection through:

    logits = F.linear(hidden_states.to(fp32), lm_head.weight.to(fp32),
                      bias.to(fp32))

Notes:
- The lm_head **weight is not modified**: it stays in the model's
  native dtype (typically bf16) in GPU memory. Only the matmul
  operands are upcast on the fly.
- This bypasses ``lm_head.quant_method.apply`` (which would error on
  dtype mismatch). For unquantized lm_heads the original ``apply`` is
  just ``F.linear`` anyway, so we lose no functionality.
- For quantized lm_heads (rare) the patch falls back to the original
  ``_get_logits`` so quant kernels still run; the result is the
  upstream behavior, not fp32 logits.
- Per-step cost: a single ``vocab_size * hidden * 2`` byte read of the
  weight tensor through HBM (sub-ms on H100/MI300 for 128k-vocab x
  4k-hidden). No extra VRAM is used for an fp32 weight copy.

Enable via the env var ``ARCTIC_FP32_LM_HEAD=1`` or the CLI flag
``--fp32-lm-head``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from arctic_platform.inference.patching import ArcticPatch

logger = init_logger(__name__)

# A quant config that excludes lm_head hands out UnquantizedLinearMethod rather
# than UnquantizedEmbeddingMethod; either way lm_head.weight is a plain tensor.
_UNQUANTIZED_METHODS = (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
_BASE_ASYNC_ENGINE_ARGS_POST_INIT = getattr(
    AsyncEngineArgs, "_orig_post_init", AsyncEngineArgs.__post_init__)
_BASE_ASYNC_ENGINE_ARGS_CREATE_ENGINE_CONFIG = getattr(
    AsyncEngineArgs,
    "_orig_create_engine_config",
    AsyncEngineArgs.create_engine_config,
)

# Module-level toggle. Set to True (e.g. via the env var or CLI flag)
# *before* ``apply_fp32_lm_head_patches`` runs and *before* the model
# is constructed. The patches are always installed but are no-ops when
# this is False.
_FP32_LM_HEAD_ENABLED = False


@dataclass
class Fp32LmHeadAsyncEngineArgs(AsyncEngineArgs):
    """AsyncEngineArgs with only the fp32 lm_head Arctic extension field."""

    fp32_lm_head: bool = False

    def __post_init__(self):
        if self.fp32_lm_head:
            os.environ["ARCTIC_FP32_LM_HEAD"] = "1"
        _BASE_ASYNC_ENGINE_ARGS_POST_INIT(self)

    def create_engine_config(self, *args, **kwargs):
        return _BASE_ASYNC_ENGINE_ARGS_CREATE_ENGINE_CONFIG(
            self, *args, **kwargs)


def set_fp32_lm_head_enabled(enabled: bool) -> None:
    """Enable or disable fp32 lm_head globally."""
    global _FP32_LM_HEAD_ENABLED
    if enabled and not _FP32_LM_HEAD_ENABLED:
        logger.info("FP32 LM head enabled: lm_head matmul will run in "
                    "fp32 (weights stay bf16; on-the-fly upcast).")
    _FP32_LM_HEAD_ENABLED = enabled


def is_fp32_lm_head_enabled() -> bool:
    return _FP32_LM_HEAD_ENABLED


def _fp32_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """``F.linear`` with all operands promoted to fp32 (no-op if already fp32).

    The weight is upcast on the fly; the underlying Parameter is not
    modified.
    """
    if weight.dtype != torch.float32:
        weight = weight.to(torch.float32)
    if hidden_states.dtype != torch.float32:
        hidden_states = hidden_states.to(torch.float32)
    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    return torch.nn.functional.linear(hidden_states, weight, bias)


class LogitsProcessorFp32Patch(ArcticPatch[LogitsProcessor]):
    """Run the lm_head matmul in fp32 when the toggle is on.

    Patching the single projection chokepoint leaves the gather, padding mask,
    soft_cap, scale and TP reduction upstream, so every caller of
    ``_apply_head`` -- ``_get_logits``, ``get_top_tokens`` and
    ``get_top_k_tokens`` -- gets fp32 logits without this module having to
    track their signatures.
    """

    _orig_apply_head = LogitsProcessor._apply_head

    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        # Quantized lm_heads use a quant-specific ``apply`` that we cannot
        # replace with a plain matmul; fall back (no fp32 upcast for them).
        if (not _FP32_LM_HEAD_ENABLED
                or not isinstance(lm_head.quant_method, _UNQUANTIZED_METHODS)):
            return self._orig_apply_head(lm_head, hidden_states,
                                         embedding_bias)
        return _fp32_linear(hidden_states, lm_head.weight, embedding_bias)


def apply_fp32_lm_head_patches() -> None:
    """Install the fp32 lm_head patch.

    The patch is always installed (so the toggle can be flipped at
    runtime via ``set_fp32_lm_head_enabled(True)`` before model
    construction) but is a no-op when the toggle is False.
    """
    LogitsProcessorFp32Patch.apply_patch()


def ensure_fp32_lm_head_vllm_patches(enabled: bool | None = None) -> None:
    """Install the fp32 lm_head patch without the full Arctic patch stack."""
    if enabled is None:
        enabled = os.getenv("ARCTIC_FP32_LM_HEAD", "0") == "1"
    set_fp32_lm_head_enabled(bool(enabled))

    patches = getattr(LogitsProcessor, "_arctic_patches", {})
    existing = patches.get("_apply_head")
    if existing is None:
        apply_fp32_lm_head_patches()
    elif existing is not LogitsProcessorFp32Patch:
        raise RuntimeError(
            "LogitsProcessor._apply_head is already patched by "
            f"{existing.__name__}; cannot install fp32 lm_head patch."
        )
