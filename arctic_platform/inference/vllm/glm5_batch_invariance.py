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

"""Opt-in: let GLM-5.3-Flash (``glm5next``) run under ``VLLM_BATCH_INVARIANT=1``.

vLLM refuses batch-invariant mode for every sparse MLA backend. On Hopper ``FLASH_ATTN_MLA_SPARSE`` runs each query
token as its own one-token sequence over its top-k cached tokens, the same call in prefill and decode; the only
batch-dependent choice is FA3's split-KV heuristic, which ``num_splits=1`` turns off, as vLLM's dense FlashAttention
backend does in this mode. The KDA layers use the GatedDeltaNet backend (allowed by
:mod:`arctic_platform.inference.vllm.gdn_batch_invariance`) and their own autotuned Triton kernels, pinned here to
one config. The sparse indexer's backends (its fp8 key cache and kpool tail) are allowed as well: their kernels have
no autotuning, and whether their top-k selection is batch-invariant is measured, not assumed. The hyper-connection
pre-norm GEMM (DeepGEMM ``tf32_hc_prenorm_gemm``) splits its reduction over more blocks the fewer tokens a step has,
and small steps take a fused post + pre-norm kernel instead; every step runs the one-split unfused path here. The MoE
router's fp32 logits go through vLLM's batch-invariant matmul instead of a kernel chosen by token count. With
``ARCTIC_GLM5_BATCH_INVARIANT=1`` all of this is applied; the GDN patch is applied too.
"""

from __future__ import annotations

import os

import torch

ENV_VAR = "ARCTIC_GLM5_BATCH_INVARIANT"
_KDA_MODULES = (
    "vllm.models.glm5next.nvidia.ops.third_party.kda.kernels",
    "vllm.models.glm5next.nvidia.ops.third_party.kda.fused_recurrent",
)

_PINNED = False
_APPLIED = False


def glm5_batch_invariance_requested() -> bool:
    return os.environ.get(ENV_VAR, "0") == "1"


def pin_kda_kernels() -> int:
    """Pin GLM-5.3-Flash's KDA Triton autotuners to their first config; returns how many were pinned."""
    global _PINNED
    if _PINNED:
        return 0
    import importlib

    from triton.runtime.autotuner import Autotuner

    from arctic_platform.inference.vllm.gdn_batch_invariance import _pin_autotuner

    pinned = 0
    for name in _KDA_MODULES:
        module = importlib.import_module(name)
        for obj in list(vars(module).values()):
            pinned += _pin_autotuner(obj, Autotuner)
    _PINNED = True
    return pinned


def _single_split(flash_attn_varlen_func):
    def call(*args, **kwargs):
        kwargs["num_splits"] = 1
        return flash_attn_varlen_func(*args, **kwargs)

    call.__wrapped__ = flash_attn_varlen_func
    return call


def _batch_invariant_fp8_block_linear() -> None:
    """Run every block-FP8 linear layer on vLLM's Triton block-scaled GEMM.

    On CUDA vLLM picks ``FlashInferFp8DeepGEMMDynamicBlockScaledKernel``, which runs FlashInfer below 32 tokens and
    DeepGEMM from 32 on, so a token's projection depends on the step's size. The Triton kernel uses one fixed tile
    config under ``VLLM_BATCH_INVARIANT`` (vLLM already forces it on ROCm for that reason).
    """
    from vllm.model_executor.kernels import linear

    original = linear.choose_scaled_mm_linear_kernel

    def choose(config, possible_kernels, compute_capability=None, force_kernel=None, *, quantization):
        if quantization == "fp8_block_w8a8" and force_kernel is None:
            force_kernel = linear.TritonFp8BlockScaledMMKernel
        return original(config, possible_kernels, compute_capability, force_kernel, quantization=quantization)

    linear.choose_scaled_mm_linear_kernel = choose


def _batch_invariant_router_gate() -> None:
    """Compute the MoE router's fp32 logits with the batch-invariant matmul.

    ``GateLinear`` picks a kernel by token count (a CuteDSL kernel up to 16 tokens, a specialized one up to 32, then
    cuBLAS ``mm(out_dtype=fp32)``, which batch-invariant mode does not replace), so a token's routing scores depend
    on how many tokens share the step.
    """
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    from arctic_platform.inference.vllm.batch_invariant_fp32_logits import linear_fp32_out

    original = GateLinear.forward

    def forward(self, x):
        if self.out_dtype != torch.float32 or getattr(self, "bias", None) is not None:
            return original(self, x)
        if self.weight.dtype == torch.float32:
            output = torch.mm(x.float(), self.weight.t())
        else:
            output = linear_fp32_out(x.to(self.weight.dtype), self.weight)
        return self._return(output)

    GateLinear.forward = forward


def ensure_glm5_batch_invariance() -> None:
    global _APPLIED
    if _APPLIED or not glm5_batch_invariance_requested():
        return
    os.environ.setdefault("ARCTIC_GDN_BATCH_INVARIANT", "1")
    # The fused grouped top-k CUDA kernel's routing weights depend on the step; vLLM's torch path sorts for batch
    # invariance.
    os.environ.setdefault("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")
    from arctic_platform.inference.vllm.gdn_batch_invariance import ensure_gdn_batch_invariance

    ensure_gdn_batch_invariance()

    from vllm.v1.attention.backends.mla import flashattn_mla_sparse
    from vllm.v1.attention.backends.mla import indexer

    for backend in (
        flashattn_mla_sparse.FlashAttnMLASparseBackend,
        indexer.Glm5NextIndexerBackend,
        indexer.KpoolTailBackend,
    ):
        backend.supports_batch_invariance = classmethod(lambda cls: True)
    flashattn_mla_sparse.flash_attn_varlen_func = _single_split(flashattn_mla_sparse.flash_attn_varlen_func)
    from vllm.model_executor.kernels.mhc import tilelang_kernels
    from vllm.model_executor.kernels.mhc import warmup

    # The hyper-connection pre-norm GEMM splits its reduction by the step's token count; callers import these from
    # both modules at call time.
    for module in (tilelang_kernels, warmup):
        module.compute_num_split = lambda *args, **kwargs: 1
        module.compute_mhc_pre_num_splits = lambda *args, **kwargs: 1
    # Small steps would take a fused post + pre-norm kernel with its own split; every step takes the split-free path.
    tilelang_kernels.mhc_fused_post_pre_split_config = lambda *args, **kwargs: None
    _batch_invariant_router_gate()
    _batch_invariant_fp8_block_linear()
    pin_kda_kernels()
    _APPLIED = True
