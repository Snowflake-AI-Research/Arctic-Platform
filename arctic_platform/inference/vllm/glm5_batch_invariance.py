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
pre-norm GEMM (DeepGEMM ``tf32_hc_prenorm_gemm``) splits its reduction over more blocks the fewer tokens a step has;
it runs one split here. With
``ARCTIC_GLM5_BATCH_INVARIANT=1`` all of this is applied; the GDN patch is applied too.
"""

from __future__ import annotations

import os

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


def ensure_glm5_batch_invariance() -> None:
    global _APPLIED
    if _APPLIED or not glm5_batch_invariance_requested():
        return
    os.environ.setdefault("ARCTIC_GDN_BATCH_INVARIANT", "1")
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

    # The hyper-connection pre-norm GEMM splits its reduction by the step's token count; callers import these at
    # call time.
    tilelang_kernels.compute_num_split = lambda *args, **kwargs: 1
    tilelang_kernels.compute_mhc_pre_num_splits = lambda *args, **kwargs: 1
    pin_kda_kernels()
    _APPLIED = True
