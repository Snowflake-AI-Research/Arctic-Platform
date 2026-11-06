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

"""Opt-in: let GatedDeltaNet hybrids run under ``VLLM_BATCH_INVARIANT=1``.

vLLM refuses batch-invariant mode for the GDN backend. Its kernels are
row-local except for two shape-dependent choices: Triton autotuning in the
vendored FLA ops, and the gated RMSNorm's rows-per-block tiling. Pinning both
makes a token's result independent of how many tokens share the call, which
is what the ``vllm_aligned`` trainer forward relies on. The trainer calls
:func:`pin_gdn_kernels` too, so both processes run the same tile configs.
"""

import os

ENV_VAR = "ARCTIC_GDN_BATCH_INVARIANT"

_FLA_AUTOTUNED_MODULES = (
    "chunk_delta_h",
    "chunk_o",
    "chunk_scaled_dot_kkt",
    "cumsum",
    "l2norm",
    "solve_tril",
    "wy_fast",
)

_PINNED = False
_APPLIED = False


def gdn_batch_invariance_requested() -> bool:
    return os.environ.get(ENV_VAR, "0") == "1"


def _pin_autotuner(obj, autotuner_cls) -> int:
    pinned = 0
    seen: set[int] = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        if isinstance(obj, autotuner_cls) and len(obj.configs) > 1:
            obj.configs = obj.configs[:1]
            obj.cache.clear()
            pinned += 1
        obj = getattr(obj, "fn", None)
    return pinned


def pin_gdn_kernels() -> int:
    """Pin FLA autotuners to their first config and gated-norm tiling to one row."""
    global _PINNED
    if _PINNED:
        return 0
    import importlib

    from triton.runtime.autotuner import Autotuner

    pinned = 0
    for name in _FLA_AUTOTUNED_MODULES:
        module = importlib.import_module(f"vllm.third_party.flash_linear_attention.ops.{name}")
        for obj in list(vars(module).values()):
            pinned += _pin_autotuner(obj, Autotuner)

    from vllm.third_party.flash_linear_attention.ops import layernorm_guard

    layernorm_guard.calc_rows_per_block = lambda M, device: 1
    _PINNED = True
    return pinned


def ensure_gdn_batch_invariance() -> None:
    global _APPLIED
    if _APPLIED or not gdn_batch_invariance_requested():
        return
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend

    GDNAttentionBackend.supports_batch_invariance = classmethod(lambda cls: True)
    pin_gdn_kernels()
    _APPLIED = True
