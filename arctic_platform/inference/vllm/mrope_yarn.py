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

from contextvars import ContextVar

from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

from arctic_platform.inference.patching import ArcticPatch

# Set for the duration of MRotaryEmbedding.__init__. _compute_inv_freq runs
# inside that call, before the instance can store the factor itself.
_POSITIONS_FACTOR: ContextVar[int | None] = ContextVar(
    "arctic_mrope_positions_factor", default=None
)


# YaRN's correction range reads max_position_embeddings, but MRotaryEmbedding
# passes the enlarged cache length (original * positions_factor) into the base
# class. vLLM 0.32 uses 4 only when multimodal pruning needs positions past the
# token count, and 1 otherwise. Divide by that factor while computing inv_freq
# so the range stays the model's real length. The cache length is unchanged.
# Remove once the vLLM pin includes
# https://github.com/vllm-project/vllm/pull/58879.
class MRotaryEmbeddingPatch(ArcticPatch[MRotaryEmbedding]):

    _orig_init = MRotaryEmbedding.__init__
    _orig_compute_inv_freq = MRotaryEmbedding._compute_inv_freq

    def __init__(self, *args, positions_factor: int = 4, **kwargs):
        token = _POSITIONS_FACTOR.set(positions_factor)
        try:
            self._orig_init(*args, positions_factor=positions_factor, **kwargs)
        finally:
            _POSITIONS_FACTOR.reset(token)
        self._arctic_mrope_positions_factor = positions_factor

    def _compute_inv_freq(self, base):
        if self.scaling_factor is None:
            return self._orig_compute_inv_freq(base)
        factor = getattr(self, "_arctic_mrope_positions_factor", None)
        if factor is None:
            factor = _POSITIONS_FACTOR.get()
        # Instances built before this patch, and the historical M-RoPE cache,
        # were enlarged by 4.
        if factor is None:
            factor = 4
        cache_max_position_embeddings = self.max_position_embeddings
        self.max_position_embeddings //= factor
        try:
            return self._orig_compute_inv_freq(base)
        finally:
            self.max_position_embeddings = cache_max_position_embeddings


_PATCHED = False


def ensure_mrope_yarn_fix() -> None:
    global _PATCHED
    if not _PATCHED:
        MRotaryEmbeddingPatch.apply_patch()
        _PATCHED = True
