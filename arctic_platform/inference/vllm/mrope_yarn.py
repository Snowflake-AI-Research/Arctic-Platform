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

from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

from arctic_platform.inference.patching import ArcticPatch


# Remove once the vLLM pin includes https://github.com/vllm-project/vllm/pull/58879.
class MRotaryEmbeddingPatch(ArcticPatch[MRotaryEmbedding]):

    _orig_compute_inv_freq = MRotaryEmbedding._compute_inv_freq

    def _compute_inv_freq(self, base):
        if self.scaling_factor is None:
            return self._orig_compute_inv_freq(base)
        cache_max_position_embeddings = self.max_position_embeddings
        self.max_position_embeddings //= 4
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
